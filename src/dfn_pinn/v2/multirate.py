"""Multi-protocol inverse problem: one DFNPINN per C-rate, shared physical parameters.

Each protocol (current, ramp, duration) gets its own networks and scales, so
the per-model formulation is exactly the one of ``train.py``; only the
log-multipliers of the physical parameters (``DFNPINN.log_mult``) are shared
between the models. The loss is the sum over protocols of the physics loss
(adaptive group weights per model) plus a voltage-data term per protocol.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from .evaluate import compare
from .model import DFNPINN
from .params import CellParams, Protocol, Scales
from .residuals import Residuals
from .train import (GROUPS, AdaptiveWeights, Sampler, TrainConfig, residual_terms, term_losses,
                    total_loss)


def build_models(cfg: TrainConfig, cell: CellParams, protocols, dtype):
    models = []
    for k, protocol in enumerate(protocols):
        torch.manual_seed(cfg.seed + 101 * k)
        m = DFNPINN(Scales(cell, protocol), cfg.width, cfg.depth, cfg.act, cfg.projection,
                    ic_tau_s=cfg.ic_tau_s, inventory=cfg.inventory, fourier_t=cfg.fourier_t,
                    width_scalar=cfg.width_scalar or None, short_t=tuple(cfg.short_t),
                    collector_bc=cfg.collector_bc, fourier_period=cfg.fourier_period).to(dtype)
        if models:
            m.log_mult = models[0].log_mult        # share the physical parameters
        models.append(m)
    return models


def train_multirate(cfg: TrainConfig, cell: CellParams, protocols, data_paths, out_dir: Path,
                    references=None, log=print, init_states=None, resume=None):
    """protocols: list of Protocol; data_paths: npz files with t [s] and V [V] per protocol.

    resume: a checkpoint written by this function (models, optimizer, step, weights) to continue
    an interrupted run in place (same config, same out_dir layout).
    """
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    models = build_models(cfg, cell, protocols, dtype)
    if init_states:
        for m, state in zip(models, init_states):
            if state is not None:
                m.load_state_dict(state, strict=False)
    shared = models[0]
    shared.set_trainable_parameters(cfg.inverse_params, cfg.inverse_init)
    log(f"Multi-rate inverse: {len(models)} protocols "
        + ", ".join(f"{p.current_A:g} A / {p.t_end_s:g} s" for p in protocols))
    log(f"Learning {cfg.inverse_params}; initial {shared.parameter_values()}")

    residuals, samplers, weights, data = [], [], [], []
    for k, (m, path) in enumerate(zip(models, data_paths)):
        res = Residuals(m, cfg.kinetics)
        res.salt_scale = cfg.salt_scale_mol_m3 if cfg.salt_conservation else None
        res.soft_current = cfg.soft_current and not cfg.projection
        residuals.append(res)
        samplers.append(Sampler(cfg, dtype, torch.Generator().manual_seed(cfg.seed + 12345 + 7 * k)))
        weights.append(AdaptiveWeights(GROUPS, cfg.adaptive_alpha))
        raw = np.load(path, allow_pickle=False)
        t_d, V_d = np.asarray(raw["t"], float), np.asarray(raw["V"], float)
        if cfg.data_noise_mV > 0:
            V_d = V_d + np.random.default_rng(cfg.seed + 7 + 13 * k).normal(0.0, cfg.data_noise_mV * 1e-3, V_d.shape)
        data.append((torch.as_tensor(t_d / m.sc.t_end, dtype=dtype).view(-1, 1),
                     torch.as_tensor(V_d, dtype=dtype).view(-1, 1)))

    net_params = {}
    for m in models:
        for name, p in m.named_parameters():
            if not name.startswith("log_mult"):
                net_params[id(p)] = p
    phys_params = [shared.log_mult[n] for n in cfg.inverse_params]
    groups = [{"params": list(net_params.values()), "lr": cfg.lr}]
    if phys_params:
        groups.append({"params": phys_params, "lr": cfg.param_lr})
    opt = torch.optim.Adam(groups)
    gamma = (cfg.lr_final / cfg.lr) ** (1 / max(cfg.adam_steps, 1))
    start_step = 1
    history, evals = [], []
    if resume is not None:
        ck = torch.load(resume, weights_only=False)
        for m, state in zip(models, ck["models"]):
            m.load_state_dict(state, strict=False)
        opt.load_state_dict(ck["optimizer"])
        for aw, w in zip(weights, ck.get("group_weights", [])):
            aw.w.update(w)
        start_step = int(ck.get("step", 0)) + 1
        # closed-form learning rate for the resumed step (re-stepping a fresh scheduler on the stored,
        # already-decayed lr decayed it twice: the f4_multirate3 run of 2026-10-04 trained with lr x 0.4
        # from step 1000 on because of this; see docs/V2_RESULTS.md 3.3)
        for group, base in zip(opt.param_groups, [cfg.lr, cfg.param_lr]):
            group["lr"] = base * gamma ** (start_step - 1)
        prev = Path(resume).parent / "history.json"
        if prev.exists():
            saved = json.loads(prev.read_text())
            history, evals = saved.get("history", []), saved.get("evals", [])
        log(f"Resumed from {resume} at step {start_step} (lr {opt.param_groups[0]['lr']:.3e}); "
            f"parameters {shared.parameter_values()}")
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)   # continues from the current lr

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "train": cfg.to_dict(), "cell": cell.to_dict(), "protocols": [p.to_dict() for p in protocols],
        "data_paths": [str(p) for p in data_paths],
        "scales": [m.sc.summary() for m in models]}, indent=2))
    t0 = time.perf_counter() - (history[-1]["time_s"] if history else 0.0)   # wall time continues on resume

    def save(name, extra=None):
        torch.save({"models": [m.state_dict() for m in models], "optimizer": opt.state_dict(), "step": current_step[0],
                    "train": cfg.to_dict(), "cell": cell.to_dict(), "protocols": [p.to_dict() for p in protocols],
                    "group_weights": [w.w for w in weights], "parameters": shared.parameter_values(),
                    **(extra or {})}, out_dir / name)

    def evaluate(step):
        if references is None:
            return None
        rec = {"step": step}
        for k, (m, ref) in enumerate(zip(models, references)):
            if ref is None:
                continue
            metrics, _ = compare(m, ref, max_times=60, kinetics=cfg.kinetics)
            label = f"{protocols[k].current_A:g}A"
            rec[label] = metrics
            log(f"  eval@{step} {label}: V rmse {metrics['V_rmse_mV']:.2f} mV (max {metrics['V_max_mV']:.2f}); "
                f"ce max {metrics['ce_max']:.1f}; j rmse rel n {metrics['j_n_rmse_rel']:.3f} p {metrics['j_p_rmse_rel']:.3f}")
        evals.append(rec)
        return rec

    current_step = [start_step - 1]
    for step in range(start_step, cfg.adam_steps + 1):
        current_step[0] = step
        opt.zero_grad(set_to_none=True)
        step_losses, misfits, total = {}, {}, 0.0
        for k, (m, res, sampler, aw) in enumerate(zip(models, residuals, samplers, weights)):
            batch = sampler.draw()
            terms = residual_terms(res, batch, cell)
            terms["data_V"] = (m.terminal_voltage(data[k][0]) - data[k][1]) / (cfg.data_scale_mV * 1e-3)
            losses = term_losses(terms)
            if cfg.adaptive and (step == 1 or step % cfg.adaptive_every == 0):
                aw.update(m, losses)
            loss_k = total_loss(losses, aw.w if cfg.adaptive else cfg.fixed_group_weights, cfg.weights)
            loss_k.backward()                      # gradients accumulate in the shared parameters
            total += float(loss_k.detach())
            label = f"{protocols[k].current_A:g}A"
            step_losses[label] = {n: float(v.detach()) for n, v in losses.items()}
            misfits[label] = float(losses["data_V"].detach()) ** 0.5 * cfg.data_scale_mV
        for n, p in zip(cfg.inverse_params, phys_params):   # fields first, parameters later (hierarchical release)
            if step <= max(cfg.param_warmup_steps, cfg.param_release_steps.get(n, 0)):
                p.grad = None                      # Adam skips a parameter without gradient
        opt.step()
        sched.step()
        with torch.no_grad():
            for n, p in zip(cfg.inverse_params, phys_params):
                b = cfg.param_bounds.get(n, cfg.param_log_bound)
                    p.clamp_(0.0 if n == "R0" else -b, b)   # R0 >= 0; per-parameter bound overrides
        if step % cfg.log_every == 0 or step == 1:
            params = shared.parameter_values()
            rec = {"step": step, "loss": total, "time_s": time.perf_counter() - t0, "losses": step_losses,
                   "misfit_mV": misfits, "parameters": params, "w": [dict(w.w) for w in weights]}
            history.append(rec)
            log(f"step {step:6d} loss {total:.3e} | misfit mV " + " ".join(f"{k} {v:.2f}" for k, v in misfits.items())
                + " | " + " ".join(f"{n}={params[n]:.4g}" for n in cfg.inverse_params) + f" | {rec['time_s']:.0f}s")
        if step % cfg.eval_every == 0:
            evaluate(step)
        if cfg.latest_every and step % cfg.latest_every == 0:
            save("latest.pt")                      # rolling checkpoint for --resume
            (out_dir / "history.json").write_text(json.dumps({"history": history, "evals": evals}, indent=1))
        if step % cfg.checkpoint_every == 0:
            save(f"adam_{step:06d}.pt")
            (out_dir / "history.json").write_text(json.dumps({"history": history, "evals": evals}, indent=1))

    final = evaluate(cfg.adam_steps)
    save("final.pt", {"final_eval": final})
    (out_dir / "history.json").write_text(json.dumps({"history": history, "evals": evals}, indent=1))
    return models, history, evals
