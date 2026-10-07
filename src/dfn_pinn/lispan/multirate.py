"""Multi-rate inverse problem for the Li-SPAN PINN: one LiSPANPINN per discharge rate, all sharing a single
set of physical parameters (the ParameterDict ``log_mult``), trained jointly on the physics residuals of
every rate and on the voltage data of every rate.

Why: the identifiability study (docs/LI_SPAN_MODEL_FORMULATION.md 6.5) shows that a single rate cannot
separate the reaction kinetics (Tafel shift ~ ln I) from the ohmic terms (~ I) once the OCV parameters are
free; 0.1 C + 1 C data make the 8-parameter set {k0_m, b_m, U0_1, Z_CC} identifiable to < 1 %.

Usage (stage 1 = scripts/lispan_train.py with ``fixed_multipliers`` per rate, from forward checkpoints):

    python scripts/lispan_inverse_multirate.py --config configs/lispan_inverse_multirate.json
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from ..v2.train import AdaptiveWeights
from .params import LiSPANParams, LiSPANProtocol
from .pinn import GROUPS, LiSPANPINN, LiSPANTrainConfig, Sampler, evaluate, huber, residuals


def rate_tag(protocol: LiSPANProtocol) -> str:
    return f"{protocol.crate:g}C"


def build_models(cfg: LiSPANTrainConfig, params: LiSPANParams, protocols, t_ends, dtype):
    models = nn.ModuleList()
    for prot, t_end in zip(protocols, t_ends):
        m = LiSPANPINN(params, prot, t_end, cfg.width, cfg.depth, cfg.act, cfg.fourier_t, cfg.fourier_period,
                       tuple(cfg.short_t), cfg.ic_tau_s, cfg.quad_order).to(dtype)
        m.salt_total_scale = cfg.salt_total_scale
        m.charge_total_scale = cfg.charge_total_scale
        if cfg.salt_sep_natural_scale:
            m.salt_scale_sep = m.salt_scale * params.L_cat / params.L_sep
        models.append(m)
    shared = models[0].log_mult
    for m in models[1:]:
        m.log_mult = shared          # one parameter set for all rates
    return models


def load_data(path, t_end, noise_mV, seed, dtype):
    raw = np.load(path, allow_pickle=False)
    t_d, V_d = np.asarray(raw["t"], float), np.asarray(raw["V"], float)
    if noise_mV > 0:
        V_d = V_d + np.random.default_rng(seed + 7).normal(0.0, noise_mV * 1e-3, V_d.shape)
    keep = t_d <= t_end
    truth = json.loads(str(raw["truth_json"])) if "truth_json" in raw.files else {}
    return (torch.as_tensor(t_d[keep] / t_end, dtype=dtype).view(-1, 1),
            torch.as_tensor(V_d[keep], dtype=dtype).view(-1, 1)), truth


def gauss_newton_parameters(models, data, names, cfg, log, h=1e-2):
    """Levenberg-Marquardt iterations on the physical parameters with the networks frozen.

    The parameters enter the PINN voltage directly (root solve for Delta phi, contact resistance, electrolyte
    conductivity) and through the fields; with the fields frozen the data residuals are cheap to evaluate, so
    a finite-difference Jacobian (2 x n_params voltage evaluations) and an LM step resolve correlated
    parameter directions (e.g. k0_1 - U0_1, corr -0.88) in a few iterations, where Adam crawls along the
    valley.  Returns the rms data misfit [mV] before and after."""
    lm = models[0].log_mult

    def resid(theta):
        with torch.no_grad():
            for n, v in zip(names, theta):
                lm[n].fill_(float(v))
            r = torch.cat([(m.voltage(d[0]) - d[1]).view(-1) for m, d in zip(models, data)]) / 1e-3
        return r.double()

    theta = torch.tensor([float(lm[n]) for n in names], dtype=torch.float64)
    r = resid(theta)
    cost0 = cost = float((r ** 2).mean())
    for _ in range(cfg.gn_iters):
        J = torch.stack([(resid(theta + h * e) - resid(theta - h * e)) / (2 * h)
                         for e in torch.eye(len(names), dtype=torch.float64)], dim=1)
        g = J.T @ r
        A = J.T @ J
        lam, accepted = 1e-3, False
        for _ in range(8):
            delta = torch.linalg.solve(A + lam * torch.diag(torch.diag(A)) + 1e-12 * torch.eye(len(names), dtype=A.dtype), -g)
            delta = delta.clamp(-cfg.gn_max_step, cfg.gn_max_step)
            trial = (theta + delta).clamp(-cfg.param_log_bound, cfg.param_log_bound)
            r_new = resid(trial)
            c_new = float((r_new ** 2).mean())
            if c_new < cost:
                theta, r, cost, accepted = trial, r_new, c_new, True
                break
            lam *= 10.0
        if not accepted:
            break
    resid(theta)      # leave the accepted values in place
    return cost0 ** 0.5, cost ** 0.5


class TrustRegionLM:
    """Levenberg-Marquardt on the physical parameters with acceptance on the RE-EQUILIBRATED misfit.

    With model-form error the frozen-field Jacobian (no field response d u*/d theta) points away from the true
    optimum and repeated frozen-field steps drift (k0 x 6, misfit 18 -> 24 mV on the measured Li-SPAN data).  Here a
    proposal theta_ref + delta is followed by gn_every physics-only training steps of the fields; the step is kept
    if the misfit then is below the reference misfit, otherwise fields and parameters are restored and the damping
    is raised (x 4); accepted steps lower it (/ 3)."""

    def __init__(self, models, data, names, cfg, log, h=1e-2):
        self.models, self.data, self.names, self.cfg, self.log, self.h = models, data, names, cfg, log, h
        self.lm = models[0].log_mult
        self.lam, self.ref, self.pending = 1e-2, None, False

    def _set(self, theta):
        with torch.no_grad():
            for n, v in zip(self.names, theta):
                self.lm[n].fill_(float(v))

    def _resid(self, theta=None):
        if theta is not None:
            self._set(theta)
        with torch.no_grad():
            r = torch.cat([(m.voltage(d[0]) - d[1]).view(-1) for m, d in zip(self.models, self.data)]) / 1e-3
        return r.double()

    def _theta(self):
        return torch.tensor([float(self.lm[n].detach()) for n in self.names], dtype=torch.float64)

    def _fields(self):
        return [{k: v.detach().clone() for k, v in m.state_dict().items() if "log_mult" not in k} for m in self.models]

    def _restore(self, fields):
        for m, st in zip(self.models, fields):
            m.load_state_dict(st, strict=False)

    def step(self, step):
        theta = self._theta()
        cur = float((self._resid() ** 2).mean()) ** 0.5
        verdict = "start"
        if self.ref is not None and self.pending:
            if cur <= self.ref[1]:
                verdict = f"accepted ({self.ref[1]:.3f} -> {cur:.3f} mV)"
                self.ref = (theta, cur, self._fields())
                self.lam = max(self.lam / 3.0, 1e-4)
            else:
                verdict = f"rejected ({self.ref[1]:.3f} -> {cur:.3f} mV)"
                self._restore(self.ref[2]); self._set(self.ref[0])
                self.lam = min(self.lam * 4.0, 1e4)
        else:
            self.ref = (theta, cur, self._fields())
        theta0, ref_misfit = self.ref[0], self.ref[1]
        r = self._resid(theta0)
        J = torch.stack([(self._resid(theta0 + self.h * e) - self._resid(theta0 - self.h * e)) / (2 * self.h)
                         for e in torch.eye(len(self.names), dtype=torch.float64)], dim=1)
        A, g = J.T @ J, J.T @ r
        delta = torch.linalg.solve(A + self.lam * torch.diag(torch.diag(A)) + 1e-12 * torch.eye(len(self.names), dtype=A.dtype), -g)
        delta = delta.clamp(-self.cfg.gn_max_step, self.cfg.gn_max_step)
        trial = (theta0 + delta).clamp(-self.cfg.param_log_bound, self.cfg.param_log_bound)
        frozen = float((self._resid(trial) ** 2).mean()) ** 0.5
        self.pending = True
        vals = self.models[0].parameter_values()
        self.log(f"  TR step {step}: {verdict}; reference misfit {ref_misfit:.3f} mV, lambda {self.lam:.3g}, proposal "
                 f"(frozen-field misfit {frozen:.3f} mV): " + ", ".join(f"{n} {vals[n]:.4g}" for n in self.names))


def train_multirate(cfg: LiSPANTrainConfig, params: LiSPANParams, protocols, t_ends, data_paths, out_dir: Path,
                    init_states=None, references=None, log=print, resume=None):
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    torch.manual_seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    models = build_models(cfg, params, protocols, t_ends, dtype)
    tags = [rate_tag(p) for p in protocols]
    if init_states:
        for m, st in zip(models, init_states):
            if st is not None:
                m.load_state_dict(st, strict=False)
    # data: a different noise draw per rate (seed offset by the rate index)
    data, truth = [], {}
    for k, (path, m) in enumerate(zip(data_paths, models)):
        d, tr = load_data(path, m.t_end, cfg.data_noise_mV, cfg.seed + 100 * k, dtype)
        data.append(d)
        truth.update(tr)
    models[0].set_trainable_parameters(cfg.inverse_params, cfg.inverse_init)
    log(f"Multi-rate inverse: rates {tags}, learning {cfg.inverse_params}, initial "
        f"{ {n: round(v, 5) for n, v in models[0].parameter_values().items() if n in cfg.inverse_params} }")
    sampler = Sampler(cfg, dtype, gen)
    net_params = [p_ for n, p_ in models.named_parameters() if "log_mult" not in n]
    phys_params = [models[0].log_mult[n] for n in cfg.inverse_params]
    opt = torch.optim.Adam([{"params": net_params, "lr": cfg.lr}, {"params": phys_params, "lr": cfg.param_lr}])
    gamma = (cfg.lr_final / cfg.lr) ** (1 / max(cfg.adam_steps, 1))
    gamma_p = (cfg.param_lr_final / cfg.param_lr) ** (1 / max(cfg.adam_steps, 1)) if cfg.param_lr_final > 0 else gamma
    group_names = {}
    for tag in tags:
        for g, names in GROUPS.items():
            group_names[f"{tag}:{g}"] = [f"{tag}:{n}" for n in names]
        group_names[f"{tag}:data"] = [f"{tag}:data_V"]
    aw = AdaptiveWeights(group_names, cfg.adaptive_alpha)
    history, evals, start_step = [], [], 1
    if resume is not None:
        ck = torch.load(resume, weights_only=False)
        for m, st in zip(models, ck["models"]):
            m.load_state_dict(st, strict=False)
        opt.load_state_dict(ck["optimizer"])
        aw.w.update(ck.get("group_weights", {}))
        start_step = int(ck.get("step", 0)) + 1
        prev = Path(resume).parent / "history.json"
        if prev.exists():
            saved = json.loads(prev.read_text()); history, evals = saved.get("history", []), saved.get("evals", [])
        log(f"Resumed from {resume} at step {start_step}")

    def set_lr(step):
        opt.param_groups[0]["lr"] = cfg.lr * gamma ** (step - 1)
        opt.param_groups[1]["lr"] = cfg.param_lr * gamma_p ** (step - 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter() - (history[-1]["time_s"] if history else 0.0)
    step_holder = [start_step - 1]

    def save(name):
        torch.save({"models": [m.state_dict() for m in models], "optimizer": opt.state_dict(), "train": cfg.to_dict(),
                    "params": params.to_dict(), "protocols": [p.to_dict() for p in protocols], "t_ends": list(t_ends),
                    "group_weights": aw.w, "parameters": models[0].parameter_values(), "step": step_holder[0]},
                   out_dir / name)
        (out_dir / "history.json").write_text(json.dumps({"history": history, "evals": evals}))

    n_bad = 0
    tr = None
    for step in range(start_step, cfg.adam_steps + 1):
        step_holder[0] = step
        set_lr(step)
        if cfg.gn_mode == "tr" and cfg.gn_every > 0 and (step == start_step or step % cfg.gn_every == 0) \
                and (cfg.gn_until <= 0 or step <= cfg.gn_until):
            if tr is None:
                tr = TrustRegionLM(models, data, cfg.inverse_params, cfg, log)
            tr.step(step)
        elif cfg.gn_mode != "tr" and cfg.gn_every > 0 and cfg.gn_iters > 0 and (step == 1 or step % cfg.gn_every == 0) \
                and (cfg.gn_until <= 0 or step <= cfg.gn_until):
            before, after = gauss_newton_parameters(models, data, cfg.inverse_params, cfg, log)
            for p_ in phys_params:              # stale Adam moments would pull the parameters back
                opt.state.pop(p_, None)
            log(f"  LM step {step}: data misfit {before:.3f} -> {after:.3f} mV rms | "
                + ", ".join(f"{n} {v:.4g}" for n, v in models[0].parameter_values().items() if n in cfg.inverse_params))
        losses = {}
        for tag, m, d in zip(tags, models, data):
            terms = residuals(m, sampler.draw())
            for k, v in terms.items():
                losses[f"{tag}:{k}"] = (huber(v, cfg.huber_delta) if k in ("charge_s", "charge_e") else v.square()).mean()
            losses[f"{tag}:data_V"] = ((m.voltage(d[0]) - d[1]) / (1e-3 * cfg.data_scale_mV)).square().mean()
        if cfg.adaptive and (step == start_step or step % cfg.adaptive_every == 0):
            aw.update(models, losses)
        loss = sum(aw.w[g] * sum(losses[n] for n in names if n in losses) for g, names in group_names.items())
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            n_bad += 1
            if n_bad <= 5:
                log(f"step {step}: non-finite loss, batch skipped")
            continue
        if cfg.data_to_fields:
            loss.backward()
        else:
            # physics residuals train the fields (and see the parameters); the data misfit moves only the parameters,
            # so the fields remain a physics-consistent solution at the current parameters even when the model
            # cannot reproduce the data (model-form error), instead of being bent towards the data
            data_loss = sum(aw.w[f"{tag}:data"] * losses[f"{tag}:data_V"] for tag in tags)
            # ... and the parameters only by the data (a physics gradient on the parameters would make them chase
            # the current fields instead of the data: drift observed in lispan_inv8p_experiment_physfields_*)
            gd = torch.autograd.grad(data_loss, phys_params, retain_graph=True, allow_unused=True)
            (loss - data_loss).backward()
            for p_, g_ in zip(phys_params, gd):
                p_.grad = None if cfg.gn_mode == "tr" else g_      # trust-region mode: parameters move by LM steps only
        if any((p_.grad is not None and not torch.isfinite(p_.grad).all()) for p_ in net_params + phys_params):
            n_bad += 1
            if n_bad <= 5:
                log(f"step {step}: non-finite gradients, batch skipped")
            opt.zero_grad(set_to_none=True)
            continue
        if step <= cfg.param_warmup_steps:
            for p_ in phys_params:
                p_.grad = None
        if step <= cfg.net_freeze_steps:
            # parameters first, with the stage-1 fields frozen: a large initial misfit (e.g. 125 mV at 1 C from a
            # 1.5x contact resistance) is otherwise absorbed by the electrolyte fields before the parameters move
            for p_ in net_params:
                p_.grad = None
        torch.nn.utils.clip_grad_norm_(net_params, 10.0)
        opt.step()
        with torch.no_grad():
            for n in cfg.inverse_params:
                models[0].log_mult[n].clamp_(-cfg.param_log_bound, cfg.param_log_bound)
        rec = {"step": step, "loss": float(loss.detach()), "time_s": time.perf_counter() - t0,
               **{k: float(v.detach()) for k, v in losses.items()}, "params": models[0].parameter_values()}
        history.append(rec)
        if step % cfg.log_every == 0 or step == start_step:
            msg = (f"step {step:6d} loss {rec['loss']:.3e} | "
                   + " ".join(f"{tag} data {rec[tag + ':data_V']:.2e} ode {rec[tag + ':ode1'] + rec[tag + ':ode2'] + rec[tag + ':ode3']:.1e}"
                              f" salt {rec[tag + ':salt_c'] + rec[tag + ':salt_s']:.1e}" for tag in tags)
                   + f" | {rec['time_s']:.0f}s | "
                   + ", ".join(f"{n} {rec['params'][n]:.4g}" for n in cfg.inverse_params))
            log(msg)
        if references is not None and (step % cfg.eval_every == 0 or step == cfg.adam_steps):
            ev = {"step": step}
            for tag, m, ref in zip(tags, models, references):
                if ref is not None:
                    ev.update({f"{tag}:{k}": v for k, v in evaluate(m, ref, dtype).items()})
            evals.append(ev)
            log("  eval: " + "; ".join(f"{tag} V {ev.get(tag + ':V_rmse_mV', float('nan')):.3f} mV rms / "
                                        f"{ev.get(tag + ':V_max_mV', float('nan')):.2f} max, c_e {ev.get(tag + ':c_e_rmse', float('nan')):.2f}"
                                        for tag in tags))
        if cfg.latest_every and step % cfg.latest_every == 0:
            save("latest.pt")
        if cfg.checkpoint_every and step % cfg.checkpoint_every == 0:
            save(f"step_{step}.pt")
    save("final.pt")
    est = models[0].parameter_values()
    result = {"rates": tags, "estimates": {n: est[n] for n in cfg.inverse_params},
              "truth": {n: truth.get(n) for n in cfg.inverse_params}, "initial": cfg.inverse_init,
              "data": list(data_paths), "noise_mV": cfg.data_noise_mV, "final_eval": evals[-1] if evals else {}}
    result["errors_pct_or_mV"] = {n: (1e3 * (est[n] - truth[n]) if n.startswith("U0") else 100.0 * (est[n] / truth[n] - 1.0))
                                  for n in cfg.inverse_params if truth.get(n) is not None}
    (out_dir / "inverse_result.json").write_text(json.dumps(result, indent=1))
    log("Estimates: " + ", ".join(f"{n} {est[n]:.4g}" for n in cfg.inverse_params))
    log("Errors vs truth [% (mV for U0)]: " + ", ".join(f"{n} {v:+.2f}" for n, v in result["errors_pct_or_mV"].items()))
    return models, history, evals
