"""Training loop for the v2 DFN PINN (forward problem)."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .model import DFNPINN
from .params import CellParams, Protocol, Scales
from .residuals import Residuals


@dataclass
class TrainConfig:
    seed: int = 0
    dtype: str = "float32"
    threads: int = 0                 # 0 -> torch default
    width: int = 64
    depth: int = 4
    act: str = "tanh"
    kinetics: str = "inverse"        # "inverse", "direct" or "hard" (phi_e defined by inverse BV in the electrodes)
    projection: bool = True
    collector_bc: str = "soft"       # "hard": zero-slope electrolyte x-features at the collectors (model._xe)
    fourier_period: float = 1.0      # period of the first Fourier time feature in units of t_end (2 = no aliasing of t_end with 0)
    inventory: str = "soft"          # "soft": learned theta_bar + residual; "derived": j from d theta_bar/dt
    fourier_t: int = 0               # number of Fourier time-feature frequencies (0 = none)
    width_scalar: int = 0            # width of the scalar/1-D networks (0 = width // 2)
    short_t: list = field(default_factory=list)   # short-time feature constants tau [s], e.g. [50, 200]
    early2_fraction: float = 0.0     # extra fraction of time samples in [0, early2_window]
    early2_window: float = 0.02      # in t_hat (0.02 = 60 s)
    ic_tau_s: float = 60.0
    n_electrode: int = 512           # collocation points per electrode per step
    n_separator: int = 128
    n_particle: int = 1024           # per particle domain per step
    n_radial: int = 8                # radial points sharing one (x, t) pair
    n_boundary: int = 128
    early_fraction: float = 0.3      # fraction of times drawn in [0, early_window]
    early_window: float = 0.1        # in t_hat
    x_edge_fraction: float = 0.0     # fraction of electrode x samples near the separator
    x_edge_width: float = 0.15       # width of that band (local coordinate)
    x_collector_fraction: float = 0.0  # fraction of electrode x samples near the current collector
    x_collector_width: float = 0.1     # width of that band (local coordinate)
    adam_steps: int = 20000
    lr: float = 2e-3
    lr_final: float = 2e-5
    weights: dict = field(default_factory=dict)  # optional per-term weights
    adaptive: bool = True            # gradient-norm-based term-group balancing
    fixed_group_weights: dict = field(default_factory=dict)  # used when adaptive is False
    causal: bool = False             # causal time weighting (Wang et al. 2022)
    causal_bins: int = 16
    causal_tol: float = 0.99
    adaptive_every: int = 250
    adaptive_alpha: float = 0.9
    lbfgs_iters: int = 0
    lbfgs_batch_mult: int = 4
    lbfgs_resample_every: int = 25   # iterations per collocation batch
    lbfgs_history: int = 50
    lbfgs_eval_every: int = 200      # iterations between reference evaluations (0 = only at the end)
    log_every: int = 250
    eval_every: int = 2000
    checkpoint_every: int = 5000     # numbered checkpoints adam_XXXXXX.pt
    latest_every: int = 250          # rolling latest.pt (model + optimizer + step) for --resume
    # ---- inverse problem (voltage data) ----
    inverse_params: list = field(default_factory=list)   # e.g. ["D_p", "k_n"]
    inverse_init: dict = field(default_factory=dict)     # initial multipliers, e.g. {"D_p": 3.0}
    data_path: str = ""                                   # npz with t [s] and V [V]; "" -> forward problem
    data_noise_mV: float = 0.0                            # synthetic Gaussian noise added to V
    data_scale_mV: float = 1.0                            # data residual = (V_pred - V_data) / data_scale
    param_lr: float = 5e-3
    param_warmup_steps: int = 0      # keep the physical parameters frozen for the first N steps
    param_release_steps: dict = field(default_factory=dict)  # per-parameter release step, e.g. {"D_e": 5000}
    param_log_bound: float = 2.302585092994046   # |log multiplier| <= ln 10 (factor 0.1 - 10)
    param_bounds: dict = field(default_factory=dict)   # per-parameter override, e.g. {"R0": 2.0} (R0 unit = R0_SCALE = 5 mOhm)
    soft_current: bool = True        # soft electrode-current term when projection is off (group "current")
    salt_conservation: bool = True   # soft global salt-inventory term (group "conservation")
    salt_scale_mol_m3: float = 1.0

    def to_dict(self):
        return asdict(self)


GROUPS = {
    "mass": ["mass_n", "mass_s", "mass_p", "bc_flux_0", "bc_flux_L", "if1_flux", "if2_flux"],
    "charge_e": ["charge_n", "charge_s", "charge_p", "bc_ie_0", "bc_ie_L", "if1_ie", "if2_ie", "if1_phie", "if2_phie"],
    "solid": ["solid_n", "solid_p"],
    "particle": ["particle_n", "particle_p"],
    "inventory": ["inv_n", "inv_p"],
    "kinetics": ["kin_n", "kin_p"],
    "data": ["data_V"],
    "conservation": ["salt"],
    "current": ["current_n", "current_p"],
}


class Sampler:
    def __init__(self, cfg: TrainConfig, dtype, generator):
        self.cfg, self.dtype, self.g = cfg, dtype, generator

    def u(self, n, d=1):
        return torch.rand(n, d, generator=self.g, dtype=self.dtype)

    def times(self, n):
        n_early = int(round(self.cfg.early_fraction * n))
        n_early2 = int(round(self.cfg.early2_fraction * n))
        t = self.u(n)
        t[:n_early] *= self.cfg.early_window
        t[n_early:n_early + n_early2] *= self.cfg.early2_window
        return t

    def xs(self, n, k):
        """Electrode coordinates; a fraction is concentrated near the separator side."""
        x = self.u(n)
        m = int(round(self.cfg.x_edge_fraction * n))
        if m:
            near = self.cfg.x_edge_width * self.u(m)          # distance from the separator
            x[:m] = (1 - near) if k == "n" else near           # n: separator at x=1; p: at x=0
        mc = int(round(self.cfg.x_collector_fraction * n))
        if mc:
            near = self.cfg.x_collector_width * self.u(mc)    # distance from the collector
            x[m:m + mc] = near if k == "n" else (1 - near)     # n: collector at x=0; p: at x=1
        return x

    def draw(self, mult=1):
        c = self.cfg
        out = {}
        for k in ("n", "p"):
            n = c.n_electrode * mult
            out[f"e_{k}"] = (self.xs(n, k), self.times(n))
            n_pairs = max(c.n_particle * mult // c.n_radial, 1)
            rho = self.u(n_pairs, c.n_radial) ** (1 / 3)   # uniform in particle volume
            out[f"p_{k}"] = (rho ** 2, self.xs(n_pairs, k), self.times(n_pairs))
        n = c.n_separator * mult
        out["s"] = self.u(n)
        out["s_t"] = self.times(n)
        out["b_t"] = self.times(c.n_boundary * mult)
        return out


def residual_terms(res: Residuals, batch, cell: CellParams, return_times=False):
    """Residual tensors per term (and, optionally, the t_hat of every sample)."""
    terms, times = {}, {}
    for k in ("n", "p"):
        xk, t = (v.clone().requires_grad_(True) for v in batch[f"e_{k}"])
        e = res.electrode(k, xk, t)
        terms[f"mass_{k}"] = e["r_mass"]
        terms[f"charge_{k}"] = e["r_charge"]
        terms[f"solid_{k}"] = e["r_solid"]
        if e["r_inv"] is not None:
            terms[f"inv_{k}"] = e["r_inv"]
        if e["r_kin"] is not None:
            terms[f"kin_{k}"] = e["r_kin"]
        for name in ("mass", "charge", "solid", "inv", "kin"):
            if f"{name}_{k}" in terms:
                times[f"{name}_{k}"] = t.detach()
        s, xk2, t2 = batch[f"p_{k}"]
        terms[f"particle_{k}"] = res.particle_grouped(k, s, xk2.clone(), t2.clone().requires_grad_(True))
        times[f"particle_{k}"] = t2.expand(s.shape).reshape(-1, 1)
    Xs = (cell.X1 + (cell.X2 - cell.X1) * batch["s"]).requires_grad_(True)
    ts = batch["s_t"].clone().requires_grad_(True)
    sep = res.separator(Xs, ts)
    terms["mass_s"] = sep["r_mass"]
    terms["charge_s"] = sep["r_charge"]
    times["mass_s"] = times["charge_s"] = ts.detach()
    tb = batch["b_t"].clone().requires_grad_(True)
    bnd = res.boundaries(tb)
    terms.update(bnd)
    for name in bnd:
        times[name] = tb.detach()
    if res.salt_scale is not None:
        # evenly strided subset of the boundary times (the first entries are the
        # early-window samples, so a plain [:64] slice would miss late times)
        t_salt = batch["b_t"][::max(1, len(batch["b_t"]) // 64)]
        terms["salt"] = res.salt_inventory(t_salt, res.salt_scale)
        times["salt"] = t_salt
    if res.soft_current:
        for k in ("n", "p"):
            terms[f"current_{k}"] = res.electrode_current(k, batch["b_t"])
            times[f"current_{k}"] = batch["b_t"]
    return (terms, times) if return_times else terms


def term_losses(terms):
    return {k: v.square().mean() for k, v in terms.items()}


class AdaptiveWeights:
    """Group weights lambda_g ~ mean_g' |grad L_g'| / |grad L_g| (Wang et al. 2021 style)."""

    def __init__(self, groups, alpha=0.9):
        self.groups, self.alpha = groups, alpha
        self.w = {g: 1.0 for g in groups}

    def update(self, model, losses):
        params = [p for p in model.parameters() if p.requires_grad]
        norms = {}
        for g, names in self.groups.items():
            present = [losses[n] for n in names if n in losses]
            if not present:
                continue
            Lg = sum(present)
            grads = torch.autograd.grad(Lg, params, retain_graph=True, allow_unused=True)
            norms[g] = math.sqrt(sum(float((gr ** 2).sum()) for gr in grads if gr is not None)) + 1e-12
        mean_norm = sum(norms.values()) / len(norms)
        for g in norms:
            target = min(max(mean_norm / norms[g], 1e-2), 1e3)
            self.w[g] = self.alpha * self.w[g] + (1 - self.alpha) * target
        return norms


class CausalWeights:
    """Causal (time-respecting) loss weights, Wang, Sankaran & Perdikaris (2022).

    Time is split into ``bins`` intervals; the loss of bin b is weighted by
    w_b = exp(-eps * sum_{k<b} L_k), so later times only matter once earlier
    times are resolved. eps grows along ``schedule`` whenever min_b w_b > tol.
    """

    def __init__(self, bins=16, schedule=(1e-2, 1e-1, 1.0, 10.0, 100.0), tol=0.99):
        self.bins, self.schedule, self.tol = bins, list(schedule), tol
        self.level = 0
        self.last_w = None

    @property
    def eps(self):
        return self.schedule[self.level]

    def bin_losses(self, terms, times, group_w, term_w):
        """Weighted loss per time bin, summed over all terms (differentiable)."""
        total = None
        for g, names in GROUPS.items():
            for n in names:
                if n not in terms:
                    continue
                r2 = terms[n].square().view(-1)
                b = torch.clamp((times[n].view(-1) * self.bins).long(), 0, self.bins - 1)
                sums = torch.zeros(self.bins, dtype=r2.dtype, device=r2.device).index_add(0, b, r2)
                counts = torch.zeros(self.bins, dtype=r2.dtype, device=r2.device).index_add(0, b, torch.ones_like(r2))
                mean = sums / counts.clamp_min(1.0)
                term = group_w.get(g, 1.0) * term_w.get(n, 1.0) * mean
                total = term if total is None else total + term
        return total

    def loss(self, terms, times, group_w, term_w):
        L = self.bin_losses(terms, times, group_w, term_w)
        with torch.no_grad():
            cum = torch.cumsum(L, 0) - L
            w = torch.exp(-self.eps * cum)
            self.last_w = w
            if float(w.min()) > self.tol and self.level < len(self.schedule) - 1:
                self.level += 1
        return (w * L).mean()


def total_loss(losses, group_w, term_w):
    total = 0.0
    for g, names in GROUPS.items():
        for n in names:
            if n in losses:
                total = total + group_w.get(g, 1.0) * term_w.get(n, 1.0) * losses[n]
    return total


def train(cfg: TrainConfig, cell: CellParams, protocol: Protocol, out_dir: Path,
          reference=None, log=print, init_state=None, init_weights=None, init_optimizer=None,
          resume=None):
    """resume: path of a ``latest.pt`` written by this function; continues the same run in place
    (model, Adam state, group weights, step counter, learning-rate schedule and history)."""
    from .evaluate import compare

    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    torch.manual_seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    gen = torch.Generator().manual_seed(cfg.seed + 12345)
    scales = Scales(cell, protocol)
    model = DFNPINN(scales, cfg.width, cfg.depth, cfg.act, cfg.projection, ic_tau_s=cfg.ic_tau_s,
                    inventory=cfg.inventory, fourier_t=cfg.fourier_t,
                    width_scalar=cfg.width_scalar or None, short_t=tuple(cfg.short_t),
                    collector_bc=cfg.collector_bc, fourier_period=cfg.fourier_period).to(dtype)
    if init_state is not None:
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        assert all(k.startswith("log_mult") for k in missing) and not unexpected, (missing, unexpected)
    data = None
    if cfg.data_path:
        raw = np.load(cfg.data_path, allow_pickle=False)
        t_d, V_d = np.asarray(raw["t"], float), np.asarray(raw["V"], float)
        if cfg.data_noise_mV > 0:
            V_d = V_d + np.random.default_rng(cfg.seed + 7).normal(0.0, cfg.data_noise_mV * 1e-3, V_d.shape)
        data = (torch.as_tensor(t_d / scales.t_end, dtype=dtype).view(-1, 1),
                torch.as_tensor(V_d, dtype=dtype).view(-1, 1))
        model.set_trainable_parameters(cfg.inverse_params, cfg.inverse_init)
        log(f"Inverse problem: learning {cfg.inverse_params}, initial {model.parameter_values()}")
    res = Residuals(model, cfg.kinetics)
    res.salt_scale = cfg.salt_scale_mol_m3 if cfg.salt_conservation else None
    res.soft_current = cfg.soft_current and not cfg.projection
    sampler = Sampler(cfg, dtype, gen)
    net_params = [p for n, p in model.named_parameters() if not n.startswith("log_mult")]
    phys_params = [model.log_mult[n] for n in cfg.inverse_params]
    groups = [{"params": net_params, "lr": cfg.lr}]
    if phys_params:
        groups.append({"params": phys_params, "lr": cfg.param_lr})
    opt = torch.optim.Adam(groups)
    if init_optimizer is not None:
        try:
            opt.load_state_dict(init_optimizer)
            for group, lr in zip(opt.param_groups, [cfg.lr, cfg.param_lr]):
                group["lr"] = lr          # keep the moments, restart the schedule at cfg.lr
            log("Restored Adam state from the warm-start checkpoint (schedule restarted).")
        except (ValueError, KeyError) as exc:
            log(f"Adam state not restored ({exc}); fresh optimizer.")
    gamma = (cfg.lr_final / cfg.lr) ** (1 / max(cfg.adam_steps, 1))
    aw = AdaptiveWeights(GROUPS, cfg.adaptive_alpha)
    if init_weights:
        aw.w.update(init_weights)
    history, evals = [], []
    start_step = 1
    if resume is not None:
        ck = torch.load(resume, weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["optimizer"])
        aw.w.update(ck.get("group_weights", {}))
        init_weights = init_weights or ck.get("group_weights")   # skip the step-1 weight update
        start_step = int(ck.get("step", 0)) + 1
        # The learning rate is recomputed from the step (closed form), not re-stepped: the stored
        # optimizer lr is already decayed, and stepping a fresh scheduler on top of it decayed it twice
        # (bug in the first version of --resume, 2026-10-05; see docs/V2_RESULTS.md 3.3 and 7).
        for group, base in zip(opt.param_groups, [cfg.lr, cfg.param_lr]):
            group["lr"] = base * gamma ** (start_step - 1)
        prev = Path(resume).parent / "history.json"
        if prev.exists():
            saved = json.loads(prev.read_text())
            history, evals = saved.get("history", []), saved.get("evals", [])
        log(f"Resumed from {resume} at step {start_step} (lr {opt.param_groups[0]['lr']:.3e})")
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)   # continues from the current lr
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"train": cfg.to_dict(), "cell": cell.to_dict(),
                                                     "protocol": protocol.to_dict(),
                                                     "scales": scales.summary()}, indent=2))
    t0 = time.perf_counter() - (history[-1]["time_s"] if history else 0.0)
    current_step = [start_step - 1]

    def save(name, extra=None):
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "train": cfg.to_dict(),
                    "cell": cell.to_dict(), "protocol": protocol.to_dict(), "group_weights": aw.w,
                    "parameters": model.parameter_values(), "step": current_step[0], **(extra or {})},
                   out_dir / name)

    def evaluate(step):
        if reference is None:
            return None
        metrics, _ = compare(model, reference, max_times=120, kinetics=cfg.kinetics)
        metrics["step"] = step
        evals.append(metrics)
        log(f"  eval@{step}: V rmse {metrics['V_rmse_mV']:.2f} mV (max {metrics['V_max_mV']:.2f}); "
            f"ce max {metrics['ce_max']:.1f}; phie max {metrics['phie_max_mV']:.1f} mV; "
            f"j rmse rel n {metrics['j_n_rmse_rel']:.3f} p {metrics['j_p_rmse_rel']:.3f}; "
            f"thsurf max n {metrics['thsurf_n_max']:.4f} p {metrics['thsurf_p_max']:.4f}")
        return metrics

    causal = CausalWeights(cfg.causal_bins, tol=cfg.causal_tol) if cfg.causal else None
    for step in range(start_step, cfg.adam_steps + 1):
        current_step[0] = step
        batch = sampler.draw()
        terms, times = residual_terms(res, batch, cell, return_times=True)
        if data is not None:
            terms["data_V"] = (model.terminal_voltage(data[0]) - data[1]) / (cfg.data_scale_mV * 1e-3)
            times["data_V"] = data[0].detach()
        losses = term_losses(terms)
        if cfg.adaptive and ((step == 1 and not init_weights) or step % cfg.adaptive_every == 0):
            aw.update(model, losses)
        group_w = aw.w if cfg.adaptive else cfg.fixed_group_weights
        if causal is not None:
            loss = causal.loss(terms, times, group_w, cfg.weights)
        else:
            loss = total_loss(losses, group_w, cfg.weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for n, p in zip(cfg.inverse_params, phys_params):   # fields first, parameters later
            if step <= max(cfg.param_warmup_steps, cfg.param_release_steps.get(n, 0)):
                p.grad = None                      # Adam skips a parameter without gradient
        opt.step()
        sched.step()
        if phys_params:
            with torch.no_grad():
                for n, p in zip(cfg.inverse_params, phys_params):
                    b = cfg.param_bounds.get(n, cfg.param_log_bound)
                    p.clamp_(0.0 if n == "R0" else -b, b)   # R0 >= 0; per-parameter bound overrides
        if step % cfg.log_every == 0 or step == 1:
            rec = {"step": step, "loss": float(loss.detach()), "time_s": time.perf_counter() - t0,
                   **{k: float(v.detach()) for k, v in losses.items()}, "w": dict(aw.w)}
            if phys_params:
                rec["parameters"] = model.parameter_values()
            history.append(rec)
            grp = {g: sum(float(losses[n].detach()) for n in names if n in losses) for g, names in GROUPS.items()
                   if any(n in losses for n in names)}
            msg = (f"step {step:6d} loss {rec['loss']:.3e} | " + " ".join(f"{g} {v:.2e}" for g, v in grp.items())
                   + f" | {rec['time_s']:.0f}s")
            if phys_params:
                msg += " | " + " ".join(f"{n}={rec['parameters'][n]:.4g}" for n in cfg.inverse_params)
            if causal is not None:
                rec["causal_eps"], rec["causal_min_w"] = causal.eps, float(causal.last_w.min())
                msg += f" | causal eps {causal.eps:g} min w {rec['causal_min_w']:.3f}"
            log(msg)
        if step % cfg.eval_every == 0:
            evaluate(step)
        if cfg.latest_every and step % cfg.latest_every == 0:
            save("latest.pt")                      # rolling checkpoint for --resume
            (out_dir / "history.json").write_text(json.dumps({"history": history, "evals": evals}, indent=1))
        if step % cfg.checkpoint_every == 0:
            save(f"adam_{step:06d}.pt")
            (out_dir / "history.json").write_text(json.dumps({"history": history, "evals": evals}, indent=1))

    if cfg.lbfgs_iters > 0:
        # Mini-batch L-BFGS: a fresh collocation batch every ``lbfgs_resample_every``
        # iterations (the history persists across .step calls). With a single fixed
        # batch L-BFGS overfits the collocation points: on the 20k checkpoint the batch
        # loss fell 43 % while the c_e error against PyBaMM grew 5x (docs/V2_RESULTS.md).
        params = [p for p in model.parameters() if p.requires_grad]
        lbfgs = torch.optim.LBFGS(params, lr=1.0, max_iter=cfg.lbfgs_resample_every,
                                  history_size=cfg.lbfgs_history, line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-12, tolerance_change=1e-15)
        count, batch = [0], [None]
        group_w = aw.w if cfg.adaptive else cfg.fixed_group_weights

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            terms = residual_terms(res, batch[0], cell)
            if data is not None:
                terms["data_V"] = (model.terminal_voltage(data[0]) - data[1]) / (cfg.data_scale_mV * 1e-3)
            L = total_loss(term_losses(terms), group_w, cfg.weights)
            L.backward()
            count[0] += 1
            if count[0] % 100 == 0:
                log(f"  lbfgs eval {count[0]} batch loss {float(L.detach()):.3e}")
            return L

        done = 0
        while done < cfg.lbfgs_iters:
            batch[0] = sampler.draw(cfg.lbfgs_batch_mult)
            lbfgs.step(closure)
            done += cfg.lbfgs_resample_every
            if cfg.lbfgs_eval_every and done % cfg.lbfgs_eval_every == 0:
                evaluate(cfg.adam_steps + done)
                save("lbfgs_latest.pt")
        evaluate(cfg.adam_steps + done)

    final = evaluate(cfg.adam_steps)
    save("final.pt", {"final_metrics": final})
    (out_dir / "history.json").write_text(json.dumps({"history": history, "evals": evals}, indent=1))
    return model, history, evals
