"""Helpers to reload finished Li-SPAN PINN runs (paper scripts: cost table, figures, numbers)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .params import LiSPANParams, LiSPANProtocol
from .pinn import LiSPANPINN, LiSPANTrainConfig


def _params(d: dict) -> LiSPANParams:
    known = {f for f in LiSPANParams.__dataclass_fields__}
    return LiSPANParams(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in known})


def _protocol(d: dict) -> LiSPANProtocol:
    return LiSPANProtocol.from_dict(d)


def load_model(checkpoint, dtype=torch.float32, ema: bool = False) -> LiSPANPINN:
    """Rebuild a single-rate LiSPANPINN from a checkpoint written by pinn.train (final.pt, step_*.pt, latest.pt)."""
    ck = torch.load(checkpoint, weights_only=False)
    cfg = LiSPANTrainConfig(**{k: v for k, v in ck["train"].items() if k in LiSPANTrainConfig.__dataclass_fields__})
    model = LiSPANPINN(_params(ck["params"]), _protocol(ck["protocol"]), ck["t_end"], cfg.width, cfg.depth, cfg.act,
                       cfg.fourier_t, cfg.fourier_period, tuple(cfg.short_t), cfg.ic_tau_s, cfg.quad_order).to(dtype)
    model.load_state_dict(ck["model"], strict=False)
    if ema and ck.get("ema"):                       # EMA holds the network weights only
        model.load_state_dict(ck["ema"], strict=False)
    model.salt_total_scale = cfg.salt_total_scale
    model.charge_total_scale = cfg.charge_total_scale
    if cfg.salt_sep_natural_scale:
        model.salt_scale_sep = model.salt_scale * model.p.L_cat / model.p.L_sep
    model.eval()
    return model


def history(run_dir) -> dict:
    """{'history': [...per log step...], 'evals': [...]} of a run (time_s excludes the gaps between resumes)."""
    return json.loads((Path(run_dir) / "history.json").read_text())


def is_stopped(run_dir) -> bool:
    return (Path(run_dir) / "STOPPED.txt").exists()


def time_to(run_dir, key="V_rmse_mV", threshold=1.0):
    """Training wall time [s] and step at which the eval metric first fell below threshold (None if never)."""
    h = history(run_dir)
    t_of = {e["step"]: e["time_s"] for e in h["history"]}
    for ev in h["evals"]:
        if ev.get(key, float("inf")) < threshold:
            s = ev["step"]
            near = min(t_of, key=lambda k: abs(k - s)) if t_of else None
            return (t_of.get(s, t_of.get(near)), s)
    return (None, None)


def load_multirate(checkpoint, dtype=torch.float32):
    """Rebuild the per-rate LiSPANPINNs of a multi-rate inverse checkpoint (dfn_pinn.lispan.multirate.save layout);
    each model carries the shared parameter multipliers of the checkpoint."""
    ck = torch.load(checkpoint, weights_only=False)
    cfg = LiSPANTrainConfig(**{k: v for k, v in ck["train"].items() if k in LiSPANTrainConfig.__dataclass_fields__})
    p = _params(ck["params"])
    models = []
    for state, prot_d, t_end in zip(ck["models"], ck["protocols"], ck["t_ends"]):
        m = LiSPANPINN(p, _protocol(prot_d), t_end, cfg.width, cfg.depth, cfg.act, cfg.fourier_t, cfg.fourier_period,
                       tuple(cfg.short_t), cfg.ic_tau_s, cfg.quad_order).to(dtype)
        m.load_state_dict(state, strict=False)
        m.eval()
        models.append(m)
    return models, ck
