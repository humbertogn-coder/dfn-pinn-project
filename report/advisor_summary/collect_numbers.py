"""Collect every number quoted in the advisor summary from saved results -> report/advisor_summary/numbers.json.

    python report/advisor_summary/collect_numbers.py

The chapter files of build/ read numbers.json (no number in the text is typed by hand, except a few literature
values quoted from the two papers, which are marked as such in the text).
"""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
RES = ROOT / "results"
RUNS = RES / "lispan_runs"
N = {}


def put(key, value, fmt="{:.2f}"):
    N[key] = fmt.format(value) if isinstance(value, (float, int, np.floating)) else value


def main():
    paper = json.loads((ROOT / "paper" / "numbers.json").read_text())
    pv = lambda k: paper[k]["value"]
    fig = json.loads((HERE / "figures" / "figure_numbers.json").read_text())

    # DFN
    r1 = fig["r1_dfn_vs_pybamm"]
    put("dfn_fwd_1", r1[0][1]); put("dfn_fwd_2", r1[1][1])
    for k, v in fig["r2_dfn_inverse_aging"]["fresh"].items():
        put(f"dfn_inv_{k}", v, "{:+.1f}")
    for k, v in fig["r2_dfn_inverse_aging"]["aged"].items():
        put(f"dfn_age_{k}", v, "{:+.1f}")
    put("dfn_age_misfit", fig["r2_dfn_inverse_aging"]["aged_misfit_mV"])

    # Li-SPAN reference vs paper
    put("ref_rms_01", pv("fig6a_rms_mV_0.1C"), "{:.0f}"); put("ref_rms_1", pv("fig6a_rms_mV_1C"), "{:.0f}")

    # forward PINN
    f = fig["r4_r5_forward"]
    for cr in ("0.1", "1"):
        for i, v in enumerate(f[cr]):
            put(f"fwd_{cr}_{i + 1}", v)
        put(f"fwd_{cr}_mean", float(np.mean(f[cr])))

    # synthetic inverse table (physical values)
    from dfn_pinn.lispan.params import LiSPANParams
    p = LiSPANParams()
    true = {"k0_1": p.k0[0], "k0_2": p.k0[1], "k0_3": p.k0[2], "b_1": p.b[0], "b_2": p.b[1], "b_3": p.b[2],
            "U0_1": p.U0[0], "Z_CC": 0.025}
    adam = json.loads((RUNS / "lispan_inv8p_multirate_freeze_20261007T073258Z" / "inverse_result.json").read_text())
    lm = json.loads((RUNS / "lispan_inv8p_multirate_lm_20261007T083347Z" / "inverse_result.json").read_text())
    ideal = json.loads((RES / "lispan" / "fit_experiment" / "fit_0.1+1C_nominal.json").read_text())
    rows = []
    for k, t in true.items():
        def val(err):
            return t + err * 1e-3 if k == "U0_1" else t * (1 + err / 100)
        start = adam["initial"][k]
        sv = t + start if k == "U0_1" else t * start
        rows.append({"param": k, "true": t, "start": sv, "adam": val(adam["errors_pct_or_mV"][k]),
                     "lm": val(lm["errors_pct_or_mV"][k]), "err_adam": adam["errors_pct_or_mV"][k],
                     "err_lm": lm["errors_pct_or_mV"][k], "noise_limit": ideal["crlb_pct_or_mV"][k]})
    N["inverse_table"] = rows

    # our reference model (Table 2 parameters, Z_CC 0.025) vs the measured Fig. 4b points (Figure r3b)
    from dfn_pinn.lispan.model import load as load_ref
    dev = []
    for cr in (0.05, 0.1, 0.2, 1.0):
        r = load_ref(RES / "lispan" / "ref_Zcc0.025" / f"discharge_{cr:g}C.npz")
        d = np.loadtxt(RES / "lispan" / "paper_digitized" / f"fig4b_exp_{cr:g}C.csv", delimiter=",", skiprows=1)
        m = (d[:, 0] >= r["Q_mAh_gS"][0]) & (d[:, 0] <= r["Q_mAh_gS"][-1])
        e = np.interp(d[m, 0], r["Q_mAh_gS"], r["V"]) - d[m, 1]
        dev.append(float(1e3 * np.sqrt(np.mean(e ** 2))))
        put(f"ref_vs_exp_mean_{cr:g}", float(1e3 * e.mean()), "{:+.0f}")
    put("ref_vs_exp_min", min(dev), "{:.0f}"); put("ref_vs_exp_max", max(dev), "{:.0f}")
    put("sim_pinn_rms_01", pv("measured_fig4b_PINN_rms_mV_0.1C"), "{:.0f}"); put("sim_pinn_rms_1", pv("measured_fig4b_PINN_rms_mV_1C"), "{:.0f}")
    put("sim_fv_rms_1", pv("measured_fig4b_FV_rms_mV_1C"), "{:.0f}")
    # measured data, Simanjuntak
    put("sim_fv_rms", pv("measured_fig4b_FV_rms_mV_0.1C"), "{:.0f}")
    put("sim_Zcc_pinn", pv("measured_fig4b_Z_CC_pinn_Ohm_m2"), "{:.3f}")
    put("sim_Zcc_fv", pv("measured_fig4b_Z_CC_fit_Ohm_m2"), "{:.3f}")

    # SPAN500 cycle 3 (paper case)
    s5 = json.loads((RES / "lispan" / "span500" / "fit_cycle3.json").read_text())
    put("s5_rms_paper", s5["fits"]["paper"]["rms_mV"], "{:.0f}"); put("s5_rms_fast", s5["fits"]["fast"]["rms_mV"], "{:.0f}")
    put("s5_wS", s5["fits"]["paper"]["estimates"]["w_S"])
    put("s5_c2c", s5["data"]["cycle_to_cycle_rms_mV"], "{:.0f}")
    put("s5_c2_miss", s5["fits"]["paper"]["check_cycle_rms_mV"], "{:.0f}")
    put("s5_1C_diff", pv("span500_1C_prediction_max_diff_mV"), "{:.0f}")
    win = json.loads((RES / "lispan" / "span500" / "fit_cycle3_pinnwindow.json").read_text())
    put("s5_win_fv", win["rms_mV"], "{:.1f}")
    pinn_run = sorted(RUNS.glob("lispan_inv_span500_c3_paper_wS_b5_*"))[-1]
    put("s5_win_pinn", pv("span500_pinn_fv_rms_mV"), "{:.1f}")

    # rate analysis
    ra = json.loads((RES / "lispan" / "span_chem" / "rate_analysis.json").read_text())
    c1 = fig["c1_rates"]
    put("rate_loss_total", c1["total_1C_mV"], "{:.0f}")
    put("rate_loss_zcc", c1["contact resistance"], "{:.0f}")
    put("rate_loss_kin", c1["reaction kinetics"], "{:.0f}")
    put("rate_loss_el", c1["electrolyte transport"], "{:.0f}")
    for r, v in c1["left_S2_pct"].items():
        put(f"rate_left_{r}", v, "{:.0f}")
    for r in ("0.05", "0.1", "0.2", "1"):
        put(f"rate_Q_{r}", ra["rates"][r]["Q_end_mAh_gS"], "{:.0f}")
    put("rate_Q_1_noZ", ra["variants_1C"]["no contact resistance"]["Q_end_mAh_gS"], "{:.0f}")
    put("rate_Q_1_fastk", ra["variants_1C"]["fast kinetics (k0 x100)"]["Q_end_mAh_gS"], "{:.0f}")
    put("rate_Q_1_el", ra["variants_1C"]["ideal electrolyte (x100 transport)"]["Q_end_mAh_gS"], "{:.0f}")
    smin, smax = ra["rates"]["1"]["salt_min_max_mol_m3"]
    put("rate_salt_dev_1", 100 * max(1000 - smin, smax - 1000) / 1000, "{:.0f}")
    lmin, lmax = ra["rates"]["1"]["Li2S_end_min_max"]
    put("rate_li2s_spread_1", 100 * (lmax - lmin) / lmax, "{:.0f}")
    put("rate_pinn_V_01", ra["pinn_vs_fv"]["0.1C"]["V_rms_mV"]); put("rate_pinn_V_1", ra["pinn_vs_fv"]["1C"]["V_rms_mV"])
    put("rate_pinn_species_max", max(max(v["field_rms"][k] for k in ("c_S4", "c_S3", "c_S2", "c_S1"))
                                     for v in ra["pinn_vs_fv"].values()), "{:.1f}")
    est = ra["estimates"]
    for k in ("k0_1", "k0_2", "k0_3"):
        put(f"rate_{k}_mult", est[k], "{:.1f}")

    # chemistry data sets (filled when the runs exist)
    import lispan_span500 as sp5
    U0n = sp5.base_params(1.0).U0
    chem = {}
    for label in ("s500_c1", "s500_c2", "s500_c3", "si300_c2", "si320_c2", "si500_c2"):
        if label == "s500_c3":
            fitd = {"estimates": s5["fits"]["paper"]["estimates"], "rms_mV": s5["fits"]["paper"]["rms_mV"],
                    "window": {"rms_mV": win["rms_mV"], "estimates": win["estimates"]}, "Q_end_mAh_gSPAN": s5["data"]["Q_end_mAh_gSPAN"]}
            pinn = json.loads((pinn_run / "inverse_result.json").read_text())
        else:
            fp = RES / "lispan" / "span_chem" / f"fit_{label}.json"
            if not fp.exists():
                continue
            fitd = json.loads(fp.read_text())
            runs = [r for r in sorted(RUNS.glob(f"lispan_chem_{label}_inv_*")) if (r / "inverse_result.json").exists()]
            pinn = json.loads((runs[-1] / "inverse_result.json").read_text()) if runs else None
        src = pinn["estimates"] if pinn else fitd.get("window", fitd)["estimates"]
        rms_pinn = None
        if pinn and label != "s500_c3":       # FV model at the PINN's parameters on the PINN's data window
            dw = np.load(RES / "lispan" / "span_chem" / f"V_{label}_pinnwindow.npz")
            base = replace(sp5.base_params(1.0), Z_CC=fitd["Z_CC_Ohm_m2"])
            Qm, Vm, _, _ = sp5.simulate({**fitd["estimates"], **pinn["estimates"]}, base)
            r = np.where(dw["Q_mAh_gSPAN"] <= Qm[-1], np.interp(dw["Q_mAh_gSPAN"], Qm, Vm), 1.0) - dw["V"]
            rms_pinn = float(1e3 * np.sqrt(np.mean(r ** 2)))
        elif label == "s500_c3":
            rms_pinn = paper["span500_pinn_fv_rms_mV"]["value"]
        chem[label] = {"tool": "PINN" if pinn else "FV", "w_S": fitd["estimates"]["w_S"],
                       "U0": [U0n[i] + src[f"U0_{i + 1}"] for i in range(3)],
                       "b_mult": [src[f"b_{i + 1}"] for i in range(3)],
                       "rms_full_fv": fitd["rms_mV"], "rms_window_fv": fitd.get("window", {}).get("rms_mV"),
                       "rms_window_pinn": rms_pinn,
                       "U0_fv_window": [U0n[i] + fitd["window"]["estimates"][f"U0_{i + 1}"] for i in range(3)],
                       "Q_end": fitd.get("Q_end_mAh_gSPAN")}
    N["chem"] = chem
    # Coulombic efficiency and capacities per cycle (Nat. Mater. source data, Fig. 4e)
    import pandas as pd
    e = pd.read_excel(ROOT / "data" / "span_natmater2026" / "41563_2026_2484_MOESM7_ESM.xlsx", sheet_name="Fig.4e",
                      header=None).iloc[1:4].apply(pd.to_numeric, errors="coerce")
    for name, col in (("500", 11), ("300", 13), ("320", 15)):
        put(f"ce_{name}_c1", float(e.iloc[0, col]), "{:.0f}")
    for name, col in (("300", 3), ("320", 5), ("500", 7)):
        put(f"cap_{name}_c1", float(e.iloc[0, col]), "{:.0f}"); put(f"cap_{name}_c2", float(e.iloc[1, col]), "{:.0f}")
        put(f"cap_{name}_c3", float(e.iloc[2, col]), "{:.0f}")
    (HERE / "numbers.json").write_text(json.dumps(N, indent=1))
    print(json.dumps({k: v for k, v in N.items() if k not in ("inverse_table",)}, indent=1))


if __name__ == "__main__":
    main()
