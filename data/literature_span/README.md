# Published multi-rate SPAN discharge data (Phase 2 of the identifiability proposal)

Curves extracted from published figures for the calibrate-then-predict tests (`scripts/lispan_lit_rates.py`).
The PDFs are not stored here; each extraction script takes the PDF (or the figure raster) as an argument.

| folder | source | material / cell | rates | extraction | accuracy |
| --- | --- | --- | --- | --- | --- |
| `chen2019/` | Chen et al., Nat. Commun. 10, 1021 (2019), Fig. 3b (ether) and 3d (carbonate) | Se0.06SPAN (Se-doped SPAN), 80:10:5:5 (CMC/SBR), ~1 mg/cm2 active; 1 M LiPF6 EC/DMC or 1 M LiTFSI DOL/DME + 2 wt% LiNO3; Li metal; 1-3 V | 0.2 (first cycle of the cell, not used), 0.4, 1, 2, 4, 6, 10 A/g (per g S+Se) | exact: polylines read from the PDF vector paths (`scripts/extract_chen2019_fig3.py`, page 5) | line width only |
| `ma2026/` | Ma et al., Small 22, e74169 (2026), Fig. 6c (SPAN) and 6d (SPAN aerogel, "SPAN-A") | SPAN made at 350 C (S:PAN 3:1), 80:10:10 (Super P, PVDF) on carbon-coated Al, ~1.9 mg S/cm2; 1 M LiPF6 EC/DMC/EMC, 40 uL; Li metal; 1-3 V | 0.1 (first cycle of the cell), 0.2, 0.3, 0.5, 1, 2 C; 1 C = 1.675 A/g S (assumed; consistent with the paper's utilisation, Fig. 6b) | digitized from the 300 dpi raster (`scripts/digitize_ma2026_fig6.py`): nearest-colour classification, discharge branch tracked from 1.0 V | 3.6 mAh/g x 5.4 mV per pixel; points below 60 mAh/g dropped (all curves cross there); end capacities 0.2-2.3 % below the values in the text (SPAN 2 C: +1.7 %) |

Units: capacity in mAh per g of the paper's basis (S+Se for Chen, S for Ma), voltage in V vs Li/Li+.

Caveats:
* Chen 2019: Se doping changes the chemistry slightly (Se-S bonds; Se contributes capacity), so model parameters are
  effective. The same material is shown in two electrolytes, which allows a direct electrolyte comparison.
* Ma 2026: the profiles are most likely the first cycle at each rate (the capacities quoted in the text are the first
  point of each rate step in Fig. 6a, and the SPAN-A 0.1 / 0.2 C curves end there once the ~0.8 % low reading of the
  curve panel is removed; at 0.3 C and above the three cycles of a step are indistinguishable). The 0.1 C curve is then
  the first cycle of the cell (SPAN-A: 1513 mAh/g vs ~1380 when 0.1 C is
  repeated at the end of the test), so 0.2 C is used as the slow calibration rate. Electrode thickness and porosity
  are not reported. EIS of both materials (Fig. 4f,g) gives independent resistances for comparison.
* Not used (assessed, see docs/PHASE2_LITERATURE_DATA.md): Hao et al., ACS Nano 19, 25385 (2025) (rate profiles only
  for an FeS2/SPAN composite whose FeS2 conversion is not in the model; plain-SPAN profiles are in its SI, not
  available) and Moschner et al., Adv. Sci. 12, 2415436 (2025) (only C/50 and C/10 voltage curves; rate tests shown as
  capacities only).
