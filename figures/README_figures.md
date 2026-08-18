# Regenerated Figures — PIEF → PIEL Rebranding

All 10 figures referenced in `main-10.tex` (via `\includegraphics`), regenerated
from scratch with "PIEL" labeling throughout — no "PIEF" anywhere. The original
plotting code was never part of this session (confirmed by searching past
conversations), so these are fresh regenerations built directly from your
uploaded `PIEF_pipeline_v3.py` pipeline and data files, not edits of an
original script.

Generation script: `generate_figures.py` (in the parent output folder).

## Verification status per figure

| File | Source | Status |
|---|---|---|
| `fig1_latency_vs_load.png` | FIFO/Greedy/PIEL: re-run via `PIEF_pipeline_v3_final.py`. Water-Filling/DQN(trained): published Table 6 values. | FIFO/Greedy/PIEL re-run and matched Table 6 to full precision |
| `fig2_energy_vs_load.png` | same | same |
| `fig3_convergence.png` | Re-run: `pief_solve(..., random_init=True)`, moderate load | Matches paper's 7.073 → 0.400 over 7 iterations exactly |
| `fig5_ablation.png` | Re-run: all 4 ablation variants | Matches Table 8 exactly (latency, energy, success, entropy, iteration count — all 4 variants) |
| `fig_lambda_sensitivity.png` | Re-run: λ ∈ {0.5,...,500} at moderate and high load | Reproduces paper's claim exactly: 100%/0% success, completely flat across the full 1000× range |
| `fig_weight_pareto.png` | Re-run: the exact (α,β,γ) grid recovered from a prior session (`weight_grid` in the "Manuscript rejected with resubmission ban" conversation), then rerun fresh against your pipeline | **Exact** match: α=0.90 endpoint (308.42ms/11.8546J) matches Table 8's "High-alpha" row precisely; α=0.10 endpoint (325.93ms/9.7939J) matches the paper's stated 325.9ms/9.79J to the decimal |
| `fig_taxi_demand.png` | `hourly_demand_v3.csv` (uploaded, used directly, no re-derivation needed) | Exact: mean 9,911 tasks/hr, hour-18 peak of 15,110 tasks, both match the paper's caption verbatim |
| `fig_trace_convergence.png` | `trace_convergence_v3.csv` (uploaded, used directly) | Exact: 3.912 → 0.217 over 12 iterations, matches paper exactly |
| `fig7_rsu_spatial_distribution.png` | Percentages taken from `trace_results_v3.txt`'s "Spatial origin distribution" block (already verified against Table 9 in the prior pipeline-check pass) | Exact: RSU0+RSU1 shares are 85.0% / 83.6% / 80.9% per scenario, all within the paper's stated 78–85% range |
| `fig6_trace_vs_synthetic.png` | Combines the (already-verified) Table 6 and Table 9 values | All underlying numbers independently verified in the prior pipeline-check pass |

## What's exact vs. what necessarily comes from the published tables

**Exact — re-derived from your actual code/data and matched to full precision:**
Figs 1–3, 5, the λ-sensitivity sweep, the weight-Pareto sweep (now exact, not
approximate — the real sweep grid was recovered from a prior session and
rerun), the taxi demand profile, trace convergence, and spatial distribution.

**Taken from the published tables, not independently re-derivable in this
session:** the Water-Filling baseline and the trained-DQN numbers used in
Figs 1, 2, and 6. Their source scripts (a separate water-filling
implementation, and the specific DQN training run behind Table 6/9's
"DQN (trained)" column) weren't part of this upload. The trained-DQN
*checkpoint* (`dqn_v3_best.pt`) was uploaded and loads correctly, but
reproducing Table 6/9's exact DQN numbers from it would require re-running
`fair_score()`/`evaluate_dqn()` against the same task realizations used
originally, which introduces its own risk of drift from the published values
without a way to cross-check. Using the already-published, already-verified
numbers directly avoids that risk.

## Style notes

- Colour/marker scheme: FIFO (grey), Greedy (light grey), Water-Filling
  (orange), DQN trained (blue), PIEL (green) — consistent across every figure.
- All labels say "PIEL", not "PIEF", including the ablation study's "Full
  PIEL" bar and every legend entry.
- Fig 6 (trace vs. synthetic) uses solid bars for VEC synthetic and hatched
  bars for the NYC trace, per the paper's caption convention.
