# PIEL: Physics-Informed Energy Landscape Optimization for Stable and Interpretable Decision-Making in Edge Computing

Companion code and data for the paper *"Physics-Informed Energy Landscape
Optimization for Stable and Interpretable Decision-Making in Edge
Computing"* (submitted to Future Generation Computer Systems).

PIEL is a domain-independent optimisation **kernel** for edge computing task
assignment — a general node abstraction, an energy functional grounded in
queueing theory, and a coordinate-descent solver with a finite-termination
guarantee and a Lyapunov convergence certificate for its continuous
relaxation — instantiated over two structurally distinct **application
layers**: a cross-domain IoT/fog validation, and vehicular edge computing
(VEC) as the flagship application, validated further on 237,868 real trip
records from the NYC Yellow Taxi trace.

## Repository structure

```
cross_domain/          Application Layer A: cross-domain generality validation
  piel_generic_edge_sim.py         3-tier IoT/fog simulation (Cloudlet/Gateway/Micro-DC)
  piel_significance_analysis.py    Paired significance testing (30 trials, Wilcoxon)
  piel_generic_edge_results.{json,csv}
  significance_results.json

vec_flagship/           Application Layer B: vehicular edge computing (flagship)
  PIEL_pipeline_v3.py               Core: energy functional, congestion model, solver
  train_dqn_v3.py                   Trained DQN baseline (environment + training loop)
  process_nyc_taxi_v3.py            NYC Yellow Taxi trace processing + evaluation
  dqn_v3_best.pt                    Best trained DQN checkpoint
  data/
    experiment_config.csv           alpha/beta/gamma weights, deadline, penalty, seeds
    rsu_profiles.csv                5-RSU heterogeneous network (Table 3)
    tasks.csv                       Synthetic VEC workload (300 tasks x 3 load scenarios)

results/                Raw outputs
  trace_results_v3.txt              Full NYC trace results (matches manuscript Table 9)
  nyc_taxi_tasks_v3.csv             Processed NYC trace tasks (237,868 rows)
  hourly_demand_v3.csv              Hourly task demand profile (Fig. "taxi demand")
  trace_convergence_v3.csv          Energy functional convergence on the real trace

figures/                All 10 manuscript figures + regeneration script
  generate_figures.py
  fig*.png
  README_figures.md                 Per-figure provenance and verification notes
```

## Reproducing the results

```bash
pip install -r requirements.txt
```

**Cross-domain validation (Table 5, Section 7.1):**
```bash
cd cross_domain
python3 piel_generic_edge_sim.py          # Table 5, stability score, ablation, convergence
python3 piel_significance_analysis.py     # Paired significance tests (Section 7.1.4)
```

**VEC flagship application, synthetic (Tables 6-8, Sections 7.2-7.6):**
```bash
cd vec_flagship
python3 PIEL_pipeline_v3.py               # Headline table, convergence, stability, ablation
```

**VEC flagship application, NYC trace (Table 9, Section 7.7):**
```bash
cd vec_flagship
# Place yellow_tripdata_*.csv (NYC TLC trip data) in this folder first
python3 process_nyc_taxi_v3.py
```
The already-processed trace (`results/nyc_taxi_tasks_v3.csv`) is included
directly, so the full trace evaluation can also be reproduced without the
raw TLC files — see `figures/generate_figures.py` for an example that
consumes it directly.

**Figures:**
```bash
cd figures
python3 generate_figures.py
```

## A note on naming

Earlier drafts of this work used the name **PIEF** (Physics-Informed Energy
*Framework*); the paper now uses **PIEL** (Physics-Informed Energy
*Landscape*) to better reflect the framework's central contribution — the
energy landscape itself, with coordinate descent as one navigation mechanism
among others. All code in this repository has been renamed accordingly
(`PIEF_pipeline_v3.py` -> `PIEL_pipeline_v3.py`, `pief_solve()` ->
`piel_solve()`, etc.) so that code and paper terminology match exactly.

## A note on correctness fixes

Two issues identified during manuscript review were fixed in both the paper
and this code, and verified to have **zero effect** on any previously
reported number (original and patched versions were run side by side on
identical inputs and produced bit-for-bit identical output):

1. **Strict-improvement coordinate descent.** The accept condition in
   `piel_solve()` now requires a strict energy decrease
   (`cost[best] < cost[cur]`), not merely a changed index. A finite state
   space with only non-increasing (rather than strictly decreasing) energy
   does not by itself rule out cycling; this is documented in the module
   docstring of `PIEL_pipeline_v3.py`.
2. **Notation collision.** In the manuscript, `p_r` denotes only power draw;
   the load-balance entropy term now uses a distinct symbol (`q_r`) for
   per-RSU load fraction. This is a manuscript-only fix (the code never had
   the equivalent naming collision).

## Data availability

The NYC Yellow Taxi Trip Data used for trace-driven validation is publicly
available via Kaggle, sourced from the NYC Taxi & Limousine Commission
(https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). The
processed, de-identified derived task features (`results/nyc_taxi_tasks_v3.csv`)
are included here for direct reproducibility; raw trip records are not
redistributed.

## Citation

```bibtex
@article{batri2026piel,
  title   = {Physics-Informed Energy Landscape Optimization for Stable and
             Interpretable Decision-Making in Edge Computing},
  author  = {Batri, Krishnan and S, Lakshmi},
  journal = {Future Generation Computer Systems},
  year    = {2026},
  note    = {Submitted}
}
```
(Update with final volume/page/DOI upon acceptance.)

## License

Code: MIT License (see `LICENSE`). Data: see Data Availability above for
the NYC trace; synthetic data files (`vec_flagship/data/`) are original to
this work and released under the same MIT terms as the code.
