"""
Significance analysis for the cross-domain (generic edge computing) synthetic
dataset (Section 7.1 / Table 5 of the manuscript).

The headline results in Table 5 come from a SINGLE random task-set
realisation (seed=42) per load level -- point estimates, no repeated trials,
no confidence intervals, no significance test against baselines. This script
addresses that gap directly: it re-runs the identical experiment across many
independent random task-set realisations per load level, then reports
(a) mean +/- std and a 95% CI for each method, and (b) a paired Wilcoxon
signed-rank test (PIEL vs. each baseline, paired by trial/seed) on both
latency and energy at each load level.

Wilcoxon signed-rank (not a paired t-test) is used as the primary test
because it makes no normality assumption on the per-trial differences,
appropriate for engineering benchmark data with a moderate number of trials
and no guarantee of a symmetric, Gaussian-like difference distribution. A
paired t-test is reported alongside as a supplementary, more familiar
reference point.
"""
import numpy as np
from scipy import stats
import json

from piel_generic_edge_sim import (
    gen_tasks, batch_window, evaluate_assignment, coordinate_descent,
    fifo_assign, greedy_assign, water_filling_assign,
    N_TASKS, LOAD_LEVELS
)

N_TRIALS = 30  # independent random task-set realisations per load level


def run_trial(ell, seed):
    sigma, c, d = gen_tasks(N_TASKS, seed)
    Tbatch = batch_window(c, ell)
    out = {}
    for name, fn in [
        ("FIFO", lambda: fifo_assign(N_TASKS)),
        ("Greedy", lambda: greedy_assign(sigma, c, d, Tbatch)),
        ("Water-Filling", lambda: water_filling_assign(sigma, c, d, Tbatch)),
        ("PIEL", lambda: coordinate_descent(sigma, c, d, Tbatch, seed=seed)[0]),
    ]:
        x = fn()
        lat, en, succ, rho = evaluate_assignment(x, sigma, c, d, Tbatch)
        out[name] = dict(lat=float(np.mean(lat)), energy=float(np.mean(en)),
                          success=float(100 * np.mean(succ)))
    return out


def mean_ci95(arr):
    arr = np.asarray(arr, dtype=float)
    m = arr.mean()
    se = stats.sem(arr)
    lo, hi = stats.t.interval(0.95, len(arr) - 1, loc=m, scale=se) if len(arr) > 1 and se > 0 else (m, m)
    return m, arr.std(ddof=1), lo, hi


def paired_tests(piel_vals, baseline_vals):
    piel_vals = np.asarray(piel_vals, dtype=float)
    baseline_vals = np.asarray(baseline_vals, dtype=float)
    diffs = piel_vals - baseline_vals
    # Wilcoxon signed-rank (primary; drop exact zeros automatically)
    try:
        w_stat, w_p = stats.wilcoxon(piel_vals, baseline_vals)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    # Paired t-test (supplementary)
    t_stat, t_p = stats.ttest_rel(piel_vals, baseline_vals)
    return dict(mean_diff=float(diffs.mean()), wilcoxon_stat=float(w_stat), wilcoxon_p=float(w_p),
                ttest_stat=float(t_stat), ttest_p=float(t_p))


if __name__ == "__main__":
    results_summary = {}
    print(f"Running {N_TRIALS} independent trials per load level "
          f"(varying task-generation seed only; PIEL's own coordinate-descent "
          f"seed is tied to the trial seed, consistent with Table 5's protocol)...\n")

    for load_name, ell in LOAD_LEVELS.items():
        trial_data = {m: {"lat": [], "energy": []} for m in ["FIFO", "Greedy", "Water-Filling", "PIEL"]}
        for trial in range(N_TRIALS):
            seed = 42 + trial  # seed=42 trial reproduces the exact Table 5 point estimate
            res = run_trial(ell, seed)
            for method, vals in res.items():
                trial_data[method]["lat"].append(vals["lat"])
                trial_data[method]["energy"].append(vals["energy"])

        print(f"=== {load_name} (n={N_TRIALS} trials) ===")
        summary = {}
        for method in ["FIFO", "Greedy", "Water-Filling", "PIEL"]:
            lat_m, lat_sd, lat_lo, lat_hi = mean_ci95(trial_data[method]["lat"])
            en_m, en_sd, en_lo, en_hi = mean_ci95(trial_data[method]["energy"])
            summary[method] = dict(
                lat_mean=lat_m, lat_std=lat_sd, lat_ci95=(lat_lo, lat_hi),
                energy_mean=en_m, energy_std=en_sd, energy_ci95=(en_lo, en_hi),
            )
            print(f"  {method:15s} lat={lat_m:8.2f} +/- {lat_sd:5.2f} ms  "
                  f"[95% CI {lat_lo:.2f}, {lat_hi:.2f}]   "
                  f"energy={en_m:7.3f} +/- {en_sd:5.3f} J")

        print(f"\n  Paired significance tests (PIEL vs. baseline, n={N_TRIALS} paired trials):")
        sig_tests = {}
        for baseline in ["FIFO", "Greedy", "Water-Filling"]:
            lat_test = paired_tests(trial_data["PIEL"]["lat"], trial_data[baseline]["lat"])
            en_test = paired_tests(trial_data["PIEL"]["energy"], trial_data[baseline]["energy"])
            sig_tests[baseline] = dict(latency=lat_test, energy=en_test)
            sig_marker_lat = "***" if lat_test["wilcoxon_p"] < 0.001 else ("**" if lat_test["wilcoxon_p"] < 0.01 else ("*" if lat_test["wilcoxon_p"] < 0.05 else "ns"))
            sig_marker_en = "***" if en_test["wilcoxon_p"] < 0.001 else ("**" if en_test["wilcoxon_p"] < 0.01 else ("*" if en_test["wilcoxon_p"] < 0.05 else "ns"))
            print(f"    PIEL vs {baseline:15s} latency: mean_diff={lat_test['mean_diff']:+7.2f} ms, "
                  f"Wilcoxon p={lat_test['wilcoxon_p']:.2e} [{sig_marker_lat}]   "
                  f"energy: mean_diff={en_test['mean_diff']:+6.3f} J, Wilcoxon p={en_test['wilcoxon_p']:.2e} [{sig_marker_en}]")
        print()

        results_summary[load_name] = dict(summary=summary, significance=sig_tests)

    with open("significance_results.json", "w") as fh:
        json.dump(results_summary, fh, indent=2, default=str)
    print("Saved -> significance_results.json")
