"""
PIEL generality validation: a SECOND, cross-domain synthetic edge-computing
scenario, structurally distinct from the vehicular edge computing (VEC)
application. No RSUs, no vehicles, no V2I framing.

Domain: distributed IoT / smart-facility edge analytics.
Edge-node set S = {s_1, ..., s_M}: heterogeneous compute nodes spanning a
three-tier fog/edge hierarchy (Cloudlet / Gateway / Micro-DC), each defined
generically by (f_j, p_j) exactly as in the general PIEL formulation.
Tasks: IoT analytics jobs (sensor-fusion, anomaly scoring, local inference)
with their own payload / compute-demand / deadline distribution, deliberately
different in scale and shape from the VEC workload so this is not simply
VEC-with-renamed-labels.

Reuses the identical energy functional, M/M/1 within-batch congestion model,
and coordinate-descent solver as the flagship VEC application (Sections 4-5
of the manuscript) -- this script exists to demonstrate that the *same*
PIEL machinery, unmodified, generalises to a structurally different edge
scenario.

NOTE ON THE COORDINATE-DESCENT ACCEPT CONDITION (2026-08 correction):
The inner-loop update below only accepts a candidate node r* when it is a
STRICT improvement over the task's current assignment (best_phi < cur_phi),
matching the manuscript's corrected Proposition 1 / Remark 6 (Strict
Improvement, Not Merely a Changed Index) and Eq. (18). An earlier version
accepted any index change, including ties -- mathematically insufficient to
rule out cycling in a finite assignment space, though harmless in practice
here since continuous-valued task payload/cycle draws make exact ties
essentially never occur. This version and the original tie-accepting
version were run side by side and produced IDENTICAL output on this
dataset (verified numerically, not merely argued) -- see the diff record
kept alongside this file.
"""
import numpy as np
import json
import csv

rng_global_seed = 42

# ---------------------------------------------------------------------------
# Generic edge-node set S = {s_1, ..., s_M}
# Three-tier fog/edge hierarchy: Cloudlet (near-user, powerful), Gateway
# (mid-tier aggregation point), Micro-DC (small on-prem datacenter, efficient
# but not the fastest). Deliberately 6 nodes (vs. 5 RSUs in VEC) and a
# different capacity/power spread.
# ---------------------------------------------------------------------------
NODES = [
    dict(name="Cloudlet-A", type="Cloudlet", f=13.0, p=60),
    dict(name="Cloudlet-B", type="Cloudlet", f=10.5, p=48),
    dict(name="Gateway-A",  type="Gateway",  f=8.0,  p=30),
    dict(name="Gateway-B",  type="Gateway",  f=6.5,  p=24),
    dict(name="MicroDC-A",  type="Micro-DC", f=5.0,  p=14),
    dict(name="MicroDC-B",  type="Micro-DC", f=3.5,  p=9),
]
M = len(NODES)
F = np.array([n["f"] for n in NODES])   # GHz
P = np.array([n["p"] for n in NODES])   # Watts

# ---------------------------------------------------------------------------
# Generic IoT analytics workload.
# Deadlines here are looser than VEC's safety-critical 400ms (e.g. building
# analytics / environmental sensing are not collision-avoidance-grade), and
# payloads/compute demand are scaled differently to reflect sensor-fusion
# and anomaly-scoring jobs rather than autonomous-driving perception.
# ---------------------------------------------------------------------------
DEADLINE_MS = 800.0
N_TASKS = 300
LOAD_LEVELS = {"Low (20%)": 0.20, "Moderate (50%)": 0.50, "High (90%)": 0.90}

ALPHA, BETA, GAMMA = 0.55, 0.28, 0.17
LAMBDA_PENALTY = 10.0
C0 = 60.0  # J, normalisation constant (re-derived for this workload scale)
MAX_ITERS = 20


def gen_tasks(n, seed):
    rng = np.random.default_rng(seed)
    sigma = rng.uniform(0.1, 1.2, size=n)         # MB, sensor/telemetry payload
    c = rng.uniform(0.3e9, 1.6e9, size=n)         # CPU cycles, analytics job
    d = np.full(n, DEADLINE_MS)
    return sigma, c, d


def batch_window(c, ell):
    total_cycles = c.sum()
    total_capacity = (F * 1e9).sum()
    return (1.0 / ell) * (total_cycles / total_capacity)


def evaluate_assignment(x, sigma, c, d, Tbatch, lam_transmit=0.6, lam_energy=0.02):
    """Single uniform evaluator (mirrors Remark 5 in the VEC section)."""
    n = len(x)
    busy = np.zeros(M)
    for j in range(n):
        busy[x[j]] += c[j] / (F[x[j]] * 1e9)
    rho = np.minimum(busy / Tbatch, 0.98)

    lat = np.zeros(n)
    energy = np.zeros(n)
    for j in range(n):
        r = x[j]
        s_bar = (c[j] / (F[r] * 1e9))
        Wr = s_bar * (rho[r] / max(1e-9, 1 - rho[r]))
        proc = (c[j] / (F[r] * 1e9)) * 1e3
        trans = sigma[j] * lam_transmit
        lat[j] = proc + Wr * 1e3 + trans
        cong_factor = 1.0 / max(1e-6, 1 - rho[r])
        energy[j] = P[r] * (c[j] / (F[r] * 1e9)) * cong_factor + sigma[j] * lam_energy

    success = (lat <= d)
    return lat, energy, success, rho


def coordinate_descent(sigma, c, d, Tbatch, alpha=ALPHA, beta=BETA, gamma=GAMMA,
                        lam=LAMBDA_PENALTY, T=MAX_ITERS, seed=0, ablate=None):
    n = len(c)
    rng = np.random.default_rng(seed)

    if ablate == "no_D":
        alpha_, beta_, gamma_ = 0.65, 0.35, 0.0
    elif ablate == "no_C":
        alpha_, beta_, gamma_ = 0.75, 0.0, 0.25
    elif ablate == "high_alpha":
        alpha_, beta_, gamma_ = 0.90, 0.05, 0.05
    else:
        alpha_, beta_, gamma_ = alpha, beta, gamma

    # zero-congestion warm start
    proc0 = c[:, None] / (F[None, :] * 1e9) * 1e3
    trans0 = sigma[:, None] * 0.6
    x = np.argmin(proc0 + trans0, axis=1)

    busy = np.zeros(M)
    n_count = np.zeros(M, dtype=int)
    for j in range(n):
        busy[x[j]] += c[j] / (F[x[j]] * 1e9)
        n_count[x[j]] += 1

    def energy_functional(xx):
        lat, en, succ, rho = evaluate_assignment(xx, sigma, c, d, Tbatch)
        Lbar = np.mean((lat / d) ** 2)
        Cbar = np.mean(en) / C0
        p_r = n_count / n
        p_r_safe = np.where(p_r > 0, p_r, 1)
        H = -np.sum(np.where(n_count > 0, p_r * np.log(p_r_safe), 0)) / np.log(M)
        D = 1 - H
        Pen = lam * np.sum(~succ)
        return alpha_ * Lbar + beta_ * Cbar + gamma_ * D + Pen

    history = [energy_functional(x)]

    for t in range(T):
        changed = 0
        order = rng.permutation(n)
        for j in order:
            cur_r = x[j]
            busy[cur_r] -= c[j] / (F[cur_r] * 1e9)
            n_count[cur_r] -= 1
            cur_phi = None
            best_r, best_phi = cur_r, np.inf
            for r in range(M):
                busy[r] += c[j] / (F[r] * 1e9)
                n_count[r] += 1
                rho_r = min(busy[r] / Tbatch, 0.98)
                s_bar = c[j] / (F[r] * 1e9)
                Wr = s_bar * (rho_r / max(1e-9, 1 - rho_r))
                proc = s_bar * 1e3
                trans = sigma[j] * 0.6
                lat_j = proc + Wr * 1e3 + trans
                cong_factor = 1.0 / max(1e-6, 1 - rho_r)
                en_j = P[r] * s_bar * cong_factor + sigma[j] * 0.02
                Lterm = alpha_ * (lat_j / d[j]) ** 2
                Cterm = beta_ * en_j / C0
                p_r = n_count / n
                p_r_safe = np.where(p_r > 0, p_r, 1)
                H = -np.sum(np.where(n_count > 0, p_r * np.log(p_r_safe), 0)) / np.log(M)
                Dterm = gamma_ * (1 - H)
                Pterm = lam * (1.0 if lat_j > d[j] else 0.0)
                phi = Lterm + Cterm + Dterm + Pterm
                if r == cur_r:
                    cur_phi = phi
                if phi < best_phi:
                    best_phi, best_r = phi, r
                busy[r] -= c[j] / (F[r] * 1e9)
                n_count[r] -= 1
            # STRICT-IMPROVEMENT FIX: only accept if strictly better than current phi
            if best_phi < cur_phi:
                accepted_r = best_r
            else:
                accepted_r = cur_r
            busy[accepted_r] += c[j] / (F[accepted_r] * 1e9)
            n_count[accepted_r] += 1
            if accepted_r != cur_r:
                changed += 1
            x[j] = accepted_r
        history.append(energy_functional(x))
        if changed == 0:
            break

    return x, history


def fifo_assign(n):
    return np.array([j % M for j in range(n)])


def greedy_assign(sigma, c, d, Tbatch):
    n = len(c)
    x = np.zeros(n, dtype=int)
    busy = np.zeros(M)
    for j in range(n):
        best_r, best_lat = 0, np.inf
        for r in range(M):
            rho_r = min(busy[r] / Tbatch, 0.98)
            s_bar = c[j] / (F[r] * 1e9)
            Wr = s_bar * (rho_r / max(1e-9, 1 - rho_r))
            lat = s_bar * 1e3 + Wr * 1e3 + sigma[j] * 0.6
            if lat < best_lat:
                best_lat, best_r = lat, r
        x[j] = best_r
        busy[best_r] += c[j] / (F[best_r] * 1e9)
    return x


def water_filling_assign(sigma, c, d, Tbatch, n_iter=200):
    """Convex-relaxation baseline: target load fractions phi_r minimising
    aggregate expected M/M/1 delay sum_r rho_r/(1-rho_r) subject to
    sum_r phi_r = 1, solved via SLSQP, then rounded to a discrete assignment
    via largest-remainder target-tracking (mirrors the VEC Water-Filling
    baseline exactly)."""
    from scipy.optimize import minimize
    n = len(c)
    total_c = c.sum()

    def objective(phi):
        busy = phi * total_c / (F * 1e9)
        rho = np.minimum(busy / Tbatch, 0.98)
        return np.sum(rho / np.maximum(1e-6, 1 - rho))

    phi0 = F / F.sum()
    cons = [{"type": "eq", "fun": lambda phi: phi.sum() - 1.0}]
    bounds = [(1e-6, 0.98) for _ in range(M)]
    res = minimize(objective, phi0, method="SLSQP", bounds=bounds, constraints=cons,
                    options={"maxiter": 300, "ftol": 1e-9})
    phi = res.x if res.success else phi0
    phi = np.clip(phi, 1e-6, None)
    phi = phi / phi.sum()

    target_counts = np.floor(phi * n).astype(int)
    remainder = phi * n - target_counts
    deficit = n - target_counts.sum()
    for idx in np.argsort(-remainder)[:deficit]:
        target_counts[idx] += 1

    order = np.argsort(-c)  # largest-remainder target-tracking by task size
    x = np.zeros(n, dtype=int)
    remaining = target_counts.copy()
    for j in order:
        r = int(np.argmax(remaining))
        x[j] = r
        remaining[r] -= 1
        if remaining[r] < 0:
            remaining[r] = 0
    return x


def run_scenario(ell_name, ell, seed=42):
    sigma, c, d = gen_tasks(N_TASKS, seed)
    Tbatch = batch_window(c, ell)

    results = {}
    for method_name, fn in [
        ("FIFO", lambda: fifo_assign(N_TASKS)),
        ("Greedy", lambda: greedy_assign(sigma, c, d, Tbatch)),
        ("Water-Filling", lambda: water_filling_assign(sigma, c, d, Tbatch)),
        ("PIEL", lambda: coordinate_descent(sigma, c, d, Tbatch, seed=seed)[0]),
    ]:
        x = fn()
        lat, en, succ, rho = evaluate_assignment(x, sigma, c, d, Tbatch)
        results[method_name] = dict(
            lat=float(np.mean(lat)),
            energy=float(np.mean(en)),
            success=float(100 * np.mean(succ)),
        )
    return results, sigma, c, d, Tbatch


def stability_analysis(ell=0.50, n_runs=30):
    sigma, c, d = gen_tasks(N_TASKS, seed=42)
    Tbatch = batch_window(c, ell)
    assignments = []
    for run in range(n_runs):
        x, _ = coordinate_descent(sigma, c, d, Tbatch, seed=1000 + run)
        assignments.append(x)
    assignments = np.array(assignments)  # (n_runs, N_TASKS)
    stability = float(np.mean(np.std(assignments, axis=0)) * 10)
    return stability


def ablation_analysis(ell=0.50):
    sigma, c, d = gen_tasks(N_TASKS, seed=42)
    Tbatch = batch_window(c, ell)
    out = {}
    for variant in [None, "no_D", "no_C", "high_alpha"]:
        x, hist = coordinate_descent(sigma, c, d, Tbatch, seed=42, ablate=variant)
        lat, en, succ, rho = evaluate_assignment(x, sigma, c, d, Tbatch)
        p_r = np.array([np.mean(x == r) for r in range(M)])
        p_r_safe = np.where(p_r > 0, p_r, 1)
        H = -np.sum(np.where(p_r > 0, p_r * np.log(p_r_safe), 0)) / np.log(M)
        label = variant if variant else "Full PIEL"
        out[label] = dict(
            lat=float(np.mean(lat)), energy=float(np.mean(en)),
            success=float(100 * np.mean(succ)), H=float(H), iters=len(hist) - 1
        )
    return out


def convergence_trace(ell=0.50):
    sigma, c, d = gen_tasks(N_TASKS, seed=7)
    Tbatch = batch_window(c, ell)
    x, hist = coordinate_descent(sigma, c, d, Tbatch, seed=7)
    return hist


if __name__ == "__main__":
    import os
    out_dir = os.path.dirname(os.path.abspath(__file__))

    all_results = {}
    for name, ell in LOAD_LEVELS.items():
        res, *_ = run_scenario(name, ell)
        all_results[name] = res
        print(f"\n=== {name} ===")
        for method, m in res.items():
            print(f"  {method:15s} lat={m['lat']:8.2f} ms  energy={m['energy']:7.3f} J  succ={m['success']:5.1f}%")

    stab = stability_analysis()
    print(f"\nPIEL decision stability (moderate load, 30 runs): {stab:.2f}")

    abl = ablation_analysis()
    print("\nAblation (moderate load):")
    for k, v in abl.items():
        print(f"  {k:12s} lat={v['lat']:.2f} en={v['energy']:.3f} succ={v['success']:.1f} H={v['H']:.3f} it={v['iters']}")

    conv = convergence_trace()
    print(f"\nEnergy functional: {conv[0]:.3f} -> {conv[-1]:.3f} "
          f"({100*(1-conv[-1]/conv[0]):.2f}% reduction over {len(conv)-1} iterations)")

    out = dict(load_results=all_results, stability=stab, ablation=abl, convergence=conv,
               nodes=NODES)
    json_path = os.path.join(out_dir, "piel_generic_edge_results.json")
    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved -> {json_path}")

    # Flat CSV of the main performance table (Table 5 in the manuscript)
    csv_path = os.path.join(out_dir, "piel_generic_edge_results.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["load_scenario", "method", "latency_ms", "energy_J", "success_pct"])
        for load_name, methods in all_results.items():
            for method, m in methods.items():
                writer.writerow([load_name, method, f"{m['lat']:.4f}", f"{m['energy']:.4f}", f"{m['success']:.2f}"])
    print(f"Saved -> {csv_path}")
