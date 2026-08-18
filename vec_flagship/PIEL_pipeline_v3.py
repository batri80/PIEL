"""
PIEL pipeline v3 -- genuine within-batch congestion model.

Key changes from the original pipeline:
  1. D(x) is now normalised load-distribution ENTROPY (1 - H), correctly
     minimised when load is balanced, not when it's concentrated.
  2. Queueing delay is no longer a static pre-existing backlog (which made
     queuing latency identical across RSUs regardless of capacity, since
     queue = cap * load * qf made the cap terms cancel in queue/cap).
     Instead it's a proper M/M/1-style delay driven by how much load is
     ACTUALLY assigned to each RSU within this batch, relative to a batch
     time window T_batch derived from total demand / total capacity / the
     target load level. This makes concentration genuinely costly, so
     load-balancing is a real trade-off instead of a free lunch.
  3. evaluate() is now a pure function of the final assignment vector alone
     (steady-state utilisation), used IDENTICALLY for every method. This
     structurally eliminates the earlier decide-vs-evaluate mismatch bug
     (Greedy's decisions no longer can silently diverge from its scoring).
  4. [2026-08] piel_solve()'s coordinate-descent accept condition now
     requires a STRICT improvement (cost[best] < cost[cur]), not merely a
     changed index (best != cur). The previous condition could in principle
     accept a tie (cost[best] == cost[cur]) as a "move", which is not a
     genuine energy decrease; a finite state space plus non-increasing (as
     opposed to strictly decreasing) energy does not by itself rule out
     cycling. This is what the paper's Proposition 1 / Remark on strict
     improvement (Eq. 18) actually requires for the finite-termination
     argument to be rigorous. VERIFIED to have zero effect on every
     reported result: the original and patched versions were run side by
     side on identical inputs -- the full VEC synthetic suite (all three
     load levels, convergence trace, 30-run stability, full ablation study)
     and all three NYC trace scenarios (using the actual 237,868-task
     processed trace) -- and produced bit-for-bit identical output in every
     case. Expected, since task payloads/cycles are continuous-valued
     (Uniform-distributed synthetic tasks; real-valued trip-derived trace
     tasks) and no two RSUs share an identical (capacity, power) profile,
     so an exact tie in `cost` has probability essentially zero and was
     never actually triggered on this data.
"""
import numpy as np
import csv, os, sys
from collections import Counter

# ============================================================================
#  CSV LOADERS (unchanged from original pipeline)
# ============================================================================

def load_config(path):
    cfg = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            raw = row['value'].strip()
            try:
                cfg[row['parameter']] = int(raw) if '.' not in raw else float(raw)
            except ValueError:
                cfg[row['parameter']] = raw
    return cfg

def load_rsu_profiles(path):
    ids, caps, watts, types = [], [], [], []
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            ids.append(int(row['rsu_id'])); caps.append(float(row['capacity_ghz']))
            watts.append(float(row['power_watts'])); types.append(row['type'])
    return {'ids': ids, 'cap': np.array(caps), 'watts': np.array(watts),
            'types': types, 'n': len(ids)}

def load_tasks(path, scenario_filter=None):
    task_ids, scenarios, orders, sizes, cycles = [], [], [], [], []
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if scenario_filter and row['load_scenario'] != scenario_filter:
                continue
            task_ids.append(int(row['task_id'])); scenarios.append(row['load_scenario'])
            orders.append(int(row['arrival_order'])); sizes.append(float(row['size_mb']))
            cycles.append(float(row['cpu_cycles']))
    return {'task_id': task_ids, 'scenario': scenarios, 'order': orders,
            'size': np.array(sizes), 'cycles': np.array(cycles)}

def get_scenario_names(tasks_path):
    names = set()
    with open(tasks_path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            names.add(row['load_scenario'])
    return sorted(names)

def load_factor_from_name(scenario_name):
    try:
        return float(scenario_name.split('_')[1]) / 100.0
    except (IndexError, ValueError):
        return 0.50


# ============================================================================
#  CONGESTION MODEL
# ============================================================================

def compute_T_batch(load_factor, tasks, rsu):
    """
    Batch time window (seconds) this set of tasks is assumed to arrive over.
    Derived from total task demand / total system capacity / target load, so
    "load_factor" keeps its intended meaning: at load_factor=1.0, the batch
    exactly saturates the combined capacity of all RSUs if load were
    perfectly balanced. Lower load_factor -> longer window -> less congestion.
    """
    total_cycles = tasks['cycles'].sum()
    total_cap    = rsu['cap'].sum() * 1e9          # combined cycles/sec
    T_full       = total_cycles / total_cap         # seconds at 100% util
    return T_full / load_factor


def compute_avg_service_time(tasks, rsu):
    """Average per-task processing time (s) on each RSU, used as the M/M/1
    mean service time in the queueing-delay formula."""
    return tasks['cycles'].mean() / (rsu['cap'] * 1e9)


def compute_busy(assignments, tasks, rsu):
    """Total processing time (s) currently assigned to each RSU."""
    busy = np.zeros(rsu['n'])
    proc = tasks['cycles'] / (rsu['cap'][assignments] * 1e9)
    np.add.at(busy, assignments, proc)
    return busy


def latency_vec(task_idx, tasks, busy_excl, rsu, cfg, T_batch, avg_service_time):
    """Latency (ms) for task_idx on each candidate RSU, given the
    background load `busy_excl` (seconds, excluding task_idx itself)."""
    proc  = (tasks['cycles'][task_idx] / (rsu['cap'] * 1e9)) * 1000
    rho   = np.minimum(busy_excl / T_batch, 0.98)
    Wq_ms = avg_service_time * (rho / (1.0 - rho)) * 1000
    trans = tasks['size'][task_idx] * cfg['trans_latency_mspermb']
    return proc + Wq_ms + trans


def energy_vec(task_idx, tasks, busy_excl, rsu, cfg, T_batch):
    """Energy (J) for task_idx on each candidate RSU."""
    proc_time  = tasks['cycles'][task_idx] / (rsu['cap'] * 1e9)
    rho        = np.minimum(busy_excl / T_batch, 0.98)
    congestion = 1.0 / (1.0 - rho)
    return rsu['watts'] * proc_time * congestion + tasks['size'][task_idx] * cfg['trans_energy_jpermb']


def busy_excluding_self(task_idx, assignments, busy_all, tasks, rsu):
    """busy_all with task_idx's own contribution removed from its current RSU."""
    busy_excl = busy_all.copy()
    cur = assignments[task_idx]
    own = tasks['cycles'][task_idx] / (rsu['cap'][cur] * 1e9)
    busy_excl[cur] = max(busy_excl[cur] - own, 0.0)
    return busy_excl


def load_entropy(assignments, n_rsu):
    """Normalised Shannon entropy of the per-RSU load distribution.
    1.0 = perfectly balanced, 0.0 = fully concentrated on one RSU."""
    n = len(assignments)
    counts = np.bincount(assignments, minlength=n_rsu).astype(float)
    p = counts[counts > 0] / n
    if len(p) <= 1:
        return 0.0
    H = -np.sum(p * np.log(p))
    return float(H / np.log(n_rsu))


def entropy_from_counts(counts, n, n_rsu):
    """Same entropy formula as load_entropy(), but O(R) given per-RSU
    counts directly instead of O(N) recomputed from the full assignment
    array. This is the hot path inside coordinate descent."""
    p = counts[counts > 0] / n
    if len(p) <= 1:
        return 0.0
    H = -np.sum(p * np.log(p))
    return float(H / np.log(n_rsu))


def stability_delta_from_counts(counts, cur_rsu, candidate_rsu, n, n_rsu):
    """D(x) = 1 - H(x) if task currently at cur_rsu moves to candidate_rsu,
    computed from the small (size-R) counts array in O(R), not O(N).
    ~1000x+ faster than the original stability_delta() at real trace scale
    (100K+ tasks) since it never touches the full assignment array."""
    if cur_rsu == candidate_rsu:
        return 1.0 - entropy_from_counts(counts, n, n_rsu)
    test_counts = counts.copy()
    test_counts[cur_rsu] -= 1
    test_counts[candidate_rsu] += 1
    return 1.0 - entropy_from_counts(test_counts, n, n_rsu)


def stability_delta(assignments, task_idx, candidate_rsu, n_rsu):
    """D(x) = 1 - H(x) if task_idx -> candidate_rsu. Minimised when balanced.
    NOTE: O(N) -- kept only for total_energy_functional() (called once per
    iteration, not per-task). The coordinate-descent hot loop uses
    stability_delta_from_counts() instead."""
    test = assignments.copy()
    test[task_idx] = candidate_rsu
    return 1.0 - load_entropy(test, n_rsu)


# ============================================================================
#  UNIFORM EVALUATION (pure function of final assignment vector -- used
#  identically for every method, eliminating decide-vs-evaluate mismatch)
# ============================================================================

def compute_lat_eng(assignments, tasks, rsu, cfg, T_batch, avg_service_time):
    n = len(assignments)
    busy_all = compute_busy(assignments, tasks, rsu)
    lat = np.zeros(n); eng = np.zeros(n)
    for i in range(n):
        r = assignments[i]
        busy_excl_r = busy_excluding_self(i, assignments, busy_all, tasks, rsu)[r]
        rho = min(busy_excl_r / T_batch, 0.98)
        Wq_ms = avg_service_time[r] * (rho / (1.0 - rho)) * 1000
        proc  = (tasks['cycles'][i] / (rsu['cap'][r] * 1e9)) * 1000
        trans = tasks['size'][i] * cfg['trans_latency_mspermb']
        lat[i] = proc + Wq_ms + trans
        proc_time = tasks['cycles'][i] / (rsu['cap'][r] * 1e9)
        congestion = 1.0 / (1.0 - rho)
        eng[i] = rsu['watts'][r] * proc_time * congestion + tasks['size'][i] * cfg['trans_energy_jpermb']
    return lat, eng


def evaluate(assignments, tasks, rsu, cfg, T_batch, avg_service_time):
    lat, eng = compute_lat_eng(assignments, tasks, rsu, cfg, T_batch, avg_service_time)
    return {
        'latency': float(lat.mean()),
        'energy' : float(eng.mean()),
        'success': float((lat <= cfg['deadline_ms']).mean() * 100),
    }


def total_energy_functional(assignments, tasks, rsu, cfg, T_batch, avg_service_time):
    lat, eng = compute_lat_eng(assignments, tasks, rsu, cfg, T_batch, avg_service_time)
    L = ((lat / cfg['deadline_ms']) ** 2).mean()
    C = eng.mean() / 50.0
    D = 1.0 - load_entropy(assignments, rsu['n'])
    penalty = float(np.mean(lat > cfg['deadline_ms']) * 10.0)
    return cfg['alpha'] * L + cfg['beta'] * C + cfg['gamma'] * D + penalty


# ============================================================================
#  PIEL OPTIMISER (coordinate descent on E(x))
# ============================================================================

def piel_solve(tasks, rsu, cfg, T_batch, avg_service_time, seed=42,
               alpha=None, beta=None, gamma=None, random_init=False,
               return_history=True, verbose=False):
    a = alpha if alpha is not None else cfg['alpha']
    b = beta  if beta  is not None else cfg['beta']
    g = gamma if gamma is not None else cfg['gamma']

    rng = np.random.default_rng(seed)
    n = len(tasks['size'])

    if random_init:
        assignments = rng.integers(0, rsu['n'], size=n)
    else:
        # Warm start: min processing+transmission time, ignoring congestion
        # (a reasonable zero-congestion initial guess)
        assignments = np.array([
            int(np.argmin((tasks['cycles'][i] / (rsu['cap'] * 1e9)) * 1000
                          + tasks['size'][i] * cfg['trans_latency_mspermb']))
            for i in range(n)
        ])

    busy_all = compute_busy(assignments, tasks, rsu)
    counts = np.bincount(assignments, minlength=rsu['n']).astype(float)
    history = [total_energy_functional(assignments, tasks, rsu, cfg, T_batch, avg_service_time)] if return_history else None

    for _ in range(int(cfg['max_iter'])):
        changed = 0
        for i in rng.permutation(n):
            cur = assignments[i]
            own_cur = tasks['cycles'][i] / (rsu['cap'][cur] * 1e9)
            busy_excl = busy_all.copy()
            busy_excl[cur] = max(busy_excl[cur] - own_cur, 0.0)

            lat_raw = latency_vec(i, tasks, busy_excl, rsu, cfg, T_batch, avg_service_time)
            L_vec = (lat_raw / cfg['deadline_ms']) ** 2
            C_vec = energy_vec(i, tasks, busy_excl, rsu, cfg, T_batch) / 50.0
            D_vec = np.array([stability_delta_from_counts(counts, cur, r, n, rsu['n'])
                              for r in range(rsu['n'])])
            penalty = np.where(lat_raw > cfg['deadline_ms'], 10.0, 0.0)

            cost = a * L_vec + b * C_vec + g * D_vec + penalty
            best = int(np.argmin(cost))
            if best != cur and cost[best] < cost[cur]:  # strict improvement only (see module docstring, item 4)
                own_new = tasks['cycles'][i] / (rsu['cap'][best] * 1e9)
                busy_all[cur] -= own_cur
                busy_all[best] += own_new
                counts[cur] -= 1
                counts[best] += 1
                assignments[i] = best
                changed += 1
        if return_history:
            history.append(total_energy_functional(assignments, tasks, rsu, cfg, T_batch, avg_service_time))
        if verbose:
            print(f"      iter {_+1}: {changed:,} tasks moved", flush=True)
        if changed == 0:
            break

    return (assignments, history) if return_history else assignments


# ============================================================================
#  BASELINES
# ============================================================================

def fifo_assign(n, n_rsu):
    return np.arange(n) % n_rsu


def greedy_assign(tasks, rsu, cfg, T_batch, avg_service_time):
    """
    Sequential greedy: process tasks in arrival order, each picks the RSU
    minimising ITS OWN latency given load already committed by earlier
    decisions (no lookahead / no re-optimisation). Final scoring uses the
    same uniform evaluate() as every other method, based on the FINAL
    assignment's steady-state congestion -- so there's no way for decisions
    and scoring to diverge.
    """
    n = len(tasks['size'])
    assignments = np.zeros(n, dtype=int)
    busy_all = np.zeros(rsu['n'])
    for i in range(n):
        lat = latency_vec(i, tasks, busy_all, rsu, cfg, T_batch, avg_service_time)
        chosen = int(np.argmin(lat))
        assignments[i] = chosen
        busy_all[chosen] += tasks['cycles'][i] / (rsu['cap'][chosen] * 1e9)
    return assignments


def dqn_approx_assign(tasks, rsu, seed=42):
    """
    DRL-inspired stochastic proxy: softmax over raw RSU capacity, WITHOUT
    congestion awareness (distinguishing it from Greedy/PIEL). This is the
    placeholder later replaced by a genuinely trained DQN.
    """
    rng = np.random.default_rng(seed)
    score = rsu['cap']
    probs = np.exp(score) / np.exp(score).sum()
    return rng.choice(rsu['n'], size=len(tasks['size']), p=probs)


def run_all_methods(tasks, rsu, cfg, T_batch, avg_service_time, seed=42):
    asgn_p, _ = piel_solve(tasks, rsu, cfg, T_batch, avg_service_time, seed=seed)
    return {
        'FIFO'  : evaluate(fifo_assign(len(tasks['size']), rsu['n']), tasks, rsu, cfg, T_batch, avg_service_time),
        'Greedy': evaluate(greedy_assign(tasks, rsu, cfg, T_batch, avg_service_time), tasks, rsu, cfg, T_batch, avg_service_time),
        'DQN'   : evaluate(dqn_approx_assign(tasks, rsu, seed), tasks, rsu, cfg, T_batch, avg_service_time),
        'PIEL'  : evaluate(asgn_p, tasks, rsu, cfg, T_batch, avg_service_time),
    }, asgn_p


# ============================================================================
#  MAIN
# ============================================================================
if __name__ == '__main__':
    DATA_DIR = 'data'
    cfg = load_config(os.path.join(DATA_DIR, 'experiment_config.csv'))
    rsu = load_rsu_profiles(os.path.join(DATA_DIR, 'rsu_profiles.csv'))
    scenario_names = get_scenario_names(os.path.join(DATA_DIR, 'tasks.csv'))
    scenario_names = sorted(scenario_names, key=load_factor_from_name)

    print("="*68)
    print("  PIEL v3 -- genuine within-batch congestion model")
    print(f"  alpha={cfg['alpha']} beta={cfg['beta']} gamma={cfg['gamma']}")
    print("="*68)

    load_results = {}
    T_batches = {}
    avg_sts = {}
    for sc in scenario_names:
        lf = load_factor_from_name(sc)
        tasks = load_tasks(os.path.join(DATA_DIR, 'tasks.csv'), scenario_filter=sc)
        T_batch = compute_T_batch(lf, tasks, rsu)
        avg_st  = compute_avg_service_time(tasks, rsu)
        T_batches[sc] = T_batch; avg_sts[sc] = avg_st
        results, asgn_p = run_all_methods(tasks, rsu, cfg, T_batch, avg_st, seed=42)
        load_results[sc] = results
        print(f"\n  Scenario: {sc}  (T_batch={T_batch:.2f}s)")
        print(f"  {'Method':<10} {'Latency(ms)':>12} {'Energy(J)':>10} {'Success(%)':>11}")
        for m in ['FIFO', 'Greedy', 'DQN', 'PIEL']:
            r = results[m]
            print(f"  {m:<10} {r['latency']:>12.2f} {r['energy']:>10.4f} {r['success']:>10.1f}%")
        print(f"  PIEL assignment distribution: {Counter(asgn_p.tolist())}  H={load_entropy(asgn_p, rsu['n']):.3f}")

    # ── Exp 3: Energy convergence (moderate load, random init) ─────────────
    print("\n\n[Exp 3] Energy Functional Convergence")
    print("-"*68)
    mid_sc = 'moderate_50'
    tasks_conv = load_tasks(os.path.join(DATA_DIR, 'tasks.csv'), scenario_filter=mid_sc)
    _, conv_hist = piel_solve(tasks_conv, rsu, cfg, T_batches[mid_sc], avg_sts[mid_sc],
                               seed=42, random_init=True)
    print(f"  E(x) start: {conv_hist[0]:.5f}")
    print(f"  E(x) final: {conv_hist[-1]:.5f}")
    print(f"  Iterations: {len(conv_hist)-1}")
    print(f"  Reduction : {(1-conv_hist[-1]/conv_hist[0])*100:.2f}%")

    # ── Exp 4: Decision stability across 30 runs (moderate load) ───────────
    print("\n\n[Exp 4] Decision Stability (30 independent runs)")
    print("-"*68)
    tasks_stab = load_tasks(os.path.join(DATA_DIR, 'tasks.csv'), scenario_filter=mid_sc)
    T_b, avg_st = T_batches[mid_sc], avg_sts[mid_sc]
    n_runs = int(cfg['n_runs'])
    stab_runs = {m: [] for m in ['FIFO', 'Greedy', 'DQN', 'PIEL']}
    for run_idx in range(n_runs):
        s = run_idx * 7 + 13
        stab_runs['FIFO'].append(fifo_assign(len(tasks_stab['size']), rsu['n']))
        stab_runs['Greedy'].append(greedy_assign(tasks_stab, rsu, cfg, T_b, avg_st))
        stab_runs['DQN'].append(dqn_approx_assign(tasks_stab, rsu, seed=s))
        asgn_p = piel_solve(tasks_stab, rsu, cfg, T_b, avg_st, seed=s, return_history=False)
        stab_runs['PIEL'].append(asgn_p)
    stab_scores = {}
    for m, runs in stab_runs.items():
        arr = np.stack(runs)
        stab_scores[m] = round(arr.std(axis=0).mean() * 10, 2)
    print(f"  {'Method':<10} {'Stability Score (lower=more stable)':>38}")
    for m, s in stab_scores.items():
        print(f"  {m:<10} {s:>38.2f}")

    # ── Exp 5: Ablation study (moderate load) ───────────────────────────────
    print("\n\n[Exp 5] Ablation Study")
    print("-"*68)
    ablation_weights = {
        'Full PIEL (a.55 b.28 g.17)': (0.55, 0.28, 0.17),
        'No D(x)   (a.65 b.35 g.00)': (0.65, 0.35, 0.00),
        'No C(x)   (a.75 b.00 g.25)': (0.75, 0.00, 0.25),
        'High alpha(a.90 b.05 g.05)': (0.90, 0.05, 0.05),
    }
    for name, (a, b, g) in ablation_weights.items():
        asgn_ab, hist_ab = piel_solve(tasks_stab, rsu, cfg, T_b, avg_st, seed=42,
                                       alpha=a, beta=b, gamma=g)
        ev = evaluate(asgn_ab, tasks_stab, rsu, cfg, T_b, avg_st)
        H = load_entropy(asgn_ab, rsu['n'])
        print(f"  {name:<28} lat={ev['latency']:>8.2f} energy={ev['energy']:>7.3f} "
              f"success={ev['success']:>6.1f}% H={H:.3f} iters={len(hist_ab)-1}")

