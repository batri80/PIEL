"""
================================================================================
  NYC Yellow Taxi -> PIEL Trace-Driven Evaluation (v3 -- corrected model)
  Run this on your local machine, in the same folder as your
  yellow_tripdata_*.csv files and PIEL_pipeline_v3.py.

  WHAT CHANGED FROM THE ORIGINAL process_nyc_taxi.py:
    - Trip -> task mapping (payload/cycles/RSU-zone/hour->load logic) is
      UNCHANGED -- that part was never in question.
    - The assignment/scoring model underneath now uses PIEL_pipeline_v3's
      corrected congestion model:
        * D(x) is entropy-based and correctly signed (rewards balance,
          not concentration)
        * Queueing delay is genuine M/M/1-style congestion driven by how
          much load actually lands on each RSU within the batch, not a
          static pre-existing backlog (which made queuing delay identical
          across RSUs regardless of capacity)
        * Every method (FIFO/Greedy/DQN/PIEL) is scored with the SAME
          uniform evaluate() function, so there's no decide-vs-evaluate
          mismatch like the one found in the original Greedy baseline.
    - Optionally loads your trained DQN checkpoint (train_dqn_v3.py /
      dqn_v3_best.pt) instead of the naive capacity-softmax proxy, if the
      checkpoint file is present.

  USAGE:
    1. Put this file, PIEL_pipeline_v3.py, and (optionally) train_dqn_v3.py
       + dqn_v3_best.pt in the same folder as your yellow_tripdata_*.csv
    2. pip install pandas numpy torch   (torch only needed for the real DQN)
    3. python process_nyc_taxi_v3.py

  OUTPUT:
    nyc_taxi_tasks_v3.csv        -- processed task trace (for your records)
    trace_results_v3.txt         -- all numbers, LaTeX-ready table
================================================================================
"""
import numpy as np
import pandas as pd
import os, glob, sys
from collections import Counter

from PIEL_pipeline_v3 import (
    load_config, load_rsu_profiles, compute_T_batch, compute_avg_service_time,
    evaluate, load_entropy, run_all_methods, piel_solve, greedy_assign,
    fifo_assign, dqn_approx_assign
)

np.random.seed(42)
rng = np.random.default_rng(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')

cfg = load_config(os.path.join(DATA_DIR, 'experiment_config.csv'))
rsu = load_rsu_profiles(os.path.join(DATA_DIR, 'rsu_profiles.csv'))
N_RSU = rsu['n']

# NYC bounding box -> 5 RSU coverage zones (UNCHANGED from original script)
RSU_ZONES = [
    {'lat_min':40.740,'lat_max':40.780,'lon_min':-74.010,'lon_max':-73.960},  # Midtown
    {'lat_min':40.700,'lat_max':40.740,'lon_min':-74.020,'lon_max':-73.960},  # Lower Manhattan
    {'lat_min':40.780,'lat_max':40.840,'lon_min':-73.970,'lon_max':-73.920},  # Upper Manhattan
    {'lat_min':40.650,'lat_max':40.700,'lon_min':-74.020,'lon_max':-73.900},  # Brooklyn/Queens
    {'lat_min':40.840,'lat_max':40.920,'lon_min':-73.950,'lon_max':-73.800},  # Bronx
]

# NYC TLC 2015 Annual Report, Exhibit 14 -- hourly demand multiplier (UNCHANGED)
TLC_HOUR_MULT = {
     0:0.31, 1:0.21, 2:0.16, 3:0.13, 4:0.14, 5:0.22,
     6:0.51, 7:0.84, 8:1.00, 9:0.92,10:0.79,11:0.82,
    12:0.86,13:0.83,14:0.80,15:0.87,16:0.95,17:1.00,
    18:1.05,19:0.95,20:0.84,21:0.79,22:0.62,23:0.41
}
def hour_to_scenario(h):
    m = TLC_HOUR_MULT.get(h, 0.5)
    if   m < 0.40: return 'low_20'
    elif m < 0.80: return 'moderate_50'
    else:          return 'high_90'

def load_factor_for_scenario(name):
    return {'low_20': 0.20, 'moderate_50': 0.50, 'high_90': 0.90}[name]


# ============================================================================
#  CSV PARSING (UNCHANGED trip -> task mapping logic)
# ============================================================================

def find_csv_files():
    files = []
    for name in ['yellow_tripdata_2015-01.csv', 'yellow_tripdata_2016-01.csv',
                 'yellow_tripdata_2016-02.csv', 'yellow_tripdata_2016-03.csv']:
        p = os.path.join(SCRIPT_DIR, name)
        if os.path.exists(p):
            files.append(p)
    for p in glob.glob(os.path.join(SCRIPT_DIR, 'yellow_tripdata_*.csv')):
        if p not in files:
            files.append(p)
    return sorted(files)


def load_and_parse(filepath, sample_n=80000):
    print(f"  Reading {os.path.basename(filepath)} ...", end=' ', flush=True)
    hdr = pd.read_csv(filepath, nrows=0)
    cols = [c.strip() for c in hdr.columns]
    col_lower = {c.lower(): c for c in cols}

    def find_col(*candidates):
        for cand in candidates:
            if cand.lower() in col_lower:
                return col_lower[cand.lower()]
        return None

    pickup_col  = find_col('tpep_pickup_datetime', 'pickup_datetime')
    dropoff_col = find_col('tpep_dropoff_datetime', 'dropoff_datetime')
    dist_col    = find_col('trip_distance')
    lon_col     = find_col('pickup_longitude')
    lat_col     = find_col('pickup_latitude')
    zone_col    = find_col('PULocationID', 'pulocationid')

    use_cols = [c for c in [pickup_col, dropoff_col, dist_col, lon_col, lat_col, zone_col] if c]

    total_rows = sum(1 for _ in open(filepath, encoding='utf-8', errors='replace')) - 1
    skip_every = max(1, total_rows // sample_n)
    skipfn = lambda i: i > 0 and i % skip_every != 0

    df = pd.read_csv(filepath, usecols=use_cols, skiprows=skipfn,
                      low_memory=False, encoding_errors='replace')
    print(f"{len(df):,} rows sampled from {total_rows:,} total")

    rename_map = {}
    if pickup_col:  rename_map[pickup_col]  = 'pickup_dt'
    if dropoff_col: rename_map[dropoff_col] = 'dropoff_dt'
    if dist_col:    rename_map[dist_col]    = 'trip_dist'
    if lon_col:     rename_map[lon_col]     = 'lon'
    if lat_col:     rename_map[lat_col]     = 'lat'
    if zone_col:    rename_map[zone_col]    = 'zone_id'
    df.rename(columns=rename_map, inplace=True)

    df['pickup_dt']  = pd.to_datetime(df['pickup_dt'], errors='coerce')
    df['dropoff_dt'] = pd.to_datetime(df.get('dropoff_dt', pd.Series(dtype='object')), errors='coerce')
    df['trip_dist']  = pd.to_numeric(df.get('trip_dist', 0), errors='coerce')
    df = df.dropna(subset=['pickup_dt', 'trip_dist'])

    if 'dropoff_dt' in df.columns:
        df['duration_min'] = (df['dropoff_dt'] - df['pickup_dt']).dt.total_seconds() / 60.0
    else:
        df['duration_min'] = df['trip_dist'] / 12.0 * 60

    df = df[(df['trip_dist'] >= 0.1) & (df['trip_dist'] <= 30.0) &
            (df['duration_min'] >= 1.0) & (df['duration_min'] <= 90.0)].copy()

    df['hour']          = df['pickup_dt'].dt.hour
    df['load_scenario'] = df['hour'].apply(hour_to_scenario)

    df['size_mb'] = np.clip(df['duration_min'] * 0.5 + rng.normal(0, 0.1, len(df)), 0.2, 4.0).round(4)
    df['cpu_cycles'] = np.clip(df['trip_dist'] * 3e8 + rng.normal(0, 8e7, len(df)), 5e8, 2.0e9).astype(np.int64)

    has_gps = 'lon' in df.columns and 'lat' in df.columns
    if has_gps:
        def gps_to_rsu(lon, lat):
            for i, z in enumerate(RSU_ZONES):
                if z['lat_min'] <= lat <= z['lat_max'] and z['lon_min'] <= lon <= z['lon_max']:
                    return i
            return int(rng.integers(0, N_RSU))
        df['nearest_rsu'] = df.apply(lambda r: gps_to_rsu(r.get('lon', 0), r.get('lat', 0)), axis=1)
    elif 'zone_id' in df.columns:
        def zone_to_rsu(z):
            try: z = int(z)
            except: return int(rng.integers(0, N_RSU))
            if   1   <= z <= 69:  return int(rng.integers(0, 2))
            elif 70  <= z <= 109: return 4
            elif 110 <= z <= 168: return 3
            elif 169 <= z <= 243: return int(rng.integers(2, 4))
            elif 244 <= z <= 263: return 4
            return int(rng.integers(0, N_RSU))
        df['nearest_rsu'] = df['zone_id'].apply(zone_to_rsu)
    else:
        df['nearest_rsu'] = rng.integers(0, N_RSU, len(df))

    return df[['load_scenario', 'size_mb', 'cpu_cycles', 'nearest_rsu', 'hour']]


# ============================================================================
#  ASSIGNMENT + EVALUATION USING THE CORRECTED v3 MODEL
# ============================================================================

def df_to_tasks(df):
    return {'size': df['size_mb'].to_numpy(), 'cycles': df['cpu_cycles'].to_numpy().astype(float)}


def evaluate_scenario(df_scenario, load_factor, seed=42, dqn_qnet=None):
    tasks = df_to_tasks(df_scenario)
    n = len(tasks['size'])
    print(f"    [{n:,} tasks] computing congestion model...", flush=True)
    T_batch = compute_T_batch(load_factor, tasks, rsu)
    avg_st  = compute_avg_service_time(tasks, rsu)

    print(f"    [{n:,} tasks] running PIEL coordinate descent (this is the slow part)...", flush=True)
    import time
    t0 = time.time()
    asgn_pief, _ = piel_solve(tasks, rsu, cfg, T_batch, avg_st, seed=seed, verbose=True)
    print(f"    PIEL done in {time.time()-t0:.1f}s", flush=True)

    print(f"    [{n:,} tasks] running FIFO / Greedy / DQN...", flush=True)
    results = {
        'FIFO'  : evaluate(fifo_assign(n, N_RSU), tasks, rsu, cfg, T_batch, avg_st),
        'Greedy': evaluate(greedy_assign(tasks, rsu, cfg, T_batch, avg_st), tasks, rsu, cfg, T_batch, avg_st),
        'PIEL'  : evaluate(asgn_pief, tasks, rsu, cfg, T_batch, avg_st),
    }
    if dqn_qnet is not None:
        # Real trained DQN, if a checkpoint was supplied (see main() below)
        from train_dqn_v3 import evaluate_dqn
        res = evaluate_dqn(dqn_qnet, tasks, load_factor, n_tasks=n)
        results['DQN'] = evaluate(res['assignments'], tasks, rsu, cfg, T_batch, avg_st)
    else:
        results['DQN'] = evaluate(dqn_approx_assign(tasks, rsu, seed=seed), tasks, rsu, cfg, T_batch, avg_st)

    H_pief = load_entropy(asgn_pief, N_RSU)
    return results, H_pief, n


def main():
    files = find_csv_files()
    if not files:
        print("ERROR: no yellow_tripdata_*.csv files found in this folder.")
        print("Place your NYC Yellow Taxi CSV files alongside this script and re-run.")
        sys.exit(1)

    print(f"Found {len(files)} trip file(s):")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    dfs = [load_and_parse(f) for f in files]
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal processed tasks: {len(df_all):,}")
    df_all.to_csv('nyc_taxi_tasks_v3.csv', index=False)

    # ── NEW: hourly demand histogram (for Fig. "taxi_demand") ──────────────
    hourly_counts = df_all['hour'].value_counts().sort_index()
    hourly_df = hourly_counts.reindex(range(24), fill_value=0).reset_index()
    hourly_df.columns = ['hour', 'task_count']
    hourly_df.to_csv('hourly_demand_v3.csv', index=False)
    print(f"Saved hourly_demand_v3.csv ({hourly_df['task_count'].sum():,} tasks across 24 hours)")

    # Optional: load a real trained DQN checkpoint if present
    dqn_qnet = None
    ckpt_path = 'dqn_v3_best.pt'
    if os.path.exists(ckpt_path):
        try:
            import torch
            from train_dqn_v3 import QNet, STATE_DIM, ACTION_DIM
            dqn_qnet = QNet(STATE_DIM, ACTION_DIM)
            dqn_qnet.load_state_dict(torch.load(ckpt_path))
            dqn_qnet.eval()
            print(f"\nLoaded trained DQN checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"\nCould not load {ckpt_path} ({e}); using naive DQN-proxy instead.")

    lines = []
    lines.append("="*72)
    lines.append("  NYC Yellow Taxi Trace -- PIEL v3 Evaluation")
    lines.append("="*72)

    for sc in ['low_20', 'moderate_50', 'high_90']:
        df_sc = df_all[df_all['load_scenario'] == sc].reset_index(drop=True)
        if len(df_sc) == 0:
            lines.append(f"\n  Scenario: {sc} -- no tasks found, skipping.")
            continue
        lf = load_factor_for_scenario(sc)
        results, H_pief, n = evaluate_scenario(df_sc, lf, seed=42, dqn_qnet=dqn_qnet)

        lines.append(f"\n  Scenario: {sc}  (n_tasks={n:,})")
        lines.append(f"  {'Method':<10} {'Latency(ms)':>12} {'Energy(J)':>10} {'Success(%)':>11}")
        for m in ['FIFO', 'Greedy', 'DQN', 'PIEL']:
            r = results[m]
            lines.append(f"  {m:<10} {r['latency']:>12.2f} {r['energy']:>10.4f} {r['success']:>10.1f}%")
        lines.append(f"  PIEL load-balance entropy H(x) = {H_pief:.3f}  (1.0 = perfectly balanced)")

        zone_names = ['Midtown Manhattan', 'Lower Manhattan', 'Upper Manhattan', 'Brooklyn/Queens', 'Bronx']
        zc = Counter(df_sc['nearest_rsu'].tolist())
        lines.append("  Spatial origin distribution (nearest_rsu, informational only -- "
                      "assignment is NOT constrained to this):")
        for i, name in enumerate(zone_names):
            pct = 100 * zc.get(i, 0) / n
            lines.append(f"    RSU {i} ({name}): {zc.get(i,0):,} trips ({pct:.1f}%)")

    out_text = "\n".join(lines)
    print("\n" + out_text)
    with open('trace_results_v3.txt', 'w') as f:
        f.write(out_text + "\n")
    print("\nSaved: nyc_taxi_tasks_v3.csv, trace_results_v3.txt")

    # ── NEW: convergence history on the real trace (for Fig. "trace_convergence") ──
    print("\nCapturing PIEL convergence history on moderate_50 trace (random init)...")
    df_mod = df_all[df_all['load_scenario'] == 'moderate_50'].reset_index(drop=True)
    if len(df_mod) > 0:
        tasks_mod = df_to_tasks(df_mod)
        T_batch_mod = compute_T_batch(0.50, tasks_mod, rsu)
        avg_st_mod = compute_avg_service_time(tasks_mod, rsu)
        _, history = piel_solve(tasks_mod, rsu, cfg, T_batch_mod, avg_st_mod,
                                 seed=42, random_init=True, verbose=True)
        hist_df = pd.DataFrame({'iteration': range(len(history)), 'energy': history})
        hist_df.to_csv('trace_convergence_v3.csv', index=False)
        print(f"Saved trace_convergence_v3.csv ({len(history)-1} iterations, "
              f"{history[0]:.4f} -> {history[-1]:.4f}, "
              f"{(1-history[-1]/history[0])*100:.2f}% reduction)")
    else:
        print("No moderate_50 tasks found; skipping convergence capture.")


if __name__ == '__main__':
    main()
