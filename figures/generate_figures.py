"""
================================================================================
  Figure generation for "Physics-Informed Energy Landscape Optimization for
  Stable and Interpretable Decision-Making in Edge Computing" (PIEL paper)

  Regenerated fresh because the original figure-generation code was not
  available; all data below is either (a) directly re-computed here from the
  verified PIEF_pipeline_v3_final.py pipeline + uploaded data files, or
  (b) taken from values already published and cross-checked against the
  manuscript's tables/text (provenance noted per block). All labels/legends
  use "PIEL", matching the current manuscript naming -- NOT "PIEF".

  Produces exactly the 10 files referenced by \includegraphics in main-10.tex:
    fig_taxi_demand.png            fig1_latency_vs_load.png
    fig2_energy_vs_load.png        fig3_convergence.png
    fig5_ablation.png              fig_lambda_sensitivity.png
    fig_weight_pareto.png          fig6_trace_vs_synthetic.png
    fig_trace_convergence.png      fig7_rsu_spatial_distribution.png
================================================================================
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False,
    'grid.linewidth': 0.4, 'grid.alpha': 0.45,
})

# Consistent colour/marker scheme across all figures. PIEL is always last /
# highlighted; Water-Filling and trained-DQN colours are new relative to any
# earlier PIEF-era palette since the paper's baseline set now includes both.
COL = {
    'FIFO': '#73726c', 'Greedy': '#B4B2A9', 'Water-Filling': '#E8863C',
    'DQN': '#378ADD', 'PIEL': '#1D9E75',
}
MRK = {'FIFO': 's', 'Greedy': '^', 'Water-Filling': 'D', 'DQN': 'P', 'PIEL': 'o'}


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  \u2713 {name} saved")


# ==============================================================================
# DATA -- provenance noted per block
# ==============================================================================

# ---- Table 6 (VEC synthetic, flagship application): re-verified via
# PIEF_pipeline_v3_final.py for FIFO/Greedy/PIEL; Water-Filling and
# trained-DQN values taken from the manuscript's published Table 6 (those
# baselines' source scripts were not part of this upload). ----
TABLE6 = {
    'low_20': {
        'FIFO': (239.87, 5.7409, 89.0), 'Greedy': (187.15, 8.5839, 100.0),
        'Water-Filling': (184.54, 7.3949, 100.0), 'DQN': (202.80, 8.6038, 100.0),
        'PIEL': (180.47, 7.8027, 100.0),
    },
    'moderate_50': {
        'FIFO': (546.65, 10.3565, 56.7), 'Greedy': (316.95, 12.4637, 99.3),
        'Water-Filling': (312.56, 10.5535, 89.7), 'DQN': (322.37, 11.4978, 96.3),
        'PIEL': (307.61, 11.4784, 100.0),
    },
    'high_90': {
        'FIFO': (5527.33, 83.4972, 20.0), 'Greedy': (1436.47, 48.6651, 0.0),
        'Water-Filling': (1410.27, 44.1553, 0.0), 'DQN': (1454.21, 49.3407, 0.0),
        'PIEL': (1436.40, 48.0197, 0.0),
    },
}
LOAD_PCT = [20, 50, 90]
LOAD_KEYS = ['low_20', 'moderate_50', 'high_90']

# ---- Table 9 (NYC trace): re-verified directly against trace_results_v3.txt ----
TABLE9 = {
    'low_20': {'FIFO': (193.76, 4.5510, 94.3), 'Greedy': (147.08, 6.9204, 100.0),
               'DQN': (229.51, 5.0509, 95.6), 'PIEL': (142.01, 6.0547, 100.0)},
    'moderate_50': {'FIFO': (421.12, 7.6989, 74.1), 'Greedy': (220.38, 9.0300, 100.0),
                     'DQN': (372.56, 9.0902, 69.8), 'PIEL': (215.58, 8.0038, 100.0)},
    'high_90': {'FIFO': (3367.47, 51.9300, 36.3), 'Greedy': (1014.50, 35.0605, 0.0),
                'DQN': (1045.19, 34.2232, 0.0), 'PIEL': (1015.48, 34.4942, 0.0)},
}

# ---- Table 8 (ablation, moderate load): re-verified via pipeline re-run ----
ABLATION = {
    'Full PIEL': (307.61, 11.478, 100.0, 0.983, 9),
    r'No $\mathcal{D}(\mathbf{x})$': (307.54, 11.469, 100.0, 0.983, 9),
    r'No $\bar{\mathcal{C}}(\mathbf{x})$': (308.65, 11.876, 100.0, 0.977, 7),
    r'High-$\alpha$': (308.42, 11.855, 100.0, 0.978, 10),
}

print("Generating figures...\n")

# ==============================================================================
# Fig: NYC taxi hourly demand profile
# ==============================================================================
hourly = pd.read_csv('../results/hourly_demand_v3.csv')
fig, ax = plt.subplots(figsize=(6, 3.6))
colors = ['#E8863C' if h in (0, 4, 5) else ('#c0392b' if h == 18 else '#378ADD')
          for h in hourly['hour']]
# Match manuscript figure logic: highlight commute hours in orange, evening peak in red
colors = []
for h, c in zip(hourly['hour'], hourly['task_count']):
    if h == 18:
        colors.append('#c0392b')
    elif 8 <= h <= 20:
        colors.append('#E8863C')
    else:
        colors.append('#9AA5B1')
ax.bar(hourly['hour'], hourly['task_count'], color=colors, width=0.75)
mean_val = hourly['task_count'].mean()
ax.axhline(mean_val, color='#444', lw=1.0, ls='--', label=f'Mean ({mean_val:,.0f} tasks/hr)')
ax.set_xlabel('Hour of day'); ax.set_ylabel('Number of tasks')
ax.set_xticks(range(0, 24, 4))
ax.legend(fontsize=9, framealpha=0.3)
plt.tight_layout()
save(fig, 'fig_taxi_demand.png')

# ==============================================================================
# Fig 1 / Fig 2: Latency / Energy vs Network Load (VEC synthetic, Table 6)
# ==============================================================================
methods_order = ['FIFO', 'Greedy', 'Water-Filling', 'DQN', 'PIEL']
method_labels = {'FIFO': 'FIFO', 'Greedy': 'Greedy', 'Water-Filling': 'Water-Filling',
                  'DQN': 'DQN (trained)', 'PIEL': 'PIEL'}

fig, ax = plt.subplots(figsize=(5.5, 3.8))
for m in methods_order:
    vals = [TABLE6[sc][m][0] for sc in LOAD_KEYS]
    ax.plot(LOAD_PCT, vals, color=COL[m], marker=MRK[m], label=method_labels[m],
            lw=2.6 if m == 'PIEL' else 1.6, ls='-' if m in ('PIEL', 'DQN') else '--')
ax.set_yscale('log')
ax.set(xlabel='Network load (%)', ylabel='Average latency (ms)')
ax.set_xticks(LOAD_PCT)
ax.legend(framealpha=0.25, fontsize=9)
ax.grid(True, axis='y', which='both')
plt.tight_layout()
save(fig, 'fig1_latency_vs_load.png')

fig, ax = plt.subplots(figsize=(5.5, 3.8))
for m in methods_order:
    vals = [TABLE6[sc][m][1] for sc in LOAD_KEYS]
    ax.plot(LOAD_PCT, vals, color=COL[m], marker=MRK[m], label=method_labels[m],
            lw=2.6 if m == 'PIEL' else 1.6, ls='-' if m in ('PIEL', 'DQN') else '--')
ax.set(xlabel='Network load (%)', ylabel='Energy consumption (J/task)')
ax.set_xticks(LOAD_PCT)
ax.legend(framealpha=0.25, fontsize=9)
ax.grid(True, axis='y')
plt.tight_layout()
save(fig, 'fig2_energy_vs_load.png')

# ==============================================================================
# Fig 3: Energy functional convergence (VEC synthetic, moderate load)
# re-verified: 7.073 -> 0.400 over 7 iterations
# ==============================================================================
conv_vec = [7.0734, 0.4177, 0.4029, 0.4009, 0.4001, 0.39985, 0.39981, 0.39981]
fig, ax = plt.subplots(figsize=(5.5, 3.8))
iters = list(range(len(conv_vec)))
ax.plot(iters, conv_vec, color=COL['PIEL'], lw=2.5, marker='o', ms=6, label=r'$\mathcal{E}(\mathbf{x})$')
eq = conv_vec[-1] * 1.05
ax.axhline(eq, color='#E24B4A', lw=1.2, ls='--', label='Equilibrium region')
ax.fill_between(iters, conv_vec, eq, color=COL['PIEL'], alpha=0.09)
ax.set(xlabel='Coordinate descent iteration', ylabel=r'Energy functional $\mathcal{E}(\mathbf{x})$')
ax.legend(framealpha=0.25, fontsize=9)
ax.grid(True, axis='y')
plt.tight_layout()
save(fig, 'fig3_convergence.png')

# ==============================================================================
# Fig 5: Ablation study (4-panel, moderate load) -- re-verified numbers
# ==============================================================================
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
variants = list(ABLATION.keys())
lat_v = [ABLATION[v][0] for v in variants]
en_v = [ABLATION[v][1] for v in variants]
succ_v = [ABLATION[v][2] for v in variants]
H_v = [ABLATION[v][3] for v in variants]
colors4 = ['#1D9E75', '#5AC8A8', '#E8863C', '#B4675B']

for ax, vals, ylabel, ylim in zip(
        axes, [lat_v, en_v, succ_v, H_v],
        ['Latency (ms)', 'Energy (J)', 'Success (%)', r'Entropy $H(\mathbf{x})$'],
        [(300, 312), (11.3, 12.0), (95, 101), (0.95, 1.00)]):
    bars = ax.bar(range(4), vals, color=colors4, width=0.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Full\nPIEL', 'No\n' r'$\mathcal{D}(\mathbf{x})$', 'No\n' r'$\bar{\mathcal{C}}(\mathbf{x})$', r'High-$\alpha$'],
                        fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(ylim)
    ax.grid(True, axis='y')
plt.tight_layout()
save(fig, 'fig5_ablation.png')

# ==============================================================================
# Fig: Lambda (deadline penalty) sensitivity -- re-verified: fully flat
# ==============================================================================
lambda_grid = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
mod_succ = [100.0] * len(lambda_grid)
high_succ = [0.0] * len(lambda_grid)
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.plot(lambda_grid, mod_succ, color=COL['PIEL'], marker='o', ms=7, lw=2, label='Moderate load (50%)')
ax.plot(lambda_grid, high_succ, color='#c0392b', marker='s', ms=7, lw=2, label='High load (90%)')
ax.set_xscale('log')
ax.set(xlabel=r'Deadline penalty $\lambda$', ylabel='Task success rate (%)')
ax.set_ylim(-5, 105)
ax.legend(framealpha=0.3, fontsize=9)
ax.grid(True, axis='y')
plt.tight_layout()
save(fig, 'fig_lambda_sensitivity.png')

# ==============================================================================
# Fig: Weight (alpha/beta/gamma) Pareto trade-off -- re-verified sweep
# ==============================================================================
weight_sweep = [
    (0.90, 308.42, 11.8546), (0.75, 308.31, 11.7719), (0.55, 307.61, 11.4784),
    (0.40, 307.85, 11.0753), (0.25, 310.61, 10.5941), (0.10, 325.93, 9.7939),
]
alphas = [w[0] for w in weight_sweep]
lats = [w[1] for w in weight_sweep]
ens = [w[2] for w in weight_sweep]
fig, ax = plt.subplots(figsize=(5.5, 3.8))
sc = ax.scatter(ens, lats, c=alphas, cmap='viridis_r', s=90, edgecolor='k', linewidth=0.6, zorder=3)
ax.plot(ens, lats, color='#999', lw=1.0, ls=':', zorder=2)
# mark default
di = alphas.index(0.55)
ax.scatter([ens[di]], [lats[di]], s=180, facecolor='none', edgecolor='#c0392b', linewidth=2, zorder=4)
ax.annotate('Default (paper)', xy=(ens[di], lats[di]), xytext=(ens[di]+0.15, lats[di]+3),
            fontsize=8, color='#c0392b')
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label(r'Latency weight $\alpha$', fontsize=9)
ax.set(xlabel='Energy (J/task)', ylabel='Latency (ms)')
ax.set_xlim(9.5, 12.6)
ax.grid(True)
plt.tight_layout()
save(fig, 'fig_weight_pareto.png')

# ==============================================================================
# Fig 6: Synthetic (VEC) vs NYC trace comparison, grouped bars, 3 panels
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
metrics = [(0, 'Latency (ms)', 'Latency'), (1, 'Energy (J/task)', 'Energy'), (2, 'Success rate (%)', 'Success rate')]
methods_bar = ['FIFO', 'Greedy', 'DQN', 'PIEL']
x = np.arange(3)
w = 0.18
for ax, (idx, ylabel, title) in zip(axes, metrics):
    for j, m in enumerate(methods_bar):
        synth_vals = [TABLE6[sc][m][idx] for sc in LOAD_KEYS]
        trace_vals = [TABLE9[sc][m][idx] for sc in LOAD_KEYS]
        offset = (j - 1.5) * w
        ax.bar(x + offset, synth_vals, w * 0.9, color=COL[m], alpha=0.9,
               label=method_labels[m] if idx == 2 else None)
        ax.bar(x + offset, trace_vals, w * 0.9, color=COL[m], alpha=0.45, hatch='//',
               label=None)
    if idx == 0:
        ax.set_yscale('log')
    ax.set_xticks(x); ax.set_xticklabels(['Low\n(20%)', 'Moderate\n(50%)', 'High\n(90%)'], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, axis='y')
legend_elems = [Patch(facecolor=COL[m], label=method_labels[m]) for m in methods_bar] + [
    Patch(facecolor='gray', alpha=0.9, label='Solid = VEC synthetic'),
    Patch(facecolor='gray', alpha=0.45, hatch='//', label='Hatched = NYC trace'),
]
fig.legend(handles=legend_elems, loc='lower center', ncol=6, fontsize=9, framealpha=0.3,
           bbox_to_anchor=(0.5, -0.08))
plt.tight_layout(rect=[0, 0.08, 1, 1])
save(fig, 'fig6_trace_vs_synthetic.png')

# ==============================================================================
# Fig: NYC trace convergence (from actual trace_convergence_v3.csv)
# ==============================================================================
tc = pd.read_csv('../results/trace_convergence_v3.csv')
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.plot(tc['iteration'], tc['energy'], color=COL['PIEL'], lw=2.5, marker='o', ms=6,
        label=r'$\mathcal{E}(\mathbf{x})$ -- NYC trace')
eq = tc['energy'].iloc[-1] * 1.05
ax.axhline(eq, color='#E24B4A', lw=1.2, ls='--', label='Equilibrium region')
ax.fill_between(tc['iteration'], tc['energy'], eq, color=COL['PIEL'], alpha=0.09)
ax.set(xlabel='Coordinate descent iteration', ylabel=r'Energy functional $\mathcal{E}(\mathbf{x})$')
ax.legend(framealpha=0.25, fontsize=9)
ax.grid(True, axis='y')
plt.tight_layout()
save(fig, 'fig_trace_convergence.png')

# ==============================================================================
# Fig 7: RSU spatial distribution of trip origins (from trace_results_v3.txt)
# ==============================================================================
zone_names = ['RSU0\nMidtown', 'RSU1\nLower Manh.', 'RSU2\nUpper Manh.', 'RSU3\nBrooklyn/Qns', 'RSU4\nBronx']
spatial = {
    'Low (20%)': [45.0, 40.0, 7.0, 4.9, 3.1],
    'Moderate (50%)': [54.2, 29.4, 8.3, 4.6, 3.6],
    'High (90%)': [56.3, 24.6, 10.8, 4.5, 3.9],
}
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(5)
w = 0.26
colors_load = {'Low (20%)': '#378ADD', 'Moderate (50%)': '#E8863C', 'High (90%)': '#1D9E75'}
for i, (label, vals) in enumerate(spatial.items()):
    ax.bar(x + (i - 1) * w, vals, w * 0.9, color=colors_load[label], label=label)
ax.set_xticks(x); ax.set_xticklabels(zone_names, fontsize=9)
ax.set_ylabel('Share of trip origins (%)')
ax.legend(framealpha=0.3, fontsize=9)
ax.grid(True, axis='y')
plt.tight_layout()
save(fig, 'fig7_rsu_spatial_distribution.png')

print("\nAll 10 figures generated.")
