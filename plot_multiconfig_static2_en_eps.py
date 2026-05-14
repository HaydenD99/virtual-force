"""
English EPS plots for multiconfig static results
=================================================
Reads:
  result/multiconfig2/static_K20.json
  result/multiconfig2/static_K30.json
  result/multiconfig2/static_K40.json

Outputs EPS only.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = ['Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['ps.fonttype'] = 42

OUT_DIR = 'result/multiconfig2'
K_LIST = [20, 30, 40]
ALG_ORDER = ['LB-BVF', 'GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']
COLORS = {
    'LB-BVF': '#ee6f63',
    'GA-3D-LB': '#5fa8e8',
    'PSO-3D-LB': '#63cfa0',
    'SSA-3D-LB': '#b08adf',
}
MARKS = {'LB-BVF': 'X', 'GA-3D-LB': 's', 'PSO-3D-LB': '^', 'SSA-3D-LB': 'D'}
LABELS = {'LB-BVF': 'LB-BVF', 'GA-3D-LB': 'DGA-CF', 'PSO-3D-LB': 'DPSO-CF', 'SSA-3D-LB': 'NSSA-CF'}


def load_json(k):
    path = os.path.join(OUT_DIR, f'static_K{k}.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def collect(data, key):
    seeds = sorted(data.keys(), key=lambda x: int(x))
    out = {}
    for alg in ALG_ORDER:
        out[alg] = np.array([data[s][alg][key] for s in seeds], dtype=float)
    return out


def plot_metric(metric_key, ylabel, out_tag):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False)

    for ax, k in zip(axes, K_LIST):
        data = load_json(k)
        arr_map = collect(data, metric_key)

        for alg in ALG_ORDER:
            vals = arr_map[alg]
            ax.plot([0], [vals.mean()], marker=MARKS[alg], color=COLORS[alg],
                    linestyle='None', markersize=8, label=LABELS[alg])

        ax.set_title(f'K = {k}', fontweight='bold')
        ax.set_xticks([])
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(os.path.join(OUT_DIR, f'{out_tag}.eps'), format='eps', bbox_inches='tight')
    plt.close(fig)


def main():
    plot_metric('min_rate', 'Min User Rate (Mbps)', 'fig_static_min_rate_en')
    plot_metric('jfi', 'JFI', 'fig_static_jfi_en')
    plot_metric('joint_score', 'Joint Score', 'fig_static_joint_score_en')
    print(f'English EPS static figures saved to: {OUT_DIR}/')


if __name__ == '__main__':
    main()
