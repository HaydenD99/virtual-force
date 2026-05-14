"""
Dynamic large-scale plus redraw script (EPS only)
===============================================
- No experiment rerun
- Read existing raw_results_plus.json
- Export EPS only
- Keep original plotting logic and layout
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

OUT_DIR = 'result/dynamic_large_scale_plus'
JSON_PATH = os.path.join(OUT_DIR, 'raw_results_plus.json')

NUM_STEPS = 10
DT = 5.0
ALG_NAMES = ['DE2VF-MR+', 'DGA-CF', 'DPSO-CF', 'NSSA-CF']
COLORS = {
    'DE2VF-MR+': '#ee6f63',
    'DGA-CF': '#5fa8e8',
    'DPSO-CF': '#63cfa0',
    'NSSA-CF': '#b08adf',
}
MARKS = {
    'DE2VF-MR+': 'X',
    'DGA-CF': 's',
    'DPSO-CF': '^',
    'NSSA-CF': 'D',
}


def plot_metrics(all_records, seeds):
    t = np.arange(NUM_STEPS + 1) * DT
    metric_defs = [
        ('min_rate', 'Min User Rate (Mbps)', 'fig_timeseries_min_rate'),
        ('jfi', 'JFI_eff', 'fig_timeseries_jfi'),
        ('energy_cumul', 'Cumulative Energy (kJ)', 'fig_timeseries_energy', 1 / 1000),
        ('joint_score', 'Joint Score', 'fig_timeseries_joint_score'),
    ]

    for item in metric_defs:
        if len(item) == 3:
            key, ylabel, file_tag = item
            scale = 1.0
        else:
            key, ylabel, file_tag, scale = item

        fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
        for a in ALG_NAMES:
            arr = np.array([all_records[s][a][key] for s in seeds], dtype=float) * scale
            m = arr.mean(axis=0)
            ax.plot(t, m, color=COLORS[a], marker=MARKS[a], linewidth=2.3,
                    markersize=6, label=a)

        ax.set_xlabel('Time (s)', fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.tick_params(axis='both', labelsize=13)
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(os.path.join(OUT_DIR, f'{file_tag}.eps'), format='eps', bbox_inches='tight')
        plt.close(fig)


def main():
    if not os.path.exists(JSON_PATH):
        print(f'Missing result file: {JSON_PATH}')
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    seeds = sorted([int(s) for s in raw.keys()])
    all_records = {int(s): raw[str(s)] for s in raw.keys()}

    plot_metrics(all_records, seeds)
    print(f'EPS figures saved to: {OUT_DIR}/')


if __name__ == '__main__':
    main()
