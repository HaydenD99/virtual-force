"""
English EPS-only redraw script from raw_results_mr2_quick.json
=============================================================
- No re-run of experiments
- Outputs EPS only
- Keeps current plotting style/ranges
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
JSON_PATH = os.path.join(OUT_DIR, 'raw_results_mr2_quick.json')

NUM_STEPS = 10
DT = 5.0

ALG_ORDER = ['DE2VF', 'DGA-CF', 'DPSO-CF', 'NSSA-CF']
COLORS = {
    'DE2VF': '#ee6f63',
    'DGA-CF': '#5fa8e8',
    'DPSO-CF': '#63cfa0',
    'NSSA-CF': '#b08adf'
}
MARKS = {
    'DE2VF': 'X',
    'DGA-CF': 's',
    'DPSO-CF': '^',
    'NSSA-CF': 'D'
}


def _normalize_key(raw_data):
    out = {}
    for s, rec in raw_data.items():
        if 'DE2VF' not in rec and 'DE2VF-MR2' in rec:
            rec['DE2VF'] = rec['DE2VF-MR2']
        out[s] = rec
    return out


def _collect(all_rec, key):
    seeds = sorted(all_rec.keys(), key=lambda x: int(x))
    arr_map = {}
    for alg in ALG_ORDER:
        series = []
        for s in seeds:
            if alg not in all_rec[s]:
                continue
            series.append(all_rec[s][alg][key])
        if series:
            arr_map[alg] = np.array(series)
    return arr_map


def plot_one(all_rec, key, ylabel, out_tag, scale=1.0):
    t = np.arange(NUM_STEPS + 1) * DT
    arr_map = _collect(all_rec, key)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
    for alg in ALG_ORDER:
        if alg not in arr_map:
            continue
        arr = arr_map[alg] * scale
        m = arr.mean(axis=0)
        ax.plot(t, m, color=COLORS[alg], marker=MARKS[alg], linewidth=2.3, markersize=6, label=alg)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylabel)

    if key == 'jfi':
        ax.set_ylim(0.75, 0.95)
    if key == 'energy_cumul':
        ax.set_ylim(0, 5000)
        ax.set_yticks(np.arange(0, 5001, 1000))
    if key == 'min_rate':
        ax.set_ylim(32, 65)
    if key == 'joint_score':
        ax.set_ylim(0.2, 1.0)

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{out_tag}.eps'), format='eps', bbox_inches='tight')
    plt.close(fig)


def plot_energy_eff(all_rec):
    t = np.arange(1, NUM_STEPS + 1) * DT
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))

    seeds = sorted(all_rec.keys(), key=lambda x: int(x))
    for alg in ALG_ORDER:
        series = []
        for s in seeds:
            if alg not in all_rec[s]:
                continue
            js = np.array(all_rec[s][alg]['joint_score'], dtype=float)[1:]
            ec_mj = (np.array(all_rec[s][alg]['energy_cumul'], dtype=float) / 1e6)[1:]
            ee = js / (ec_mj + 1e-9)
            series.append(ee)

        if not series:
            continue

        arr = np.array(series)
        m = arr.mean(axis=0)
        ax.plot(t, m, color=COLORS[alg], marker=MARKS[alg], linewidth=2.3, markersize=6, label=alg)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy Efficiency')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_energy_eff.eps'), format='eps', bbox_inches='tight')
    plt.close(fig)


def main():
    if not os.path.exists(JSON_PATH):
        print(f'Missing result file: {JSON_PATH}')
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    all_rec = _normalize_key(raw)

    plot_one(all_rec, 'min_rate', 'Min User Rate (Mbps)', 'fig_mr2_min_rate_en')
    plot_one(all_rec, 'jfi', r'$JFI_{\mathrm{eff}}$', 'fig_mr2_jfi_en')
    plot_one(all_rec, 'energy_cumul', 'Cumulative Energy (kJ)', 'fig_mr2_energy_en', scale=1 / 1000)
    plot_one(all_rec, 'joint_score', 'Joint Score', 'fig_mr2_joint_score_en')
    plot_energy_eff(all_rec)

    print(f'English EPS figures saved to: {OUT_DIR}/')


if __name__ == '__main__':
    main()
