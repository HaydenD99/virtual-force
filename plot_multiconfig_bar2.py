"""
多配置柱状图 v2
===============
静态: L=9, G=6,  K=20/30/40
动态: L=6, G=9,  K=20/30/40
X轴: K 配置组 (低→高负载),  每组 4 根算法柱

Output: result/multiconfig2/fig_static_multibar.eps
        result/multiconfig2/fig_dynamic_multibar.eps
"""
import numpy as np
import json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams['font.family'] = ['Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['ps.fonttype'] = 42

OUT_DIR   = 'result/multiconfig2'
K_LIST    = [20, 30, 40]
ALG_S     = ['LB-BVF',        'GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']
ALG_D     = ['Dynamic LB-BVF','GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']
ALG_LABELS= ['BVF', 'DGA-CF', 'DPSO-CF', 'NSSA-CF']
ALG_COLORS= ['#ee6f63', '#5fa8e8', '#63cfa0', '#b08adf']



def load_stats(path, alg_list, is_dynamic=False):
    with open(path) as f:
        data = json.load(f)
    stats = {}
    for alg in alg_list:
        if is_dynamic:
            mr  = [data[s][alg]['min_rate']    for s in data]
            jfi = [data[s][alg]['jfi']         for s in data]
            js  = [data[s][alg]['joint_score'] for s in data]
        else:
            mr  = [data[s][alg]['min_rate']    for s in data]
            jfi = [data[s][alg]['jfi']         for s in data]
            js  = [data[s][alg]['joint_score'] for s in data]
        stats[alg] = {
            'mr':  (np.mean(mr),  np.std(mr)),
            'jfi': (np.mean(jfi), np.std(jfi)),
            'js':  (np.mean(js),  np.std(js)),
        }
    return stats


def draw_grouped_bar(ax, cfg_stats_list, cfg_labels, alg_names,
                     metric_key, ylabel):
    n_cfg = len(cfg_stats_list)
    n_alg = len(alg_names)
    grp_w = 0.72
    bar_w = grp_w / n_alg
    x_ctrs = np.arange(n_cfg)

    for ai, (alg, clr, lbl) in enumerate(
            zip(alg_names, ALG_COLORS, ALG_LABELS)):
        offsets = (ai - (n_alg - 1) / 2) * bar_w
        means = [s[alg][metric_key][0] for s in cfg_stats_list]
        stds  = [s[alg][metric_key][1] for s in cfg_stats_list]
        ax.bar(x_ctrs + offsets, means, bar_w * 0.90,
               color=clr,
               edgecolor='white', linewidth=0.6,
               yerr=stds, capsize=3,
               error_kw=dict(elinewidth=1.0, ecolor='#555', capthick=1.0),
               label=lbl, zorder=3)
        fmt = '.2f' if metric_key in ('jfi', 'js') else '.1f'
        for xp, m, s in zip(x_ctrs + offsets, means, stds):
            ax.text(xp, m + s + max(means) * 0.013,
                    f'{m:{fmt}}', ha='center', va='bottom',
                    fontsize=6.8, fontweight='bold', color='#222')

    ax.set_xticks(x_ctrs)
    ax.set_xticklabels(cfg_labels, fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_scenario(alg_list, prefix, L, G, title_prefix,
                  seeds_note, is_dynamic=False, split_subplots=False):
    cfg_stats = []
    for K in K_LIST:
        path = os.path.join(OUT_DIR, f'{prefix}_K{K}.json')
        cfg_stats.append(load_stats(path, alg_list, is_dynamic))

    cfg_labels = [f'K={K}\n(L={L}, G={G})' for K in K_LIST]

    metric_defs = [
        ('mr', 'Min Rate (Mbps)', 'min_rate'),
        ('jfi', 'JFI$_{eff}$', 'jfi'),
        ('js', 'Joint Score', 'joint_score'),
    ]

    if split_subplots:
        for metric_key, ylabel, suffix in metric_defs:
            fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.6))
            draw_grouped_bar(ax, cfg_stats, cfg_labels, alg_list,
                             metric_key, ylabel)
            handles = [mpatches.Patch(facecolor=c, edgecolor='#888', label=l)
                       for c, l in zip(ALG_COLORS, ALG_LABELS)]
            ax.legend(handles=handles, loc='upper right', ncol=1,
                      fontsize=9.2, frameon=True)
            plt.tight_layout()
            p = os.path.join(OUT_DIR, f'fig_{prefix}_{suffix}.eps')
            plt.savefig(p, format='eps', bbox_inches='tight')
            print(f'  ✓ {p}')
            plt.close(fig)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (metric_key, ylabel, _) in zip(axes, metric_defs):
            draw_grouped_bar(ax, cfg_stats, cfg_labels, alg_list,
                             metric_key, ylabel)

        handles = [mpatches.Patch(facecolor=c, edgecolor='#888', label=l)
                   for c, l in zip(ALG_COLORS, ALG_LABELS)]
        fig.legend(handles=handles, loc='lower center', ncol=4,
                   fontsize=10.5, frameon=True, bbox_to_anchor=(0.5, -0.04))
        plt.tight_layout()
        p = os.path.join(OUT_DIR, f'fig_{prefix}_multibar.eps')
        plt.savefig(p, format='eps', bbox_inches='tight')
        print(f'  ✓ {p}')
        plt.close(fig)


if __name__ == '__main__':
    needed = ([os.path.join(OUT_DIR, f'static_K{K}.json')  for K in K_LIST] +
              [os.path.join(OUT_DIR, f'dynamic_K{K}.json') for K in K_LIST])
    missing = [p for p in needed if not os.path.exists(p)]
    if missing:
        print("缺少文件, 请先运行实验脚本:"); [print(f"  {p}") for p in missing]
        import sys; sys.exit(1)

    print("静态场景图 ...")
    plot_scenario(ALG_S, 'static',  L=9, G=6,
                  title_prefix='Static UAV Deployment',
                  seeds_note='50 seeds per config',
                  split_subplots=True)

    print("动态场景图 ...")
    plot_scenario(ALG_D, 'dynamic', L=6, G=9,
                  title_prefix='Dynamic UAV Re-Deployment',
                  seeds_note='30 seeds per config, avg over 10 steps',
                  is_dynamic=True)

    print(f"\n完成! → {OUT_DIR}/")
