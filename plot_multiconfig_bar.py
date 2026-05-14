"""
多配置对比柱状图
================
加载静态/动态各三组配置结果, 生成分组柱状图

静态 : A(K=40,L=6,G=9)  B(K=50,L=8,G=9)  C(K=40,L=9,G=9)
动态 : A(K=30,L=9,G=6)  B(K=40,L=12,G=6)  C(K=20,L=6,G=6)

图布局 (每个场景独立一张 3-subplot 图):
  subplot1: Min Rate (Mbps)
  subplot2: JFI_eff
  subplot3: Joint Score
  x 轴: 3 个配置组, 每组 4 根算法柱

输出: result/multiconfig/fig_static_multibar.png/pdf
      result/multiconfig/fig_dynamic_multibar.png/pdf
"""

import numpy as np
import json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = 'result/multiconfig'
os.makedirs(OUT_DIR, exist_ok=True)

# ── 算法样式 ──────────────────────────────────────────────────────────
ALG_S = ['LB-BVF', 'GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']
ALG_D = ['Dynamic LB-BVF', 'GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']
ALG_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
ALG_LABELS = ['LB-BVF (Ours)', 'GA-3D', 'PSO-3D', 'SSA-3D']
HATCHES    = ['', '///', '\\\\\\', 'xxx']


# ══════════════════════════════════════════════════════════════════════
#  数据加载
# ══════════════════════════════════════════════════════════════════════

def load_static_cfg(cfg_name):
    """从 JSON 计算各算法在所有种子上的均值 / 标准差"""
    path = os.path.join(OUT_DIR, f'static_cfg{cfg_name}.json')
    with open(path) as f:
        data = json.load(f)
    stats = {}
    for alg in ALG_S:
        mr  = [data[s][alg]['min_rate']    for s in data]
        jfi = [data[s][alg]['jfi']         for s in data]
        js  = [data[s][alg]['joint_score'] for s in data]
        stats[alg] = {
            'mr':  (np.mean(mr),  np.std(mr)),
            'jfi': (np.mean(jfi), np.std(jfi)),
            'js':  (np.mean(js),  np.std(js)),
        }
    return stats


def load_static_A():
    """加载已有静态 Config A 结果"""
    path = 'result/static_full_comparison/raw_results.json'
    with open(path) as f:
        data = json.load(f)
    stats = {}
    for alg in ALG_S:
        mr  = [data[s][alg]['min_rate']    for s in data]
        jfi = [data[s][alg]['jfi']         for s in data]
        js  = [data[s][alg]['joint_score'] for s in data]
        stats[alg] = {
            'mr':  (np.mean(mr),  np.std(mr)),
            'jfi': (np.mean(jfi), np.std(jfi)),
            'js':  (np.mean(js),  np.std(js)),
        }
    return stats


def load_dynamic_cfg(cfg_name):
    path = os.path.join(OUT_DIR, f'dynamic_cfg{cfg_name}.json')
    with open(path) as f:
        data = json.load(f)
    stats = {}
    for alg in ALG_D:
        mr  = [data[s][alg]['min_rate']    for s in data]
        jfi = [data[s][alg]['jfi']         for s in data]
        js  = [data[s][alg]['joint_score'] for s in data]
        stats[alg] = {
            'mr':  (np.mean(mr),  np.std(mr)),
            'jfi': (np.mean(jfi), np.std(jfi)),
            'js':  (np.mean(js),  np.std(js)),
        }
    return stats


def load_dynamic_A():
    """加载已有动态 Config A 结果 (dynamic_large_scale)"""
    path = 'result/dynamic_large_scale/raw_results.json'
    with open(path) as f:
        data = json.load(f)
    stats = {}
    for alg in ALG_D:
        mr_all, jfi_all, js_all = [], [], []
        for s in data:
            # 取步骤 1~end 的均值
            mr_all.append( np.mean(data[s][alg]['min_rate'][1:]))
            jfi_all.append(np.mean(data[s][alg]['jfi'][1:]))
            js_all.append( np.mean(data[s][alg]['joint_score'][1:]))
        stats[alg] = {
            'mr':  (np.mean(mr_all),  np.std(mr_all)),
            'jfi': (np.mean(jfi_all), np.std(jfi_all)),
            'js':  (np.mean(js_all),  np.std(js_all)),
        }
    return stats


# ══════════════════════════════════════════════════════════════════════
#  通用分组柱状图绘制
# ══════════════════════════════════════════════════════════════════════

def draw_grouped_bar(ax, cfg_stats_list, cfg_labels, alg_names,
                     metric_key, ylabel, title):
    """
    cfg_stats_list : list of dict  (每个配置的 stats)
    cfg_labels     : x 轴分组标签
    alg_names      : 算法名列表
    metric_key     : 'mr' | 'jfi' | 'js'
    """
    n_cfg  = len(cfg_stats_list)
    n_alg  = len(alg_names)
    grp_w  = 0.7          # 每组总宽度
    bar_w  = grp_w / n_alg
    x_ctrs = np.arange(n_cfg)   # 每组中心

    for ai, (alg, clr, hatch, lbl) in enumerate(
            zip(alg_names, ALG_COLORS, HATCHES, ALG_LABELS)):
        offsets = (ai - (n_alg-1)/2) * bar_w
        means = [stats[alg][metric_key][0] for stats in cfg_stats_list]
        stds  = [stats[alg][metric_key][1] for stats in cfg_stats_list]
        ax.bar(x_ctrs + offsets, means, bar_w * 0.92,
               color=clr, hatch=hatch,
               edgecolor='white', linewidth=0.6,
               yerr=stds, capsize=3,
               error_kw=dict(elinewidth=1.0, ecolor='#555', capthick=1.0),
               label=lbl, zorder=3)
        # 数值
        for xp, m, s in zip(x_ctrs + offsets, means, stds):
            ax.text(xp, m + s + (max(means)*0.012 if max(means) > 0 else 0.004),
                    f'{m:.2f}' if metric_key in ('jfi', 'js') else f'{m:.1f}',
                    ha='center', va='bottom', fontsize=6.5,
                    fontweight='bold', color='#222', rotation=0)

    ax.set_xticks(x_ctrs)
    ax.set_xticklabels(cfg_labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ══════════════════════════════════════════════════════════════════════
#  静态场景图
# ══════════════════════════════════════════════════════════════════════

def plot_static():
    stats_A = load_static_A()
    stats_B = load_static_cfg('B')
    stats_C = load_static_cfg('C')
    cfg_list   = [stats_A, stats_B, stats_C]
    cfg_labels = ['Config A\n(K=40, L=6, G=9)',
                  'Config B\n(K=50, L=8, G=9)',
                  'Config C\n(K=40, L=9, G=9)']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    draw_grouped_bar(axes[0], cfg_list, cfg_labels, ALG_S,
                     'mr', 'Min Rate (Mbps)', '(a) Minimum User Rate')
    draw_grouped_bar(axes[1], cfg_list, cfg_labels, ALG_S,
                     'jfi', 'JFI$_{eff}$', '(b) Load Balancing (JFI$_{eff}$)')
    draw_grouped_bar(axes[2], cfg_list, cfg_labels, ALG_S,
                     'js', 'Joint Score', '(c) Joint Score')

    # 共享图例
    handles = [mpatches.Patch(facecolor=c, hatch=h, edgecolor='#888', label=l)
               for c, h, l in zip(ALG_COLORS, HATCHES, ALG_LABELS)]
    fig.legend(handles=handles, loc='lower center',
               ncol=4, fontsize=10.5, frameon=True,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Static UAV Deployment: Multi-Configuration Comparison\n'
                 '(100 / 50 / 50 independent seeds)',
                 fontsize=12.5, fontweight='bold', y=1.01)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        p = os.path.join(OUT_DIR, f'fig_static_multibar.{ext}')
        plt.savefig(p, dpi=200, bbox_inches='tight')
        print(f'  ✓ {p}')
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  动态场景图
# ══════════════════════════════════════════════════════════════════════

def plot_dynamic():
    stats_A = load_dynamic_A()
    stats_B = load_dynamic_cfg('B')
    stats_C = load_dynamic_cfg('C')
    cfg_list   = [stats_A, stats_B, stats_C]
    cfg_labels = ['Config A\n(K=30, L=9, G=6)',
                  'Config B\n(K=40, L=12, G=6)',
                  'Config C\n(K=20, L=6, G=6)']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    draw_grouped_bar(axes[0], cfg_list, cfg_labels, ALG_D,
                     'mr', 'Min Rate (Mbps)', '(a) Minimum User Rate')
    draw_grouped_bar(axes[1], cfg_list, cfg_labels, ALG_D,
                     'jfi', 'JFI$_{eff}$', '(b) Load Balancing (JFI$_{eff}$)')
    draw_grouped_bar(axes[2], cfg_list, cfg_labels, ALG_D,
                     'js', 'Joint Score', '(c) Joint Score')

    alg_labels_d = ['DLB-BVF (Ours)', 'GA-3D', 'PSO-3D', 'SSA-3D']
    handles = [mpatches.Patch(facecolor=c, hatch=h, edgecolor='#888', label=l)
               for c, h, l in zip(ALG_COLORS, HATCHES, alg_labels_d)]
    fig.legend(handles=handles, loc='lower center',
               ncol=4, fontsize=10.5, frameon=True,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Dynamic UAV Re-Deployment: Multi-Configuration Comparison\n'
                 '(100 / 30 / 30 seeds,  avg over 10 time steps)',
                 fontsize=12.5, fontweight='bold', y=1.01)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        p = os.path.join(OUT_DIR, f'fig_dynamic_multibar.{ext}')
        plt.savefig(p, dpi=200, bbox_inches='tight')
        print(f'  ✓ {p}')
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys

    # 检查所需文件是否存在
    needed = [
        'result/static_full_comparison/raw_results.json',
        'result/dynamic_large_scale/raw_results.json',
        os.path.join(OUT_DIR, 'static_cfgB.json'),
        os.path.join(OUT_DIR, 'static_cfgC.json'),
        os.path.join(OUT_DIR, 'dynamic_cfgB.json'),
        os.path.join(OUT_DIR, 'dynamic_cfgC.json'),
    ]
    missing = [p for p in needed if not os.path.exists(p)]
    if missing:
        print("缺少以下文件, 请先运行对应实验脚本:")
        for p in missing: print(f"  {p}")
        sys.exit(1)

    print("生成静态多配置柱状图 ...")
    plot_static()
    print("生成动态多配置柱状图 ...")
    plot_dynamic()
    print("\n完成! 结果保存在:", OUT_DIR)
