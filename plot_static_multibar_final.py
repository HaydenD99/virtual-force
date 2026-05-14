"""
静态多配置最终图
================
上排: 分组柱状图  (MinRate / JFI / JointScore)
下排: 箱线图      (JFI_eff  /  JointScore)
调整：替换回原版四色，精准锁定各图例在指定 Y 轴坐标
"""
import numpy as np
import json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR   = 'result/multiconfig2'
K_LIST    = [20, 30, 40]
ALG_S     = ['LB-BVF', 'GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']
ALG_LABELS= ['LB-BVF', 'GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']

# 替换为你要求的颜色方案
COLORS    = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']


# ── 数据加载 ──────────────────────────────────────────────────────────
def load_raw(K):
    path = os.path.join(OUT_DIR, f'static_K{K}.json')
    with open(path) as f:
        return json.load(f)

def get_stats(data, alg, key):
    vals = [data[s][alg][key] for s in data]
    return np.mean(vals), np.std(vals), vals   # mean, std, raw list

# ── 独立图例添加 ──────────────────────────────────────────────────────
def add_subplot_legend(ax, plot_type='bar', legend_y=None):
    handles = [mpatches.Patch(facecolor=c, edgecolor='#555', linewidth=0.6, label=l)
               for c, l in zip(COLORS, ALG_LABELS)]
    
    if plot_type == 'bar':
        # 柱状图：单列置于右上角。如果有 legend_y，将图例下边缘锁在特定的 Y 值
        if legend_y is not None:
            ax.legend(handles=handles, loc='lower right', ncol=1, 
                      fontsize=8.5, framealpha=0.90, handlelength=1.2,
                      bbox_to_anchor=(1.0, legend_y), bbox_transform=ax.get_yaxis_transform())
        else:
            ax.legend(handles=handles, loc='upper right', ncol=1, 
                      fontsize=8.5, framealpha=0.90, handlelength=1.2)
            
    elif plot_type == 'box':
        # 箱线图：ncol=4 形成一横排置于顶部居中。如果有 legend_y，锁在该 Y 值之上
        if legend_y is not None:
            ax.legend(handles=handles, loc='lower center', ncol=4, 
                      fontsize=8.5, framealpha=0.90, handlelength=1.2, columnspacing=1.0,
                      bbox_to_anchor=(0.5, legend_y), bbox_transform=ax.get_yaxis_transform())
        else:
            ax.legend(handles=handles, loc='upper center', ncol=4, 
                      fontsize=8.5, framealpha=0.90, handlelength=1.2, columnspacing=1.0)


# ── 柱状图子图 ────────────────────────────────────────────────────────
def bar_subplot(ax, datasets, cfg_labels, alg_list, metric_key, ylabel, sublabel, legend_y=None, ylim_top=None):
    n_cfg = len(datasets)
    n_alg = len(alg_list)
    grp_w = 0.72
    bar_w = grp_w / n_alg
    x_ctrs = np.arange(n_cfg)
    
    max_y_val = 0  

    for ai, (alg, clr, lbl) in enumerate(zip(alg_list, COLORS, ALG_LABELS)):
        offsets = (ai - (n_alg - 1) / 2) * bar_w
        means, stds = [], []
        for data in datasets:
            m, s, _ = get_stats(data, alg, metric_key)
            means.append(m); stds.append(s)
            
        ax.bar(x_ctrs + offsets, means, bar_w * 0.90,
               color=clr, edgecolor='#444', linewidth=0.7, 
               yerr=stds, capsize=3,
               error_kw=dict(elinewidth=1.2, ecolor='#222', capthick=1.2),
               zorder=3)
        
        fmt = '.2f' if metric_key in ('jfi', 'joint_score') else '.1f'
        for xp, m, s in zip(x_ctrs + offsets, means, stds):
            text_y = m + s + max(means) * 0.015
            ax.text(xp, text_y,
                    f'{m:{fmt}}', ha='center', va='bottom',
                    fontsize=6.5, fontweight='bold', color='#222')
            max_y_val = max(max_y_val, text_y)

    ax.set_xticks(x_ctrs)
    ax.set_xticklabels(cfg_labels, fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=11)
    if sublabel:
        ax.text(0.02, 0.97, sublabel, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top')
    
    # 使用传入的高度上限，或自动扩展
    if ylim_top is not None:
        ax.set_ylim(bottom=0, top=ylim_top)
    else:
        ax.set_ylim(bottom=0, top=max_y_val * 1.35)
        
    ax.yaxis.grid(True, linestyle='-', color='#eee', zorder=0) 
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    add_subplot_legend(ax, plot_type='bar', legend_y=legend_y)


# ── 箱线图子图 ────────────────────────────────────────────────────────
def box_subplot(ax, datasets, cfg_labels, alg_list, metric_key, ylabel, sublabel, legend_y=None, ylim_top=None):
    n_cfg = len(datasets)
    n_alg = len(alg_list)
    grp_w = 0.70
    step  = grp_w / n_alg
    box_w = step * 0.75  
    x_ctrs = np.arange(n_cfg, dtype=float)
    
    max_y_val = float('-inf')
    min_y_val = float('inf')

    for ai, (alg, clr) in enumerate(zip(alg_list, COLORS)):
        offset = (ai - (n_alg - 1) / 2.0) * step
        for ci, data in enumerate(datasets):
            vals = np.array([data[s][alg][metric_key] for s in data])
            xc   = x_ctrs[ci] + offset
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            iqr  = q3 - q1
            inner = vals[(vals >= q1 - 1.5*iqr) & (vals <= q3 + 1.5*iqr)]
            wlo   = inner.min() if len(inner) else vals.min()
            whi   = inner.max() if len(inner) else vals.max()
            out  = vals[(vals < wlo) | (vals > whi)]
            
            max_y_val = max(max_y_val, whi, out.max() if len(out) else whi)
            min_y_val = min(min_y_val, wlo, out.min() if len(out) else wlo)

            rect = plt.Rectangle((xc - box_w/2, q1), box_w, iqr,
                                  facecolor=clr, edgecolor='#444',
                                  alpha=0.9, linewidth=0.8, zorder=3)
            ax.add_patch(rect)
            ax.plot([xc - box_w/2, xc + box_w/2], [med, med],
                    color='#222', linewidth=1.5, zorder=4)
            ax.plot([xc, xc], [wlo, q1], color='#444', linewidth=1.0, linestyle='--', zorder=3)
            ax.plot([xc, xc], [q3, whi], color='#444', linewidth=1.0, linestyle='--', zorder=3)
            ax.plot([xc - box_w*0.25, xc + box_w*0.25], [wlo, wlo],
                    color='#444', linewidth=1.2, zorder=3)
            ax.plot([xc - box_w*0.25, xc + box_w*0.25], [whi, whi],
                    color='#444', linewidth=1.2, zorder=3)
            if len(out):
                ax.scatter([xc]*len(out), out, s=12, marker='o',
                           facecolor=clr, edgecolor='#444', linewidth=0.6, alpha=0.65, zorder=5)

    half = grp_w / 2 + step * 0.7
    ax.set_xlim(x_ctrs[0] - half, x_ctrs[-1] + half)
    
    y_range = max_y_val - min_y_val
    if ylim_top is not None:
        ax.set_ylim(bottom=min_y_val - y_range * 0.1, top=ylim_top)
    else:
        ax.set_ylim(bottom=min_y_val - y_range * 0.1, top=max_y_val + y_range * 0.28)

    ax.set_xticks(x_ctrs)
    ax.set_xticklabels(cfg_labels, fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=11)
    if sublabel:
        ax.text(0.02, 0.97, sublabel, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top')
    ax.yaxis.grid(True, linestyle='-', color='#eee', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    add_subplot_legend(ax, plot_type='box', legend_y=legend_y)


# ── 主绘图 ────────────────────────────────────────────────────────────
try:
    datasets   = [load_raw(K) for K in K_LIST]
except FileNotFoundError:
    print("未找到数据文件，请确保执行路径包含 result/multiconfig2/static_K*.json")
    datasets = []

cfg_labels = [f'K={K}' for K in K_LIST]

if datasets:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10),
                             gridspec_kw={'hspace': 0.35, 'wspace': 0.28})

    # 第 1 张图：柱状图 Min Rate，图例锁定 Y=80，同时把 Y 轴拉高至 135 保证有空间
    bar_subplot(axes[0, 0], datasets, cfg_labels, ALG_S,
                'min_rate',    'Min Rate (Mbps)', '', legend_y=80, ylim_top=135)
                
    # 第 2 张图：柱状图 JFI，图例锁定 Y=1.0，Y 轴拉高至 1.4
    bar_subplot(axes[0, 1], datasets, cfg_labels, ALG_S,
                'jfi',         'JFI$_{eff}$',     '', legend_y=1.0, ylim_top=1.4)
                
    # 第 3 张图：柱状图 Joint Score，图例锁定 Y=1.2，Y 轴拉高至 1.65
    bar_subplot(axes[0, 2], datasets, cfg_labels, ALG_S,
                'joint_score', 'Joint Score',     '', legend_y=1.2, ylim_top=1.65)


    # 第 4 张图：箱线图 Min Rate，使用自动匹配高度和内部顶部横排居中
    box_subplot(axes[1, 0], datasets, cfg_labels, ALG_S,
                'min_rate',    'Min Rate (Mbps)', '')
                
    # 第 5 张图：箱线图 JFI，横排图例锁定在 Y=1.0，Y 轴拉高至 1.15
    box_subplot(axes[1, 1], datasets, cfg_labels, ALG_S,
                'jfi',         'JFI$_{eff}$',     '', legend_y=1.0, ylim_top=1.15) 
                
    # 第 6 张图：箱线图 Joint Score，横排图例锁定在 Y=1.2，Y 轴拉高至 1.4
    box_subplot(axes[1, 2], datasets, cfg_labels, ALG_S,
                'joint_score', 'Joint Score',     '', legend_y=1.2, ylim_top=1.4)

    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ['png', 'pdf']:
        p = os.path.join(OUT_DIR, f'fig_static_final.{ext}')
        plt.savefig(p, dpi=200, bbox_inches='tight')
        print(f'  ✓ {p}')
    plt.close()
    print("完成")