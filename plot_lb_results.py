"""
LB-BVF 结果可视化 (4方对比: Initial / V6 / V3-Style / LB-BVF-MC)

Figure 1: JFI 对比图 (2 子图)
  1a. 分种子分组柱状图: 4方 JFI_eff
  1b. 代表性种子的 MC 收敛曲线

Figure 2: 用户速率 CDF 图 (4方聚合)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from balanced_virtual_force_optimizer_v6 import BalancedVirtualForceOptimizerV6, create_v6_config
from load_balanced_bvf_v3style import LoadBalancedBVF_V3Style, create_lb_v3style_config
from load_balanced_bvf_v6 import LoadBalancedBVF_V6, create_lb_v6_config

K, L, G = 40, 6, 9
EVAL_SEED = 99999
W_MIN, W_JFI, REF_RATE = 0.5, 0.5, 60.0

OUTPUT_DIR = "result/lb_experiment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.dpi': 150,
})

CLR_INIT = '#d62728'
CLR_V6   = '#1f77b4'
CLR_V3   = '#ff7f0e'
CLR_MC   = '#2ca02c'


def generate_hotspot(seed, sq=1000):
    np.random.seed(seed)
    h_ue, h_ap, h_uav = 1.65, 15.0, 50.0
    spacing = sq / 4
    gap = []
    for i in range(3):
        for j in range(3):
            gap.append([(i + 1) * spacing, (j + 1) * spacing, h_ap])
    ground_AP_pos = np.array(gap)

    n_hot = int(K * 0.75)
    n_uni = K - n_hot
    centers = [[sq * 0.25, sq * 0.30], [sq * 0.70, sq * 0.75]]
    per = n_hot // 2
    hot = []
    for cx, cy in centers:
        pts = np.random.normal([cx, cy], sq * 0.05, (per, 2))
        hot.append(pts)
    hot_xy = np.vstack(hot)[:n_hot]
    uni_xy = np.random.uniform(50, sq - 50, (n_uni, 2))
    UE_xy = np.clip(np.vstack([hot_xy, uni_xy]), 30, sq - 30)
    UE_pos = np.column_stack([UE_xy, np.full(K, h_ue)])

    l_side = int(np.ceil(np.sqrt(L)))
    usp = sq / (l_side + 1)
    uavs = []
    for i in range(l_side):
        for j in range(l_side):
            if len(uavs) >= L:
                break
            x = (i + 1) * usp + np.random.uniform(-15, 15)
            y = (j + 1) * usp + np.random.uniform(-15, 15)
            uavs.append([np.clip(x, 60, sq - 60), np.clip(y, 60, sq - 60), h_uav])
    UAV_pos = np.array(uavs[:L])
    return UE_pos, ground_AP_pos, UAV_pos, np.random.get_state()


def eval_full(evaluator, UE_pos, ground_AP_pos, UAV_pos):
    state = np.random.get_state()
    np.random.seed(EVAL_SEED)
    all_AP = np.vstack([ground_AP_pos, UAV_pos])
    _, _, betas = evaluator.compute_channel_model(UE_pos, all_AP)
    mask = evaluator.compute_AP_selection_mask(betas)
    rates, sum_rate = evaluator.compute_user_rates(UE_pos, all_AP, mask)
    np.random.set_state(state)

    mask_uav = mask[:, G:]
    mask_ground = mask[:, :G]
    eff = np.zeros(L)
    for l in range(L):
        served = np.where(mask_uav[:, l])[0]
        for k in served:
            ub = betas[k, G + l]
            sg = np.where(mask_ground[k])[0]
            gb = betas[k, sg].sum() if len(sg) > 0 else 0.0
            eff[l] += ub / (gb + ub + 1e-12)
    s = eff.sum()
    jfi = float(s ** 2 / (L * (eff ** 2).sum() + 1e-12)) if s > 1e-10 else 1.0
    joint = W_MIN * (float(rates.min()) / REF_RATE) + W_JFI * jfi
    return {'rates': rates.copy(), 'min_rate': float(rates.min()),
            'sum_rate': float(sum_rate), 'jfi_eff': jfi, 'joint': joint}


def main():
    seeds = [42, 51, 62, 71, 75, 33, 88, 99, 107, 123]
    conv_seeds = [88, 62, 42]   # MC 收敛展示种子

    base_cfg = create_v6_config()
    base_cfg.update({'num_UE': K, 'num_UAV': L, 'num_ground_AP': G,
                     'tau_p': K, 'max_iterations': 80, 'num_serving_APs': 3})
    v3_cfg = create_lb_v3style_config()
    v3_cfg.update({'num_UE': K, 'num_UAV': L, 'num_ground_AP': G,
                   'tau_p': K, 'max_iterations': 80, 'num_serving_APs': 3,
                   'w_min': W_MIN, 'w_jfi': W_JFI, 'ref_rate': REF_RATE})
    mc_cfg = create_lb_v6_config()
    mc_cfg.update({'num_UE': K, 'num_UAV': L, 'num_ground_AP': G,
                   'tau_p': K, 'max_iterations': 80, 'num_serving_APs': 3,
                   'w_min': W_MIN, 'w_jfi': W_JFI, 'ref_rate': REF_RATE})

    init_jfis, v6_jfis, v3_jfis, mc_jfis = [], [], [], []
    init_rates_all, v6_rates_all, v3_rates_all, mc_rates_all = [], [], [], []
    conv_data = {}

    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Seed={seed}")
        UE_pos, gAP, init_UAV, opt_state = generate_hotspot(seed)
        ev = BalancedVirtualForceOptimizerV6(base_cfg)

        # Initial
        r = eval_full(ev, UE_pos, gAP, init_UAV)
        init_jfis.append(r['jfi_eff']); init_rates_all.append(r['rates'])
        print(f"  Init:     min={r['min_rate']:.1f}  JFI={r['jfi_eff']:.4f}")

        # V6
        np.random.set_state(opt_state)
        res = BalancedVirtualForceOptimizerV6(base_cfg).optimize(UE_pos, gAP, init_UAV.copy())
        r = eval_full(ev, UE_pos, gAP, res['optimized_UAV_pos'])
        v6_jfis.append(r['jfi_eff']); v6_rates_all.append(r['rates'])
        print(f"  V6:       min={r['min_rate']:.1f}  JFI={r['jfi_eff']:.4f}  Joint={r['joint']:.4f}")

        # V3-Style
        np.random.set_state(opt_state)
        res = LoadBalancedBVF_V3Style(v3_cfg).optimize(UE_pos, gAP, init_UAV.copy())
        r = eval_full(ev, UE_pos, gAP, res['optimized_UAV_pos'])
        v3_jfis.append(r['jfi_eff']); v3_rates_all.append(r['rates'])
        print(f"  V3-Style: min={r['min_rate']:.1f}  JFI={r['jfi_eff']:.4f}  Joint={r['joint']:.4f}")

        # LB-BVF-MC
        np.random.set_state(opt_state)
        res = LoadBalancedBVF_V6(mc_cfg).optimize(UE_pos, gAP, init_UAV.copy())
        r = eval_full(ev, UE_pos, gAP, res['optimized_UAV_pos'])
        mc_jfis.append(r['jfi_eff']); mc_rates_all.append(r['rates'])
        print(f"  MC:       min={r['min_rate']:.1f}  JFI={r['jfi_eff']:.4f}  Joint={r['joint']:.4f}")
        if seed in conv_seeds:
            conv_data[seed] = {'mc_history': res.get('mc_jfi_history', [])}

    init_jfis = np.array(init_jfis); v6_jfis = np.array(v6_jfis)
    v3_jfis  = np.array(v3_jfis);  mc_jfis  = np.array(mc_jfis)
    all_init  = np.concatenate(init_rates_all)
    all_v6    = np.concatenate(v6_rates_all)
    all_v3    = np.concatenate(v3_rates_all)
    all_mc    = np.concatenate(mc_rates_all)

    print(f"\n  JFI means: Init={init_jfis.mean():.3f}  V6={v6_jfis.mean():.3f}"
          f"  V3={v3_jfis.mean():.3f}  MC={mc_jfis.mean():.3f}")

    # ════════════════════════════════════════════════════════════
    # Figure 1: JFI bar + MC convergence
    # ════════════════════════════════════════════════════════════
    fig1 = plt.figure(figsize=(16, 5.5))
    gs = gridspec.GridSpec(1, 2, figure=fig1, width_ratios=[3, 2], wspace=0.35)

    # ── 1a: 4 方柱状图 ──
    ax1 = fig1.add_subplot(gs[0])
    n = len(seeds)
    x = np.arange(n)
    w = 0.18

    ax1.bar(x - 1.5*w, init_jfis, w, label='Initial',         color=CLR_INIT, alpha=0.85, edgecolor='white', lw=0.4)
    ax1.bar(x - 0.5*w, v6_jfis,   w, label='BVF V6',          color=CLR_V6,   alpha=0.85, edgecolor='white', lw=0.4)
    ax1.bar(x + 0.5*w, v3_jfis,   w, label='LB-BVF V3-Style', color=CLR_V3,   alpha=0.85, edgecolor='white', lw=0.4)
    ax1.bar(x + 1.5*w, mc_jfis,   w, label='LB-BVF-MC (Ours)',color=CLR_MC,   alpha=0.85, edgecolor='white', lw=0.4)

    for mean, col in [(init_jfis.mean(), CLR_INIT), (v6_jfis.mean(), CLR_V6),
                      (v3_jfis.mean(), CLR_V3), (mc_jfis.mean(), CLR_MC)]:
        ax1.axhline(mean, color=col, ls='--', lw=1.2, alpha=0.65)
    ax1.axhline(0.9, color='dimgray', ls=':', lw=1.4, alpha=0.55, label='Target JFI=0.9')

    # 均值文字 (右侧)
    xr = n - 0.05
    offsets = [0.012, -0.022, 0.012, -0.022]
    for mean, col, off in zip(
            [init_jfis.mean(), v6_jfis.mean(), v3_jfis.mean(), mc_jfis.mean()],
            [CLR_INIT, CLR_V6, CLR_V3, CLR_MC], offsets):
        ax1.text(xr, mean + off, f'μ={mean:.3f}', color=col, fontsize=7.5, ha='right')

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'S{s}' for s in seeds])
    ax1.set_ylabel('JFI$_{eff}$ (Effective Load Fairness)')
    ax1.set_title(f'(a)  UAV Load JFI — Per Scenario\n'
                  f'K={K}, L={L}, G={G} Ground APs · Hotspot 75%')
    ax1.set_ylim(0, 1.10)
    ax1.legend(loc='lower right', fontsize=8.5, ncol=2)
    ax1.grid(axis='y', alpha=0.3, lw=0.6)

    # ── 1b: MC 收敛曲线 ──
    ax2 = fig1.add_subplot(gs[1])
    conv_colors = [CLR_MC, '#9467bd', '#e377c2']
    for cidx, seed in enumerate(conv_seeds):
        if seed not in conv_data:
            continue
        hist = conv_data[seed].get('mc_history', [])
        if not hist:
            continue
        jfis_h = [h[1] for h in hist]
        xs = list(range(len(hist)))
        col = conv_colors[cidx]
        ax2.plot(xs, jfis_h, '-o', color=col, lw=2.0, ms=5,
                 label=f'Seed {seed}', zorder=3)
        ax2.annotate(f'{jfis_h[0]:.3f}', (xs[0], jfis_h[0]),
                     textcoords='offset points', xytext=(-15, 6),
                     fontsize=7.5, color=col)
        ax2.annotate(f'{jfis_h[-1]:.3f}', (xs[-1], jfis_h[-1]),
                     textcoords='offset points', xytext=(3, 4),
                     fontsize=7.5, color=col)

    ax2.axhline(0.9, color='dimgray', ls=':', lw=1.4, alpha=0.55, label='Target=0.9')
    ax2.set_xlabel('MC Round  (0 = V6 Baseline P1)')
    ax2.set_ylabel('JFI$_{eff}$')
    ax2.set_title('(b)  LB-BVF-MC Convergence\n(JFI$_{eff}$ per Round)')
    ax2.set_ylim(0.5, 1.06)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(alpha=0.3, lw=0.6)

    fig1.suptitle('Load-Balanced BVF: UAV Effective Load JFI Improvement',
                  fontsize=13, fontweight='bold', y=1.01)
    fig1.tight_layout()
    p1 = os.path.join(OUTPUT_DIR, 'jfi_comparison.png')
    fig1.savefig(p1, dpi=180, bbox_inches='tight')
    print(f"\nSaved: {p1}")
    plt.close(fig1)

    # ════════════════════════════════════════════════════════════
    # Figure 2: CDF (4 方)
    # ════════════════════════════════════════════════════════════
    fig2, ax = plt.subplots(figsize=(9, 6.2))

    def plot_cdf(data, color, label, lw=2.0, ls='-', zo=2):
        s = np.sort(data)
        c = np.arange(1, len(s) + 1) / len(s)
        ax.plot(s, c, color=color, lw=lw, ls=ls, label=label, zorder=zo)

    plot_cdf(all_init, CLR_INIT, 'Initial Deployment',  lw=1.6, ls='--')
    plot_cdf(all_v6,   CLR_V6,   'BVF V6',              lw=1.8, ls='-.')
    plot_cdf(all_v3,   CLR_V3,   'LB-BVF V3-Style',     lw=2.0, ls=':',  zo=3)
    plot_cdf(all_mc,   CLR_MC,   'LB-BVF-MC (Ours)',    lw=2.5, ls='-',  zo=4)

    # 底部 10% 标注
    p10 = {n: np.percentile(d, 10) for n, d in
           [('init', all_init), ('v6', all_v6), ('v3', all_v3), ('mc', all_mc)]}
    p90 = {n: np.percentile(d, 90) for n, d in
           [('v6', all_v6), ('v3', all_v3), ('mc', all_mc)]}

    ax.axhline(0.10, color='gray', ls=':', lw=1.1, alpha=0.5)
    ax.axhline(0.90, color='gray', ls=':', lw=1.1, alpha=0.5)

    for val, col in [(p10['init'], CLR_INIT), (p10['v6'], CLR_V6),
                     (p10['v3'], CLR_V3),   (p10['mc'],  CLR_MC)]:
        ax.plot(val, 0.10, 'v', color=col, ms=7, zorder=5)
    for val, col in [(p90['v6'], CLR_V6), (p90['v3'], CLR_V3), (p90['mc'], CLR_MC)]:
        ax.plot(val, 0.90, '^', color=col, ms=7, zorder=5)

    # 10th pct 标注框
    ax.text(p10['init'] - 1, 0.15,
            f"Init: {p10['init']:.1f}", color=CLR_INIT, fontsize=8, ha='right')
    ax.text((p10['v6'] + p10['v3']) / 2, 0.20,
            f"V6: {p10['v6']:.1f}\nV3: {p10['v3']:.1f}\nMC: {p10['mc']:.1f}",
            color='black', fontsize=8, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.9))

    # 90th pct 标注框
    ax.text(max(p90['mc'], p90['v3']) + 2, 0.90,
            f"90th pct\nV6: {p90['v6']:.1f}\nV3: {p90['v3']:.1f}\nMC: {p90['mc']:.1f}",
            va='center', fontsize=8, color='dimgray',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.85))

    ax.set_xlabel('Per-User Downlink Rate (Mbps)', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title(
        f'User Rate CDF — Aggregated over {len(seeds)} Hotspot Scenarios\n'
        f'K={K} UEs, L={L} UAVs, G={G} Ground APs  (Total {K*len(seeds)} samples)',
        fontsize=12)
    ax.legend(fontsize=10.5, loc='lower right')
    ax.grid(alpha=0.3, lw=0.6)
    ax.set_xlim(left=0)
    ax.set_ylim(-0.01, 1.02)

    fig2.tight_layout()
    p2 = os.path.join(OUTPUT_DIR, 'user_rate_cdf.png')
    fig2.savefig(p2, dpi=180, bbox_inches='tight')
    print(f"Saved: {p2}")
    plt.close(fig2)

    # ════════════════════════════════════════════════════════════
    # Figure 3: JointScore 散点+柱状综合对比
    # ════════════════════════════════════════════════════════════
    fig3, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 3a: JointScore 分种子
    ax3a = axes[0]
    for jfis, col, lbl, mk in [
            (v6_jfis,  CLR_V6,  'BVF V6',          'o'),
            (v3_jfis,  CLR_V3,  'LB-BVF V3-Style', 's'),
            (mc_jfis,  CLR_MC,  'LB-BVF-MC',       '^')]:
        # 计算 joint from jfi and min (reuse stored data)
        pass   # will use separate joints list below

    # 重新计算 joint scores per seed from arrays
    v6_joints  = [W_MIN*(np.concatenate([v6_rates_all[i]])).min() / REF_RATE + W_JFI*v6_jfis[i]
                  for i in range(len(seeds))]
    v3_joints  = [W_MIN*(np.concatenate([v3_rates_all[i]])).min() / REF_RATE + W_JFI*v3_jfis[i]
                  for i in range(len(seeds))]
    mc_joints  = [W_MIN*(np.concatenate([mc_rates_all[i]])).min() / REF_RATE + W_JFI*mc_jfis[i]
                  for i in range(len(seeds))]

    xs = np.arange(len(seeds))
    ax3a.plot(xs, v6_joints,  'o-', color=CLR_V6,  lw=1.5, ms=5, label='BVF V6')
    ax3a.plot(xs, v3_joints,  's--', color=CLR_V3, lw=1.5, ms=5, label='LB-BVF V3-Style')
    ax3a.plot(xs, mc_joints,  '^-', color=CLR_MC,  lw=2.0, ms=6, label='LB-BVF-MC')
    ax3a.set_xticks(xs)
    ax3a.set_xticklabels([f'S{s}' for s in seeds], fontsize=8.5)
    ax3a.set_ylabel('JointScore = 0.5·(min/60) + 0.5·JFI$_{eff}$')
    ax3a.set_title('(a)  JointScore per Scenario')
    ax3a.legend(fontsize=9)
    ax3a.grid(alpha=0.3, lw=0.6)

    # 3b: 平均指标 Bar chart (JFI_eff 和 JointScore, 均在 0-1 量级)
    ax3b = axes[1]
    v6_mins = np.mean([v6_rates_all[i].min() for i in range(len(seeds))])
    v3_mins = np.mean([v3_rates_all[i].min() for i in range(len(seeds))])
    mc_mins = np.mean([mc_rates_all[i].min() for i in range(len(seeds))])

    metrics = ['Avg JFI$_{eff}$', 'Avg JointScore']
    v6_vals = [v6_jfis.mean(), np.mean(v6_joints)]
    v3_vals = [v3_jfis.mean(), np.mean(v3_joints)]
    mc_vals = [mc_jfis.mean(), np.mean(mc_joints)]

    xb = np.arange(len(metrics))
    wb = 0.25
    ax3b.bar(xb - wb, v6_vals, wb, label='BVF V6',          color=CLR_V6, alpha=0.85, edgecolor='white')
    ax3b.bar(xb,      v3_vals, wb, label='LB-BVF V3-Style', color=CLR_V3, alpha=0.85, edgecolor='white')
    ax3b.bar(xb + wb, mc_vals, wb, label='LB-BVF-MC',       color=CLR_MC, alpha=0.85, edgecolor='white')

    for xi, (a, b, c) in enumerate(zip(v6_vals, v3_vals, mc_vals)):
        for v, off, col in [(a, -wb, CLR_V6), (b, 0, CLR_V3), (c, wb, CLR_MC)]:
            ax3b.text(xi + off, v + 0.008, f'{v:.3f}', ha='center',
                      va='bottom', fontsize=8, color=col, fontweight='bold')

    ax3b.set_xticks(xb)
    ax3b.set_xticklabels(metrics, fontsize=10.5)
    ax3b.set_title(f'(b)  Average Performance (10 Seeds)\n'
                   f'Avg min-rate: V6={v6_mins:.1f}  V3={v3_mins:.1f}  MC={mc_mins:.1f} Mbps')
    ax3b.legend(fontsize=9)
    ax3b.set_ylim(0, 1.12)
    ax3b.grid(axis='y', alpha=0.3, lw=0.6)

    fig3.suptitle('JointScore Comparison: BVF V6 / V3-Style / LB-BVF-MC',
                  fontsize=13, fontweight='bold')
    fig3.tight_layout()
    p3 = os.path.join(OUTPUT_DIR, 'joint_score_comparison.png')
    fig3.savefig(p3, dpi=180, bbox_inches='tight')
    print(f"Saved: {p3}")
    plt.close(fig3)

    print("\nAll done.")


if __name__ == "__main__":
    main()
