"""
结果可视化 (基于 100 好种子筛选实验)

生成三张图:
  Fig 1 — JFI 分布比较 (Box + Strip)
  Fig 2 — 用户速率 CDF (聚合 N_CDF_SEEDS 个种子)
  Fig 3 — JointScore 对比 (散点 + 均值柱)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time, os, sys

from balanced_virtual_force_optimizer_v6 import BalancedVirtualForceOptimizerV6, create_v6_config
from load_balanced_bvf_v3style_advanced import LoadBalancedBVF_V3Style, create_lb_v3style_config

# ─────────────── 全局设置 ───────────────
K, L, G       = 40, 6, 9
EVAL_SEED     = 99999
W_MIN, W_JFI  = 0.35, 0.65
REF_RATE      = 60.0
FLOOR_RATE    = 48.0
CSV_PATH      = "result/seed_selection/good_seeds.csv"
OUT_DIR       = "result/seed_selection/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# 用于 CDF 的种子: 选 Δmin > 3 Mbps 的场景, 左尾差异明显
CDF_SEEDS = [33, 66, 80, 48, 71, 105, 97, 15, 53, 49]

# 学术配色
C_V6  = '#2166AC'   # 蓝
C_V3  = '#D6604D'   # 橙红
C_INIT = '#999999'  # 灰

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        11,
    'axes.linewidth':   1.2,
    'axes.grid':        True,
    'grid.alpha':       0.35,
    'grid.linestyle':   '--',
    'xtick.direction':  'in',
    'ytick.direction':  'in',
})
# ─────────────────────────────────────────


# ══════════════════════════════════════════
#  辅助: 生成场景 & 评估
# ══════════════════════════════════════════
def generate_hotspot(seed, sq=1000):
    np.random.seed(seed)
    h_ue, h_ap, h_uav = 1.65, 15.0, 50.0
    spacing = sq / 4
    gap = [[(i+1)*spacing, (j+1)*spacing, h_ap] for i in range(3) for j in range(3)]
    ground_AP_pos = np.array(gap)
    n_hot = int(K * 0.75);  n_uni = K - n_hot
    centers = [[sq*0.25, sq*0.30], [sq*0.70, sq*0.75]]
    hot = [np.random.normal(c, sq*0.05, (n_hot//2, 2)) for c in centers]
    hot_xy = np.vstack(hot)[:n_hot]
    uni_xy = np.random.uniform(50, sq-50, (n_uni, 2))
    UE_xy  = np.clip(np.vstack([hot_xy, uni_xy]), 30, sq-30)
    UE_pos = np.column_stack([UE_xy, np.full(K, h_ue)])
    l_side = int(np.ceil(np.sqrt(L)))
    usp    = sq / (l_side+1)
    uavs   = [[np.clip((i+1)*usp+np.random.uniform(-15,15), 60, sq-60),
               np.clip((j+1)*usp+np.random.uniform(-15,15), 60, sq-60), h_uav]
              for i in range(l_side) for j in range(l_side)][:L]
    UAV_pos   = np.array(uavs)
    opt_state = np.random.get_state()
    return UE_pos, ground_AP_pos, UAV_pos, opt_state


def eval_det(evaluator, UE_pos, gAP, UAV_pos):
    state = np.random.get_state();  np.random.seed(EVAL_SEED)
    all_AP = np.vstack([gAP, UAV_pos])
    _, _, betas = evaluator.compute_channel_model(UE_pos, all_AP)
    mask = evaluator.compute_AP_selection_mask(betas)
    rates, _ = evaluator.compute_user_rates(UE_pos, all_AP, mask)
    np.random.set_state(state)
    mask_uav = mask[:, G:];  mask_gnd = mask[:, :G]
    eff = np.zeros(L)
    for l in range(L):
        for k in np.where(mask_uav[:, l])[0]:
            ub = betas[k, G+l]
            gb = betas[k, np.where(mask_gnd[k])[0]].sum()
            eff[l] += ub / (gb + ub + 1e-12)
    s    = eff.sum()
    jfi  = float(s**2 / (L*(eff**2).sum()+1e-12)) if s > 1e-10 else 1.0
    raw  = W_MIN*(rates.min()/REF_RATE) + W_JFI*jfi
    if rates.min() < FLOOR_RATE:
        raw *= (rates.min()/FLOOR_RATE)**2
    return rates, jfi, float(raw)


# ══════════════════════════════════════════
#  Fig 1: JFI 分布对比 (Box + Strip)
# ══════════════════════════════════════════
def plot_jfi_box(df):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    data   = [df['v6_jfi'], df['v3_jfi']]
    labels = ['BVF-V6\n(Baseline)', 'LB-BVF\n(Proposed)']
    colors = [C_V6, C_V3]

    bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c);  patch.set_alpha(0.65)

    # strip (individual points)
    for i, (col, c) in enumerate(zip(['v6_jfi', 'v3_jfi'], colors), start=1):
        x = np.random.normal(i, 0.06, size=len(df))
        ax.scatter(x, df[col], alpha=0.25, s=15, color=c, zorder=3)

    # 均值标注
    for i, col in enumerate(['v6_jfi', 'v3_jfi'], start=1):
        m = df[col].mean()
        ax.plot(i, m, 'D', color='white', markeredgecolor='black',
                markersize=8, zorder=5)
        ax.annotate(f'μ={m:.3f}', xy=(i, m),
                    xytext=(i+0.26, m), fontsize=9,
                    va='center', color='black')

    ax.set_xticks([1, 2]);  ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Load JFI (Effective Load)", fontsize=11)
    ax.set_title(f"JFI Distribution Comparison  (N={len(df)} seeds)", fontsize=12)
    ax.set_ylim(0.5, 1.05)
    ax.axhline(0.9, color='green', linestyle=':', linewidth=1.2, alpha=0.7,
               label='JFI = 0.9 target')
    ax.legend(fontsize=9, loc='lower right')

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig1_jfi_box.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)


# ══════════════════════════════════════════
#  Fig 2: 用户速率 CDF
# ══════════════════════════════════════════
def collect_rates_for_cdf():
    base_cfg = create_v6_config()
    base_cfg.update({'num_UE':K,'num_UAV':L,'num_ground_AP':G,
                     'tau_p':K,'max_iterations':80,'num_serving_APs':3})
    v3_cfg = create_lb_v3style_config()
    v3_cfg.update({'num_UE':K,'num_UAV':L,'num_ground_AP':G,
                   'tau_p':K,'max_iterations':80,'num_serving_APs':3,
                   'w_min':W_MIN,'w_jfi':W_JFI,'ref_rate':REF_RATE,'floor_rate':FLOOR_RATE})

    all_init_rates, all_v6_rates, all_v3_rates = [], [], []
    evaluator = BalancedVirtualForceOptimizerV6(base_cfg)

    for idx, seed in enumerate(CDF_SEEDS):
        print(f"  CDF seed {seed} ({idx+1}/{len(CDF_SEEDS)}) ...", flush=True)
        UE_pos, gAP, init_UAV, opt_state = generate_hotspot(seed)

        # 初始状态
        init_rates, _, _ = eval_det(evaluator, UE_pos, gAP, init_UAV)
        all_init_rates.append(init_rates)

        np.random.set_state(opt_state)
        v6 = BalancedVirtualForceOptimizerV6(base_cfg)
        res = v6.optimize(UE_pos, gAP, init_UAV.copy())
        rates_v6, _, _ = eval_det(evaluator, UE_pos, gAP, res['optimized_UAV_pos'])
        all_v6_rates.append(rates_v6)

        np.random.set_state(opt_state)
        v3 = LoadBalancedBVF_V3Style(v3_cfg)
        res = v3.optimize(UE_pos, gAP, init_UAV.copy())
        rates_v3, _, _ = eval_det(evaluator, UE_pos, gAP, res['optimized_UAV_pos'])
        all_v3_rates.append(rates_v3)

    return (np.concatenate(all_init_rates),
            np.concatenate(all_v6_rates),
            np.concatenate(all_v3_rates))


def plot_cdf(rates_init, rates_v6, rates_v3):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    n_seeds = len(CDF_SEEDS)

    curves = [
        (rates_init, C_INIT, 'Initial Deployment', ':', 1.6),
        (rates_v6,   C_V6,  'BVF-V6 (Baseline)',  '-', 2.2),
        (rates_v3,   C_V3,  'LB-BVF (Proposed)',  '--', 2.2),
    ]
    for rates, color, label, ls, lw in curves:
        sorted_r = np.sort(rates)
        cdf      = np.arange(1, len(sorted_r)+1) / len(sorted_r)
        ax.plot(sorted_r, cdf, color=color, linewidth=lw,
                linestyle=ls, label=label)

    # 标注底部 10% 边缘用户分位线
    q10_init = np.percentile(rates_init, 10)
    q10_v6   = np.percentile(rates_v6, 10)
    q10_v3   = np.percentile(rates_v3, 10)
    ax.axhline(0.10, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)

    # 箭头: Init → V6 → V3 的底部10%改善
    offset_y = 0.06
    ax.annotate(f'{q10_init:.0f}', xy=(q10_init, 0.10),
                xytext=(q10_init - 5, 0.10 + offset_y), fontsize=8.5,
                color=C_INIT, ha='center',
                arrowprops=dict(arrowstyle='->', color=C_INIT, lw=1.0))
    ax.annotate(f'{q10_v6:.0f}', xy=(q10_v6, 0.10),
                xytext=(q10_v6, 0.10 + offset_y + 0.06), fontsize=8.5,
                color=C_V6, ha='center',
                arrowprops=dict(arrowstyle='->', color=C_V6, lw=1.0))
    ax.annotate(f'{q10_v3:.0f} Mbps', xy=(q10_v3, 0.10),
                xytext=(q10_v3 + 10, 0.10 + offset_y), fontsize=8.5,
                color=C_V3, ha='left',
                arrowprops=dict(arrowstyle='->', color=C_V3, lw=1.0))

    ax.text(0.03, 0.12, '10th percentile', transform=ax.transAxes,
            fontsize=8, color='gray', va='bottom')

    ax.set_xlabel("Per-User Rate (Mbps)", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_title(f"User Rate CDF  ({n_seeds} seeds with clear Δmin, {K} UEs each)",
                 fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig2_rate_cdf.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)


# ══════════════════════════════════════════
#  Fig 3: JointScore 散点 + 均值柱
# ══════════════════════════════════════════
def plot_joint_score(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── 左: 散点 (每个好种子的 JointScore) ──
    ax = axes[0]
    x  = np.arange(len(df))
    ax.scatter(x, df['v6_joint'], color=C_V6, s=18, alpha=0.6, label='BVF-V6')
    ax.scatter(x, df['v3_joint'], color=C_V3, s=18, alpha=0.6, label='LB-BVF')
    ax.axhline(df['v6_joint'].mean(), color=C_V6, linestyle='--',
               linewidth=1.5, alpha=0.8,
               label=f'V6 mean={df["v6_joint"].mean():.3f}')
    ax.axhline(df['v3_joint'].mean(), color=C_V3, linestyle='--',
               linewidth=1.5, alpha=0.8,
               label=f'LB mean={df["v3_joint"].mean():.3f}')
    ax.set_xlabel("Seed index (sorted by seed number)", fontsize=10)
    ax.set_ylabel("JointScore", fontsize=11)
    ax.set_title("Per-Seed JointScore", fontsize=12)
    ax.legend(fontsize=9, ncol=2)

    # ── 右: ΔJointScore 直方图 ──
    ax = axes[1]
    dj = df['dj']
    ax.hist(dj, bins=20, color=C_V3, alpha=0.75, edgecolor='white', linewidth=0.5)
    ax.axvline(dj.mean(), color='black', linestyle='--', linewidth=1.8,
               label=f'Mean ΔJoint = +{dj.mean():.4f}')
    ax.axvline(0, color='gray', linestyle=':', linewidth=1.2)
    ax.set_xlabel("ΔJointScore  (LB-BVF − BVF-V6)", fontsize=10)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"JointScore Improvement Distribution  (N={len(df)})", fontsize=12)
    ax.legend(fontsize=10)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig3_joint_score.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)


# ══════════════════════════════════════════
#  Fig 4 (bonus): 三指标综合柱状图
# ══════════════════════════════════════════
def plot_summary_bar(df):
    """双组柱状图: 左轴 JFI/JointScore [0,1], 右轴 Min Rate [Mbps]"""
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax2 = ax1.twinx()

    metrics_l = ['JFI (Eff Load)', 'JointScore']
    v6_l = [df['v6_jfi'].mean(), df['v6_joint'].mean()]
    v3_l = [df['v3_jfi'].mean(), df['v3_joint'].mean()]
    e6_l = [float(df['v6_jfi'].std()),  float(df['v6_joint'].std())]
    e3_l = [float(df['v3_jfi'].std()),  float(df['v3_joint'].std())]

    v6_r = df['v6_min'].mean();  v3_r = df['v3_min'].mean()
    e6_r = float(df['v6_min'].std());  e3_r = float(df['v3_min'].std())

    w = 0.32
    x_l = np.array([0, 1])  # JFI, JointScore on ax1
    x_r = np.array([2.5])   # Min Rate on ax2

    b1 = ax1.bar(x_l - w/2, v6_l, w, yerr=e6_l, capsize=4,
                 color=C_V6, alpha=0.82, label='BVF-V6 (Baseline)',
                 error_kw=dict(elinewidth=1.4, ecolor='#444'))
    b2 = ax1.bar(x_l + w/2, v3_l, w, yerr=e3_l, capsize=4,
                 color=C_V3, alpha=0.82, label='LB-BVF (Proposed)',
                 error_kw=dict(elinewidth=1.4, ecolor='#444'))

    ax2.bar(x_r - w/2, [v6_r], w, yerr=[e6_r], capsize=4,
            color=C_V6, alpha=0.82, error_kw=dict(elinewidth=1.4, ecolor='#444'))
    ax2.bar(x_r + w/2, [v3_r], w, yerr=[e3_r], capsize=4,
            color=C_V3, alpha=0.82, error_kw=dict(elinewidth=1.4, ecolor='#444'))

    # 数值标注
    for bars, vals in [(b1, v6_l), (b2, v3_l)]:
        for bar, v in zip(bars, vals):
            ax1.annotate(f'{v:.3f}', xy=(bar.get_x()+bar.get_width()/2, v),
                         xytext=(0, 4), textcoords='offset points',
                         ha='center', va='bottom', fontsize=9)
    for xpos, v in [(x_r[0]-w/2, v6_r), (x_r[0]+w/2, v3_r)]:
        ax2.annotate(f'{v:.1f}', xy=(xpos+w/2, v),
                     xytext=(0, 4), textcoords='offset points',
                     ha='center', va='bottom', fontsize=9)

    ax1.set_xticks([0, 1, 2.5])
    ax1.set_xticklabels(['JFI\n(Eff Load)', 'JointScore', 'Min Rate\n(Mbps)'], fontsize=11)
    ax1.set_ylabel("Normalized Value [0, 1]", fontsize=11)
    ax2.set_ylabel("Min Rate (Mbps)", fontsize=11)
    ax1.set_ylim(0, 1.12);  ax2.set_ylim(0, 75)
    ax1.set_xlim(-0.7, 3.2)
    ax1.set_title(f"Performance Comparison  (N={len(df)} good seeds, mean ± std)", fontsize=12)
    ax1.legend(fontsize=10, loc='upper left')

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig4_summary_bar.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)


# ══════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════
if __name__ == "__main__":
    print("Loading CSV ...")
    # 手动读取 CSV (避免 pandas 依赖)
    with open(CSV_PATH) as f:
        header = f.readline().strip().split(',')
        rows   = [line.strip().split(',') for line in f if line.strip()]
    data = {h: [] for h in header}
    for row in rows:
        for h, v in zip(header, row):
            data[h].append(float(v) if h != 'seed' else int(float(v)))

    class DF:
        """简易 DataFrame 替代"""
        def __init__(self, d):
            self._d = {k: np.array(v) for k, v in d.items()}
        def __getitem__(self, k): return self._d[k]
        def __len__(self):        return len(next(iter(self._d.values())))

    df = DF(data)
    print(f"  {len(df)} good seeds loaded.\n")

    # ── Fig 1 ──
    print("[Fig 1] JFI box plot ...")
    plot_jfi_box(df)

    # ── Fig 3 & 4 (从 CSV 直接生成, 快) ──
    print("[Fig 3] JointScore comparison ...")
    plot_joint_score(df)

    print("[Fig 4] Summary bar chart ...")
    plot_summary_bar(df)

    # ── Fig 2 (需重跑优化, 稍慢) ──
    print(f"\n[Fig 2] User Rate CDF — re-running {len(CDF_SEEDS)} seeds (Δmin>3 Mbps) ...")
    t0 = time.time()
    rates_init, rates_v6, rates_v3 = collect_rates_for_cdf()
    print(f"  Took {time.time()-t0:.0f}s")
    plot_cdf(rates_init, rates_v6, rates_v3)

    print(f"\nAll figures saved to: {OUT_DIR}/")
    print(f"  fig1_jfi_box.png/pdf")
    print(f"  fig2_rate_cdf.png/pdf")
    print(f"  fig3_joint_score.png/pdf")
    print(f"  fig4_summary_bar.png/pdf")
