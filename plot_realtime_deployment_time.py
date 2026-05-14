"""
实时部署计算时间对比 (真实部署设置)
=====================================
真实部署场景: nbrOfRealizations=1 (仅用大尺度衰落估计, 无 Monte Carlo)
BVF : max_iterations=20  → 亚秒级 (<1s)
对手: max_iterations=20  → 仍需遍历种群, 秒级以上

K=40, L=6, G=9,  30 seeds
输出: result/computation_time/fig_time_realworld.png/pdf
"""
import numpy as np
import json, os, sys, io, time, warnings
warnings.filterwarnings('ignore')

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from load_balanced_bvf_v3style_advanced import LoadBalancedBVF_V3Style, create_lb_v3style_config
from heuristic_lb_3d import GA3D_LB, PSO3D_LB, SSA3D_LB, create_heuristic_config

K, L, G   = 40, 6, 9
N_SEEDS   = 30
NBR_R     = 1       # 实际部署: 仅大尺度衰落
MAX_ITER  = 20      # 所有算法统一迭代上限
OUT_DIR   = 'result/computation_time'
os.makedirs(OUT_DIR, exist_ok=True)

ALG_NAMES = ['LB-BVF', 'GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']

LABELS= ['BVF', 'DGA-CF', 'DPSO-CF', 'NSSA-CF']
COLORS= ['#ee6f63', '#5fa8e8', '#63cfa0', '#b08adf']

# ── 计时 ──────────────────────────────────────────────────────────────
times = {a: [] for a in ALG_NAMES}

for seed in range(1, N_SEEDS + 1):
    np.random.seed(seed)
    sp = 1000 / 4
    gAP = np.array([[(i+1)*sp, (j+1)*sp, 15.0]
                    for i in range(3) for j in range(3)])
    n_hot = int(K * 0.75); n_uni = K - n_hot
    hot = np.vstack([np.random.normal(c, 50, (n_hot//2, 2))
                     for c in [[250, 300], [700, 750]]])[:n_hot]
    uni = np.random.uniform(50, 950, (n_uni, 2))
    UE_pos = np.column_stack([np.clip(np.vstack([hot, uni]), 30, 970),
                               np.ones(K) * 1.65])
    l_side = int(np.ceil(np.sqrt(L)))
    usp = 1000 / (l_side + 1)
    UAV_init = np.array([
        [np.clip((i+1)*usp + np.random.uniform(-15, 15), 60, 940),
         np.clip((j+1)*usp + np.random.uniform(-15, 15), 60, 940), 50.0]
        for i in range(l_side) for j in range(l_side)
    ])[:L]

    # LB-BVF
    cfg = create_lb_v3style_config()
    cfg.update({'num_UE': K, 'num_UAV': L, 'num_ground_AP': G, 'tau_p': K,
                'max_iterations': MAX_ITER, 'num_serving_APs': 3,
                'nbrOfRealizations': NBR_R})
    t0 = time.perf_counter()
    old = sys.stdout; sys.stdout = io.StringIO()
    LoadBalancedBVF_V3Style(cfg).optimize(UE_pos, gAP, UAV_init.copy())
    sys.stdout = old
    times['LB-BVF'].append(time.perf_counter() - t0)

    # GA / PSO / SSA
    hcfg = create_heuristic_config(K=K, L=L, G=G)
    hcfg.update({'num_serving_APs': 3,
                 'nbrOfRealizations_inner': NBR_R,
                 'nbrOfRealizations_final': NBR_R,
                 'max_iterations':   MAX_ITER,
                 'newssa_max_iter':  MAX_ITER,
                 'max_generations':  MAX_ITER})
    for name, Cls in [('GA-3D-LB', GA3D_LB),
                      ('PSO-3D-LB', PSO3D_LB),
                      ('SSA-3D-LB', SSA3D_LB)]:
        t0 = time.perf_counter()
        old = sys.stdout; sys.stdout = io.StringIO()
        Cls(hcfg).optimize(UE_pos, gAP, UAV_init.copy())
        sys.stdout = old
        times[name].append(time.perf_counter() - t0)

    print(f"  Seed {seed:2d}/{N_SEEDS}  BVF={times['LB-BVF'][-1]:.3f}s  "
          f"GA={times['GA-3D-LB'][-1]:.1f}s", flush=True)

# ── 统计 ──────────────────────────────────────────────────────────────
means = [np.mean(times[a]) for a in ALG_NAMES]
stds  = [np.std(times[a])  for a in ALG_NAMES]
print("\n实时部署计算时间 (nbrR=1, max_iter=20, 30 seeds):")
for a, m, s in zip(ALG_NAMES, means, stds):
    print(f"  {a:<14}: {m:.3f} ± {s:.3f} s")

# ── 绘图 ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.8, 5.0))
x = np.arange(len(ALG_NAMES))

bars = ax.bar(x, means, 0.52,
              color=COLORS, edgecolor='white', linewidth=0.8,
              yerr=stds, capsize=6,
              error_kw=dict(elinewidth=1.4, ecolor='#444', capthick=1.4),
              zorder=3)

# 数值标注
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + max(means) * 0.013,
            f'{m:.2f}s', ha='center', va='bottom',
            fontsize=10.5, fontweight='bold', color='#111')


ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=11.5)
ax.set_ylabel('Computation Time (s)', fontsize=12)
ax.set_title('Real-Time UAV Deployment: Computation Time\n'
             r'(K=40, L=6, G=9,  $N_\mathrm{real}$=1,  max\_iter=20,  30 seeds)',
             fontsize=11, fontweight='bold', pad=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.45, zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, max(means) + max(stds) + max(means) * 0.22)

plt.tight_layout()
for ext in ['png', 'pdf']:
    p = os.path.join(OUT_DIR, f'fig_time_realworld.{ext}')
    plt.savefig(p, dpi=200, bbox_inches='tight')
    print(f'  ✓ {p}')
plt.close()

# 保存 JSON
summary = {a: {'mean': float(np.mean(times[a])), 'std': float(np.std(times[a])),
               'all': [float(v) for v in times[a]]}
           for a in ALG_NAMES}
summary['settings'] = {'nbrOfRealizations': NBR_R, 'max_iter': MAX_ITER,
                       'K': K, 'L': L, 'G': G, 'n_seeds': N_SEEDS}
with open(os.path.join(OUT_DIR, 'time_realworld.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print("完成")
