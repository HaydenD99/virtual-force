"""
静态场景计算时间对比图
从 raw_results.json 读取已有的 100 个种子计时数据，直接绘图
"""
import numpy as np
import json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

STATIC_JSON = 'result/static_full_comparison/raw_results.json'
OUT_DIR     = 'result/computation_time'
ALG_NAMES   = ['LB-BVF', 'GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']
COLORS      = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
LABELS      = ['LB-BVF', 'GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']

os.makedirs(OUT_DIR, exist_ok=True)

# ── 读取计时数据 ──────────────────────────────────────────────────────
with open(STATIC_JSON) as f:
    data = json.load(f)

times = {a: [] for a in ALG_NAMES}
for seed_data in data.values():
    for a in ALG_NAMES:
        if a in seed_data and 'time' in seed_data[a]:
            times[a].append(float(seed_data[a]['time']))

means = [np.mean(times[a]) for a in ALG_NAMES]
stds  = [np.std(times[a])  for a in ALG_NAMES]

print("静态场景计算时间 (100 seeds):")
for a, m, s in zip(ALG_NAMES, means, stds):
    print(f"  {a:<14}: {m:.2f} ± {s:.2f} s")

# ── 绘图 ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.8))

x = np.arange(len(ALG_NAMES))
bars = ax.bar(x, means, 0.52,
              color=COLORS, edgecolor='white', linewidth=0.8,
              yerr=stds, capsize=6,
              error_kw=dict(elinewidth=1.4, ecolor='#444444', capthick=1.4),
              zorder=3)

# 数值标注
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + max(means) * 0.013,
            f'{m:.1f}s',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#222222')

# 加速比标注（LB-BVF 相对最慢对手）
speedup = max(means[1:]) / means[0]
ax.annotate(
    f'{speedup:.1f}× faster\n(vs slowest baseline)',
    xy=(x[0], means[0]),
    xytext=(x[0] + 0.75, means[0] + max(means) * 0.18),
    fontsize=9.5, color='#c0392b', fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.35', fc='#fef9e7',
              ec='#c0392b', alpha=0.9)
)

ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=11)
ax.set_ylabel('Computation Time (s)', fontsize=12)
ax.set_title('Static Deployment: Computation Time Comparison\n'
             r'(K=40, L=6, G=9,  100 independent seeds)',
             fontsize=11.5, fontweight='bold', pad=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.45, zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, max(means) + max(stds) + max(means) * 0.22)

plt.tight_layout()
for ext in ['png', 'pdf']:
    path = os.path.join(OUT_DIR, f'fig_time_static.{ext}')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f'  ✓ {path}')
plt.close()
print("完成")
