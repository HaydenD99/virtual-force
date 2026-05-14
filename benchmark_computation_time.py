"""
Computation Time Benchmark
==========================
Static scenario: load timing data from existing raw_results.json
Dynamic scenario: run a dedicated timing benchmark (N_BENCH seeds × T_STEPS steps)

Static : K=40, L=6, G=9
Dynamic : K=30, L=9, G=6

Outputs: result/computation_time/
  - fig_time_static.eps        -- static deployment timing bar chart
  - fig_time_dynamic.eps       -- dynamic per-step timing bar chart
  - fig_time_combined.eps      -- combined figure
  - fig_time_realworld.eps     -- real-world timing bar chart (if time_realworld.json exists)
  - time_summary.json          -- raw timing summary
"""

import numpy as np
import json, os, sys, io, time
import warnings
warnings.filterwarnings('ignore')

# ================= 字体与导出设置（仅改 dpi + 矢量图 + 字体） =================
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = ['Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.antialiased'] = True

# ─── 静态配置 (与 run_static_full_comparison.py 一致) ──────────────────
K_S, L_S, G_S = 40, 6, 9
STATIC_JSON   = 'result/static_full_comparison/raw_results.json'

# ─── 动态配置 (与 run_dynamic_large_scale.py 一致) ─────────────────────
K_D, L_D, G_D = 30, 9, 6
N_BENCH       = 15    # 基准测试种子数 (用于动态计时)
T_STEPS       = 5     # 每个种子测试步数
ITER_STEP     = 10    # 每步内优化迭代次数 (与正式实验一致)
DT            = 5.0
USER_SIGMA    = 8.0

OUT_DIR = 'result/computation_time'
STATIC_ALGS  = ['LB-BVF', 'GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']
DYNAMIC_ALGS = ['Dynamic LB-BVF', 'GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']

COLORS = {
    'LB-BVF':         '#ee6f63',
    'Dynamic LB-BVF': '#ee6f63',
    'GA-3D-LB':       '#5fa8e8',
    'PSO-3D-LB':      '#63cfa0',
    'SSA-3D-LB':      '#b08adf',
}
LABELS = {
    'LB-BVF':         'BVF',
    'Dynamic LB-BVF': 'BVF',
    'GA-3D-LB':       'DGA-CF',
    'PSO-3D-LB':      'DPSO-CF',
    'SSA-3D-LB':      'NSSA-CF',
}


# ═══════════════════════════════════════════════════════════════════════
#  STEP 1 : 读取静态计时
# ═══════════════════════════════════════════════════════════════════════

def load_static_times(json_path):
    """从已有结果文件中提取每个算法的计算时间列表"""
    with open(json_path) as f:
        data = json.load(f)

    times = {a: [] for a in STATIC_ALGS}
    for seed_str, seed_data in data.items():
        for a in STATIC_ALGS:
            if a in seed_data and 'time' in seed_data[a]:
                times[a].append(float(seed_data[a]['time']))

    print(f"\n[静态] 已从 {json_path} 加载 {len(list(times.values())[0])} 个种子的计时数据")
    for a in STATIC_ALGS:
        arr = np.array(times[a])
        print(f"  {a:<18}: mean={arr.mean():.2f}s  std={arr.std():.2f}s  "
              f"[{arr.min():.1f}, {arr.max():.1f}]")
    return times


# ═══════════════════════════════════════════════════════════════════════
#  STEP 2 : 动态每步计时基准测试
# ═══════════════════════════════════════════════════════════════════════

def benchmark_dynamic_times():
    """运行小规模计时实验, 记录每步优化所需的平均时间"""
    from balanced_virtual_force_optimizer_v6 import (
        BalancedVirtualForceOptimizerV6, create_v6_config)
    from dynamic_lb_bvf import DynamicLoadBalancedBVF, create_dynamic_lb_config
    from heuristic_lb_3d import (GA3D_LB, PSO3D_LB, SSA3D_LB,
                                 create_heuristic_config, one_step_optimize)
    from run_dynamic_energy_comparison import UAVEnergyModel, brownian_motion_users

    energy_model = UAVEnergyModel()
    times_per_step = {a: [] for a in DYNAMIC_ALGS}

    for seed_idx in range(N_BENCH):
        seed = seed_idx + 1000   # 与训练种子隔离, 避免重叠
        np.random.seed(seed)

        # ── 初始化场景 ──
        UE_pos = np.column_stack([
            np.random.uniform(50, 950, (K_D, 2)),
            np.ones(K_D) * 1.65
        ])

        gx = np.linspace(200, 800, 3)
        gy = np.linspace(333, 667, 2)
        GX, GY = np.meshgrid(gx, gy)
        gAP = np.column_stack([
            GX.flatten()[:G_D],
            GY.flatten()[:G_D],
            np.ones(G_D) * 15.0
        ])

        ux = np.linspace(200, 800, 3)
        uy = np.linspace(200, 800, 3)
        UX, UY = np.meshgrid(ux, uy)
        UAV_init = np.column_stack([
            UX.flatten()[:L_D],
            UY.flatten()[:L_D],
            np.ones(L_D) * 50.0
        ])

        # ── 实例化算法 (仅创建一次) ──
        dlb_cfg = create_dynamic_lb_config(K=K_D, L=L_D, G=G_D)
        dlb_cfg.update({
            'time_step': DT,
            'max_iterations': 80,
            'nbrOfRealizations': 20
        })
        dlb = DynamicLoadBalancedBVF(dlb_cfg, energy_model)

        hcfg = create_heuristic_config(K=K_D, L=L_D, G=G_D)
        hcfg.update({
            'num_serving_APs': 3,
            'nbrOfRealizations_inner': 10,
            'nbrOfRealizations_final': 25,
            'max_iterations': ITER_STEP,
            'newssa_max_iter': ITER_STEP,
            'max_generations': ITER_STEP
        })
        ga = GA3D_LB(hcfg)
        pso = PSO3D_LB(hcfg)
        ssa = SSA3D_LB(hcfg)

        pos = {
            'Dynamic LB-BVF': UAV_init.copy(),
            'GA-3D-LB': UAV_init.copy(),
            'PSO-3D-LB': UAV_init.copy(),
            'SSA-3D-LB': UAV_init.copy()
        }
        cur_UE = UE_pos.copy()

        for step in range(T_STEPS):
            cur_UE = brownian_motion_users(cur_UE, sigma=USER_SIGMA)

            # ── Dynamic LB-BVF ──
            old = sys.stdout
            sys.stdout = io.StringIO()
            t0 = time.perf_counter()
            new_pos, _, _, _, _ = dlb.optimize_one_step(
                cur_UE, gAP, pos['Dynamic LB-BVF'], max_iter=ITER_STEP, dt=DT
            )
            t_dlb = time.perf_counter() - t0
            sys.stdout = old
            pos['Dynamic LB-BVF'] = new_pos

            # ── GA ──
            old = sys.stdout
            sys.stdout = io.StringIO()
            t0 = time.perf_counter()
            new_pos_ga, _, _, _, _ = one_step_optimize(
                ga, cur_UE, gAP, pos['GA-3D-LB'],
                max_iter=ITER_STEP, energy_model=energy_model, flight_speed=10.0
            )
            t_ga = time.perf_counter() - t0
            sys.stdout = old
            pos['GA-3D-LB'] = new_pos_ga

            # ── PSO ──
            old = sys.stdout
            sys.stdout = io.StringIO()
            t0 = time.perf_counter()
            new_pos_pso, _, _, _, _ = one_step_optimize(
                pso, cur_UE, gAP, pos['PSO-3D-LB'],
                max_iter=ITER_STEP, energy_model=energy_model, flight_speed=10.0
            )
            t_pso = time.perf_counter() - t0
            sys.stdout = old
            pos['PSO-3D-LB'] = new_pos_pso

            # ── SSA ──
            old = sys.stdout
            sys.stdout = io.StringIO()
            t0 = time.perf_counter()
            new_pos_ssa, _, _, _, _ = one_step_optimize(
                ssa, cur_UE, gAP, pos['SSA-3D-LB'],
                max_iter=ITER_STEP, energy_model=energy_model, flight_speed=10.0
            )
            t_ssa = time.perf_counter() - t0
            sys.stdout = old
            pos['SSA-3D-LB'] = new_pos_ssa

            times_per_step['Dynamic LB-BVF'].append(t_dlb)
            times_per_step['GA-3D-LB'].append(t_ga)
            times_per_step['PSO-3D-LB'].append(t_pso)
            times_per_step['SSA-3D-LB'].append(t_ssa)

        print(f"  Seed {seed_idx+1}/{N_BENCH} 完成", flush=True)

    print(f"\n[动态] {N_BENCH} 个种子 × {T_STEPS} 步 = {N_BENCH*T_STEPS} 个计时样本")
    for a in DYNAMIC_ALGS:
        arr = np.array(times_per_step[a])
        print(f"  {a:<22}: mean={arr.mean():.2f}s  std={arr.std():.2f}s  "
              f"[{arr.min():.1f}, {arr.max():.1f}]")
    return times_per_step


# ═══════════════════════════════════════════════════════════════════════
#  STEP 3 : 绘图
# ═══════════════════════════════════════════════════════════════════════

def plot_time_bar(algs, times_dict, title, ylabel, ax, scenario='static'):
    """在给定 axes 上绘制计算时间柱状图"""
    means = [np.mean(times_dict[a]) for a in algs]
    stds = [np.std(times_dict[a]) for a in algs]
    x = np.arange(len(algs))
    w = 0.55

    bars = ax.bar(
        x, means, w,
        color=[COLORS[a] for a in algs],
        edgecolor='white', linewidth=0.8,
        yerr=stds, capsize=5,
        error_kw=dict(elinewidth=1.2, ecolor='#555555', capthick=1.2),
        zorder=3
    )

    # 数值标注
    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + max(means) * 0.012,
            f'{m:.1f}s',
            ha='center', va='bottom',
            fontsize=9.5, fontweight='bold', color='#333333'
        )

    # 加速比标注 (相对于最慢的对手)
    own_idx = 0
    own_mean = means[own_idx]
    comp_means = means[1:]
    speedup_max = max(comp_means) / own_mean

    ax.annotate(
        f'↑ {speedup_max:.1f}× faster\n(vs slowest baseline)',
        xy=(x[own_idx], own_mean),
        xytext=(x[own_idx] + 0.6, own_mean + max(means) * 0.15),
        fontsize=9, color='#c0392b', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.4),
        bbox=dict(boxstyle='round,pad=0.3', fc='#fef9e7', ec='#c0392b', alpha=0.85)
    )

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in algs], fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _save_fig_multi(fig, out_path_no_ext):
    fig.savefig(f'{out_path_no_ext}.eps', format='eps', bbox_inches='tight')


def plot_all(static_times, dyn_times, out_dir):
    import matplotlib.pyplot as plt

    # ── 单独静态图 ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    plot_time_bar(
        STATIC_ALGS, static_times,
        'Static Deployment: Computation Time\n'
        r'(K=40, L=6, G=9, 100 seeds)',
        'Computation Time (s)', ax, scenario='static'
    )
    plt.tight_layout()
    _save_fig_multi(fig, os.path.join(out_dir, 'fig_time_static'))
    plt.close(fig)
    print("  ✓ fig_time_static saved")

    # ── 单独动态图 ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    plot_time_bar(
        DYNAMIC_ALGS, dyn_times,
        'Dynamic Re-Deployment: Per-Step Computation Time\n'
        r'(K=30, L=9, G=6, ' + f'{N_BENCH}×{T_STEPS} steps)',
        'Per-Step Computation Time (s)', ax, scenario='dynamic'
    )
    plt.tight_layout()
    _save_fig_multi(fig, os.path.join(out_dir, 'fig_time_dynamic'))
    plt.close(fig)
    print("  ✓ fig_time_dynamic saved")

    # ── 合并双图 (论文用) ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    plot_time_bar(
        STATIC_ALGS, static_times,
        '(a) Static Deployment',
        'Computation Time (s)', axes[0], scenario='static'
    )
    plot_time_bar(
        DYNAMIC_ALGS, dyn_times,
        '(b) Dynamic Per-Step Re-Deployment',
        'Per-Step Computation Time (s)', axes[1], scenario='dynamic'
    )

    fig.suptitle(
        'Algorithm Computation Time Comparison',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    _save_fig_multi(fig, os.path.join(out_dir, 'fig_time_combined'))
    plt.close(fig)
    print("  ✓ fig_time_combined saved")


def plot_from_realworld_json(json_path, out_dir):
    import matplotlib.pyplot as plt

    with open(json_path, 'r') as f:
        data = json.load(f)

    algs = ['LB-BVF', 'GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']
    means = [float(data[a]['mean']) for a in algs]
    stds = [float(data[a]['std']) for a in algs]

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    x = np.arange(len(algs))
    bars = ax.bar(
        x, means, 0.58,
        color=[COLORS[a] for a in algs],
        edgecolor='white', linewidth=0.9,
        yerr=stds, capsize=5,
        error_kw=dict(elinewidth=1.2, ecolor='#555555', capthick=1.2),
        zorder=3
    )

    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            m + s + max(means) * 0.012,
            f'{m:.2f}s',
            ha='center', va='bottom',
            fontsize=9.5, fontweight='bold', color='#333333'
        )

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in algs], fontsize=10.5)
    ax.set_ylabel('计算时间 (s)', fontsize=11)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    _save_fig_multi(fig, os.path.join(out_dir, 'fig_time_realworld'))
    plt.close(fig)
    print('  ✓ fig_time_realworld saved')


# ═══════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    t_wall = time.time()

    realworld_json = os.path.join(OUT_DIR, 'time_realworld.json')
    if os.path.exists(realworld_json):
        print("=" * 60)
        print("检测到 time_realworld.json，直接绘制真实场景计算时间图 ...")
        plot_from_realworld_json(realworld_json, OUT_DIR)
        print(f"\n结果目录: {OUT_DIR}/")
        print("  fig_time_realworld.eps")
        raise SystemExit(0)

    # 1. 静态计时
    print("=" * 60)
    print("读取静态计算时间 ...")
    static_times = load_static_times(STATIC_JSON)

    # 2. 动态计时
    print("\n" + "=" * 60)
    print(f"动态每步计时基准  ({N_BENCH} seeds × {T_STEPS} steps) ...")
    dyn_times = benchmark_dynamic_times()

    # 3. 保存 JSON
    summary = {
        'static': {
            a: {
                'mean': float(np.mean(v)),
                'std': float(np.std(v)),
                'samples': len(v)
            }
            for a, v in static_times.items()
        },
        'dynamic': {
            a: {
                'mean': float(np.mean(v)),
                'std': float(np.std(v)),
                'samples': len(v)
            }
            for a, v in dyn_times.items()
        },
    }
    with open(os.path.join(OUT_DIR, 'time_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ time_summary.json 已保存")

    # 4. 绘图
    print("\n绘图中 ...")
    plot_all(static_times, dyn_times, OUT_DIR)

    print(f"\n{'='*60}")
    print(f"全部完成  总耗时: {time.time()-t_wall:.0f}s")
    print(f"结果目录: {OUT_DIR}/")
    print("  fig_time_static.eps")
    print("  fig_time_dynamic.eps")
    print("  fig_time_combined.eps")
    print("  time_summary.json")