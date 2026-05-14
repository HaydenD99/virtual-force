"""
动态场景三算法对比实验
========================
对比算法:
  1. Dynamic LB-BVF  (本文新算法: 负载均衡 + 公平 + 能量 + 3D高度)
  2. Energy-Aware BVF V6  (EA-BVF: 能量感知基线)
  3. BVF V6  (无能量感知基线)

场景:
  K=60 用户, G=4 地面AP, L=9 无人机
  用户每隔 dt=5s 做布朗运动 (σ=8m/步)
  共 20 时间步 = 100s

指标:
  min_rate (Mbps), JFI_eff, 单步能耗 (J), 累计能耗 (kJ), JointScore_dynamic
"""

import numpy as np
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

from balanced_virtual_force_optimizer_v6 import BalancedVirtualForceOptimizerV6, create_v6_config
from run_dynamic_energy_comparison import UAVEnergyModel, brownian_motion_users, EnergyAwareBVF_V6
from dynamic_lb_bvf import DynamicLoadBalancedBVF, create_dynamic_lb_config

# ======================================================================
#  实验参数
# ======================================================================
SEED        = 62
K, G, L     = 60, 4, 9
NUM_STEPS   = 20
DT          = 5.0          # 时间步长 (s)
ITER_STEP   = 20           # 每步迭代次数 (公平对比: 三者相同)
USER_SIGMA  = 8.0          # 用户移动标准差 (m)
ENERGY_LAMBDA = 0.3        # EA-BVF 能耗惩罚权重
OUTPUT_DIR  = 'result/dynamic_lb_comparison'


# ======================================================================
#  辅助: 确定性评估 JFI_eff
# ======================================================================

def eval_jfi_eff(evaluator, UE_pos, gAP, UAV_pos, G):
    """计算依赖度加权 JFI_eff"""
    all_AP = np.vstack([gAP, UAV_pos])
    _, _, betas = evaluator.compute_channel_model(UE_pos, all_AP)
    mask = evaluator.compute_AP_selection_mask(betas)
    mask_uav = mask[:, G:]; mask_gnd = mask[:, :G]
    betas_uav = betas[:, G:]; betas_gnd = betas[:, :G]

    L = UAV_pos.shape[0]
    ground_cov = np.zeros(len(UE_pos))
    for k in range(len(UE_pos)):
        sg = np.where(mask_gnd[k])[0]
        ground_cov[k] = betas_gnd[k, sg].sum() if len(sg) > 0 else 0.0

    eff = np.zeros(L)
    for l in range(L):
        for k in np.where(mask_uav[:, l])[0]:
            dep = betas_uav[k, l] / (ground_cov[k] + betas_uav[k, l] + 1e-12)
            eff[l] += dep

    s = eff.sum()
    return float(s**2 / (L * (eff**2).sum() + 1e-12)) if s > 1e-10 else 1.0


# ======================================================================
#  主仿真
# ======================================================================

def run_comparison(seed=SEED):
    np.random.seed(seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- 公共配置 ----------
    base_cfg = {
        'square_length': 1000,
        'num_UE': K, 'num_ground_AP': G, 'num_UAV': L,
        'num_serving_APs': 3, 'M': 4,
        'UE_height': 1.65, 'ground_AP_height': 15.0, 'UAV_height': 50.0,
        'nbrOfRealizations': 50,
        'tau_c': 200, 'tau_p': K,
    }

    energy_model = UAVEnergyModel()
    hover_E_step = energy_model.hover_energy(DT) * L  # 纯悬停能耗 (一步)

    # ---------- 初始化位置 (统一种子) ----------
    UE_xy = np.random.uniform(50, 950, (K, 2))
    UE_pos = np.column_stack([UE_xy, np.ones(K) * 1.65])

    gx = np.linspace(250, 750, 2); gy = np.linspace(250, 750, 2)
    GX, GY = np.meshgrid(gx, gy)
    gAP = np.column_stack([GX.flatten(), GY.flatten(), np.ones(G) * 15.0])

    ux = np.linspace(200, 800, 3); uy = np.linspace(200, 800, 3)
    UX, UY = np.meshgrid(ux, uy)
    UAV_init = np.column_stack([
        UX.flatten()[:L], UY.flatten()[:L], np.ones(L) * 50.0])

    # ---------- 初始化优化器 ----------
    # 1. Dynamic LB-BVF
    dlb_cfg = create_dynamic_lb_config(K=K, L=L, G=G)
    dlb_cfg.update(base_cfg)
    dlb_cfg['time_step'] = DT
    dlb = DynamicLoadBalancedBVF(dlb_cfg, energy_model)

    # 2. EA-BVF V6
    ea_cfg = create_v6_config()
    ea_cfg.update(base_cfg)
    ea_cfg.update({
        'max_iterations': ITER_STEP,
        'user_sigma': USER_SIGMA,
        'max_displacement': 5.0 * USER_SIGMA,
        'energy_lambda': ENERGY_LAMBDA,
        'E_normalize': 5000.0,
    })
    ea_bvf = EnergyAwareBVF_V6(ea_cfg, energy_model)

    # 3. BVF V6 (baseline, no energy)
    v6_cfg = create_v6_config()
    v6_cfg.update(base_cfg)
    v6_cfg['max_iterations'] = ITER_STEP
    v6_eval = BalancedVirtualForceOptimizerV6(v6_cfg)  # for channel eval

    # 当前位置
    pos = {
        'DLB': UAV_init.copy(),
        'EA':  UAV_init.copy(),
        'V6':  UAV_init.copy(),
    }
    cumul = {'DLB': 0.0, 'EA': 0.0, 'V6': 0.0}
    records = {'time': []}
    for alg in ['DLB', 'EA', 'V6']:
        records[alg] = {
            'min_rate': [], 'sum_rate': [], 'jfi': [],
            'energy_step': [], 'energy_cumul': [], 'dist': [], 'joint_score': []
        }

    # ---------- Step 0: 初始状态 ----------
    print("=" * 70)
    print("  Dynamic LB-BVF vs EA-BVF vs BVF-V6  (Dynamic Comparison)")
    print(f"  K={K} | G={G} | L={L} | steps={NUM_STEPS} × {DT}s | seed={seed}")
    print("=" * 70)

    cur_UE = UE_pos.copy()
    records['time'].append(0.0)

    np.random.seed(seed)
    all_AP0 = np.vstack([gAP, UAV_init])
    _, _, b0 = v6_eval.compute_channel_model(cur_UE, all_AP0)
    m0 = v6_eval.compute_AP_selection_mask(b0)
    r0, sr0 = v6_eval.compute_user_rates(cur_UE, all_AP0, m0)
    mr0 = float(r0.min()); sr0 = float(sr0)
    jfi0 = eval_jfi_eff(v6_eval, cur_UE, gAP, UAV_init, G)

    for alg in ['DLB', 'EA', 'V6']:
        records[alg]['min_rate'].append(mr0)
        records[alg]['sum_rate'].append(sr0)
        records[alg]['jfi'].append(jfi0)
        records[alg]['energy_step'].append(0.0)
        records[alg]['energy_cumul'].append(0.0)
        records[alg]['dist'].append(0.0)
        records[alg]['joint_score'].append(0.0)

    print(f"\nStep 0 (t=0): mr={mr0:.2f} Mbps | JFI={jfi0:.4f} | all_algs same")

    # ---------- Step 1 ~ NUM_STEPS-1 ----------
    for step in range(1, NUM_STEPS):
        t = step * DT
        records['time'].append(t)
        print(f"\n--- Step {step}/{NUM_STEPS-1}  (t={t:.0f}s) ---")

        # 用户布朗运动
        cur_UE = brownian_motion_users(cur_UE, sigma=USER_SIGMA, square_length=1000)

        # ── 1. Dynamic LB-BVF ──
        np.random.seed(seed + step * 100 + 1)
        new_pos, mr_d, sr_d, e_d, dist_d = dlb.optimize_one_step(
            cur_UE, gAP, pos['DLB'], max_iter=ITER_STEP, dt=DT)
        e_d += hover_E_step
        cumul['DLB'] += e_d
        jfi_d = eval_jfi_eff(v6_eval, cur_UE, gAP, new_pos, G)
        js_d  = dlb.joint_score_dynamic(mr_d, jfi_d, e_d)
        pos['DLB'] = new_pos
        records['DLB']['min_rate'].append(mr_d)
        records['DLB']['sum_rate'].append(sr_d)
        records['DLB']['jfi'].append(jfi_d)
        records['DLB']['energy_step'].append(e_d)
        records['DLB']['energy_cumul'].append(cumul['DLB'])
        records['DLB']['dist'].append(dist_d)
        records['DLB']['joint_score'].append(js_d)
        print(f"  DLB:  mr={mr_d:.2f} JFI={jfi_d:.4f} JS={js_d:.4f}"
              f" E={e_d:.0f}J d={dist_d:.1f}m cum={cumul['DLB']/1000:.2f}kJ")

        # ── 2. EA-BVF V6 ──
        np.random.seed(seed + step * 100 + 2)
        pos_ea, mr_ea, sr_ea, e_ea, dist_ea = ea_bvf.optimize_one_step(
            cur_UE, gAP, pos['EA'], max_iter=ITER_STEP)
        e_ea += hover_E_step
        cumul['EA'] += e_ea
        jfi_ea = eval_jfi_eff(v6_eval, cur_UE, gAP, pos_ea, G)
        js_ea  = dlb.joint_score_dynamic(mr_ea, jfi_ea, e_ea)
        pos['EA'] = pos_ea
        records['EA']['min_rate'].append(mr_ea)
        records['EA']['sum_rate'].append(sr_ea)
        records['EA']['jfi'].append(jfi_ea)
        records['EA']['energy_step'].append(e_ea)
        records['EA']['energy_cumul'].append(cumul['EA'])
        records['EA']['dist'].append(dist_ea)
        records['EA']['joint_score'].append(js_ea)
        print(f"  EA:   mr={mr_ea:.2f} JFI={jfi_ea:.4f} JS={js_ea:.4f}"
              f" E={e_ea:.0f}J d={dist_ea:.1f}m cum={cumul['EA']/1000:.2f}kJ")

        # ── 3. BVF V6 (one-step, same iter count for fairness) ──
        np.random.seed(seed + step * 100 + 3)
        v6_step_cfg = v6_cfg.copy()
        v6_step_cfg['max_iterations'] = ITER_STEP
        v6_step_inst = BalancedVirtualForceOptimizerV6(v6_step_cfg)
        import io
        old_out = sys.stdout; sys.stdout = io.StringIO()
        res6 = v6_step_inst.optimize(cur_UE, gAP, pos['V6'].copy())
        sys.stdout = old_out
        pos_v6 = res6['optimized_UAV_pos']
        mr_v6 = res6['final_min_rate']; sr_v6 = res6['final_sum_rate']
        e_v6, dist_v6 = energy_model.total_energy_for_repositioning(
            pos['V6'], pos_v6, flight_speed=10.0)
        e_v6 += hover_E_step
        cumul['V6'] += e_v6
        jfi_v6 = eval_jfi_eff(v6_eval, cur_UE, gAP, pos_v6, G)
        js_v6  = dlb.joint_score_dynamic(mr_v6, jfi_v6, e_v6)
        pos['V6'] = pos_v6
        records['V6']['min_rate'].append(mr_v6)
        records['V6']['sum_rate'].append(sr_v6)
        records['V6']['jfi'].append(jfi_v6)
        records['V6']['energy_step'].append(e_v6)
        records['V6']['energy_cumul'].append(cumul['V6'])
        records['V6']['dist'].append(dist_v6)
        records['V6']['joint_score'].append(js_v6)
        print(f"  V6:   mr={mr_v6:.2f} JFI={jfi_v6:.4f} JS={js_v6:.4f}"
              f" E={e_v6:.0f}J d={dist_v6:.1f}m cum={cumul['V6']/1000:.2f}kJ")

        sys.stdout.flush()

    return records


# ======================================================================
#  结果摘要
# ======================================================================

def print_summary(records):
    algs = ['DLB', 'EA', 'V6']
    labels = {'DLB': 'Dynamic LB-BVF', 'EA': 'EA-BVF V6', 'V6': 'BVF V6'}

    print(f"\n{'='*80}")
    print("  动态对比总结 (Step 1~end 均值)")
    print(f"{'='*80}")
    print(f"{'算法':<18} {'Avg MinRate':>12} {'Avg JFI':>10} {'Avg JS':>10}"
          f" {'总能耗kJ':>10} {'能效Mbps/kJ':>12}")
    print("-" * 80)

    for a in algs:
        r  = records[a]
        mr = np.mean(r['min_rate'][1:])
        jf = np.mean(r['jfi'][1:])
        js = np.mean(r['joint_score'][1:])
        ec = r['energy_cumul'][-1] / 1000.0
        ee = mr / (ec + 1e-6)
        print(f"{labels[a]:<18} {mr:>12.3f} {jf:>10.4f} {js:>10.4f}"
              f" {ec:>10.2f} {ee:>12.4f}")
    print("=" * 80)


# ======================================================================
#  绘图
# ======================================================================

def plot_results(records, seed, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    time_axis = records['time']
    algs   = ['DLB', 'EA', 'V6']
    labels = {'DLB': 'Dynamic LB-BVF (Ours)', 'EA': 'EA-BVF V6', 'V6': 'BVF V6'}
    colors = {'DLB': '#e74c3c', 'EA': '#3498db', 'V6': '#2ecc71'}
    marks  = {'DLB': 'o',       'EA': 's',       'V6': '^'}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Dynamic LB-BVF Comparison  (K={K}, L={L}, G={G}, seed={seed})',
                 fontsize=14, fontweight='bold')

    def _plot(ax, key, ylabel, title, scale=1.0):
        for a in algs:
            y = [v * scale for v in records[a][key]]
            ax.plot(time_axis, y, color=colors[a], marker=marks[a],
                    markersize=5, linewidth=2, label=labels[a])
        ax.set_xlabel('Time (s)'); ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], 'min_rate',    'Min User Rate (Mbps)',   '(a) Min-Rate')
    _plot(axes[0, 1], 'jfi',         'JFI_eff',                '(b) Load Fairness (JFI_eff)')
    _plot(axes[0, 2], 'joint_score', 'Joint Score',            '(c) 3-Objective Joint Score')
    _plot(axes[1, 0], 'energy_step', 'Energy per Step (J)',    '(d) Per-Step Energy')
    _plot(axes[1, 1], 'energy_cumul','Cumulative Energy (kJ)', '(e) Cumulative Energy', scale=1/1000)
    _plot(axes[1, 2], 'dist',        'Movement Distance (m)',  '(f) UAV Movement per Step')

    plt.tight_layout()
    path = os.path.join(output_dir, f'dynamic_lb_comparison_seed{seed}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {path}")

    # 能效比图
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for a in algs:
        t_plot, ee_plot = [], []
        for i, (t, mr, ec) in enumerate(
                zip(time_axis, records[a]['min_rate'], records[a]['energy_cumul'])):
            if ec > 0:
                t_plot.append(t); ee_plot.append(mr / (ec / 1000.0 + 1e-6))
        if t_plot:
            ax2.plot(t_plot, ee_plot, color=colors[a], marker=marks[a],
                     markersize=5, linewidth=2, label=labels[a])
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Energy Efficiency (Mbps / kJ)', fontsize=12)
    ax2.set_title('Energy Efficiency over Time', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(output_dir, f'dynamic_lb_efficiency_seed{seed}.png')
    plt.savefig(path2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"能效图已保存: {path2}")

    return path, path2


# ======================================================================
#  入口
# ======================================================================

if __name__ == '__main__':
    t0 = time.time()
    records = run_comparison(seed=SEED)
    print_summary(records)

    # 保存 JSON
    json_path = os.path.join(OUTPUT_DIR, f'dynamic_lb_seed{SEED}.json')
    save = {'config': {'seed': SEED, 'K': K, 'G': G, 'L': L,
                       'num_steps': NUM_STEPS, 'dt': DT,
                       'iter_per_step': ITER_STEP, 'user_sigma': USER_SIGMA},
            'time': records['time']}
    for a in ['DLB', 'EA', 'V6']:
        save[a] = records[a]
    with open(json_path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\nJSON saved: {json_path}")

    plot_results(records, SEED, OUTPUT_DIR)

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print("=" * 70)
    print("  实验完成!")
    print("=" * 70)
