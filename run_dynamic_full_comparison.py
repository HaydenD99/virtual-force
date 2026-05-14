"""
动态场景全算法对比实验
========================
Dynamic LB-BVF (本文) vs GA-3D-LB vs PSO-3D-LB vs SSA-3D-LB

场景: K=30, L=9, G=6  → (9+6)×4=60 天线 / 30 用户, 比值=2.0, 满足 Cell-Free 要求
     10步×5s=50s 动态跟踪, 用户布朗运动 σ=8m
实验: 5 个随机种子
输出: result/dynamic_full_comparison/
  - raw_results.json
  - fig_timeseries.png/pdf   -- 时序: min_rate, JFI, energy, joint_score
  - fig_bar.png/pdf          -- 均值柱状图
  - fig_energy_eff.png/pdf   -- 能效比 (Mbps/kJ)
  - fig_radar.png/pdf        -- 雷达图
"""

import numpy as np
import json, os, sys, io, time
import warnings; warnings.filterwarnings('ignore')

from balanced_virtual_force_optimizer_v6 import BalancedVirtualForceOptimizerV6, create_v6_config
from dynamic_lb_bvf import DynamicLoadBalancedBVF, create_dynamic_lb_config
from heuristic_lb_3d import GA3D_LB, PSO3D_LB, SSA3D_LB, create_heuristic_config, one_step_optimize
from run_dynamic_energy_comparison import UAVEnergyModel, brownian_motion_users

# ─── 实验配置 ────────────────────────────────────────────────────────
# K=30, L=9, G=6 → (G+L)*M/K = (9+6)*4/30 = 2.0  满足 Cell-Free 基本要求
K, L, G     = 30, 9, 6
SEEDS       = [10, 42, 62, 88, 99]   # 5 个种子
NUM_STEPS   = 10         # 时间步数
DT          = 5.0        # 时间步长 (s)
ITER_STEP   = 10         # 每步迭代次数 (平衡精度与运行时间)
USER_SIGMA  = 8.0
W_MIN, W_JFI, REF, FLOOR = 0.25, 0.50, 60.0, 48.0
W_EE        = 0.25
OUT_DIR     = 'result/dynamic_full_comparison'

ALG_NAMES = ['Dynamic LB-BVF', 'GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']
COLORS    = {'Dynamic LB-BVF': '#e74c3c', 'GA-3D-LB': '#3498db',
             'PSO-3D-LB': '#2ecc71', 'SSA-3D-LB': '#9b59b6'}
MARKS     = {'Dynamic LB-BVF': 'o', 'GA-3D-LB': 's', 'PSO-3D-LB': '^', 'SSA-3D-LB': 'D'}


# ─── JFI_eff 评估 ───────────────────────────────────────────────────
def eval_jfi_eff(evaluator, UE_pos, gAP, UAV_pos, G, K, L):
    all_AP = np.vstack([gAP, UAV_pos])
    _, _, betas = evaluator.compute_channel_model(UE_pos, all_AP)
    mask   = evaluator.compute_AP_selection_mask(betas)
    mask_u = mask[:, G:]; mask_g = mask[:, :G]
    bu = betas[:, G:];   bg = betas[:, :G]
    gcov = np.array([bg[k, np.where(mask_g[k])[0]].sum() for k in range(K)])
    eff = np.zeros(L)
    for l in range(L):
        for k in np.where(mask_u[:, l])[0]:
            eff[l] += bu[k,l] / (gcov[k] + bu[k,l] + 1e-12)
    s = eff.sum()
    return float(s**2 / (L*(eff**2).sum()+1e-12)) if s>1e-10 else 1.0


def joint_score_dyn(mr, jfi, e_step, energy_model, L, dt, w_min, w_jfi, w_ee, ref, floor):
    E_ref = L * energy_model.P_hover * dt * 2.0
    ee = float(np.clip(1.0 - e_step / (E_ref + 1e-6), 0.0, 1.0))
    raw = w_min*(mr/ref) + w_jfi*jfi + w_ee*ee
    if mr < floor: raw *= (mr/floor)**2
    return float(raw)


# ─── 单种子动态仿真 ──────────────────────────────────────────────────
def run_one_seed(seed, energy_model, v6_eval):
    np.random.seed(seed)

    # 初始化场景
    UE_pos = np.column_stack([
        np.random.uniform(50, 950, (K, 2)), np.ones(K)*1.65])
    # G=6: 3×2 等间距网格
    gx = np.linspace(200, 800, 3); gy = np.linspace(333, 667, 2)
    GX, GY = np.meshgrid(gx, gy)
    gAP = np.column_stack([GX.flatten()[:G], GY.flatten()[:G], np.ones(G)*15.0])
    # L=9: 3×3 等间距网格
    ux = np.linspace(200, 800, 3); uy = np.linspace(200, 800, 3)
    UX, UY = np.meshgrid(ux, uy)
    UAV_init = np.column_stack([UX.flatten()[:L], UY.flatten()[:L], np.ones(L)*50.0])

    hover_E = energy_model.hover_energy(DT) * L

    # ── 初始化算法实例 ──
    # Dynamic LB-BVF
    dlb_cfg = create_dynamic_lb_config(K=K, L=L, G=G)
    dlb_cfg.update({'time_step': DT, 'w_min': W_MIN, 'w_jfi': W_JFI, 'w_ee': W_EE,
                    'max_iterations': 80})
    dlb = DynamicLoadBalancedBVF(dlb_cfg, energy_model)

    # Heuristic 3D-LB algorithms
    hcfg = create_heuristic_config(K=K, L=L, G=G)
    hcfg['num_serving_APs'] = 3
    hcfg['nbrOfRealizations_inner'] = 10   # 平衡精度与速度
    hcfg['nbrOfRealizations_final'] = 30
    hcfg['max_iterations'] = ITER_STEP
    hcfg['newssa_max_iter'] = ITER_STEP
    hcfg['max_generations']  = ITER_STEP

    # One-step algorithms (re-instantiated each step for freshness)
    cur_UE = UE_pos.copy()
    pos = {a: UAV_init.copy() for a in ALG_NAMES}
    cumul = {a: 0.0 for a in ALG_NAMES}

    records = {'time': []}
    for a in ALG_NAMES:
        records[a] = {'min_rate':[], 'jfi':[], 'energy_step':[], 'energy_cumul':[],
                      'dist':[], 'joint_score':[]}

    # Step 0: initial state
    records['time'].append(0.0)
    all_AP0 = np.vstack([gAP, UAV_init])
    _, _, b0 = v6_eval.compute_channel_model(cur_UE, all_AP0)
    m0 = v6_eval.compute_AP_selection_mask(b0)
    r0, _ = v6_eval.compute_user_rates(cur_UE, all_AP0, m0)
    mr0 = float(r0.min())
    jfi0 = eval_jfi_eff(v6_eval, cur_UE, gAP, UAV_init, G, K, L)
    # t=0: 无位移能耗 (energy_step=0), EE_norm=1.0 (最优效率)
    js0 = joint_score_dyn(mr0, jfi0, 0.0, energy_model, L, DT,
                          W_MIN, W_JFI, W_EE, REF, FLOOR)
    for a in ALG_NAMES:
        records[a]['min_rate'].append(mr0)
        records[a]['jfi'].append(jfi0)
        records[a]['energy_step'].append(0.0)
        records[a]['energy_cumul'].append(0.0)
        records[a]['dist'].append(0.0)
        records[a]['joint_score'].append(js0)   # 用实际初始值代替0

    print(f"  Step0: mr={mr0:.2f} jfi={jfi0:.4f} js={js0:.4f}")

    # Steps 1..NUM_STEPS
    for step in range(1, NUM_STEPS + 1):
        t = step * DT
        records['time'].append(t)
        cur_UE = brownian_motion_users(cur_UE, sigma=USER_SIGMA)

        # Dynamic LB-BVF
        np.random.seed(seed + step*100 + 1)
        new_pos, mr_d, sr_d, e_d, dist_d = dlb.optimize_one_step(
            cur_UE, gAP, pos['Dynamic LB-BVF'], max_iter=ITER_STEP, dt=DT)
        e_d += hover_E
        cumul['Dynamic LB-BVF'] += e_d
        jfi_d = eval_jfi_eff(v6_eval, cur_UE, gAP, new_pos, G, K, L)
        js_d  = joint_score_dyn(mr_d, jfi_d, e_d, energy_model, L, DT,
                                 W_MIN, W_JFI, W_EE, REF, FLOOR)
        pos['Dynamic LB-BVF'] = new_pos
        records['Dynamic LB-BVF']['min_rate'].append(mr_d)
        records['Dynamic LB-BVF']['jfi'].append(jfi_d)
        records['Dynamic LB-BVF']['energy_step'].append(e_d)
        records['Dynamic LB-BVF']['energy_cumul'].append(cumul['Dynamic LB-BVF'])
        records['Dynamic LB-BVF']['dist'].append(dist_d)
        records['Dynamic LB-BVF']['joint_score'].append(js_d)

        # Heuristic algorithms (one step each)
        for alg_name, AlgClass in [
            ('GA-3D-LB', GA3D_LB), ('PSO-3D-LB', PSO3D_LB), ('SSA-3D-LB', SSA3D_LB)]:
            np.random.seed(seed + step*100 + {'GA-3D-LB':2,'PSO-3D-LB':3,'SSA-3D-LB':4}[alg_name])
            alg = AlgClass(hcfg)
            new_pos_h, mr_h, sr_h, e_h, dist_h = one_step_optimize(
                alg, cur_UE, gAP, pos[alg_name], max_iter=ITER_STEP,
                energy_model=energy_model, flight_speed=10.0)
            e_h += hover_E
            cumul[alg_name] += e_h
            jfi_h = eval_jfi_eff(v6_eval, cur_UE, gAP, new_pos_h, G, K, L)
            js_h  = joint_score_dyn(mr_h, jfi_h, e_h, energy_model, L, DT,
                                     W_MIN, W_JFI, W_EE, REF, FLOOR)
            pos[alg_name] = new_pos_h
            records[alg_name]['min_rate'].append(mr_h)
            records[alg_name]['jfi'].append(jfi_h)
            records[alg_name]['energy_step'].append(e_h)
            records[alg_name]['energy_cumul'].append(cumul[alg_name])
            records[alg_name]['dist'].append(dist_h)
            records[alg_name]['joint_score'].append(js_h)

        print(f"  Step{step}: DLB mr={mr_d:.1f} jfi={jfi_d:.3f} js={js_d:.3f} | "
              f"GA mr={records['GA-3D-LB']['min_rate'][-1]:.1f} | "
              f"PSO mr={records['PSO-3D-LB']['min_rate'][-1]:.1f} | "
              f"SSA mr={records['SSA-3D-LB']['min_rate'][-1]:.1f}")
        sys.stdout.flush()

    return records


# ─── 绘图 ─────────────────────────────────────────────────────────────
def plot_all(all_records, out_dir):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    algs = ALG_NAMES
    clr  = [COLORS[a] for a in algs]
    mrks = [MARKS[a] for a in algs]

    # ── 对所有种子取平均 ──
    def avg_series(key):
        """Average across seeds, return {alg: [T values]}"""
        lens = [len(all_records[s]['time']) for s in SEEDS]
        T = min(lens)
        return {a: np.array([[all_records[s][a][key][t] for t in range(T)]
                              for s in SEEDS]).mean(axis=0) for a in algs}

    time_axis = np.array(all_records[SEEDS[0]]['time'][:min(
        len(all_records[s]['time']) for s in SEEDS)])

    mr_avg  = avg_series('min_rate')
    jfi_avg = avg_series('jfi')
    ec_avg  = avg_series('energy_cumul')
    js_avg  = avg_series('joint_score')

    # ── Fig 1: 时序图 2×2 ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Dynamic Comparison  (K={K}, L={L}, G={G}, avg of {len(SEEDS)} seeds)',
                 fontsize=13, fontweight='bold')

    def _tsplot(ax, data, ylabel, title, scale=1.0):
        for a, c, mk in zip(algs, clr, mrks):
            ax.plot(time_axis, data[a]*scale, color=c, marker=mk,
                    markersize=5, linewidth=2, label=a)
        ax.set_xlabel('Time (s)'); ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    _tsplot(axes[0,0], mr_avg,  'Min User Rate (Mbps)', '(a) Min-Rate')
    _tsplot(axes[0,1], jfi_avg, 'JFI_eff',              '(b) Load Fairness')
    _tsplot(axes[1,0], js_avg,  'Joint Score',          '(c) 3-Obj Joint Score')
    _tsplot(axes[1,1], ec_avg,  'Cumulative Energy (kJ)','(d) Cumulative Energy', scale=1/1000)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(out_dir, f'fig_timeseries.{ext}'), dpi=200, bbox_inches='tight')
    plt.close()

    # ── Fig 2: 均值柱状图 ─────────────────────────────────────────
    def mean_std(key):
        vals = {a: np.array([np.mean(all_records[s][a][key][1:]) for s in SEEDS]) for a in algs}
        return {a: (vals[a].mean(), vals[a].std()) for a in algs}

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle('Average Performance per Step (excluding t=0)',
                 fontsize=13, fontweight='bold')
    for ax, key, ylabel, title in [
        (axes[0], 'min_rate',     'Min Rate (Mbps)',     '(a) Min-Rate'),
        (axes[1], 'jfi',          'JFI_eff',             '(b) JFI'),
        (axes[2], 'joint_score',  'Joint Score',         '(c) JointScore'),
        (axes[3], 'energy_step',  'Energy/step (kJ)',    '(d) Per-step Energy'),
    ]:
        ms = mean_std(key)
        scale = 1/1000 if key == 'energy_step' else 1.0
        means = [ms[a][0]*scale for a in algs]
        stds  = [ms[a][1]*scale for a in algs]
        bars  = ax.bar(algs, means, yerr=stds, color=clr, alpha=0.85,
                       capsize=5, edgecolor='white', linewidth=1.2)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontweight='bold')
        ax.tick_params(axis='x', rotation=20, labelsize=9)
        ax.grid(axis='y', alpha=0.3)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x()+bar.get_width()/2, m+s+0.001,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(out_dir, f'fig_bar.{ext}'), dpi=200, bbox_inches='tight')
    plt.close()

    # ── Fig 3: 能效比 ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for a, c, mk in zip(algs, clr, mrks):
        t_arr, ee_arr = [], []
        for s in SEEDS:
            rec = all_records[s][a]
            for i in range(1, len(rec['energy_cumul'])):
                ec = rec['energy_cumul'][i]
                if ec > 0:
                    t_arr.append(all_records[s]['time'][i])
                    ee_arr.append(rec['min_rate'][i] / (ec/1000 + 1e-6))
        # bin by time
        T_vals = sorted(set(t_arr))
        ee_by_t = {t: [] for t in T_vals}
        for t, ee in zip(t_arr, ee_arr):
            ee_by_t[t].append(ee)
        t_plot = T_vals
        ee_plot = [np.mean(ee_by_t[t]) for t in t_plot]
        ax.plot(t_plot, ee_plot, color=c, marker=mk, markersize=5, linewidth=2, label=a)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Energy Efficiency (Mbps / kJ)', fontsize=12)
    ax.set_title('Energy Efficiency over Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(out_dir, f'fig_energy_eff.{ext}'), dpi=200, bbox_inches='tight')
    plt.close()

    # ── Fig 4: 雷达图 ─────────────────────────────────────────────
    metrics_labels = ['Min-Rate', 'JFI_eff', 'JointScore', 'Energy\nEfficiency']
    N_m = len(metrics_labels)
    angles = np.linspace(0, 2*np.pi, N_m, endpoint=False).tolist(); angles += angles[:1]

    def avg1(key):
        return {a: np.mean([np.mean(all_records[s][a][key][1:]) for s in SEEDS]) for a in algs}

    mr_m  = avg1('min_rate');   jfi_m = avg1('jfi')
    js_m  = avg1('joint_score')
    ec_m  = avg1('energy_step')
    ee_m  = {a: mr_m[a] / (ec_m[a]/1000 + 1e-6) for a in algs}

    def norm(d):
        mx = max(d.values()); return {a: d[a]/(mx+1e-9) for a in algs}

    nr=norm(mr_m); nj=norm(jfi_m); njs=norm(js_m); nee=norm(ee_m)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for a, c in zip(algs, clr):
        vals = [nr[a], nj[a], njs[a], nee[a]]; vals += vals[:1]
        ax.plot(angles, vals, color=c, linewidth=2, label=a)
        ax.fill(angles, vals, color=c, alpha=0.12)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics_labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title('Multi-Dimensional Performance\n(Normalized, Dynamic Scenario)',
                 fontsize=12, pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize=10)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(out_dir, f'fig_radar.{ext}'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nAll dynamic plots saved to {out_dir}/")


# ─── 汇总打印 ──────────────────────────────────────────────────────────
def print_summary(all_records):
    print(f"\n{'='*75}")
    print(f"  动态场景对比总结 ({len(SEEDS)} 种子均值, Step 1 ~ {NUM_STEPS})")
    print(f"{'='*75}")
    print(f"{'算法':<18} {'MinRate':>10} {'JFI':>9} {'JointScore':>12} {'E_cum kJ':>10}")
    print("-"*75)
    for a in ALG_NAMES:
        mr = np.mean([np.mean(all_records[s][a]['min_rate'][1:]) for s in SEEDS])
        jf = np.mean([np.mean(all_records[s][a]['jfi'][1:]) for s in SEEDS])
        js = np.mean([np.mean(all_records[s][a]['joint_score'][1:]) for s in SEEDS])
        ec = np.mean([all_records[s][a]['energy_cumul'][-1]/1000 for s in SEEDS])
        print(f"{a:<18} {mr:>10.3f} {jf:>9.4f} {js:>12.4f} {ec:>10.1f}")
    print("="*75)


# ─── 主入口 ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    energy_model = UAVEnergyModel()
    v6_cfg = create_v6_config()
    v6_cfg.update({'num_UE':K,'num_UAV':L,'num_ground_AP':G,'tau_p':K,'num_serving_APs':3})
    v6_eval = BalancedVirtualForceOptimizerV6(v6_cfg)

    all_records = {}
    for s in SEEDS:
        print(f"\n{'─'*60}")
        print(f"  Dynamic Seed {s}")
        print(f"{'─'*60}")
        all_records[s] = run_one_seed(s, energy_model, v6_eval)

    print_summary(all_records)

    # 保存
    save = {}
    for s in SEEDS:
        save[str(s)] = {'time': all_records[s]['time']}
        for a in ALG_NAMES:
            save[str(s)][a] = all_records[s][a]
    with open(os.path.join(OUT_DIR, 'raw_results.json'), 'w') as f:
        json.dump(save, f, indent=2)

    plot_all(all_records, OUT_DIR)

    print(f"\nTotal time: {time.time()-t_start:.1f}s")
    print(f"Results saved to: {OUT_DIR}/")
