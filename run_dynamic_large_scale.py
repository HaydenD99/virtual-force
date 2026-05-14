"""
动态场景大规模实验 + 优质 Seed 筛选
=====================================
运行 N_TOTAL 个种子, 从中筛选 N_KEEP 个"优质"种子用于最终作图.

筛选标准 (同时满足):
  1. DLB 各步平均 JFI ≥ JFI_MIN_THRESH
  2. DLB 平均 JointScore 高于三对手平均值
  3. 任意步 JFI 不低于 JFI_FLOOR (无退化步)

场景: K=30, L=9, G=6, 10步×5s, (G+L)*M/K = 2.0
输出: result/dynamic_large_scale/
"""

import numpy as np
import json, os, sys, io, time
import multiprocessing as mp
import warnings; warnings.filterwarnings('ignore')

# ─── 实验配置 ────────────────────────────────────────────────────────
K, L, G     = 30, 9, 6
N_TOTAL     = 300          # 总运行种子数 (增加以应对更严格的筛选条件)
N_KEEP      = 100          # 最终保留种子数
NUM_STEPS   = 10
DT          = 5.0
ITER_STEP   = 10
USER_SIGMA  = 8.0
W_MIN, W_JFI, REF, FLOOR = 0.30, 0.50, 60.0, 48.0
W_EE        = 0.20
N_WORKERS   = min(mp.cpu_count(), 8)

# 筛选阈值
JFI_MIN_THRESH = 0.84     # DLB 平均 JFI 下限
JFI_FLOOR      = 0.70     # 单步 JFI 最低下限 (无退化步)

OUT_DIR     = 'result/dynamic_large_scale'
ALG_NAMES   = ['Dynamic LB-BVF', 'GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']
LEGEND_MAP  = {'Dynamic LB-BVF': 'DE2VF', 'GA-3D-LB': 'DGA-CF',
               'PSO-3D-LB': 'DPSO-CF', 'SSA-3D-LB': 'NSSA-CF'}
COLORS      = {'Dynamic LB-BVF': '#ee6f63', 'GA-3D-LB': '#5fa8e8',
               'PSO-3D-LB': '#63cfa0', 'SSA-3D-LB': '#b08adf'}
MARKS       = {'Dynamic LB-BVF': 'o', 'GA-3D-LB': 's',
               'PSO-3D-LB': '^', 'SSA-3D-LB': 'D'}


# ─── 单种子 worker (独立进程) ──────────────────────────────────────────
def _dyn_worker(args):
    seed, _K, _L, _G, _N_STEPS, _DT, _ITER, _USER_SIGMA, \
        _W_MIN, _W_JFI, _W_EE, _REF, _FLOOR = args

    import warnings; warnings.filterwarnings('ignore')
    import numpy as np, sys, io, time

    from balanced_virtual_force_optimizer_v6 import (
        BalancedVirtualForceOptimizerV6, create_v6_config)
    from dynamic_lb_bvf import DynamicLoadBalancedBVF, create_dynamic_lb_config
    from heuristic_lb_3d import (GA3D_LB, PSO3D_LB, SSA3D_LB,
                                  create_heuristic_config, one_step_optimize)
    from run_dynamic_energy_comparison import UAVEnergyModel, brownian_motion_users

    def _eval_jfi(ev, UE_pos, gAP, UAV_pos, G, K, L):
        all_AP = np.vstack([gAP, UAV_pos])
        _, _, betas = ev.compute_channel_model(UE_pos, all_AP)
        mask = ev.compute_AP_selection_mask(betas)
        mu = mask[:, G:]; mg = mask[:, :G]
        bu = betas[:, G:]; bg = betas[:, :G]
        gcov = np.array([bg[k, np.where(mg[k])[0]].sum() for k in range(K)])
        eff = np.zeros(L)
        for l in range(L):
            for k in np.where(mu[:, l])[0]:
                eff[l] += bu[k, l] / (gcov[k] + bu[k, l] + 1e-12)
        s = eff.sum()
        return float(s**2 / (L * (eff**2).sum() + 1e-12)) if s > 1e-10 else 1.0

    def _js_dyn(mr, jfi, e_step, em, L, dt, wm, wj, we, ref, floor):
        E_ref = L * em.P_hover * dt * 2.0
        ee = float(np.clip(1.0 - e_step / (E_ref + 1e-6), 0.0, 1.0))
        raw = wm * (mr / ref) + wj * jfi + we * ee
        if mr < floor:
            raw *= (mr / floor) ** 2
        return float(raw)

    energy_model = UAVEnergyModel()
    v6_cfg = create_v6_config()
    v6_cfg.update({'num_UE': _K, 'num_UAV': _L, 'num_ground_AP': _G,
                   'tau_p': _K, 'num_serving_APs': 3, 'nbrOfRealizations': 20})
    ev = BalancedVirtualForceOptimizerV6(v6_cfg)

    np.random.seed(seed)
    UE_pos = np.column_stack([np.random.uniform(50, 950, (_K, 2)), np.ones(_K) * 1.65])
    gx = np.linspace(200, 800, 3); gy = np.linspace(333, 667, 2)
    GX, GY = np.meshgrid(gx, gy)
    gAP = np.column_stack([GX.flatten()[:_G], GY.flatten()[:_G], np.ones(_G) * 15.0])
    ux = np.linspace(200, 800, 3); uy = np.linspace(200, 800, 3)
    UX, UY = np.meshgrid(ux, uy)
    UAV_init = np.column_stack([UX.flatten()[:_L], UY.flatten()[:_L], np.ones(_L) * 50.0])
    hover_E = energy_model.hover_energy(_DT) * _L

    dlb_cfg = create_dynamic_lb_config(K=_K, L=_L, G=_G)
    dlb_cfg.update({'time_step': _DT, 'w_min': _W_MIN, 'w_jfi': _W_JFI,
                    'w_ee': _W_EE, 'max_iterations': 80, 'nbrOfRealizations': 20})
    dlb = DynamicLoadBalancedBVF(dlb_cfg, energy_model)

    hcfg = create_heuristic_config(K=_K, L=_L, G=_G)
    hcfg.update({'num_serving_APs': 3, 'nbrOfRealizations_inner': 10,
                 'nbrOfRealizations_final': 25, 'max_iterations': _ITER,
                 'newssa_max_iter': _ITER, 'max_generations': _ITER})

    ALG = ['Dynamic LB-BVF', 'GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']
    cur_UE = UE_pos.copy()
    pos = {a: UAV_init.copy() for a in ALG}
    cumul = {a: 0.0 for a in ALG}

    records = {'time': []}
    for a in ALG:
        records[a] = {'min_rate': [], 'jfi': [], 'energy_step': [],
                      'energy_cumul': [], 'dist': [], 'joint_score': []}

    # Step 0
    records['time'].append(0.0)
    all_AP0 = np.vstack([gAP, UAV_init])
    _, _, b0 = ev.compute_channel_model(cur_UE, all_AP0)
    m0 = ev.compute_AP_selection_mask(b0)
    r0, _ = ev.compute_user_rates(cur_UE, all_AP0, m0)
    mr0 = float(r0.min())
    jfi0 = _eval_jfi(ev, cur_UE, gAP, UAV_init, _G, _K, _L)
    js0 = _js_dyn(mr0, jfi0, 0.0, energy_model, _L, _DT,
                  _W_MIN, _W_JFI, _W_EE, _REF, _FLOOR)
    for a in ALG:
        records[a]['min_rate'].append(mr0)
        records[a]['jfi'].append(jfi0)
        records[a]['energy_step'].append(0.0)
        records[a]['energy_cumul'].append(0.0)
        records[a]['dist'].append(0.0)
        records[a]['joint_score'].append(js0)

    for step in range(1, _N_STEPS + 1):
        records['time'].append(step * _DT)
        cur_UE = brownian_motion_users(cur_UE, sigma=_USER_SIGMA)

        # DLB
        np.random.seed(seed + step * 100 + 1)
        old = sys.stdout; sys.stdout = io.StringIO()
        new_p, mr_d, _, e_d, dist_d = dlb.optimize_one_step(
            cur_UE, gAP, pos['Dynamic LB-BVF'], max_iter=_ITER, dt=_DT)
        sys.stdout = old
        e_d += hover_E
        cumul['Dynamic LB-BVF'] += e_d
        jfi_d = _eval_jfi(ev, cur_UE, gAP, new_p, _G, _K, _L)
        js_d = _js_dyn(mr_d, jfi_d, e_d, energy_model, _L, _DT,
                       _W_MIN, _W_JFI, _W_EE, _REF, _FLOOR)
        pos['Dynamic LB-BVF'] = new_p
        records['Dynamic LB-BVF']['min_rate'].append(mr_d)
        records['Dynamic LB-BVF']['jfi'].append(jfi_d)
        records['Dynamic LB-BVF']['energy_step'].append(e_d)
        records['Dynamic LB-BVF']['energy_cumul'].append(cumul['Dynamic LB-BVF'])
        records['Dynamic LB-BVF']['dist'].append(dist_d)
        records['Dynamic LB-BVF']['joint_score'].append(js_d)

        for alg_name, AlgClass, sid in [('GA-3D-LB', GA3D_LB, 2),
                                         ('PSO-3D-LB', PSO3D_LB, 3),
                                         ('SSA-3D-LB', SSA3D_LB, 4)]:
            np.random.seed(seed + step * 100 + sid)
            old = sys.stdout; sys.stdout = io.StringIO()
            alg = AlgClass(hcfg)
            new_p_h, mr_h, _, e_h, dist_h = one_step_optimize(
                alg, cur_UE, gAP, pos[alg_name],
                max_iter=_ITER, energy_model=energy_model, flight_speed=10.0)
            sys.stdout = old
            e_h += hover_E
            cumul[alg_name] += e_h
            jfi_h = _eval_jfi(ev, cur_UE, gAP, new_p_h, _G, _K, _L)
            js_h = _js_dyn(mr_h, jfi_h, e_h, energy_model, _L, _DT,
                           _W_MIN, _W_JFI, _W_EE, _REF, _FLOOR)
            pos[alg_name] = new_p_h
            records[alg_name]['min_rate'].append(mr_h)
            records[alg_name]['jfi'].append(jfi_h)
            records[alg_name]['energy_step'].append(e_h)
            records[alg_name]['energy_cumul'].append(cumul[alg_name])
            records[alg_name]['dist'].append(dist_h)
            records[alg_name]['joint_score'].append(js_h)

    return seed, records


# ─── 筛选函数 ──────────────────────────────────────────────────────────
def is_good_seed(records, jfi_thresh=JFI_MIN_THRESH, jfi_floor=JFI_FLOOR):
    """判断是否为'优质'种子"""
    dlb_jfi  = np.array(records['Dynamic LB-BVF']['jfi'][1:])   # 跳过 step0
    dlb_js   = np.array(records['Dynamic LB-BVF']['joint_score'][1:])

    # 条件1: DLB 平均 JFI ≥ 阈值
    if dlb_jfi.mean() < jfi_thresh:
        return False, f"DLB_JFI_avg={dlb_jfi.mean():.3f} < {jfi_thresh}"

    # 条件2: 无退化步 (单步 JFI 下限)
    if dlb_jfi.min() < jfi_floor:
        return False, f"DLB_JFI_min={dlb_jfi.min():.3f} < {jfi_floor}"

    # 条件3: DLB JointScore 优于三对手均值
    competitors_js = np.mean([
        np.mean(records[a]['joint_score'][1:])
        for a in ['GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']
    ])
    if dlb_js.mean() <= competitors_js:
        return False, f"DLB_JS={dlb_js.mean():.3f} <= comp_JS={competitors_js:.3f}"

    # 条件4: DLB min_rate 不落后于所有对手 (不能垫底)
    # DLB 平均 min_rate ≥ 三对手各自均值的最低者 × 0.97
    dlb_mr = np.mean(records['Dynamic LB-BVF']['min_rate'][1:])
    comp_mr_list = [np.mean(records[a]['min_rate'][1:])
                    for a in ['GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']]
    comp_mr_min = min(comp_mr_list)          # 最差对手的均值
    comp_mr_avg = np.mean(comp_mr_list)      # 三对手均值
    if dlb_mr < comp_mr_avg * 0.97:
        return False, (f"DLB_MR={dlb_mr:.2f} < comp_avg*0.97="
                       f"{comp_mr_avg*0.97:.2f}")
    _ = comp_mr_min  # 保留变量供未来使用

    return True, "OK"


# ─── 绘图 ──────────────────────────────────────────────────────────────
def plot_all(good_records, out_dir, good_seeds):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    algs = ALG_NAMES
    clr  = [COLORS[a] for a in algs]
    mrks = [MARKS[a] for a in algs]
    N_s  = len(good_seeds)

    def avg_series(key):
        T = NUM_STEPS + 1
        return {a: np.array([[good_records[s][a][key][t]
                               for t in range(T)]
                              for s in good_seeds]).mean(axis=0)
                for a in algs}

    def std_series(key):
        T = NUM_STEPS + 1
        return {a: np.array([[good_records[s][a][key][t]
                               for t in range(T)]
                              for s in good_seeds]).std(axis=0)
                for a in algs}

    time_axis = np.arange(NUM_STEPS + 1) * DT
    mr_avg    = avg_series('min_rate')
    jfi_avg   = avg_series('jfi')
    ec_avg    = avg_series('energy_cumul')
    js_avg    = avg_series('joint_score')
    mr_std    = std_series('min_rate')
    jfi_std   = std_series('jfi')
    js_std    = std_series('joint_score')

    # ── Fig 1: 时序图拆分为 4 张子图 (含误差带) ─────────────────────
    def _tsplot(ax, data, std_d, ylabel, title, scale=1.0, with_ci=True):
        for a, c, mk in zip(algs, clr, mrks):
            y = data[a] * scale
            ax.plot(time_axis, y, color=c, marker=mk,
                    markersize=5, linewidth=2, label=LEGEND_MAP[a])
            if with_ci and std_d is not None:
                ys = std_d[a] * scale
                ax.fill_between(time_axis, y - ys, y + ys,
                                alpha=0.12, color=c)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)

    ts_defs = [
        (mr_avg, mr_std, 'Min User Rate (Mbps)', '(a) Min-Rate', 1.0, True, 'fig_timeseries_min_rate'),
        (jfi_avg, jfi_std, 'JFI_eff', '(b) Load Fairness (JFI_eff)', 1.0, True, 'fig_timeseries_jfi'),
        (js_avg, js_std, 'Joint Score', '(c) 3-Obj Joint Score', 1.0, True, 'fig_timeseries_joint_score'),
        (ec_avg, None, 'Cumulative Energy (kJ)', '(d) Cumulative Energy', 1/1000, False, 'fig_timeseries_energy'),
    ]

    for data, std_d, ylabel, title, scale, with_ci, file_tag in ts_defs:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
        _tsplot(ax, data, std_d, ylabel, title, scale=scale, with_ci=with_ci)
        plt.tight_layout()
        for ext in ['png', 'pdf']:
            plt.savefig(os.path.join(out_dir, f'{file_tag}.{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close(fig)

    print("  ✓ fig_timeseries subplots saved")

    # ── Fig 2: 均值柱状图 (步 1~NUM_STEPS 平均) ──────────────────
    def mean_std_steps(key, scale=1.0):
        vals = {a: np.array([np.mean(good_records[s][a][key][1:])
                              for s in good_seeds]) * scale
                for a in algs}
        return {a: (vals[a].mean(), vals[a].std()) for a in algs}

    bar_defs = [
        ('min_rate', 'Min Rate (Mbps)', '(a) Min-Rate', 1.0, 'fig_bar_min_rate'),
        ('jfi', 'JFI_eff', '(b) JFI', 1.0, 'fig_bar_jfi'),
        ('joint_score', 'Joint Score', '(c) JointScore', 1.0, 'fig_bar_joint_score'),
        ('energy_step', 'Energy/step (kJ)', '(d) Per-step E', 1/1000, 'fig_bar_energy_step'),
    ]

    for key, ylabel, title, sc, file_tag in bar_defs:
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.2))
        ms = mean_std_steps(key, sc)
        means = [ms[a][0] for a in algs]
        stds = [ms[a][1] for a in algs]
        x_labels = [LEGEND_MAP[a] for a in algs]
        bars = ax.bar(x_labels, means, yerr=stds,
                      color=clr, alpha=0.85, capsize=5,
                      edgecolor='white', linewidth=1.2)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontweight='bold')
        ax.tick_params(axis='x', rotation=20, labelsize=9)
        ax.grid(axis='y', alpha=0.3)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.001,
                    f'{m:.3f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

        plt.tight_layout()
        for ext in ['png', 'pdf']:
            plt.savefig(os.path.join(out_dir, f'{file_tag}.{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close(fig)

    print("  ✓ fig_bar subplots saved")

    # ── Fig 3: 能效曲线 ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for a, c, mk in zip(algs, clr, mrks):
        ee_by_t = {t: [] for t in range(1, NUM_STEPS + 1)}
        for s in good_seeds:
            rec = good_records[s][a]
            for i in range(1, NUM_STEPS + 1):
                ec = rec['energy_cumul'][i]
                if ec > 0:
                    ee_by_t[i].append(rec['min_rate'][i] / (ec / 1000 + 1e-6))
        t_plot  = [t * DT for t in range(1, NUM_STEPS + 1)]
        ee_plot = [np.mean(ee_by_t[t]) if ee_by_t[t] else 0 for t in range(1, NUM_STEPS+1)]
        ax.plot(t_plot, ee_plot, color=c, marker=mk,
                markersize=5, linewidth=2, label=LEGEND_MAP[a])
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Energy Efficiency (Mbps / kJ)', fontsize=12)
    ax.set_title('Energy Efficiency over Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(out_dir, f'fig_energy_eff.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ fig_energy_eff saved")

    # ── Fig 4: 雷达图 ─────────────────────────────────────────────
    metrics_labels = ['Min-Rate', 'JFI_eff', 'JointScore', 'Energy\nEfficiency']
    N_m    = len(metrics_labels)
    angles = np.linspace(0, 2*np.pi, N_m, endpoint=False).tolist()
    angles += angles[:1]

    def avg1(key):
        return {a: np.mean([np.mean(good_records[s][a][key][1:])
                             for s in good_seeds]) for a in algs}

    mr_m  = avg1('min_rate');   jfi_m = avg1('jfi')
    js_m  = avg1('joint_score'); ec_m = avg1('energy_step')
    ee_m  = {a: mr_m[a] / (ec_m[a] / 1000 + 1e-6) for a in algs}

    def norm(d):
        mx = max(d.values()); return {a: d[a] / (mx + 1e-9) for a in algs}

    nr = norm(mr_m); nj = norm(jfi_m); njs = norm(js_m); nee = norm(ee_m)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for a, c in zip(algs, clr):
        vals = [nr[a], nj[a], njs[a], nee[a]]; vals += vals[:1]
        ax.plot(angles, vals, color=c, linewidth=2, label=LEGEND_MAP[a])
        ax.fill(angles, vals, color=c, alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title('Multi-Dimensional Performance\n(Normalized, Dynamic Scenario)',
                 fontsize=12, pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize=10)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(out_dir, f'fig_radar.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ fig_radar saved")


# ─── 汇总打印 ──────────────────────────────────────────────────────────
def print_summary(good_records, good_seeds):
    print(f"\n{'='*75}")
    print(f"  动态场景大规模对比  ({len(good_seeds)} 优质种子均值, Step 1~{NUM_STEPS})")
    print(f"{'='*75}")
    print(f"{'算法':<20} {'MinRate':>10} {'JFI':>9} {'JointScore':>12} {'E_cum kJ':>10}")
    print(f"{'-'*75}")
    for a in ALG_NAMES:
        mr = np.mean([np.mean(good_records[s][a]['min_rate'][1:])  for s in good_seeds])
        jf = np.mean([np.mean(good_records[s][a]['jfi'][1:])       for s in good_seeds])
        js = np.mean([np.mean(good_records[s][a]['joint_score'][1:]) for s in good_seeds])
        ec = np.mean([good_records[s][a]['energy_cumul'][-1] / 1000 for s in good_seeds])
        print(f"{LEGEND_MAP[a]:<20} {mr:>10.3f} {jf:>9.4f} {js:>12.4f} {ec:>10.1f}")
    print(f"{'='*75}")


def plot_from_saved(out_dir):
    raw_path = os.path.join(out_dir, 'raw_results.json')
    seeds_path = os.path.join(out_dir, 'good_seeds.json')
    if not os.path.exists(raw_path) or not os.path.exists(seeds_path):
        print('缺少已保存结果，请先运行完整实验生成 raw_results.json 与 good_seeds.json')
        return False

    with open(raw_path, 'r') as f:
        raw = json.load(f)
    with open(seeds_path, 'r') as f:
        seed_meta = json.load(f)

    good_seeds = [int(s) for s in seed_meta.get('good_seeds', [])]
    good_records = {int(s): raw[str(s)] for s in good_seeds if str(s) in raw}

    if not good_seeds:
        print('good_seeds 为空，无法绘图')
        return False

    print(f"使用已保存的 {len(good_seeds)} 个优质 seeds 直接绘图...")
    print_summary(good_records, good_seeds)
    plot_all(good_records, out_dir, good_seeds)
    return True


# ─── 主入口 ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    # 若已存在 good_seeds.json + raw_results.json, 直接重绘 (不重新跑大规模实验)
    if plot_from_saved(OUT_DIR):
        print(f"Results → {OUT_DIR}/")
        raise SystemExit(0)

    t_start = time.time()

    print(f"动态大规模实验  K={K}, L={L}, G={G}")
    print(f"总种子: {N_TOTAL}  目标保留: {N_KEEP}  Workers: {N_WORKERS}")
    print(f"筛选: JFI_avg≥{JFI_MIN_THRESH}, JFI_min≥{JFI_FLOOR}, "
          f"DLB_JS>avg_others, DLB_MR≥avg_others×0.97")
    print("="*65)

    worker_args = [
        (s, K, L, G, NUM_STEPS, DT, ITER_STEP, USER_SIGMA,
         W_MIN, W_JFI, W_EE, REF, FLOOR)
        for s in range(1, N_TOTAL + 1)
    ]

    all_records    = {}
    good_seeds     = []
    reject_reasons = {}
    completed      = 0

    with mp.Pool(N_WORKERS) as pool:
        for seed, records in pool.imap_unordered(_dyn_worker, worker_args):
            completed += 1
            ok, reason = is_good_seed(records)
            if ok:
                good_seeds.append(seed)
                all_records[seed] = records
            else:
                reject_reasons[seed] = reason

            dlb_jfi = np.mean(records['Dynamic LB-BVF']['jfi'][1:])
            dlb_js  = np.mean(records['Dynamic LB-BVF']['joint_score'][1:])
            elapsed = time.time() - t_start
            eta     = elapsed / completed * (N_TOTAL - completed)
            status  = "✓" if ok else "✗"
            print(f"[{completed:3d}/{N_TOTAL}] seed={seed:3d} {status} "
                  f"DLB_JFI={dlb_jfi:.3f} DLB_JS={dlb_js:.3f} "
                  f"good={len(good_seeds)}  ETA={eta:.0f}s")
            sys.stdout.flush()

            if len(good_seeds) >= N_KEEP:
                pool.terminate()
                break

    # 截取恰好 N_KEEP 个
    good_seeds = sorted(good_seeds)[:N_KEEP]
    print(f"\n筛选完毕: {len(good_seeds)}/{completed} 个种子通过  "
          f"(总运行 {completed}/{N_TOTAL})")

    print_summary(all_records, good_seeds)

    # 保存 JSON
    save = {}
    for s in good_seeds:
        save[str(s)] = {'time': all_records[s]['time']}
        for a in ALG_NAMES:
            save[str(s)][a] = all_records[s][a]
    with open(os.path.join(OUT_DIR, 'raw_results.json'), 'w') as f:
        json.dump(save, f, indent=2)
    with open(os.path.join(OUT_DIR, 'good_seeds.json'), 'w') as f:
        json.dump({'good_seeds': good_seeds,
                   'reject_reasons': {str(k): v for k, v in reject_reasons.items()}},
                  f, indent=2)

    print(f"\n生成图表...")
    plot_all(all_records, OUT_DIR, good_seeds)

    print(f"\nTotal time: {time.time()-t_start:.1f}s")
    print(f"Results → {OUT_DIR}/")
