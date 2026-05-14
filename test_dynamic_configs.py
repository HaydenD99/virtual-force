"""
动态场景配置快速筛选实验
========================
在 Cell-Free Massive MIMO 中，需满足  (G+L)*M >> K
当前有问题的配置: K=60, G=4, L=9 → (13)*4=52 < 60，不满足条件

候选配置（M=4，num_serving_APs=3）:
  A: K=20, L=9, G=4  → 52  antennas / 20 users  ratio=2.60  ✓✓
  B: K=30, L=9, G=4  → 52  antennas / 30 users  ratio=1.73  ✓
  C: K=30, L=9, G=6  → 60  antennas / 30 users  ratio=2.00  ✓✓
  D: K=40, L=9, G=6  → 60  antennas / 40 users  ratio=1.50  ✓ (borderline)

快速运行: 2 seeds × 5 time-steps × 10 iter/step
输出: result/config_test/
"""

import numpy as np
import os, sys, io, time
import warnings; warnings.filterwarnings('ignore')

from balanced_virtual_force_optimizer_v6 import BalancedVirtualForceOptimizerV6, create_v6_config
from dynamic_lb_bvf import DynamicLoadBalancedBVF, create_dynamic_lb_config
from heuristic_lb_3d import GA3D_LB, PSO3D_LB, SSA3D_LB, create_heuristic_config, one_step_optimize
from run_dynamic_energy_comparison import UAVEnergyModel, brownian_motion_users

OUT_DIR = 'result/config_test'
os.makedirs(OUT_DIR, exist_ok=True)

# ── 候选配置 ─────────────────────────────────────────────────────────────
CONFIGS = {
    'A: K=20,L=9,G=4': dict(K=20, L=9, G=4),
    'B: K=30,L=9,G=4': dict(K=30, L=9, G=4),
    'C: K=30,L=9,G=6': dict(K=30, L=9, G=6),
    'D: K=40,L=9,G=6': dict(K=40, L=9, G=6),
}

TEST_SEEDS  = [42, 99]
NUM_STEPS   = 5
ITER_STEP   = 10
DT          = 5.0
USER_SIGMA  = 8.0
W_MIN, W_JFI, W_EE = 0.25, 0.50, 0.25
REF, FLOOR  = 60.0, 48.0
M           = 4


def gap_layout(G, sq=1000):
    """等间距地面 AP 布局"""
    if G == 4:
        xs = [250, 750]; ys = [250, 750]
    elif G == 6:
        xs = [200, 500, 800]; ys = [333, 667]
    elif G == 9:
        sp = sq / 4
        return np.array([[(i+1)*sp, (j+1)*sp, 15.0]
                         for i in range(3) for j in range(3)])
    else:
        n = int(np.ceil(np.sqrt(G)))
        sp = sq / (n + 1)
        pts = [[(i+1)*sp, (j+1)*sp] for i in range(n) for j in range(n)][:G]
        return np.array([[p[0], p[1], 15.0] for p in pts])
    return np.array([[x, y, 15.0] for x in xs for y in ys])[:G]


def uav_init_grid(L, sq=1000):
    n = int(np.ceil(np.sqrt(L)))
    sp = sq / (n + 1)
    pts = [[(i+1)*sp, (j+1)*sp] for i in range(n) for j in range(n)][:L]
    return np.array([[p[0], p[1], 50.0] for p in pts])


def eval_jfi(ev, UE_pos, gAP, UAV_pos, G, K, L):
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


def run_config(cfg_name, cfg, energy_model):
    K, L, G = cfg['K'], cfg['L'], cfg['G']
    ratio = (G + L) * M / K
    print(f"\n{'─'*65}")
    print(f"  Config {cfg_name}  |  K={K} L={L} G={G}  "
          f"|  AP_ratio=(({G}+{L})×{M})/{K}={ratio:.2f}")
    print(f"{'─'*65}")

    v6_cfg = create_v6_config()
    v6_cfg.update({'num_UE': K, 'num_UAV': L, 'num_ground_AP': G,
                   'tau_p': K, 'num_serving_APs': 3,
                   'nbrOfRealizations': 20})
    ev = BalancedVirtualForceOptimizerV6(v6_cfg)

    hcfg = create_heuristic_config(K=K, L=L, G=G)
    hcfg.update({'num_serving_APs': 3, 'nbrOfRealizations_inner': 10,
                 'nbrOfRealizations_final': 20, 'max_iterations': ITER_STEP,
                 'newssa_max_iter': ITER_STEP, 'max_generations': ITER_STEP})

    dlb_cfg = create_dynamic_lb_config(K=K, L=L, G=G)
    dlb_cfg.update({'time_step': DT, 'w_min': W_MIN, 'w_jfi': W_JFI,
                    'w_ee': W_EE, 'max_iterations': 80,
                    'nbrOfRealizations': 20})

    seed_results = {a: [] for a in ['DLB', 'GA', 'PSO', 'SSA']}
    seed_times   = {a: [] for a in ['DLB', 'GA', 'PSO', 'SSA']}

    for seed in TEST_SEEDS:
        np.random.seed(seed)
        UE_pos = np.column_stack([
            np.random.uniform(50, 950, (K, 2)), np.ones(K) * 1.65])
        gAP = gap_layout(G)
        UAV_init = uav_init_grid(L)
        hover_E = energy_model.hover_energy(DT) * L

        dlb = DynamicLoadBalancedBVF(dlb_cfg, energy_model)
        pos = {a: UAV_init.copy() for a in ['DLB', 'GA', 'PSO', 'SSA']}
        cur_UE = UE_pos.copy()

        step_mr = {a: [] for a in ['DLB', 'GA', 'PSO', 'SSA']}

        for step in range(1, NUM_STEPS + 1):
            cur_UE = brownian_motion_users(cur_UE, sigma=USER_SIGMA)

            # DLB
            np.random.seed(seed + step * 100 + 1)
            t0 = time.time()
            old = sys.stdout; sys.stdout = io.StringIO()
            new_p, mr_d, _, _, _ = dlb.optimize_one_step(
                cur_UE, gAP, pos['DLB'], max_iter=ITER_STEP, dt=DT)
            sys.stdout = old
            seed_times['DLB'].append(time.time() - t0)
            pos['DLB'] = new_p
            step_mr['DLB'].append(mr_d)

            # GA / PSO / SSA
            for name, Cls, sid in [('GA', GA3D_LB, 2),
                                    ('PSO', PSO3D_LB, 3),
                                    ('SSA', SSA3D_LB, 4)]:
                np.random.seed(seed + step * 100 + sid)
                t0 = time.time()
                old = sys.stdout; sys.stdout = io.StringIO()
                alg = Cls(hcfg)
                new_p_h, mr_h, _, _, _ = one_step_optimize(
                    alg, cur_UE, gAP, pos[name],
                    max_iter=ITER_STEP, energy_model=energy_model, flight_speed=10.0)
                sys.stdout = old
                seed_times[name].append(time.time() - t0)
                pos[name] = new_p_h
                step_mr[name].append(mr_h)

        for a in ['DLB', 'GA', 'PSO', 'SSA']:
            seed_results[a].append(np.mean(step_mr[a]))
        print(f"  seed={seed} | DLB:{np.mean(step_mr['DLB']):.2f}  "
              f"GA:{np.mean(step_mr['GA']):.2f}  "
              f"PSO:{np.mean(step_mr['PSO']):.2f}  "
              f"SSA:{np.mean(step_mr['SSA']):.2f} Mbps")

    # Summary for this config
    summary = {}
    for a in ['DLB', 'GA', 'PSO', 'SSA']:
        summary[a] = {
            'mean_mr': float(np.mean(seed_results[a])),
            'std_mr':  float(np.std(seed_results[a])),
            'avg_step_time': float(np.mean(seed_times[a])),
        }

    print(f"\n  Summary  (avg min_rate over {NUM_STEPS} steps × {len(TEST_SEEDS)} seeds):")
    print(f"  {'Algo':<6}  {'mean_mr':>8}  {'std':>6}  {'time/step':>10}")
    for a in ['DLB', 'GA', 'PSO', 'SSA']:
        s = summary[a]
        print(f"  {a:<6}  {s['mean_mr']:>8.3f}  {s['std_mr']:>6.3f}  "
              f"{s['avg_step_time']:>10.2f}s")
    print(f"  Cell-Free ratio (G+L)*M/K = {ratio:.2f}  "
          f"{'GOOD ✓✓' if ratio>=2.0 else ('OK ✓' if ratio>=1.5 else 'BAD ✗')}")

    return {'config': cfg_name, 'K': K, 'L': L, 'G': G, 'ratio': ratio,
            'summary': summary}


# ─── 主入口 ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    energy_model = UAVEnergyModel()
    all_cfg_results = []

    t_total = time.time()
    for cfg_name, cfg in CONFIGS.items():
        r = run_config(cfg_name, cfg, energy_model)
        all_cfg_results.append(r)

    # ── 综合对比表 ──────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  配置对比汇总  (DLB 主导 min-rate, 各算法均值)")
    print(f"{'='*75}")
    print(f"  {'Config':<22} {'K':>4} {'L':>4} {'G':>4} "
          f"{'ratio':>7} {'DLB_mr':>8} {'GA_mr':>8} {'PSO_mr':>8} {'SSA_mr':>8}")
    print(f"  {'-'*73}")
    for r in all_cfg_results:
        s = r['summary']
        print(f"  {r['config']:<22} {r['K']:>4} {r['L']:>4} {r['G']:>4} "
              f"{r['ratio']:>7.2f}"
              f" {s['DLB']['mean_mr']:>8.3f}"
              f" {s['GA']['mean_mr']:>8.3f}"
              f" {s['PSO']['mean_mr']:>8.3f}"
              f" {s['SSA']['mean_mr']:>8.3f}")
    print(f"{'='*75}")
    print(f"\n建议选择: ratio >= 2.0 且 DLB min_rate 表现较好的配置")
    print(f"总耗时: {time.time()-t_total:.1f}s")
