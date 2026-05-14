"""
DE2VF-MR+ 参数扫描脚本（仅对比原版 DE2VF）
==========================================
目标：在动态场景中快速筛选出对 min-rate 最有利的参数组合。

对比对象：
- baseline: DE2VF (DynamicLoadBalancedBVF)
- candidate: DE2VF-MR+ (DynamicLoadBalancedBVFMinRatePlus)

输出：result/dynamic_large_scale_plus/
- tune_mrplus_results.json
- tune_mrplus_leaderboard.txt
"""

import io
import json
import os
import sys
import itertools
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from balanced_virtual_force_optimizer_v6 import BalancedVirtualForceOptimizerV6, create_v6_config
from dynamic_lb_bvf import DynamicLoadBalancedBVF, create_dynamic_lb_config
from dynamic_lb_bvf_minrate_plus import DynamicLoadBalancedBVFMinRatePlus, create_dynamic_lb_minrate_plus_config
from run_dynamic_energy_comparison import UAVEnergyModel, brownian_motion_users

# 场景配置（与主实验一致）
K, L, G = 30, 9, 6
NUM_STEPS = 8        # 调参先用稍短时序提速
DT = 5.0
ITER_STEP = 10
USER_SIGMA = 8.0
W_MIN, W_JFI, W_EE, REF, FLOOR = 0.30, 0.50, 0.20, 60.0, 48.0

SEEDS_JSON = 'result/dynamic_large_scale/good_seeds.json'
OUT_DIR = 'result/dynamic_large_scale_plus'
N_USE_SEEDS = 8      # 快速调参默认 8 个 seeds


def eval_jfi(ev, UE_pos, gAP, UAV_pos):
    all_AP = np.vstack([gAP, UAV_pos])
    _, _, betas = ev.compute_channel_model(UE_pos, all_AP)
    mask = ev.compute_AP_selection_mask(betas)
    mu = mask[:, G:]
    mg = mask[:, :G]
    bu = betas[:, G:]
    bg = betas[:, :G]
    gcov = np.array([bg[k, np.where(mg[k])[0]].sum() for k in range(K)])
    eff = np.zeros(L)
    for l in range(L):
        for k in np.where(mu[:, l])[0]:
            eff[l] += bu[k, l] / (gcov[k] + bu[k, l] + 1e-12)
    s = eff.sum()
    return float(s**2 / (L * (eff**2).sum() + 1e-12)) if s > 1e-10 else 1.0


def joint_score_dyn(min_rate, jfi, e_step, energy_model):
    e_ref = L * energy_model.P_hover * DT * 2.0
    ee = float(np.clip(1.0 - e_step / (e_ref + 1e-6), 0.0, 1.0))
    raw = W_MIN * (min_rate / REF) + W_JFI * jfi + W_EE * ee
    if min_rate < FLOOR:
        raw *= (min_rate / FLOOR) ** 2
    return float(raw)


def run_alg_on_seed(seed, alg_name, alg_obj, ev, energy_model):
    np.random.seed(seed)
    UE_pos = np.column_stack([np.random.uniform(50, 950, (K, 2)), np.ones(K) * 1.65])

    gx = np.linspace(200, 800, 3)
    gy = np.linspace(333, 667, 2)
    GX, GY = np.meshgrid(gx, gy)
    gAP = np.column_stack([GX.flatten()[:G], GY.flatten()[:G], np.ones(G) * 15.0])

    ux = np.linspace(200, 800, 3)
    uy = np.linspace(200, 800, 3)
    UX, UY = np.meshgrid(ux, uy)
    UAV_init = np.column_stack([UX.flatten()[:L], UY.flatten()[:L], np.ones(L) * 50.0])

    hover_e = energy_model.hover_energy(DT) * L
    cur_UE = UE_pos.copy()
    pos = UAV_init.copy()
    e_cum = 0.0

    rec = {'min_rate': [], 'jfi': [], 'energy_cumul': [], 'joint_score': []}

    for step in range(NUM_STEPS + 1):
        if step == 0:
            all_ap0 = np.vstack([gAP, UAV_init])
            _, _, b0 = ev.compute_channel_model(cur_UE, all_ap0)
            m0 = ev.compute_AP_selection_mask(b0)
            r0, _ = ev.compute_user_rates(cur_UE, all_ap0, m0)
            mr0 = float(r0.min())
            jfi0 = eval_jfi(ev, cur_UE, gAP, UAV_init)
            js0 = joint_score_dyn(mr0, jfi0, 0.0, energy_model)
            rec['min_rate'].append(mr0)
            rec['jfi'].append(jfi0)
            rec['energy_cumul'].append(0.0)
            rec['joint_score'].append(js0)
            continue

        cur_UE = brownian_motion_users(cur_UE, sigma=USER_SIGMA)

        old = sys.stdout
        sys.stdout = io.StringIO()
        pos, mr, _, e, _ = alg_obj.optimize_one_step(cur_UE, gAP, pos, max_iter=ITER_STEP, dt=DT)
        sys.stdout = old

        e += hover_e
        e_cum += e
        jfi = eval_jfi(ev, cur_UE, gAP, pos)
        js = joint_score_dyn(mr, jfi, e, energy_model)

        rec['min_rate'].append(float(mr))
        rec['jfi'].append(float(jfi))
        rec['energy_cumul'].append(float(e_cum))
        rec['joint_score'].append(float(js))

    return rec


def summarize(records_by_seed):
    # 统计 step 1..T 的均值
    mr = np.array([np.mean(records_by_seed[s]['min_rate'][1:]) for s in records_by_seed])
    jf = np.array([np.mean(records_by_seed[s]['jfi'][1:]) for s in records_by_seed])
    js = np.array([np.mean(records_by_seed[s]['joint_score'][1:]) for s in records_by_seed])
    ec = np.array([records_by_seed[s]['energy_cumul'][-1] / 1000 for s in records_by_seed])
    return {
        'min_rate_mean': float(mr.mean()),
        'min_rate_std': float(mr.std()),
        'jfi_mean': float(jf.mean()),
        'joint_score_mean': float(js.mean()),
        'energy_cum_kj_mean': float(ec.mean()),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(SEEDS_JSON):
        print(f'缺少 {SEEDS_JSON}')
        return
    with open(SEEDS_JSON, 'r') as f:
        seed_meta = json.load(f)
    seeds = [int(s) for s in seed_meta.get('good_seeds', [])][:N_USE_SEEDS]
    if not seeds:
        print('good_seeds 为空，无法调参')
        return

    # baseline config
    base_cfg = create_dynamic_lb_config(K=K, L=L, G=G)
    base_cfg.update({'time_step': DT, 'max_iterations': 80, 'nbrOfRealizations': 20})

    # 待扫描参数网格（先小网格快速筛）
    grid = {
        'mr_gain': [0.22, 0.30, 0.38],
        'worst_q': [0.20, 0.30, 0.40],
        'k_mr_force': [1.0e4, 1.6e4, 2.2e4],
        'refine_step_sizes': [[8.0, 5.0, 3.0], [10.0, 7.0, 4.0, 2.5]],
    }

    combos = list(itertools.product(
        grid['mr_gain'], grid['worst_q'], grid['k_mr_force'], grid['refine_step_sizes']
    ))

    # evaluator (固定)
    v6_cfg = create_v6_config()
    v6_cfg.update({'num_UE': K, 'num_UAV': L, 'num_ground_AP': G,
                   'tau_p': K, 'num_serving_APs': 3, 'nbrOfRealizations': 20})

    print(f'Use seeds: {seeds}')
    print(f'Total param combinations: {len(combos)}')

    # 先跑 baseline 一次（用于做增益比较）
    baseline_records = {}
    for s in seeds:
        energy_model = UAVEnergyModel()
        ev = BalancedVirtualForceOptimizerV6(v6_cfg)
        baseline_alg = DynamicLoadBalancedBVF(base_cfg, energy_model)
        baseline_records[s] = run_alg_on_seed(s, 'DE2VF', baseline_alg, ev, energy_model)
    baseline_summary = summarize(baseline_records)

    results = {
        'settings': {
            'K': K, 'L': L, 'G': G,
            'num_steps': NUM_STEPS,
            'iter_step': ITER_STEP,
            'n_use_seeds': len(seeds),
            'seeds': seeds,
        },
        'baseline_DE2VF': baseline_summary,
        'candidates': []
    }

    for idx, (mr_gain, worst_q, k_force, refine_steps) in enumerate(combos, 1):
        cand_records = {}
        for s in seeds:
            energy_model = UAVEnergyModel()
            ev = BalancedVirtualForceOptimizerV6(v6_cfg)
            cfg = create_dynamic_lb_minrate_plus_config(K=K, L=L, G=G)
            cfg.update({
                'time_step': DT,
                'max_iterations': 80,
                'nbrOfRealizations': 20,
                'mr_gain': mr_gain,
                'worst_q': worst_q,
                'k_mr_force': k_force,
                'refine_step_sizes': refine_steps,
            })
            alg = DynamicLoadBalancedBVFMinRatePlus(cfg, energy_model)
            cand_records[s] = run_alg_on_seed(s, 'DE2VF-MR+', alg, ev, energy_model)

        sm = summarize(cand_records)
        sm['delta_min_rate_vs_baseline'] = sm['min_rate_mean'] - baseline_summary['min_rate_mean']
        sm['params'] = {
            'mr_gain': mr_gain,
            'worst_q': worst_q,
            'k_mr_force': k_force,
            'refine_step_sizes': refine_steps,
        }
        results['candidates'].append(sm)

        print(f"[{idx:2d}/{len(combos)}] mr={mr_gain:.2f} q={worst_q:.2f} k={k_force:.0f} -> "
              f"min_rate={sm['min_rate_mean']:.3f} (Δ {sm['delta_min_rate_vs_baseline']:+.3f})")

    # 排序: 先看 min_rate，再看 joint_score，再看 jfi
    ranked = sorted(
        results['candidates'],
        key=lambda x: (x['min_rate_mean'], x['joint_score_mean'], x['jfi_mean']),
        reverse=True
    )
    results['ranked'] = ranked

    out_json = os.path.join(OUT_DIR, 'tune_mrplus_results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    out_txt = os.path.join(OUT_DIR, 'tune_mrplus_leaderboard.txt')
    with open(out_txt, 'w') as f:
        f.write('=== Baseline (DE2VF) ===\n')
        f.write(json.dumps(baseline_summary, indent=2))
        f.write('\n\n=== Top 10 DE2VF-MR+ configs ===\n')
        for i, r in enumerate(ranked[:10], 1):
            f.write(f"\n[{i}] min_rate={r['min_rate_mean']:.3f} Δ={r['delta_min_rate_vs_baseline']:+.3f} "
                    f"jfi={r['jfi_mean']:.4f} js={r['joint_score_mean']:.4f} "
                    f"ec={r['energy_cum_kj_mean']:.2f}kJ\n")
            f.write(f"params={r['params']}\n")

    best = ranked[0]
    print('\n=== BEST CONFIG ===')
    print(best)
    print(f'\nSaved:\n- {out_json}\n- {out_txt}')


if __name__ == '__main__':
    main()
