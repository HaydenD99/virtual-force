"""
DE2VF-MR+ 局部精扫参数脚本
=========================
固定主结构，只在最优附近小范围搜索：
- worst_q
- k_mr_force
- refine_step_sizes

对比对象：DE2VF baseline
输出：
  result/dynamic_large_scale_plus/tune_mrplus_refine_results.json
  result/dynamic_large_scale_plus/tune_mrplus_refine_leaderboard.txt
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

K, L, G = 30, 9, 6
NUM_STEPS = 10
DT = 5.0
ITER_STEP = 10
USER_SIGMA = 8.0
W_MIN, W_JFI, W_EE, REF, FLOOR = 0.30, 0.50, 0.20, 60.0, 48.0

OUT_DIR = 'result/dynamic_large_scale_plus'
SEEDS_JSON = 'result/dynamic_large_scale/good_seeds.json'
N_USE_SEEDS = 8


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


def init_scene(seed):
    np.random.seed(seed)
    ue = np.column_stack([np.random.uniform(50, 950, (K, 2)), np.ones(K) * 1.65])

    gx = np.linspace(200, 800, 3)
    gy = np.linspace(333, 667, 2)
    GX, GY = np.meshgrid(gx, gy)
    gap = np.column_stack([GX.flatten()[:G], GY.flatten()[:G], np.ones(G) * 15.0])

    ux = np.linspace(200, 800, 3)
    uy = np.linspace(200, 800, 3)
    UX, UY = np.meshgrid(ux, uy)
    uav = np.column_stack([UX.flatten()[:L], UY.flatten()[:L], np.ones(L) * 50.0])
    return ue, gap, uav


def eval_config_on_seed(seed, cfg_update):
    energy_model = UAVEnergyModel()

    v6_cfg = create_v6_config()
    v6_cfg.update({'num_UE': K, 'num_UAV': L, 'num_ground_AP': G,
                   'tau_p': K, 'num_serving_APs': 3, 'nbrOfRealizations': 20})
    ev = BalancedVirtualForceOptimizerV6(v6_cfg)

    ue, gap, uav_init = init_scene(seed)

    base_cfg = create_dynamic_lb_config(K=K, L=L, G=G)
    base_cfg.update({'time_step': DT, 'max_iterations': 80, 'nbrOfRealizations': 20})
    base = DynamicLoadBalancedBVF(base_cfg, energy_model)

    plus_cfg = create_dynamic_lb_minrate_plus_config(K=K, L=L, G=G)
    plus_cfg.update({'time_step': DT, 'max_iterations': 80, 'nbrOfRealizations': 20})
    plus_cfg.update(cfg_update)
    plus = DynamicLoadBalancedBVFMinRatePlus(plus_cfg, energy_model)

    pos = {'base': uav_init.copy(), 'plus': uav_init.copy()}
    ec = {'base': 0.0, 'plus': 0.0}
    hover_e = energy_model.hover_energy(DT) * L

    rec = {
        'base': {'min_rate': [], 'jfi': [], 'joint_score': [], 'energy_cumul': []},
        'plus': {'min_rate': [], 'jfi': [], 'joint_score': [], 'energy_cumul': []},
    }

    cur_ue = ue.copy()
    for step in range(NUM_STEPS + 1):
        if step == 0:
            all_ap0 = np.vstack([gap, uav_init])
            _, _, b0 = ev.compute_channel_model(cur_ue, all_ap0)
            m0 = ev.compute_AP_selection_mask(b0)
            r0, _ = ev.compute_user_rates(cur_ue, all_ap0, m0)
            mr0 = float(r0.min())
            jfi0 = eval_jfi(ev, cur_ue, gap, uav_init)
            js0 = joint_score_dyn(mr0, jfi0, 0.0, energy_model)
            for k in ['base', 'plus']:
                rec[k]['min_rate'].append(mr0)
                rec[k]['jfi'].append(jfi0)
                rec[k]['joint_score'].append(js0)
                rec[k]['energy_cumul'].append(0.0)
            continue

        cur_ue = brownian_motion_users(cur_ue, sigma=USER_SIGMA)

        for tag, alg in [('base', base), ('plus', plus)]:
            old = sys.stdout
            sys.stdout = io.StringIO()
            p, mr, _, e, _ = alg.optimize_one_step(cur_ue, gap, pos[tag], max_iter=ITER_STEP, dt=DT)
            sys.stdout = old

            e += hover_e
            ec[tag] += e
            jfi = eval_jfi(ev, cur_ue, gap, p)
            js = joint_score_dyn(mr, jfi, e, energy_model)

            pos[tag] = p
            rec[tag]['min_rate'].append(float(mr))
            rec[tag]['jfi'].append(float(jfi))
            rec[tag]['joint_score'].append(float(js))
            rec[tag]['energy_cumul'].append(float(ec[tag]))

    return rec


def aggregate(rec_list, key):
    vals = np.array([np.mean(r[key][1:]) for r in rec_list])
    return float(vals.mean()), float(vals.std())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(SEEDS_JSON):
        print(f'missing {SEEDS_JSON}')
        return

    with open(SEEDS_JSON, 'r') as f:
        seeds = [int(s) for s in json.load(f).get('good_seeds', [])][:N_USE_SEEDS]

    grid = {
        'worst_q': [0.15, 0.20, 0.25],
        'k_mr_force': [2.0e4, 2.2e4, 2.4e4],
        'refine_step_sizes': [[8.0, 5.0, 3.0], [9.0, 6.0, 3.5]],
    }

    combos = list(itertools.product(grid['worst_q'], grid['k_mr_force'], grid['refine_step_sizes']))

    # baseline once
    baseline_records = []
    for s in seeds:
        r = eval_config_on_seed(s, cfg_update={})
        baseline_records.append(r['base'])

    b_mr, b_mr_sd = aggregate(baseline_records, 'min_rate')
    b_jfi, _ = aggregate(baseline_records, 'jfi')
    b_js, _ = aggregate(baseline_records, 'joint_score')
    b_ec = float(np.mean([r['energy_cumul'][-1] / 1000 for r in baseline_records]))

    results = []
    for i, (wq, kmf, rss) in enumerate(combos, 1):
        cfg = {'worst_q': wq, 'k_mr_force': kmf, 'refine_step_sizes': rss}
        plus_records = []
        for s in seeds:
            r = eval_config_on_seed(s, cfg_update=cfg)
            plus_records.append(r['plus'])

        p_mr, p_mr_sd = aggregate(plus_records, 'min_rate')
        p_jfi, _ = aggregate(plus_records, 'jfi')
        p_js, _ = aggregate(plus_records, 'joint_score')
        p_ec = float(np.mean([r['energy_cumul'][-1] / 1000 for r in plus_records]))

        item = {
            'params': cfg,
            'min_rate_mean': p_mr,
            'min_rate_std': p_mr_sd,
            'jfi_mean': p_jfi,
            'joint_score_mean': p_js,
            'energy_cum_kj_mean': p_ec,
            'delta_min_rate_vs_base': p_mr - b_mr,
        }
        results.append(item)
        print(f"[{i:2d}/{len(combos)}] Δmin={item['delta_min_rate_vs_base']:+.3f}  params={cfg}")

    results.sort(key=lambda x: x['delta_min_rate_vs_base'], reverse=True)

    out_json = {
        'baseline': {
            'min_rate_mean': b_mr,
            'min_rate_std': b_mr_sd,
            'jfi_mean': b_jfi,
            'joint_score_mean': b_js,
            'energy_cum_kj_mean': b_ec,
        },
        'results': results,
    }

    with open(os.path.join(OUT_DIR, 'tune_mrplus_refine_results.json'), 'w') as f:
        json.dump(out_json, f, indent=2)

    lines = []
    lines.append('=== Baseline (DE2VF) ===')
    lines.append(json.dumps(out_json['baseline'], indent=2))
    lines.append('\n=== Top 10 Refine configs ===\n')
    for i, r in enumerate(results[:10], 1):
        lines.append(
            f"[{i}] min_rate={r['min_rate_mean']:.3f} Δ={r['delta_min_rate_vs_base']:+.3f} "
            f"jfi={r['jfi_mean']:.4f} js={r['joint_score_mean']:.4f} ec={r['energy_cum_kj_mean']:.2f}kJ"
        )
        lines.append(f"params={r['params']}\n")

    with open(os.path.join(OUT_DIR, 'tune_mrplus_refine_leaderboard.txt'), 'w') as f:
        f.write('\n'.join(lines))

    print('\nSaved refine tuning results.')


if __name__ == '__main__':
    main()
