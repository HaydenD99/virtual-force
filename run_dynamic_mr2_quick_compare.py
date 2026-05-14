"""
快速对比: DE2VF vs 三个基线
===============================
使用少量 good_seeds 快速比较：
- DE2VF
- DGA-CF
- DPSO-CF
- NSSA-CF
输出四个时序子图 + 文本汇总。
"""

import io
import json
import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from balanced_virtual_force_optimizer_v6 import BalancedVirtualForceOptimizerV6, create_v6_config
from dynamic_lb_bvf_mr2 import DynamicLoadBalancedBVFMR2, create_dynamic_lb_mr2_config
from heuristic_lb_3d import GA3D_LB, PSO3D_LB, SSA3D_LB, create_heuristic_config, one_step_optimize
from run_dynamic_energy_comparison import UAVEnergyModel, brownian_motion_users

K, L, G = 30, 9, 6
NUM_STEPS = 10
DT = 5.0
ITER_STEP = 10
USER_SIGMA = 8.0
# 统一三目标权重（用于最终综合评分）
W_MIN, W_JFI, W_EE, REF, FLOOR = 1/3, 1/3, 1/3, 60.0, 48.0

OUT_DIR = 'result/dynamic_large_scale_plus'
SEEDS_JSON = 'result/dynamic_large_scale/good_seeds.json'
N_USE_SEEDS = 100

ALG_NAMES = ['DE2VF', 'DGA-CF', 'DPSO-CF', 'NSSA-CF']
COLORS = {'DE2VF': '#ee6f63', 'DGA-CF': '#5fa8e8', 'DPSO-CF': '#63cfa0', 'NSSA-CF': '#b08adf'}
MARKS = {'DE2VF': 'X', 'DGA-CF': 's', 'DPSO-CF': '^', 'NSSA-CF': 'D'}


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


def run_one_seed(seed):
    em = UAVEnergyModel()
    v6_cfg = create_v6_config()
    v6_cfg.update({'num_UE': K, 'num_UAV': L, 'num_ground_AP': G,
                   'tau_p': K, 'num_serving_APs': 3, 'nbrOfRealizations': 20})
    ev = BalancedVirtualForceOptimizerV6(v6_cfg)

    np.random.seed(seed)
    ue = np.column_stack([np.random.uniform(50, 950, (K, 2)), np.ones(K) * 1.65])

    gx = np.linspace(200, 800, 3)
    gy = np.linspace(333, 667, 2)
    GX, GY = np.meshgrid(gx, gy)
    gap = np.column_stack([GX.flatten()[:G], GY.flatten()[:G], np.ones(G) * 15.0])

    ux = np.linspace(200, 800, 3)
    uy = np.linspace(200, 800, 3)
    UX, UY = np.meshgrid(ux, uy)
    u0 = np.column_stack([UX.flatten()[:L], UY.flatten()[:L], np.ones(L) * 50.0])

    cfg2 = create_dynamic_lb_mr2_config(K=K, L=L, G=G)
    cfg2.update({'time_step': DT, 'max_iterations': 80, 'nbrOfRealizations': 20})
    mr2 = DynamicLoadBalancedBVFMR2(cfg2, em)

    hcfg = create_heuristic_config(K=K, L=L, G=G)
    hcfg.update({'num_serving_APs': 3,
                 'nbrOfRealizations_inner': 10,
                 'nbrOfRealizations_final': 25,
                 'max_iterations': ITER_STEP,
                 'newssa_max_iter': ITER_STEP,
                 'max_generations': ITER_STEP})

    pos = {a: u0.copy() for a in ALG_NAMES}
    ec = {a: 0.0 for a in ALG_NAMES}
    hover = em.hover_energy(DT) * L

    rec = {a: {'min_rate': [], 'jfi': [], 'energy_cumul': [], 'joint_score': []} for a in ALG_NAMES}

    cur_ue = ue.copy()
    for step in range(NUM_STEPS + 1):
        if step == 0:
            all_ap0 = np.vstack([gap, u0])
            _, _, b0 = ev.compute_channel_model(cur_ue, all_ap0)
            m0 = ev.compute_AP_selection_mask(b0)
            r0, _ = ev.compute_user_rates(cur_ue, all_ap0, m0)
            mr0 = float(r0.min())
            jfi0 = eval_jfi(ev, cur_ue, gap, u0)
            js0 = joint_score_dyn(mr0, jfi0, 0.0, em)
            for a in ALG_NAMES:
                rec[a]['min_rate'].append(mr0)
                rec[a]['jfi'].append(jfi0)
                rec[a]['energy_cumul'].append(0.0)
                rec[a]['joint_score'].append(js0)
            continue

        cur_ue = brownian_motion_users(cur_ue, sigma=USER_SIGMA)

        # MR2
        old = sys.stdout; sys.stdout = io.StringIO()
        p, mr, _, e, _ = mr2.optimize_one_step(cur_ue, gap, pos['DE2VF'], max_iter=ITER_STEP, dt=DT)
        sys.stdout = old

        e += hover
        ec['DE2VF'] += e
        jfi = eval_jfi(ev, cur_ue, gap, p)
        js = joint_score_dyn(mr, jfi, e, em)

        pos['DE2VF'] = p
        rec['DE2VF']['min_rate'].append(float(mr))
        rec['DE2VF']['jfi'].append(float(jfi))
        rec['DE2VF']['energy_cumul'].append(float(ec['DE2VF']))
        rec['DE2VF']['joint_score'].append(float(js))

        # baselines
        for name, Cls in [('DGA-CF', GA3D_LB), ('DPSO-CF', PSO3D_LB), ('NSSA-CF', SSA3D_LB)]:
            old = sys.stdout; sys.stdout = io.StringIO()
            alg = Cls(hcfg)
            p_h, mr_h, _, e_h, _ = one_step_optimize(
                alg, cur_ue, gap, pos[name],
                max_iter=ITER_STEP, energy_model=em, flight_speed=10.0)
            sys.stdout = old

            e_h += hover
            ec[name] += e_h
            jfi_h = eval_jfi(ev, cur_ue, gap, p_h)
            js_h = joint_score_dyn(mr_h, jfi_h, e_h, em)

            pos[name] = p_h
            rec[name]['min_rate'].append(float(mr_h))
            rec[name]['jfi'].append(float(jfi_h))
            rec[name]['energy_cumul'].append(float(ec[name]))
            rec[name]['joint_score'].append(float(js_h))

    return rec


def plot(all_rec, seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = np.arange(NUM_STEPS + 1) * DT
    defs = [
        ('min_rate', 'Min User Rate (Mbps)', 'fig_mr2_min_rate'),
        ('jfi', 'JFI_eff', 'fig_mr2_jfi'),
        ('energy_cumul', 'Cumulative Energy (kJ)', 'fig_mr2_energy', 1/1000),
        ('joint_score', 'Joint Score', 'fig_mr2_joint_score'),
    ]

    for item in defs:
        if len(item) == 3:
            key, yl, tag = item
            sc = 1.0
        else:
            key, yl, tag, sc = item
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
        for a in ALG_NAMES:
            arr = np.array([all_rec[s][a][key] for s in seeds]) * sc
            m = arr.mean(axis=0)
            sd = arr.std(axis=0)
            ax.plot(t, m, color=COLORS[a], marker=MARKS[a], linewidth=2.3, markersize=6, label=a)
            ax.fill_between(t, m-sd, m+sd, color=COLORS[a], alpha=0.10)
        ax.legend(loc='upper left', fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(yl)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        for ext in ['png', 'pdf']:
            plt.savefig(os.path.join(OUT_DIR, f'{tag}.{ext}'), dpi=220, bbox_inches='tight')
        plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SEEDS_JSON, 'r') as f:
        seeds = [int(s) for s in json.load(f).get('good_seeds', [])][:N_USE_SEEDS]

    all_rec = {}
    print(f'Run DE2VF vs baselines on {len(seeds)} seeds...')
    for i, s in enumerate(seeds, 1):
        all_rec[s] = run_one_seed(s)
        mr2_m = np.mean(all_rec[s]['DE2VF']['min_rate'][1:])
        ga_m = np.mean(all_rec[s]['DGA-CF']['min_rate'][1:])
        pso_m = np.mean(all_rec[s]['DPSO-CF']['min_rate'][1:])
        ssa_m = np.mean(all_rec[s]['NSSA-CF']['min_rate'][1:])
        print(f'[{i:2d}/{len(seeds)}] seed={s}  MR2={mr2_m:.2f}  GA={ga_m:.2f}  PSO={pso_m:.2f}  SSA={ssa_m:.2f}')

    print('\n=== Mean over seeds (steps 1..T) ===')
    print(f"{'Alg':<10} {'MinRate':>9} {'JFI':>8} {'JointScore':>11} {'E_cum(kJ)':>11}")
    for a in ALG_NAMES:
        mr = np.mean([np.mean(all_rec[s][a]['min_rate'][1:]) for s in seeds])
        jf = np.mean([np.mean(all_rec[s][a]['jfi'][1:]) for s in seeds])
        js = np.mean([np.mean(all_rec[s][a]['joint_score'][1:]) for s in seeds])
        ec = np.mean([all_rec[s][a]['energy_cumul'][-1] / 1000 for s in seeds])
        print(f"{a:<10} {mr:>9.3f} {jf:>8.4f} {js:>11.4f} {ec:>11.2f}")

    with open(os.path.join(OUT_DIR, 'raw_results_mr2_quick.json'), 'w') as f:
        json.dump({str(s): all_rec[s] for s in seeds}, f, indent=2)

    plot(all_rec, seeds)
    print(f'\nSaved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
