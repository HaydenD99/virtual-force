"""
轻量对比: DE2VF-MR+ vs 三种基线
================================
仅在少量 good_seeds 上快速比较 4 个核心指标:
- min_rate
- jfi
- energy (step + cumulative)
- joint_score

对比算法:
- DE2VF-MR+ (新变体)
- DGA-CF
- DPSO-CF
- NSSA-CF

Output directory: result/dynamic_large_scale_plus/
"""

import io
import json
import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = ['Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from balanced_virtual_force_optimizer_v6 import BalancedVirtualForceOptimizerV6, create_v6_config
from dynamic_lb_bvf_minrate_plus import (
    DynamicLoadBalancedBVFMinRatePlus,
    create_dynamic_lb_minrate_plus_config,
)
from heuristic_lb_3d import GA3D_LB, PSO3D_LB, SSA3D_LB, create_heuristic_config, one_step_optimize
from run_dynamic_energy_comparison import UAVEnergyModel, brownian_motion_users

# 与主实验保持一致
K, L, G = 30, 9, 6
NUM_STEPS = 10
DT = 5.0
ITER_STEP = 10
USER_SIGMA = 8.0
W_MIN, W_JFI, W_EE, REF, FLOOR = 0.30, 0.50, 0.20, 60.0, 48.0

OUT_DIR = 'result/dynamic_large_scale_plus'
SEEDS_JSON = 'result/dynamic_large_scale/good_seeds.json'
N_USE_SEEDS = 12  # 轻量快速对比

ALG_NAMES = ['DE2VF-MR+', 'DGA-CF', 'DPSO-CF', 'NSSA-CF']
COLORS = {
    'DE2VF-MR+': '#ff9f43',
    'DGA-CF': '#5fa8e8',
    'DPSO-CF': '#63cfa0',
    'NSSA-CF': '#b08adf',
}
MARKS = {'DE2VF-MR+': 'X', 'DGA-CF': 's', 'DPSO-CF': '^', 'NSSA-CF': 'D'}


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


def run_one_seed(seed: int):
    energy_model = UAVEnergyModel()

    v6_cfg = create_v6_config()
    v6_cfg.update({'num_UE': K, 'num_UAV': L, 'num_ground_AP': G,
                   'tau_p': K, 'num_serving_APs': 3, 'nbrOfRealizations': 20})
    ev = BalancedVirtualForceOptimizerV6(v6_cfg)

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

    cfg_plus = create_dynamic_lb_minrate_plus_config(K=K, L=L, G=G)
    cfg_plus.update({'time_step': DT, 'max_iterations': 80, 'nbrOfRealizations': 20})
    de2vf_plus = DynamicLoadBalancedBVFMinRatePlus(cfg_plus, energy_model)

    hcfg = create_heuristic_config(K=K, L=L, G=G)
    hcfg.update({'num_serving_APs': 3,
                 'nbrOfRealizations_inner': 10,
                 'nbrOfRealizations_final': 25,
                 'max_iterations': ITER_STEP,
                 'newssa_max_iter': ITER_STEP,
                 'max_generations': ITER_STEP})

    alg_obj = {'DE2VF-MR+': de2vf_plus}
    pos = {a: UAV_init.copy() for a in ALG_NAMES}
    cumul_e = {a: 0.0 for a in ALG_NAMES}
    hover_e = energy_model.hover_energy(DT) * L

    rec = {
        a: {
            'min_rate': [],
            'jfi': [],
            'energy_step': [],
            'energy_cumul': [],
            'joint_score': [],
        } for a in ALG_NAMES
    }

    cur_UE = UE_pos.copy()
    for step in range(NUM_STEPS + 1):
        if step == 0:
            # t=0: 所有算法使用完全相同的初始部署，指标必须一致
            all_ap0 = np.vstack([gAP, UAV_init])
            _, _, b0 = ev.compute_channel_model(cur_UE, all_ap0)
            m0 = ev.compute_AP_selection_mask(b0)
            r0, _ = ev.compute_user_rates(cur_UE, all_ap0, m0)
            mr0 = float(r0.min())
            jfi0 = eval_jfi(ev, cur_UE, gAP, UAV_init)
            js0 = joint_score_dyn(mr0, jfi0, 0.0, energy_model)

            for a in ALG_NAMES:
                rec[a]['min_rate'].append(mr0)
                rec[a]['jfi'].append(jfi0)
                rec[a]['energy_step'].append(0.0)
                rec[a]['energy_cumul'].append(0.0)
                rec[a]['joint_score'].append(js0)
            continue

        cur_UE = brownian_motion_users(cur_UE, sigma=USER_SIGMA)

        # DE2VF-MR+
        old = sys.stdout
        sys.stdout = io.StringIO()
        new_p, mr, _, e, _ = de2vf_plus.optimize_one_step(
            cur_UE, gAP, pos['DE2VF-MR+'], max_iter=ITER_STEP, dt=DT
        )
        sys.stdout = old

        e += hover_e
        cumul_e['DE2VF-MR+'] += e
        jfi = eval_jfi(ev, cur_UE, gAP, new_p)
        js = joint_score_dyn(mr, jfi, e, energy_model)

        pos['DE2VF-MR+'] = new_p
        rec['DE2VF-MR+']['min_rate'].append(float(mr))
        rec['DE2VF-MR+']['jfi'].append(float(jfi))
        rec['DE2VF-MR+']['energy_step'].append(float(e))
        rec['DE2VF-MR+']['energy_cumul'].append(float(cumul_e['DE2VF-MR+']))
        rec['DE2VF-MR+']['joint_score'].append(float(js))

        # 三个基线
        for name, Cls in [('DGA-CF', GA3D_LB), ('DPSO-CF', PSO3D_LB), ('NSSA-CF', SSA3D_LB)]:
            old = sys.stdout
            sys.stdout = io.StringIO()
            alg = Cls(hcfg)
            new_p_h, mr_h, _, e_h, _ = one_step_optimize(
                alg, cur_UE, gAP, pos[name],
                max_iter=ITER_STEP, energy_model=energy_model, flight_speed=10.0
            )
            sys.stdout = old

            e_h += hover_e
            cumul_e[name] += e_h
            jfi_h = eval_jfi(ev, cur_UE, gAP, new_p_h)
            js_h = joint_score_dyn(mr_h, jfi_h, e_h, energy_model)

            pos[name] = new_p_h
            rec[name]['min_rate'].append(float(mr_h))
            rec[name]['jfi'].append(float(jfi_h))
            rec[name]['energy_step'].append(float(e_h))
            rec[name]['energy_cumul'].append(float(cumul_e[name]))
            rec[name]['joint_score'].append(float(js_h))
    return rec


def plot_metrics(all_records, seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = np.arange(NUM_STEPS + 1) * DT
    metric_defs = [
        ('min_rate', 'Min User Rate (Mbps)', 'fig_timeseries_min_rate'),
        ('jfi', 'JFI_eff', 'fig_timeseries_jfi'),
        ('energy_cumul', 'Cumulative Energy (kJ)', 'fig_timeseries_energy', 1 / 1000),
        ('joint_score', 'Joint Score', 'fig_timeseries_joint_score'),
    ]

    for item in metric_defs:
        if len(item) == 3:
            key, ylabel, file_tag = item
            scale = 1.0
        else:
            key, ylabel, file_tag, scale = item

        fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
        for a in ALG_NAMES:
            arr = np.array([all_records[s][a][key] for s in seeds]) * scale
            m = arr.mean(axis=0)
            sd = arr.std(axis=0)
            ax.plot(t, m, color=COLORS[a], marker=MARKS[a], linewidth=2.3,
                    markersize=6, label=a)
            ax.fill_between(t, m - sd, m + sd, color=COLORS[a], alpha=0.10)

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(os.path.join(OUT_DIR, f'{file_tag}.eps'), format='eps', bbox_inches='tight')
        plt.close(fig)


def print_summary(all_records, seeds):
    print('\n=== Mean over seeds (steps 1..T) ===')
    print(f"{'Algorithm':<12} {'MinRate':>9} {'JFI':>8} {'JointScore':>11} {'E_cum(kJ)':>11}")
    for a in ALG_NAMES:
        mr = np.array([np.mean(all_records[s][a]['min_rate'][1:]) for s in seeds])
        jf = np.array([np.mean(all_records[s][a]['jfi'][1:]) for s in seeds])
        js = np.array([np.mean(all_records[s][a]['joint_score'][1:]) for s in seeds])
        ec = np.array([all_records[s][a]['energy_cumul'][-1] / 1000 for s in seeds])
        print(f"{a:<12} {mr.mean():>9.3f} {jf.mean():>8.4f} {js.mean():>11.4f} {ec.mean():>11.2f}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(SEEDS_JSON):
        print(f'缺少 {SEEDS_JSON}')
        return

    with open(SEEDS_JSON, 'r') as f:
        seed_meta = json.load(f)
    seeds = [int(s) for s in seed_meta.get('good_seeds', [])][:N_USE_SEEDS]

    print(f'Using {len(seeds)} seeds from good_seeds.json (quick mode)')
    all_records = {}
    for i, s in enumerate(seeds, 1):
        all_records[s] = run_one_seed(s)
        mr_plus = np.mean(all_records[s]['DE2VF-MR+']['min_rate'][1:])
        mr_dga = np.mean(all_records[s]['DGA-CF']['min_rate'][1:])
        print(f'[{i:2d}/{len(seeds)}] seed={s}  DE2VF-MR+={mr_plus:.2f}  DGA-CF={mr_dga:.2f}')

    print_summary(all_records, seeds)

    with open(os.path.join(OUT_DIR, 'raw_results_plus.json'), 'w') as f:
        json.dump({str(s): all_records[s] for s in seeds}, f, indent=2)

    plot_metrics(all_records, seeds)
    print(f'\nSaved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
