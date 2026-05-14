"""
轻量对比: DE2VF vs DE2VF-MR-S2
===============================
- 两算法同场景同 seeds 对比
- t=0 完全一致
- 输出 min_rate / jfi / energy / joint_score 时序图
"""

import io
import json
import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from balanced_virtual_force_optimizer_v6 import BalancedVirtualForceOptimizerV6, create_v6_config
from dynamic_lb_bvf import DynamicLoadBalancedBVF, create_dynamic_lb_config
from dynamic_lb_bvf_minrate_stage2 import (
    DynamicLoadBalancedBVFMinRateStage2,
    create_dynamic_lb_minrate_stage2_config,
)
from run_dynamic_energy_comparison import UAVEnergyModel, brownian_motion_users

K, L, G = 30, 9, 6
NUM_STEPS = 10
DT = 5.0
ITER_STEP = 10
USER_SIGMA = 8.0
W_MIN, W_JFI, W_EE, REF, FLOOR = 0.30, 0.50, 0.20, 60.0, 48.0

OUT_DIR = 'result/dynamic_large_scale_stage2'
SEEDS_JSON = 'result/dynamic_large_scale/good_seeds.json'
N_USE_SEEDS = 10

ALG_NAMES = ['DE2VF', 'DE2VF-MR-S2']
COLORS = {'DE2VF': '#ee6f63', 'DE2VF-MR-S2': '#ff9f43'}
MARKS = {'DE2VF': 'o', 'DE2VF-MR-S2': 'X'}


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

    cfg_base = create_dynamic_lb_config(K=K, L=L, G=G)
    cfg_base.update({'time_step': DT, 'max_iterations': 80, 'nbrOfRealizations': 20})
    de2vf = DynamicLoadBalancedBVF(cfg_base, energy_model)

    cfg_s2 = create_dynamic_lb_minrate_stage2_config(K=K, L=L, G=G)
    cfg_s2.update({'time_step': DT, 'max_iterations': 80, 'nbrOfRealizations': 20})
    de2vf_s2 = DynamicLoadBalancedBVFMinRateStage2(cfg_s2, energy_model)

    alg_obj = {'DE2VF': de2vf, 'DE2VF-MR-S2': de2vf_s2}
    pos = {a: UAV_init.copy() for a in ALG_NAMES}
    cumul_e = {a: 0.0 for a in ALG_NAMES}
    hover_e = energy_model.hover_energy(DT) * L

    rec = {a: {'min_rate': [], 'jfi': [], 'energy_step': [], 'energy_cumul': [], 'joint_score': []}
           for a in ALG_NAMES}

    cur_UE = UE_pos.copy()
    for step in range(NUM_STEPS + 1):
        if step == 0:
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

        for a in ALG_NAMES:
            old = sys.stdout
            sys.stdout = io.StringIO()
            new_p, mr, _, e, _ = alg_obj[a].optimize_one_step(
                cur_UE, gAP, pos[a], max_iter=ITER_STEP, dt=DT)
            sys.stdout = old

            e += hover_e
            cumul_e[a] += e
            jfi = eval_jfi(ev, cur_UE, gAP, new_p)
            js = joint_score_dyn(mr, jfi, e, energy_model)

            pos[a] = new_p
            rec[a]['min_rate'].append(float(mr))
            rec[a]['jfi'].append(float(jfi))
            rec[a]['energy_step'].append(float(e))
            rec[a]['energy_cumul'].append(float(cumul_e[a]))
            rec[a]['joint_score'].append(float(js))

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
            key, ylabel, tag = item
            scale = 1.0
        else:
            key, ylabel, tag, scale = item

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

        for ext in ['png', 'pdf']:
            plt.savefig(os.path.join(OUT_DIR, f'{tag}.{ext}'), dpi=220, bbox_inches='tight')
        plt.close(fig)


def print_summary(all_records, seeds):
    print('\n=== Mean over seeds (steps 1..T) ===')
    print(f"{'Algorithm':<14} {'MinRate':>9} {'JFI':>8} {'JointScore':>11} {'E_cum(kJ)':>11}")
    for a in ALG_NAMES:
        mr = np.array([np.mean(all_records[s][a]['min_rate'][1:]) for s in seeds])
        jf = np.array([np.mean(all_records[s][a]['jfi'][1:]) for s in seeds])
        js = np.array([np.mean(all_records[s][a]['joint_score'][1:]) for s in seeds])
        ec = np.array([all_records[s][a]['energy_cumul'][-1] / 1000 for s in seeds])
        print(f"{a:<14} {mr.mean():>9.3f} {jf.mean():>8.4f} {js.mean():>11.4f} {ec.mean():>11.2f}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(SEEDS_JSON):
        print(f'缺少 {SEEDS_JSON}')
        return

    with open(SEEDS_JSON, 'r') as f:
        seed_meta = json.load(f)
    seeds = [int(s) for s in seed_meta.get('good_seeds', [])][:N_USE_SEEDS]

    print(f'Using {len(seeds)} seeds from good_seeds.json (quick stage2 mode)')
    all_records = {}
    for i, s in enumerate(seeds, 1):
        all_records[s] = run_one_seed(s)
        b = np.mean(all_records[s]['DE2VF']['min_rate'][1:])
        n = np.mean(all_records[s]['DE2VF-MR-S2']['min_rate'][1:])
        print(f'[{i:2d}/{len(seeds)}] seed={s}  DE2VF={b:.2f}  DE2VF-MR-S2={n:.2f}')

    print_summary(all_records, seeds)

    with open(os.path.join(OUT_DIR, 'raw_results_stage2.json'), 'w') as f:
        json.dump({str(s): all_records[s] for s in seeds}, f, indent=2)

    plot_metrics(all_records, seeds)
    print(f'\nSaved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
