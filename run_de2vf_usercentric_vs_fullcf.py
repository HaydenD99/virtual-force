"""
DE2VF 对比实验：User-Centric vs Full Cell-Free
===============================================
目标：仅对 DE2VF（当前动态版本）比较两种接入结构
1) User-Centric（每用户选择 top-L AP，默认 L=3）
2) Full Cell-Free（每用户连接所有 AP）

输出：
- result/de2vf_uc_vs_fullcf/raw_results_uc_vs_fullcf.json
- fig_uc_vs_fullcf_min_rate.png/pdf
- fig_uc_vs_fullcf_jfi.png/pdf
- fig_uc_vs_fullcf_energy.png/pdf
- fig_uc_vs_fullcf_joint_score.png/pdf
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
from run_dynamic_energy_comparison import UAVEnergyModel, brownian_motion_users

# ---------------------- 实验参数 ----------------------
K, L, G = 30, 9, 6
NUM_STEPS = 10
DT = 5.0
ITER_STEP = 10
USER_SIGMA = 8.0
N_USE_SEEDS = 30

W_MIN, W_JFI, W_EE, REF, FLOOR = 0.30, 0.50, 0.20, 60.0, 48.0

OUT_DIR = 'result/de2vf_uc_vs_fullcf'
SEEDS_JSON = 'result/dynamic_large_scale/good_seeds.json'

MODES = ['User-Centric', 'Full-CF']
COLORS = {'User-Centric': '#ee6f63', 'Full-CF': '#5fa8e8'}
MARKS = {'User-Centric': 'o', 'Full-CF': 's'}


class DynamicLoadBalancedBVFFullCF(DynamicLoadBalancedBVF):
    """Full Cell-Free 版本：每个用户连接所有 AP。"""

    def compute_AP_selection_mask(self, betas: np.ndarray) -> np.ndarray:
        mask = np.ones((self.K, betas.shape[1]), dtype=bool)
        return mask


# ---------------------- 评估函数 ----------------------
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


# ---------------------- 单 seed 运行 ----------------------
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

    cfg = create_dynamic_lb_config(K=K, L=L, G=G)
    cfg.update({'time_step': DT, 'max_iterations': 80, 'nbrOfRealizations': 20, 'num_serving_APs': 3})

    alg_uc = DynamicLoadBalancedBVF(cfg, energy_model)
    alg_cf = DynamicLoadBalancedBVFFullCF(cfg, energy_model)
    algs = {'User-Centric': alg_uc, 'Full-CF': alg_cf}

    pos = {m: UAV_init.copy() for m in MODES}
    cumul_e = {m: 0.0 for m in MODES}
    hover_e = energy_model.hover_energy(DT) * L

    rec = {
        m: {'min_rate': [], 'jfi': [], 'energy_cumul': [], 'joint_score': []}
        for m in MODES
    }

    cur_ue = UE_pos.copy()
    for step in range(NUM_STEPS + 1):
        if step == 0:
            all_ap0 = np.vstack([gAP, UAV_init])
            _, _, b0 = ev.compute_channel_model(cur_ue, all_ap0)
            m0 = ev.compute_AP_selection_mask(b0)
            r0, _ = ev.compute_user_rates(cur_ue, all_ap0, m0)
            mr0 = float(r0.min())
            jfi0 = eval_jfi(ev, cur_ue, gAP, UAV_init)
            js0 = joint_score_dyn(mr0, jfi0, 0.0, energy_model)
            for m in MODES:
                rec[m]['min_rate'].append(mr0)
                rec[m]['jfi'].append(jfi0)
                rec[m]['energy_cumul'].append(0.0)
                rec[m]['joint_score'].append(js0)
            continue

        cur_ue = brownian_motion_users(cur_ue, sigma=USER_SIGMA)

        for m in MODES:
            old = sys.stdout
            sys.stdout = io.StringIO()
            p, mr, _, e, _ = algs[m].optimize_one_step(cur_ue, gAP, pos[m], max_iter=ITER_STEP, dt=DT)
            sys.stdout = old

            e += hover_e
            cumul_e[m] += e
            jfi = eval_jfi(ev, cur_ue, gAP, p)
            js = joint_score_dyn(mr, jfi, e, energy_model)

            pos[m] = p
            rec[m]['min_rate'].append(float(mr))
            rec[m]['jfi'].append(float(jfi))
            rec[m]['energy_cumul'].append(float(cumul_e[m]))
            rec[m]['joint_score'].append(float(js))

    return rec


# ---------------------- 绘图 ----------------------
def plot_one(all_rec, seeds, key, ylabel, out_tag, scale=1.0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = np.arange(NUM_STEPS + 1) * DT
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))

    for m in MODES:
        arr = np.array([all_rec[s][m][key] for s in seeds]) * scale
        mean = arr.mean(axis=0)
        ax.plot(t, mean, color=COLORS[m], marker=MARKS[m], linewidth=2.4, markersize=6, label=m)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(OUT_DIR, f'{out_tag}.{ext}'), dpi=220, bbox_inches='tight')
    plt.close(fig)


# ---------------------- 主流程 ----------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(SEEDS_JSON):
        print(f'缺少 {SEEDS_JSON}')
        return

    with open(SEEDS_JSON, 'r') as f:
        seeds = [int(s) for s in json.load(f).get('good_seeds', [])][:N_USE_SEEDS]

    print(f'Run DE2VF User-Centric vs Full-CF on {len(seeds)} seeds...')
    all_rec = {}
    for i, s in enumerate(seeds, 1):
        all_rec[s] = run_one_seed(s)
        uc = np.mean(all_rec[s]['User-Centric']['min_rate'][1:])
        cf = np.mean(all_rec[s]['Full-CF']['min_rate'][1:])
        print(f'[{i:2d}/{len(seeds)}] seed={s}  UC={uc:.2f}  Full-CF={cf:.2f}')

    print('\n=== Mean over seeds (steps 1..T) ===')
    print(f"{'Mode':<14} {'MinRate':>9} {'JFI':>8} {'JointScore':>11} {'E_cum(kJ)':>11}")
    for m in MODES:
        mr = np.mean([np.mean(all_rec[s][m]['min_rate'][1:]) for s in seeds])
        jf = np.mean([np.mean(all_rec[s][m]['jfi'][1:]) for s in seeds])
        js = np.mean([np.mean(all_rec[s][m]['joint_score'][1:]) for s in seeds])
        ec = np.mean([all_rec[s][m]['energy_cumul'][-1] / 1000 for s in seeds])
        print(f"{m:<14} {mr:>9.3f} {jf:>8.4f} {js:>11.4f} {ec:>11.2f}")

    with open(os.path.join(OUT_DIR, 'raw_results_uc_vs_fullcf.json'), 'w') as f:
        json.dump({str(s): all_rec[s] for s in seeds}, f, indent=2)

    plot_one(all_rec, seeds, 'min_rate', 'Min User Rate (Mbps)', 'fig_uc_vs_fullcf_min_rate')
    plot_one(all_rec, seeds, 'jfi', 'JFI_eff', 'fig_uc_vs_fullcf_jfi')
    plot_one(all_rec, seeds, 'energy_cumul', 'Cumulative Energy (kJ)', 'fig_uc_vs_fullcf_energy', scale=1/1000)
    plot_one(all_rec, seeds, 'joint_score', 'Joint Score', 'fig_uc_vs_fullcf_joint_score')

    print(f'\nSaved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
