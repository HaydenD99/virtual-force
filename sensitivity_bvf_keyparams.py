"""
BVF / DE2VF 关键参数敏感性实验
================================
按单因子方式扫描 3 个关键参数:
1) 负载阈值 theta_l  -> load_threshold
2) 阻尼系数 beta      -> damping_coeff
3) 基准步长 s0        -> step_size

固定其它参数，每次只改变一个参数。
输出三张图：
- fig4x_theta_sensitivity.png/pdf
- fig4y_beta_sensitivity.png/pdf
- fig4z_s0_sensitivity.png/pdf
以及原始结果 JSON：
- sensitivity_keyparams.json
"""

import io
import json
import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from load_balanced_bvf_v3style_advanced import LoadBalancedBVF_V3Style, create_lb_v3style_config
from dynamic_lb_bvf import DynamicLoadBalancedBVF, create_dynamic_lb_config
from run_dynamic_energy_comparison import UAVEnergyModel, brownian_motion_users

OUT_DIR = 'result/sensitivity_keyparams'

# 静态场景
K_S, L_S, G_S = 40, 6, 9
STATIC_SEEDS = list(range(1, 16))   # 可按需增大

# 动态场景
K_D, L_D, G_D = 30, 9, 6
DYNAMIC_SEEDS = list(range(1, 13))  # 可按需增大
NUM_STEPS = 10
DT = 5.0
ITER_STEP = 10
USER_SIGMA = 8.0

# JS 权重 (动态复现实验口径)
W_MIN, W_JFI, W_EE, REF, FLOOR = 0.30, 0.50, 0.20, 60.0, 48.0

# 参数扫描
THETA_VALUES = [1.10, 1.20, 1.30, 1.40, 1.50, 1.60]
BETA_VALUES  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
S0_VALUES    = [18, 22, 26, 30, 34, 38]


def static_scene(seed):
    np.random.seed(seed)
    sq = 1000.0
    h_ue, h_ap, h_uav = 1.65, 15.0, 50.0

    sp = sq / 4
    gAP = np.array([[(i + 1) * sp, (j + 1) * sp, h_ap]
                    for i in range(3) for j in range(3)])

    n_hot = int(K_S * 0.75)
    n_uni = K_S - n_hot
    ctr = [[sq * 0.25, sq * 0.30], [sq * 0.70, sq * 0.75]]
    hot = np.vstack([np.random.normal(c, sq * 0.05, (n_hot // 2, 2)) for c in ctr])[:n_hot]
    uni = np.random.uniform(50, sq - 50, (n_uni, 2))
    UE_xy = np.clip(np.vstack([hot, uni]), 30, sq - 30)
    UE_pos = np.column_stack([UE_xy, np.full(K_S, h_ue)])

    l_side = int(np.ceil(np.sqrt(L_S)))
    usp = sq / (l_side + 1)
    uavs = [[np.clip((i + 1) * usp + np.random.uniform(-15, 15), 60, sq - 60),
             np.clip((j + 1) * usp + np.random.uniform(-15, 15), 60, sq - 60), h_uav]
            for i in range(l_side) for j in range(l_side)][:L_S]
    UAV_init = np.array(uavs)
    return UE_pos, gAP, UAV_init


def dynamic_scene(seed):
    np.random.seed(seed)
    UE_pos = np.column_stack([np.random.uniform(50, 950, (K_D, 2)), np.ones(K_D) * 1.65])

    gx = np.linspace(200, 800, 3)
    gy = np.linspace(333, 667, 2)
    GX, GY = np.meshgrid(gx, gy)
    gAP = np.column_stack([GX.flatten()[:G_D], GY.flatten()[:G_D], np.ones(G_D) * 15.0])

    ux = np.linspace(200, 800, 3)
    uy = np.linspace(200, 800, 3)
    UX, UY = np.meshgrid(ux, uy)
    UAV_init = np.column_stack([UX.flatten()[:L_D], UY.flatten()[:L_D], np.ones(L_D) * 50.0])
    return UE_pos, gAP, UAV_init


def eval_jfi_dyn(ev, UE_pos, gAP, UAV_pos):
    all_AP = np.vstack([gAP, UAV_pos])
    _, _, betas = ev.compute_channel_model(UE_pos, all_AP)
    mask = ev.compute_AP_selection_mask(betas)
    mu = mask[:, G_D:]
    mg = mask[:, :G_D]
    bu = betas[:, G_D:]
    bg = betas[:, :G_D]
    gcov = np.array([bg[k, np.where(mg[k])[0]].sum() for k in range(K_D)])
    eff = np.zeros(L_D)
    for l in range(L_D):
        for k in np.where(mu[:, l])[0]:
            eff[l] += bu[k, l] / (gcov[k] + bu[k, l] + 1e-12)
    s = eff.sum()
    return float(s**2 / (L_D * (eff**2).sum() + 1e-12)) if s > 1e-10 else 1.0


def joint_score_dyn(min_rate, jfi, e_step, energy_model):
    e_ref = L_D * energy_model.P_hover * DT * 2.0
    ee = float(np.clip(1.0 - e_step / (e_ref + 1e-6), 0.0, 1.0))
    raw = W_MIN * (min_rate / REF) + W_JFI * jfi + W_EE * ee
    if min_rate < FLOOR:
        raw *= (min_rate / FLOOR) ** 2
    return float(raw)


def run_static_once(theta_l, beta, s0):
    mr_all, jfi_all, js_all, it_all = [], [], [], []

    for seed in STATIC_SEEDS:
        UE_pos, gAP, UAV_init = static_scene(seed)
        cfg = create_lb_v3style_config()
        cfg.update({
            'num_UE': K_S, 'num_UAV': L_S, 'num_ground_AP': G_S,
            'tau_p': K_S, 'num_serving_APs': 3, 'max_iterations': 80,
            'load_threshold': theta_l,
            'damping_coeff': beta,
            'step_size': s0,
        })

        old = sys.stdout
        sys.stdout = io.StringIO()
        res = LoadBalancedBVF_V3Style(cfg).optimize(UE_pos, gAP, UAV_init)
        sys.stdout = old

        mr_all.append(float(res['final_min_rate']))
        jfi_all.append(float(res['final_jfi']))
        js_all.append(float(res['final_joint_score']))
        it_all.append(float(res.get('best_iteration', cfg['max_iterations'])))

    return {
        'min_rate': float(np.mean(mr_all)),
        'jfi': float(np.mean(jfi_all)),
        'js': float(np.mean(js_all)),
        'best_iter': float(np.mean(it_all)),
    }


def run_dynamic_once(theta_l, beta, s0):
    mr_seed, jfi_seed, js_seed, e_step_seed = [], [], [], []

    from balanced_virtual_force_optimizer_v6 import BalancedVirtualForceOptimizerV6, create_v6_config

    for seed in DYNAMIC_SEEDS:
        energy_model = UAVEnergyModel()

        v6_cfg = create_v6_config()
        v6_cfg.update({'num_UE': K_D, 'num_UAV': L_D, 'num_ground_AP': G_D,
                       'tau_p': K_D, 'num_serving_APs': 3, 'nbrOfRealizations': 20})
        ev = BalancedVirtualForceOptimizerV6(v6_cfg)

        UE_pos, gAP, UAV_init = dynamic_scene(seed)
        cfg = create_dynamic_lb_config(K=K_D, L=L_D, G=G_D)
        cfg.update({
            'time_step': DT, 'max_iterations': 80, 'nbrOfRealizations': 20,
            'load_threshold': theta_l,
            'damping_coeff': beta,
            'step_size': s0,
        })
        alg = DynamicLoadBalancedBVF(cfg, energy_model)

        pos = UAV_init.copy()
        cur_ue = UE_pos.copy()
        hover_e = energy_model.hover_energy(DT) * L_D

        mr_t, jfi_t, js_t, e_step_t = [], [], [], []

        # t=0
        all_ap0 = np.vstack([gAP, UAV_init])
        _, _, b0 = ev.compute_channel_model(cur_ue, all_ap0)
        m0 = ev.compute_AP_selection_mask(b0)
        r0, _ = ev.compute_user_rates(cur_ue, all_ap0, m0)
        mr0 = float(r0.min())
        jfi0 = eval_jfi_dyn(ev, cur_ue, gAP, UAV_init)
        js0 = joint_score_dyn(mr0, jfi0, 0.0, energy_model)
        mr_t.append(mr0); jfi_t.append(jfi0); js_t.append(js0); e_step_t.append(0.0)

        for _ in range(NUM_STEPS):
            cur_ue = brownian_motion_users(cur_ue, sigma=USER_SIGMA)
            old = sys.stdout
            sys.stdout = io.StringIO()
            pos, mr, _, e, _ = alg.optimize_one_step(cur_ue, gAP, pos, max_iter=ITER_STEP, dt=DT)
            sys.stdout = old

            e += hover_e
            jfi = eval_jfi_dyn(ev, cur_ue, gAP, pos)
            js = joint_score_dyn(mr, jfi, e, energy_model)

            mr_t.append(float(mr))
            jfi_t.append(float(jfi))
            js_t.append(float(js))
            e_step_t.append(float(e))

        mr_seed.append(np.mean(mr_t[1:]))
        jfi_seed.append(np.mean(jfi_t[1:]))
        js_seed.append(np.mean(js_t[1:]))
        e_step_seed.append(np.mean(e_step_t[1:]) / 1000.0)  # kJ/step

    return {
        'avg_min_rate': float(np.mean(mr_seed)),
        'avg_jfi': float(np.mean(jfi_seed)),
        'avg_js': float(np.mean(js_seed)),
        'avg_energy_step_kj': float(np.mean(e_step_seed)),
    }


def sweep_one(param_name, values):
    out = {'static': [], 'dynamic': []}
    for v in values:
        theta_l = 1.30
        beta = 0.15
        s0 = 26
        if param_name == 'theta_l':
            theta_l = v
        elif param_name == 'beta':
            beta = v
        elif param_name == 's0':
            s0 = v

        print(f'[{param_name}] v={v} ...')
        s_res = run_static_once(theta_l, beta, s0)
        d_res = run_dynamic_once(theta_l, beta, s0)
        out['static'].append({'value': v, **s_res})
        out['dynamic'].append({'value': v, **d_res})
    return out


def plot_figure(data, x_key, title_tag, out_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    xs = [d['value'] for d in data['static']]

    # static metrics
    s_js = [d['js'] for d in data['static']]
    s_mr = [d['min_rate'] for d in data['static']]
    s_jfi = [d['jfi'] for d in data['static']]

    # dynamic metrics
    d_js = [d['avg_js'] for d in data['dynamic']]
    d_mr = [d['avg_min_rate'] for d in data['dynamic']]
    d_jfi = [d['avg_jfi'] for d in data['dynamic']]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))

    # left: static
    ax = axes[0]
    ax.plot(xs, s_js, marker='o', color='#ee6f63', label='JS')
    ax.plot(xs, s_mr, marker='s', color='#5fa8e8', label='Min-Rate')
    ax.plot(xs, s_jfi, marker='^', color='#63cfa0', label='JFI')
    ax.set_xlabel(x_key)
    ax.set_ylabel('Performance')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.set_title(f'Static ({title_tag})', fontweight='bold')

    # right: dynamic
    ax = axes[1]
    ax.plot(xs, d_js, marker='o', color='#ee6f63', label='Avg JS')
    ax.plot(xs, d_mr, marker='s', color='#5fa8e8', label='Avg Min-Rate')
    ax.plot(xs, d_jfi, marker='^', color='#63cfa0', label='Avg JFI')
    ax.set_xlabel(x_key)
    ax.set_ylabel('Performance')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.set_title(f'Dynamic ({title_tag})', fontweight='bold')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(OUT_DIR, f'{out_name}.{ext}'), dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    result_json = os.path.join(OUT_DIR, 'sensitivity_keyparams.json')
    if os.path.exists(result_json):
        print(f'检测到已有结果: {result_json}')
        print('仅根据结果文件重绘，不重新运行实验...')
        with open(result_json, 'r') as f:
            all_res = json.load(f)
    else:
        all_res = {
            'theta_l': sweep_one('theta_l', THETA_VALUES),
            'beta': sweep_one('beta', BETA_VALUES),
            's0': sweep_one('s0', S0_VALUES),
            'settings': {
                'static_seeds': STATIC_SEEDS,
                'dynamic_seeds': DYNAMIC_SEEDS,
                'num_steps': NUM_STEPS,
                'dt': DT,
                'iter_step': ITER_STEP,
            }
        }

        with open(result_json, 'w') as f:
            json.dump(all_res, f, indent=2)

    plot_figure(all_res['theta_l'], 'θ_l (load_threshold)', 'θ_l', 'fig4x_theta_sensitivity')
    plot_figure(all_res['beta'], 'β (damping_coeff)', 'β', 'fig4y_beta_sensitivity')
    plot_figure(all_res['s0'], 's0 (step_size)', 's0', 'fig4z_s0_sensitivity')

    print(f'Saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
