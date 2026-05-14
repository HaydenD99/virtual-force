"""
动态多配置对比实验
==================
Config A (K=30,L=9,G=6) 已有结果直接加载
Config B (K=40,L=12,G=6) 运行 30 seeds
Config C (K=20,L=6,G=6)  运行 30 seeds

输出: result/multiconfig/dynamic_cfgB.json
      result/multiconfig/dynamic_cfgC.json
"""

import numpy as np
import json, os, sys, io, time
import multiprocessing as mp
import warnings; warnings.filterwarnings('ignore')

N_SEEDS   = 30
N_WORKERS = min(mp.cpu_count(), 8)
NUM_STEPS = 10
DT        = 5.0
ITER_STEP = 10
USER_SIGMA = 8.0
W_MIN, W_JFI, W_EE, REF, FLOOR = 0.30, 0.50, 0.20, 60.0, 48.0
OUT_DIR   = 'result/multiconfig'
ALG_NAMES = ['Dynamic LB-BVF', 'GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']

CONFIGS = {
    'B': dict(K=40, L=12, G=6),
    'C': dict(K=20, L=6,  G=6),
}


def _dyn_worker(args):
    seed, K, L, G = args
    import warnings; warnings.filterwarnings('ignore')
    import numpy as np, sys, io, time

    from balanced_virtual_force_optimizer_v6 import (
        BalancedVirtualForceOptimizerV6, create_v6_config)
    from dynamic_lb_bvf import DynamicLoadBalancedBVF, create_dynamic_lb_config
    from heuristic_lb_3d import (GA3D_LB, PSO3D_LB, SSA3D_LB,
                                  create_heuristic_config, one_step_optimize)
    from run_dynamic_energy_comparison import UAVEnergyModel, brownian_motion_users

    W_MIN, W_JFI, W_EE, REF, FLOOR = 0.30, 0.50, 0.20, 60.0, 48.0
    NUM_STEPS = 10; DT = 5.0; ITER_STEP = 10; USER_SIGMA = 8.0

    def eval_jfi(ev, UE_pos, gAP, UAV_pos):
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
        return float(s**2 / (L*(eff**2).sum()+1e-12)) if s > 1e-10 else 1.0

    def js_dyn(mr, jfi, e_step, em):
        E_ref = L * em.P_hover * DT * 2.0
        ee = float(np.clip(1.0 - e_step/(E_ref+1e-6), 0.0, 1.0))
        raw = W_MIN*(mr/REF) + W_JFI*jfi + W_EE*ee
        if mr < FLOOR: raw *= (mr/FLOOR)**2
        return float(raw)

    energy_model = UAVEnergyModel()
    v6cfg = create_v6_config()
    v6cfg.update({'num_UE': K, 'num_UAV': L, 'num_ground_AP': G,
                  'tau_p': K, 'num_serving_APs': 3, 'nbrOfRealizations': 20})
    ev = BalancedVirtualForceOptimizerV6(v6cfg)

    np.random.seed(seed)
    UE_pos = np.column_stack([np.random.uniform(50, 950, (K, 2)), np.ones(K)*1.65])
    gx = np.linspace(200, 800, 3); gy = np.linspace(333, 667, 2)
    GX, GY = np.meshgrid(gx, gy)
    gAP = np.column_stack([GX.flatten()[:G], GY.flatten()[:G], np.ones(G)*15.0])
    l_side = int(np.ceil(np.sqrt(L)))
    ux = np.linspace(150, 850, l_side); uy = np.linspace(150, 850, l_side)
    UX, UY = np.meshgrid(ux, uy)
    UAV_init = np.column_stack([UX.flatten()[:L], UY.flatten()[:L], np.ones(L)*50.0])
    hover_E = energy_model.hover_energy(DT) * L

    dlb_cfg = create_dynamic_lb_config(K=K, L=L, G=G)
    dlb_cfg.update({'time_step': DT, 'w_min': W_MIN, 'w_jfi': W_JFI,
                    'w_ee': W_EE, 'max_iterations': 80, 'nbrOfRealizations': 20})
    dlb = DynamicLoadBalancedBVF(dlb_cfg, energy_model)

    hcfg = create_heuristic_config(K=K, L=L, G=G)
    hcfg.update({'num_serving_APs': 3, 'nbrOfRealizations_inner': 10,
                 'nbrOfRealizations_final': 25, 'max_iterations': ITER_STEP,
                 'newssa_max_iter': ITER_STEP, 'max_generations': ITER_STEP})

    cur_UE = UE_pos.copy()
    pos = {a: UAV_init.copy() for a in ALG_NAMES}
    cumul = {a: 0.0 for a in ALG_NAMES}

    records = {a: {'min_rate': [], 'jfi': [], 'energy_step': [],
                   'energy_cumul': [], 'joint_score': []}
               for a in ALG_NAMES}

    for step in range(1, NUM_STEPS + 1):
        cur_UE = brownian_motion_users(cur_UE, sigma=USER_SIGMA)

        np.random.seed(seed + step*100 + 1)
        old = sys.stdout; sys.stdout = io.StringIO()
        new_p, mr_d, _, e_d, _ = dlb.optimize_one_step(
            cur_UE, gAP, pos['Dynamic LB-BVF'], max_iter=ITER_STEP, dt=DT)
        sys.stdout = old
        e_d += hover_E; cumul['Dynamic LB-BVF'] += e_d
        jfi_d = eval_jfi(ev, cur_UE, gAP, new_p)
        js_d  = js_dyn(mr_d, jfi_d, e_d, energy_model)
        pos['Dynamic LB-BVF'] = new_p
        records['Dynamic LB-BVF']['min_rate'].append(mr_d)
        records['Dynamic LB-BVF']['jfi'].append(jfi_d)
        records['Dynamic LB-BVF']['energy_step'].append(e_d)
        records['Dynamic LB-BVF']['energy_cumul'].append(cumul['Dynamic LB-BVF'])
        records['Dynamic LB-BVF']['joint_score'].append(js_d)

        for alg_name, AlgClass, sid in [('GA-3D-LB', GA3D_LB, 2),
                                          ('PSO-3D-LB', PSO3D_LB, 3),
                                          ('SSA-3D-LB', SSA3D_LB, 4)]:
            np.random.seed(seed + step*100 + sid)
            old = sys.stdout; sys.stdout = io.StringIO()
            alg = AlgClass(hcfg)
            new_p_h, mr_h, _, e_h, _ = one_step_optimize(
                alg, cur_UE, gAP, pos[alg_name],
                max_iter=ITER_STEP, energy_model=energy_model, flight_speed=10.0)
            sys.stdout = old
            e_h += hover_E; cumul[alg_name] += e_h
            jfi_h = eval_jfi(ev, cur_UE, gAP, new_p_h)
            js_h  = js_dyn(mr_h, jfi_h, e_h, energy_model)
            pos[alg_name] = new_p_h
            records[alg_name]['min_rate'].append(mr_h)
            records[alg_name]['jfi'].append(jfi_h)
            records[alg_name]['energy_step'].append(e_h)
            records[alg_name]['energy_cumul'].append(cumul[alg_name])
            records[alg_name]['joint_score'].append(js_h)

    # 每个种子返回各算法在 NUM_STEPS 步的均值
    summary = {}
    for a in ALG_NAMES:
        summary[a] = {
            'min_rate':    float(np.mean(records[a]['min_rate'])),
            'jfi':         float(np.mean(records[a]['jfi'])),
            'joint_score': float(np.mean(records[a]['joint_score'])),
            'energy_cumul':float(records[a]['energy_cumul'][-1]),
        }
    return seed, summary


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    for cfg_name, dims in CONFIGS.items():
        K, L, G = dims['K'], dims['L'], dims['G']
        ratio = (G+L)*4/K
        print(f"\n=== Dynamic Config {cfg_name}: K={K}, L={L}, G={G}  "
              f"(G+L)*M/K={ratio:.2f} ===")
        t0 = time.time()

        worker_args = [(s, K, L, G) for s in range(1, N_SEEDS+1)]
        all_results = {}

        with mp.Pool(N_WORKERS) as pool:
            for i, (seed, res) in enumerate(
                    pool.imap_unordered(_dyn_worker, worker_args), 1):
                all_results[seed] = res
                mr  = res['Dynamic LB-BVF']['min_rate']
                jfi = res['Dynamic LB-BVF']['jfi']
                print(f"  [{i:2d}/{N_SEEDS}] seed={seed}  "
                      f"DLB: mr={mr:.1f} jfi={jfi:.3f}", flush=True)

        print(f"\n  Config {cfg_name} 完成 ({time.time()-t0:.0f}s)")
        for alg in ALG_NAMES:
            mr  = np.mean([all_results[s][alg]['min_rate']    for s in all_results])
            jfi = np.mean([all_results[s][alg]['jfi']         for s in all_results])
            js  = np.mean([all_results[s][alg]['joint_score'] for s in all_results])
            print(f"  {alg:<22}: mr={mr:.2f}  jfi={jfi:.4f}  js={js:.4f}")

        save = {str(s): all_results[s] for s in all_results}
        out_path = os.path.join(OUT_DIR, f'dynamic_cfg{cfg_name}.json')
        with open(out_path, 'w') as f:
            json.dump(save, f, indent=2)
        print(f"  → {out_path}")

    print("\n全部动态配置完成")
