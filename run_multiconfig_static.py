"""
静态多配置对比实验
==================
Config A (K=40,L=6,G=9) 已有结果直接加载
Config B (K=50,L=8,G=9) 运行 50 seeds
Config C (K=40,L=9,G=9) 运行 50 seeds

输出: result/multiconfig/static_cfgB.json
      result/multiconfig/static_cfgC.json
"""

import numpy as np
import json, os, sys, io, time
import multiprocessing as mp
import warnings; warnings.filterwarnings('ignore')

N_SEEDS  = 50
N_WORKERS = min(mp.cpu_count(), 8)
EVAL_SEED = 99999
W_MIN, W_JFI, REF, FLOOR = 0.35, 0.65, 60.0, 48.0
OUT_DIR  = 'result/multiconfig'
ALG_NAMES = ['LB-BVF', 'GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']

CONFIGS = {
    'B': dict(K=50, L=8, G=9),
    'C': dict(K=40, L=9, G=9),
}


def _worker(args):
    seed, K, L, G, W_MIN, W_JFI, REF, FLOOR, EVAL_SEED = args
    import warnings; warnings.filterwarnings('ignore')
    import numpy as np, sys, io, time

    from balanced_virtual_force_optimizer_v6 import (
        BalancedVirtualForceOptimizerV6, create_v6_config)
    from load_balanced_bvf_v3style_advanced import (
        LoadBalancedBVF_V3Style, create_lb_v3style_config)
    from heuristic_lb_3d import GA3D_LB, PSO3D_LB, SSA3D_LB, create_heuristic_config

    sq = 1000

    def gen_scene(s):
        np.random.seed(s)
        sp = sq / 4
        gAP = np.array([[(i+1)*sp, (j+1)*sp, 15.0]
                         for i in range(3) for j in range(3)])[:G]
        n_hot = int(K*0.75); n_uni = K - n_hot
        ctr = [[sq*0.25, sq*0.30], [sq*0.70, sq*0.75]]
        n_per = (n_hot + 1) // 2   # ceiling, 保证足够热点用户
        hot = np.vstack([np.random.normal(c, sq*0.05, (n_per, 2))
                         for c in ctr])[:n_hot]
        uni = np.random.uniform(50, sq-50, (n_uni, 2))
        UE_xy = np.clip(np.vstack([hot, uni]), 30, sq-30)
        UE_pos = np.column_stack([UE_xy, np.full(K, 1.65)])
        l_side = int(np.ceil(np.sqrt(L)))
        usp = sq / (l_side + 1)
        uavs = [[np.clip((i+1)*usp + np.random.uniform(-15, 15), 60, sq-60),
                 np.clip((j+1)*usp + np.random.uniform(-15, 15), 60, sq-60), 50.0]
                for i in range(l_side) for j in range(l_side)][:L]
        return UE_pos, gAP, np.array(uavs)

    def det_eval(ev, UE_pos, gAP, UAV_pos):
        st = np.random.get_state(); np.random.seed(EVAL_SEED)
        all_AP = np.vstack([gAP, UAV_pos])
        _, _, betas = ev.compute_channel_model(UE_pos, all_AP)
        mask = ev.compute_AP_selection_mask(betas)
        rates, _ = ev.compute_user_rates(UE_pos, all_AP, mask)
        np.random.set_state(st)
        mu = mask[:, G:]; mg = mask[:, :G]
        bu = betas[:, G:]; bg = betas[:, :G]
        gcov = np.array([bg[k, np.where(mg[k])[0]].sum() for k in range(K)])
        eff = np.zeros(L)
        for l in range(L):
            for k in np.where(mu[:, l])[0]:
                eff[l] += bu[k, l] / (gcov[k] + bu[k, l] + 1e-12)
        s = eff.sum()
        jfi = float(s**2 / (L*(eff**2).sum()+1e-12)) if s > 1e-10 else 1.0
        mr = float(rates.min())
        raw = W_MIN*(mr/REF) + W_JFI*jfi
        if mr < FLOOR: raw *= (mr/FLOOR)**2
        return mr, jfi, float(raw), rates.tolist()

    v6cfg = create_v6_config()
    v6cfg.update({'num_UE': K, 'num_UAV': L, 'num_ground_AP': G,
                  'tau_p': K, 'num_serving_APs': 3})
    ev = BalancedVirtualForceOptimizerV6(v6cfg)
    UE_pos, gAP, init_UAV = gen_scene(seed)
    results = {}

    # LB-BVF
    cfg = create_lb_v3style_config()
    cfg.update({'num_UE': K, 'num_UAV': L, 'num_ground_AP': G,
                'tau_p': K, 'max_iterations': 80, 'num_serving_APs': 3,
                'w_min': W_MIN, 'w_jfi': W_JFI,
                'ref_rate': REF, 'floor_rate': FLOOR})
    t0 = time.time()
    old = sys.stdout; sys.stdout = io.StringIO()
    res = LoadBalancedBVF_V3Style(cfg).optimize(UE_pos, gAP, init_UAV.copy())
    sys.stdout = old
    mr, jfi, js, rates = det_eval(ev, UE_pos, gAP, res['optimized_UAV_pos'])
    results['LB-BVF'] = {'min_rate': mr, 'jfi': jfi, 'joint_score': js,
                         'rates': rates, 'time': time.time()-t0}

    hcfg = create_heuristic_config(K=K, L=L, G=G)
    hcfg['num_serving_APs'] = 3
    for name, Cls in [('GA-3D-LB', GA3D_LB),
                      ('PSO-3D-LB', PSO3D_LB),
                      ('SSA-3D-LB', SSA3D_LB)]:
        t0 = time.time()
        old = sys.stdout; sys.stdout = io.StringIO()
        res = Cls(hcfg).optimize(UE_pos, gAP, init_UAV.copy())
        sys.stdout = old
        mr, jfi, js, rates = det_eval(ev, UE_pos, gAP, res['optimized_UAV_pos'])
        results[name] = {'min_rate': mr, 'jfi': jfi, 'joint_score': js,
                         'rates': rates, 'time': time.time()-t0}
    return seed, results


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    for cfg_name, dims in CONFIGS.items():
        K, L, G = dims['K'], dims['L'], dims['G']
        ratio = (G+L)*4/K
        print(f"\n=== Static Config {cfg_name}: K={K}, L={L}, G={G}  "
              f"(G+L)*M/K={ratio:.2f} ===")
        t0 = time.time()

        worker_args = [(s, K, L, G, W_MIN, W_JFI, REF, FLOOR, EVAL_SEED)
                       for s in range(1, N_SEEDS+1)]
        all_results = {}

        with mp.Pool(N_WORKERS) as pool:
            for i, (seed, res) in enumerate(
                    pool.imap_unordered(_worker, worker_args), 1):
                all_results[seed] = res
                mr = res['LB-BVF']['min_rate']
                jfi = res['LB-BVF']['jfi']
                print(f"  [{i:2d}/{N_SEEDS}] seed={seed}  "
                      f"LB-BVF: mr={mr:.1f} jfi={jfi:.3f}", flush=True)

        # 汇总
        print(f"\n  Config {cfg_name} 完成 ({time.time()-t0:.0f}s)")
        for alg in ALG_NAMES:
            mr  = np.mean([all_results[s][alg]['min_rate']   for s in all_results])
            jfi = np.mean([all_results[s][alg]['jfi']        for s in all_results])
            js  = np.mean([all_results[s][alg]['joint_score'] for s in all_results])
            print(f"  {alg:<16}: mr={mr:.2f}  jfi={jfi:.4f}  js={js:.4f}")

        # 保存
        save = {str(s): {a: all_results[s][a] for a in ALG_NAMES}
                for s in all_results}
        out_path = os.path.join(OUT_DIR, f'static_cfg{cfg_name}.json')
        with open(out_path, 'w') as f:
            json.dump(save, f, indent=2)
        print(f"  → {out_path}")

    print("\n全部静态配置完成")
