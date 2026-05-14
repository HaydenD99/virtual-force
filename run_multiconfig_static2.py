"""
静态多配置对比 v2
=================
固定 L=9 (UAV), G=6 (地面AP), 变化 K=20/30/40
(G+L)*M/K = 60/K → 3.0 / 2.0 / 1.5  均满足 Cell-Free 要求

输出: result/multiconfig2/static_K20.json
      result/multiconfig2/static_K30.json
      result/multiconfig2/static_K40.json
"""
import numpy as np
import json, os, sys, io, time
import multiprocessing as mp
import warnings; warnings.filterwarnings('ignore')

L, G     = 9, 6          # 固定
K_LIST   = [20, 30, 40]  # 变化
N_SEEDS  = 50
N_WORKERS = min(mp.cpu_count(), 8)
EVAL_SEED = 99999
W_MIN, W_JFI, REF, FLOOR = 0.35, 0.65, 60.0, 48.0
OUT_DIR  = 'result/multiconfig2'
ALG_NAMES = ['LB-BVF', 'GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']


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
        # G=6: 2×3 grid
        gx = np.linspace(200, 800, 3); gy = np.linspace(333, 667, 2)
        GX, GY = np.meshgrid(gx, gy)
        gAP = np.column_stack([GX.flatten()[:G], GY.flatten()[:G],
                                np.ones(G) * 15.0])
        # UE: 热点+均匀混合
        n_hot = int(K * 0.75); n_uni = K - n_hot
        n_per = (n_hot + 1) // 2
        ctr = [[sq*0.25, sq*0.30], [sq*0.70, sq*0.75]]
        hot = np.vstack([np.random.normal(c, sq*0.05, (n_per, 2))
                         for c in ctr])[:n_hot]
        uni = np.random.uniform(50, sq-50, (n_uni, 2))
        UE_xy = np.clip(np.vstack([hot, uni]), 30, sq-30)
        UE_pos = np.column_stack([UE_xy, np.full(K, 1.65)])
        # L=9: 3×3 grid
        ux = np.linspace(200, 800, 3); uy = np.linspace(200, 800, 3)
        UX, UY = np.meshgrid(ux, uy)
        UAV_pos = np.column_stack([
            UX.flatten()[:L] + np.random.uniform(-15, 15, L),
            UY.flatten()[:L] + np.random.uniform(-15, 15, L),
            np.ones(L) * 50.0])
        UAV_pos[:, :2] = np.clip(UAV_pos[:, :2], 60, sq-60)
        return UE_pos, gAP, UAV_pos

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
    for K in K_LIST:
        ratio = (G+L)*4/K
        print(f"\n=== Static  K={K}, L={L}, G={G}  (G+L)*M/K={ratio:.2f} ===")
        t0 = time.time()
        worker_args = [(s, K, L, G, W_MIN, W_JFI, REF, FLOOR, EVAL_SEED)
                       for s in range(1, N_SEEDS+1)]
        all_results = {}
        with mp.Pool(N_WORKERS) as pool:
            for i, (seed, res) in enumerate(
                    pool.imap_unordered(_worker, worker_args), 1):
                all_results[seed] = res
                print(f"  [{i:2d}/{N_SEEDS}] seed={seed}  "
                      f"LB-BVF: mr={res['LB-BVF']['min_rate']:.1f} "
                      f"jfi={res['LB-BVF']['jfi']:.3f}", flush=True)

        print(f"\n  K={K} 完成 ({time.time()-t0:.0f}s)")
        for alg in ALG_NAMES:
            mr  = np.mean([all_results[s][alg]['min_rate']    for s in all_results])
            jfi = np.mean([all_results[s][alg]['jfi']         for s in all_results])
            js  = np.mean([all_results[s][alg]['joint_score'] for s in all_results])
            print(f"  {alg:<16}: mr={mr:.2f}  jfi={jfi:.4f}  js={js:.4f}")

        save = {str(s): {a: all_results[s][a] for a in ALG_NAMES}
                for s in all_results}
        out_path = os.path.join(OUT_DIR, f'static_K{K}.json')
        with open(out_path, 'w') as f:
            json.dump(save, f, indent=2)
        print(f"  → {out_path}")

    print("\n全部静态配置完成")
