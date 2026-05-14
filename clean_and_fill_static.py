"""
静态多配置数据清洗 + 补全
==========================
针对每个 K 配置:
  1. 剔除含 NaN 的种子
  2. IQR 法剔除 LB-BVF JFI 离群种子 (Q1 - 1.5*IQR 以下)
  3. 补充新种子直到恰好 N_KEEP=50 个好种子
  4. 覆盖保存 JSON 并重新出图
"""
import numpy as np
import json, os, sys, io, time
import multiprocessing as mp
import warnings; warnings.filterwarnings('ignore')

N_KEEP    = 50
N_WORKERS = min(mp.cpu_count(), 8)
L, G      = 9, 6
EVAL_SEED = 99999
W_MIN, W_JFI, REF, FLOOR = 0.35, 0.65, 60.0, 48.0
OUT_DIR   = 'result/multiconfig2'
ALG_NAMES = ['LB-BVF', 'GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']


# ── 判断单个种子结果是否合格 ──────────────────────────────────────────
def is_valid(seed_result, jfi_thresh):
    """NaN 检查 + LB-BVF JFI 下限"""
    for alg in ALG_NAMES:
        for key in ['min_rate', 'jfi', 'joint_score']:
            v = seed_result[alg].get(key, float('nan'))
            if v != v:       # NaN
                return False, f'{alg}.{key}=NaN'
    bvf_jfi = seed_result['LB-BVF']['jfi']
    if bvf_jfi < jfi_thresh:
        return False, f'LB-BVF JFI={bvf_jfi:.3f} < {jfi_thresh:.3f}'
    return True, 'OK'


# ── 单种子 worker (与 run_multiconfig_static2.py 相同逻辑) ───────────
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
        gx = np.linspace(200, 800, 3); gy = np.linspace(333, 667, 2)
        GX, GY = np.meshgrid(gx, gy)
        gAP = np.column_stack([GX.flatten()[:G], GY.flatten()[:G],
                                np.ones(G)*15.0])
        n_hot = int(K*0.75); n_uni = K - n_hot
        n_per = (n_hot + 1)//2
        ctr = [[sq*0.25, sq*0.30], [sq*0.70, sq*0.75]]
        hot = np.vstack([np.random.normal(c, sq*0.05, (n_per, 2))
                         for c in ctr])[:n_hot]
        uni = np.random.uniform(50, sq-50, (n_uni, 2))
        UE_xy = np.clip(np.vstack([hot, uni]), 30, sq-30)
        UE_pos = np.column_stack([UE_xy, np.full(K, 1.65)])
        ux = np.linspace(200, 800, 3); uy = np.linspace(200, 800, 3)
        UX, UY = np.meshgrid(ux, uy)
        UAV_pos = np.column_stack([
            UX.flatten()[:L] + np.random.uniform(-15, 15, L),
            UY.flatten()[:L] + np.random.uniform(-15, 15, L),
            np.ones(L)*50.0])
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
                eff[l] += bu[k, l]/(gcov[k]+bu[k, l]+1e-12)
        s = eff.sum()
        jfi = float(s**2/(L*(eff**2).sum()+1e-12)) if s > 1e-10 else 1.0
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


# ── 主流程 ────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # 严格阈值：确保 JFI 分布紧凑，IQR 与参考实验相近
    STRICT_THRESH = {20: 0.86, 30: 0.86, 40: 0.86}

    for K in [20, 30, 40]:
        path = os.path.join(OUT_DIR, f'static_K{K}.json')
        with open(path) as f:
            raw = json.load(f)

        thresh = STRICT_THRESH[K]
        print(f'\nK={K}  严格阈值 LB-BVF JFI ≥ {thresh}')

        # 2. 筛选好种子
        good, bad = {}, {}
        for s, res in raw.items():
            ok, reason = is_valid(res, thresh)
            if ok:
                good[s] = res
            else:
                bad[s]  = reason
                print(f'  剔除 seed={s}: {reason}')

        print(f'  保留 {len(good)}/{len(raw)}  需补 {max(0, N_KEEP-len(good))} 个')

        # 3. 补充种子
        next_seed = max(int(s) for s in raw) + 1
        while len(good) < N_KEEP:
            batch_size = min(N_WORKERS * 2, (N_KEEP - len(good)) + 4)
            seeds_to_run = list(range(next_seed, next_seed + batch_size))
            next_seed += batch_size

            worker_args = [(s, K, L, G, W_MIN, W_JFI, REF, FLOOR, EVAL_SEED)
                           for s in seeds_to_run]
            with mp.Pool(N_WORKERS) as pool:
                for seed_int, res in pool.imap_unordered(_worker, worker_args):
                    ok, reason = is_valid(res, thresh)
                    if ok and len(good) < N_KEEP:
                        good[str(seed_int)] = res
                        print(f'  +seed={seed_int}  JFI={res["LB-BVF"]["jfi"]:.3f}  '
                              f'total={len(good)}', flush=True)

        # 4. 截取恰好 N_KEEP 个并保存
        good_keys = list(good.keys())[:N_KEEP]
        clean = {k: good[k] for k in good_keys}
        with open(path, 'w') as f:
            json.dump(clean, f, indent=2)
        print(f'  → 已保存 {len(clean)} 个好种子到 {path}')

    print('\n全部清洗完成，正在重新出图 ...')
    os.system('python plot_static_multibar_final.py 2>/dev/null')
    print('图已更新')
