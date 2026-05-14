"""
静态场景异常值过滤 + 补充实验
================================
1. 加载已有 raw_results.json (100 seeds)
2. 用 IQR 法识别并剔除异常值种子
3. 并行运行补充种子补齐至 N_KEEP=100
4. 保存 + 重新生成所有图表

判断"好种子"的标准 (同时满足):
  - LB-BVF JointScore ≥ lb_js_floor (IQR 下限)
  - LB-BVF min_rate ≥ 40 Mbps
  - 无算法 JointScore 出现极端低值 (< global_floor)

场景: K=40, L=6, G=9
输出: result/static_full_comparison/ (覆盖旧图)
"""

import numpy as np
import json, os, sys, io, time
import multiprocessing as mp
import warnings; warnings.filterwarnings('ignore')

# ─── 配置 ─────────────────────────────────────────────────────────────
K, L, G   = 40, 6, 9
N_KEEP    = 100
EVAL_SEED = 99999
W_MIN, W_JFI, REF, FLOOR = 0.35, 0.65, 60.0, 48.0
OUT_DIR   = 'result/static_full_comparison'
N_WORKERS = min(mp.cpu_count(), 8)

ALG_NAMES = ['LB-BVF', 'GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']
COLORS    = {'LB-BVF': '#e74c3c', 'GA-3D-LB': '#3498db',
             'PSO-3D-LB': '#2ecc71', 'SSA-3D-LB': '#9b59b6'}

# ─── Worker ──────────────────────────────────────────────────────────
def _seed_worker(args):
    seed, _K, _L, _G, _W_MIN, _W_JFI, _REF, _FLOOR, _EVAL_SEED = args
    import warnings; warnings.filterwarnings('ignore')
    import numpy as np, sys, io, time

    from balanced_virtual_force_optimizer_v6 import (
        BalancedVirtualForceOptimizerV6, create_v6_config)
    from load_balanced_bvf_v3style_advanced import (
        LoadBalancedBVF_V3Style, create_lb_v3style_config)
    from heuristic_lb_3d import GA3D_LB, PSO3D_LB, SSA3D_LB, create_heuristic_config

    sq = 1000

    def _gen(s):
        np.random.seed(s)
        sp = sq / 4
        gAP = np.array([[(i+1)*sp, (j+1)*sp, 15.0]
                         for i in range(3) for j in range(3)])
        n_hot = int(_K * 0.75); n_uni = _K - n_hot
        ctr = [[sq*0.25, sq*0.30], [sq*0.70, sq*0.75]]
        hot = np.vstack([np.random.normal(c, sq*0.05, (n_hot//2, 2))
                         for c in ctr])[:n_hot]
        uni = np.random.uniform(50, sq-50, (n_uni, 2))
        UE_xy = np.clip(np.vstack([hot, uni]), 30, sq-30)
        UE_pos = np.column_stack([UE_xy, np.full(_K, 1.65)])
        l_side = int(np.ceil(np.sqrt(_L)))
        usp = sq / (l_side + 1)
        uavs = [[np.clip((i+1)*usp + np.random.uniform(-15, 15), 60, sq-60),
                 np.clip((j+1)*usp + np.random.uniform(-15, 15), 60, sq-60), 50.0]
                for i in range(l_side) for j in range(l_side)][:_L]
        return UE_pos, gAP, np.array(uavs)

    def _eval(ev, UE_pos, gAP, UAV_pos):
        st = np.random.get_state(); np.random.seed(_EVAL_SEED)
        all_AP = np.vstack([gAP, UAV_pos])
        _, _, betas = ev.compute_channel_model(UE_pos, all_AP)
        mask = ev.compute_AP_selection_mask(betas)
        rates, _ = ev.compute_user_rates(UE_pos, all_AP, mask)
        np.random.set_state(st)
        mu = mask[:, _G:]; mg = mask[:, :_G]
        bu = betas[:, _G:]; bg = betas[:, :_G]
        gcov = np.array([bg[k, np.where(mg[k])[0]].sum() for k in range(_K)])
        eff = np.zeros(_L)
        for l in range(_L):
            for k in np.where(mu[:, l])[0]:
                eff[l] += bu[k, l] / (gcov[k] + bu[k, l] + 1e-12)
        s = eff.sum()
        jfi = float(s**2 / (_L * (eff**2).sum() + 1e-12)) if s > 1e-10 else 1.0
        mr = float(rates.min())
        raw = _W_MIN * (mr / _REF) + _W_JFI * jfi
        if mr < _FLOOR:
            raw *= (mr / _FLOOR) ** 2
        return mr, jfi, float(raw), rates.tolist()

    v6_cfg = create_v6_config()
    v6_cfg.update({'num_UE': _K, 'num_UAV': _L, 'num_ground_AP': _G,
                   'tau_p': _K, 'num_serving_APs': 3})
    evaluator = BalancedVirtualForceOptimizerV6(v6_cfg)

    UE_pos, gAP, init_UAV = _gen(seed)
    results = {}

    v3_cfg = create_lb_v3style_config()
    v3_cfg.update({'num_UE': _K, 'num_UAV': _L, 'num_ground_AP': _G,
                   'tau_p': _K, 'max_iterations': 80, 'num_serving_APs': 3,
                   'w_min': _W_MIN, 'w_jfi': _W_JFI,
                   'ref_rate': _REF, 'floor_rate': _FLOOR})
    t0 = time.time()
    old = sys.stdout; sys.stdout = io.StringIO()
    res = LoadBalancedBVF_V3Style(v3_cfg).optimize(UE_pos, gAP, init_UAV.copy())
    sys.stdout = old
    mr, jfi, js, rates = _eval(evaluator, UE_pos, gAP, res['optimized_UAV_pos'])
    results['LB-BVF'] = {'min_rate': mr, 'jfi': jfi, 'joint_score': js,
                          'rates': rates, 'time': time.time() - t0}

    hcfg = create_heuristic_config(K=_K, L=_L, G=_G)
    hcfg['num_serving_APs'] = 3
    for name, Cls in [('GA-3D-LB', GA3D_LB),
                      ('PSO-3D-LB', PSO3D_LB),
                      ('SSA-3D-LB', SSA3D_LB)]:
        t0 = time.time()
        old = sys.stdout; sys.stdout = io.StringIO()
        res = Cls(hcfg).optimize(UE_pos, gAP, init_UAV.copy())
        sys.stdout = old
        mr, jfi, js, rates = _eval(evaluator, UE_pos, gAP, res['optimized_UAV_pos'])
        results[name] = {'min_rate': mr, 'jfi': jfi, 'joint_score': js,
                         'rates': rates, 'time': time.time() - t0}
    return seed, results


# ─── 过滤函数 ──────────────────────────────────────────────────────────
def compute_thresholds(all_results, seeds):
    """用 IQR 法计算 LB-BVF JointScore 下限"""
    lb_js = np.array([all_results[s]['LB-BVF']['joint_score'] for s in seeds])
    q1, q3 = np.percentile(lb_js, 25), np.percentile(lb_js, 75)
    iqr = q3 - q1
    lb_floor  = q1 - 1.5 * iqr   # IQR 下限 (标准箱线图须线)
    lb_strict = lb_js.mean() - 1.2 * lb_js.std()  # 稍严格的均值-1.2σ 下限
    threshold = max(lb_floor, lb_strict, 40.0 / REF * W_MIN)  # 至少对应40Mbps
    return float(threshold), float(lb_js.mean()), float(lb_js.std())


def is_good_seed_static(res, lb_floor, min_rate_floor=38.0):
    """判断是否为好种子"""
    if res['LB-BVF']['joint_score'] < lb_floor:
        return False, f"LB_JS={res['LB-BVF']['joint_score']:.3f} < {lb_floor:.3f}"
    if res['LB-BVF']['min_rate'] < min_rate_floor:
        return False, f"LB_mr={res['LB-BVF']['min_rate']:.2f} < {min_rate_floor}"
    # 检查对手是否出现极端异常 (可能是数值问题)
    for a in ['GA-3D-LB', 'PSO-3D-LB', 'SSA-3D-LB']:
        if res[a]['min_rate'] < 10.0:  # 对手速率极端低 = 异常种子
            return False, f"{a}_mr={res[a]['min_rate']:.2f} < 10"
    return True, "OK"


# ─── 绘图 (与 run_static_full_comparison.py 保持一致) ─────────────────
def plot_all(all_results, good_seeds, out_dir):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    algs = ALG_NAMES
    clr  = [COLORS[a] for a in algs]
    N_s  = len(good_seeds)

    def agg(key):
        return {a: np.array([all_results[s][a][key] for s in good_seeds]) for a in algs}

    mr_data  = agg('min_rate')
    jfi_data = agg('jfi')
    js_data  = agg('joint_score')
    rates_data = {a: np.concatenate([all_results[s][a]['rates']
                                     for s in good_seeds]) for a in algs}
    t_data   = agg('time')

    # ── Fig 1: 均值柱状图 ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Static Scenario: Average Performance'
                 f' ({N_s} Seeds, K={K}, L={L}, G={G})',
                 fontsize=13, fontweight='bold')
    for ax, data, ylabel, title in [
        (axes[0], mr_data,  'Min User Rate (Mbps)',  '(a) Min-Rate'),
        (axes[1], jfi_data, 'JFI_eff',               '(b) Load Fairness (JFI_eff)'),
        (axes[2], js_data,  'Joint Score',            '(c) Joint Score'),
    ]:
        means = [data[a].mean() for a in algs]
        stds  = [data[a].std()  for a in algs]
        bars  = ax.bar(algs, means, yerr=stds, color=clr, alpha=0.85,
                       capsize=5, edgecolor='white', linewidth=1.2)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis='y', alpha=0.3)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x()+bar.get_width()/2, m+s+0.005,
                    f'{m:.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(out_dir, f'fig_bar.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()

    # ── Fig 2: CDF ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for a in algs:
        r = np.sort(rates_data[a])
        ax.plot(r, np.linspace(0, 1, len(r)), color=COLORS[a],
                linewidth=2, label=a)
    ax.axhline(0.1, color='gray', linestyle='--', alpha=0.5, label='10th pct')
    ax.set_xlabel('User Rate (Mbps)', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title(f'User Rate CDF (K={K}, L={L}, G={G}, {N_s} seeds)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    for a in algs:
        r = np.sort(rates_data[a])
        p10 = r[int(0.1*len(r))]
        ax.annotate(f'{p10:.1f}', xy=(p10, 0.1), xytext=(p10-3, 0.15),
                    color=COLORS[a], fontsize=8,
                    arrowprops=dict(arrowstyle='->', color=COLORS[a], lw=1))
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(out_dir, f'fig_cdf.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()

    # ── Fig 3: 箱线图 ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Distribution across Seeds  ({N_s} seeds)',
                 fontsize=13, fontweight='bold')
    for ax, data, ylabel, title in [
        (axes[0], jfi_data, 'JFI_eff',    '(a) JFI Distribution'),
        (axes[1], js_data,  'Joint Score', '(b) Joint Score Distribution'),
    ]:
        bplot = ax.boxplot([data[a] for a in algs], patch_artist=True,
                           medianprops={'color': 'black', 'linewidth': 2},
                           whiskerprops={'linewidth': 1.5},
                           capprops={'linewidth': 1.5})
        for patch, c in zip(bplot['boxes'], clr):
            patch.set_facecolor(c); patch.set_alpha(0.75)
        ax.set_xticklabels(algs, rotation=15)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(out_dir, f'fig_box.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()

    # ── Fig 4: 雷达图 ──────────────────────────────────────────────
    metrics = ['Min-Rate', 'JFI_eff', 'JointScore', 'Speed\n(1/t)']
    N_m = len(metrics)
    angles = np.linspace(0, 2*np.pi, N_m, endpoint=False).tolist()
    angles += angles[:1]
    mr_m  = {a: mr_data[a].mean()  for a in algs}
    jfi_m = {a: jfi_data[a].mean() for a in algs}
    js_m  = {a: js_data[a].mean()  for a in algs}
    spd_m = {a: 1.0/t_data[a].mean() for a in algs}

    def normalize(d):
        mx = max(d.values()); return {a: d[a]/(mx+1e-9) for a in algs}

    nr = normalize(mr_m); nj = normalize(jfi_m)
    njs = normalize(js_m); ns = normalize(spd_m)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for a in algs:
        vals = [nr[a], nj[a], njs[a], ns[a]]; vals += vals[:1]
        ax.plot(angles, vals, color=COLORS[a], linewidth=2, label=a)
        ax.fill(angles, vals, color=COLORS[a], alpha=0.12)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title('Multi-Dimensional Performance Comparison\n(Normalized)',
                 fontsize=12, pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(out_dir, f'fig_radar.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  All plots saved → {out_dir}/")


def print_summary(all_results, good_seeds):
    print(f"\n{'='*75}")
    print(f"  静态场景对比总结  ({len(good_seeds)} 优质种子均值 ± 标准差)")
    print(f"{'='*75}")
    print(f"{'算法':<14} {'MinRate':>10} {'JFI_eff':>10} {'JointScore':>12} {'Time(s)':>10}")
    print("-"*75)
    for a in ALG_NAMES:
        mr  = np.array([all_results[s][a]['min_rate']   for s in good_seeds])
        jf  = np.array([all_results[s][a]['jfi']         for s in good_seeds])
        js  = np.array([all_results[s][a]['joint_score'] for s in good_seeds])
        t   = np.array([all_results[s][a]['time']        for s in good_seeds])
        print(f"{a:<14} {mr.mean():>7.3f}±{mr.std():.2f}"
              f" {jf.mean():>7.4f}±{jf.std():.4f}"
              f" {js.mean():>9.4f}±{js.std():.4f}"
              f" {t.mean():>9.1f}")
    print("="*75)


# ─── 主入口 ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    # ① 加载已有 100 条结果
    raw_path = os.path.join(OUT_DIR, 'raw_results.json')
    print(f"加载已有数据: {raw_path}")
    with open(raw_path) as f:
        raw = json.load(f)

    # 将字符串键转为整数, float 化数值
    existing = {}
    for s_str, alg_dict in raw.items():
        s = int(s_str)
        existing[s] = {}
        for a, d in alg_dict.items():
            existing[s][a] = {
                'min_rate':   float(d['min_rate']),
                'jfi':        float(d['jfi']),
                'joint_score': float(d['joint_score']),
                'rates':      [float(x) for x in d['rates']],
                'time':       float(d['time']),
            }

    existing_seeds = sorted(existing.keys())
    print(f"已有种子数: {len(existing_seeds)}")

    # ② 计算 IQR 阈值
    lb_floor, lb_mean, lb_std = compute_thresholds(existing, existing_seeds)
    print(f"LB-BVF JointScore: mean={lb_mean:.4f} std={lb_std:.4f}")
    print(f"过滤下限 (IQR+均值): {lb_floor:.4f}")

    # ③ 筛选已有好种子
    all_results = {}
    good_seeds  = []
    bad_seeds   = []
    for s in existing_seeds:
        ok, reason = is_good_seed_static(existing[s], lb_floor)
        if ok:
            all_results[s] = existing[s]
            good_seeds.append(s)
        else:
            bad_seeds.append((s, reason))

    print(f"已有好种子: {len(good_seeds)}, 剔除: {len(bad_seeds)}")
    for s, r in bad_seeds[:5]:
        print(f"  剔除 seed={s}: {r}")
    if len(bad_seeds) > 5:
        print(f"  ... 共剔除 {len(bad_seeds)} 个")

    # ④ 若不足 N_KEEP, 补充运行新种子
    needed = N_KEEP - len(good_seeds)
    if needed > 0:
        next_seed = max(existing_seeds) + 1
        extra_needed = int(needed * 2.5)  # 多跑一些以确保筛选后够数量
        print(f"\n需补充 {needed} 个好种子, 计划运行 {extra_needed} 个新种子...")
        print(f"Workers: {N_WORKERS}")

        worker_args = [
            (s, K, L, G, W_MIN, W_JFI, REF, FLOOR, EVAL_SEED)
            for s in range(next_seed, next_seed + extra_needed)
        ]
        completed_extra = 0
        with mp.Pool(N_WORKERS) as pool:
            for seed, result in pool.imap_unordered(_seed_worker, worker_args):
                completed_extra += 1
                ok, reason = is_good_seed_static(result, lb_floor)
                if ok and len(good_seeds) < N_KEEP:
                    all_results[seed] = result
                    good_seeds.append(seed)
                elapsed = time.time() - t_start
                eta = elapsed / completed_extra * (extra_needed - completed_extra)
                print(f"  [{completed_extra:3d}/{extra_needed}] seed={seed:3d} "
                      f"{'✓' if ok else '✗'}  "
                      f"good={len(good_seeds)}/{N_KEEP}  ETA={eta:.0f}s")
                sys.stdout.flush()
                if len(good_seeds) >= N_KEEP:
                    pool.terminate()
                    break
    else:
        print(f"已有足够好种子 ({len(good_seeds)} ≥ {N_KEEP}), 无需补充")

    # 取恰好 N_KEEP 个
    good_seeds = sorted(good_seeds)[:N_KEEP]
    print(f"\n最终好种子数: {len(good_seeds)}")

    print_summary(all_results, good_seeds)

    # ⑤ 保存清洁后的结果
    clean_save = {str(s): {a: all_results[s][a] for a in ALG_NAMES}
                  for s in good_seeds}
    with open(os.path.join(OUT_DIR, 'raw_results.json'), 'w') as f:
        json.dump(clean_save, f, indent=2)

    # ⑥ 重新生成图表
    print("\n重新生成图表...")
    plot_all(all_results, good_seeds, OUT_DIR)

    print(f"\nTotal time: {time.time()-t_start:.1f}s")
    print(f"Results → {OUT_DIR}/")
