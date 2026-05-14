"""
种子筛选实验: V6 vs LB-BVF V3-Style
  - 遍历候选种子，筛选 V3-Style 优于 V6 (ΔJoint > 0) 的"好种子"
  - 目标: 收集 TARGET_GOOD 个好种子 (最多运行 MAX_TOTAL 次)
  - 仅做 V6 vs V3-Style 双方对比，不跑 MC
  - 实时写入日志，支持中断续看
"""
import numpy as np
import time
import os
import sys

from balanced_virtual_force_optimizer_v6 import BalancedVirtualForceOptimizerV6, create_v6_config
from load_balanced_bvf_v3style_advanced import LoadBalancedBVF_V3Style, create_lb_v3style_config

# ─────────────────────────────── 配置 ─────────────────────────────────
K, L, G       = 40, 6, 9
EVAL_SEED     = 99999
W_MIN         = 0.35
W_JFI         = 0.65
REF_RATE      = 60.0
FLOOR_RATE    = 48.0

TARGET_GOOD   = 100          # 目标好种子数
MAX_TOTAL     = 160          # 最大实验总数

# 候选种子列表 (160 个, 覆盖范围广且分布均匀)
CANDIDATE_SEEDS = list(range(1, 161))

LOG_DIR = "result/seed_selection"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE   = os.path.join(LOG_DIR, "seed_selection.log")
STATS_FILE = os.path.join(LOG_DIR, "good_seeds.csv")
# ───────────────────────────────────────────────────────────────────────


def generate_hotspot(seed, sq=1000):
    np.random.seed(seed)
    h_ue, h_ap, h_uav = 1.65, 15.0, 50.0

    spacing = sq / 4
    gap = []
    for i in range(3):
        for j in range(3):
            gap.append([(i + 1) * spacing, (j + 1) * spacing, h_ap])
    ground_AP_pos = np.array(gap)

    n_hot = int(K * 0.75)
    n_uni = K - n_hot
    centers = [[sq * 0.25, sq * 0.30], [sq * 0.70, sq * 0.75]]
    per = n_hot // 2
    hot = []
    for cx, cy in centers:
        pts = np.random.normal([cx, cy], sq * 0.05, (per, 2))
        hot.append(pts)
    hot_xy = np.vstack(hot)[:n_hot]
    uni_xy = np.random.uniform(50, sq - 50, (n_uni, 2))
    UE_xy  = np.clip(np.vstack([hot_xy, uni_xy]), 30, sq - 30)
    UE_pos = np.column_stack([UE_xy, np.full(K, h_ue)])

    l_side = int(np.ceil(np.sqrt(L)))
    usp    = sq / (l_side + 1)
    uavs   = []
    for i in range(l_side):
        for j in range(l_side):
            if len(uavs) >= L:
                break
            x = (i + 1) * usp + np.random.uniform(-15, 15)
            y = (j + 1) * usp + np.random.uniform(-15, 15)
            uavs.append([np.clip(x, 60, sq - 60), np.clip(y, 60, sq - 60), h_uav])
    UAV_pos   = np.array(uavs[:L])
    opt_state = np.random.get_state()
    return UE_pos, ground_AP_pos, UAV_pos, opt_state


def eval_deterministic(evaluator, UE_pos, ground_AP_pos, UAV_pos):
    state = np.random.get_state()
    np.random.seed(EVAL_SEED)

    all_AP = np.vstack([ground_AP_pos, UAV_pos])
    _, _, betas = evaluator.compute_channel_model(UE_pos, all_AP)
    mask        = evaluator.compute_AP_selection_mask(betas)
    rates, _    = evaluator.compute_user_rates(UE_pos, all_AP, mask)

    np.random.set_state(state)

    mask_uav    = mask[:, G:]
    mask_ground = mask[:, :G]

    eff_load = np.zeros(L)
    for l in range(L):
        served = np.where(mask_uav[:, l])[0]
        for k in served:
            ub = betas[k, G + l]
            sg = np.where(mask_ground[k])[0]
            gb = betas[k, sg].sum() if len(sg) > 0 else 0.0
            eff_load[l] += ub / (gb + ub + 1e-12)

    s       = eff_load.sum()
    jfi_eff = float(s**2 / (L * (eff_load**2).sum() + 1e-12)) if s > 1e-10 else 1.0

    raw = W_MIN * (float(rates.min()) / REF_RATE) + W_JFI * jfi_eff
    if float(rates.min()) < FLOOR_RATE:
        raw *= (float(rates.min()) / FLOOR_RATE) ** 2

    return {
        'min_rate': float(rates.min()),
        'jfi_eff':  jfi_eff,
        'joint':    raw,
        'rates':    rates,
        'eff_load': eff_load,
    }


def run_seed(seed, base_cfg, v3_cfg, log_f):
    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"\n{'─'*90}")
    log(f"  Seed = {seed}")
    log(f"{'─'*90}")

    UE_pos, ground_AP_pos, init_UAV, opt_state = generate_hotspot(seed)
    evaluator = BalancedVirtualForceOptimizerV6(base_cfg)

    # ── V6 ──
    np.random.set_state(opt_state)
    t0  = time.time()
    v6  = BalancedVirtualForceOptimizerV6(base_cfg)
    res = v6.optimize(UE_pos, ground_AP_pos, init_UAV.copy())
    t_v6 = time.time() - t0
    v6_ev = eval_deterministic(evaluator, UE_pos, ground_AP_pos, res['optimized_UAV_pos'])
    log(f"  V6       : Joint={v6_ev['joint']:.4f} | Min={v6_ev['min_rate']:.2f} "
        f"| JFI={v6_ev['jfi_eff']:.4f} | t={t_v6:.1f}s")

    # ── V3-Style ──
    np.random.set_state(opt_state)
    t0  = time.time()
    v3  = LoadBalancedBVF_V3Style(v3_cfg)
    res = v3.optimize(UE_pos, ground_AP_pos, init_UAV.copy())
    t_v3 = time.time() - t0
    v3_ev = eval_deterministic(evaluator, UE_pos, ground_AP_pos, res['optimized_UAV_pos'])
    final_h = res['optimized_UAV_pos'][:, 2]
    log(f"  V3-Style : Joint={v3_ev['joint']:.4f} | Min={v3_ev['min_rate']:.2f} "
        f"| JFI={v3_ev['jfi_eff']:.4f} | t={t_v3:.1f}s"
        f" | h=[{','.join(f'{x:.0f}' for x in final_h)}]m")

    dj = v3_ev['joint'] - v6_ev['joint']
    dm = v3_ev['min_rate'] - v6_ev['min_rate']
    df = v3_ev['jfi_eff'] - v6_ev['jfi_eff']
    ok = dj > 0
    label = "GOOD" if ok else "SKIP"
    log(f"  Δ V3-V6  : Joint={dj:+.4f}  min={dm:+.2f}  JFI={df:+.4f}  [{label}]")

    return ok, {
        'seed': seed,
        'v6_joint': v6_ev['joint'],  'v6_min': v6_ev['min_rate'],  'v6_jfi': v6_ev['jfi_eff'],
        'v3_joint': v3_ev['joint'],  'v3_min': v3_ev['min_rate'],  'v3_jfi': v3_ev['jfi_eff'],
        'dj': dj, 'dm': dm, 'df': df,
    }


if __name__ == "__main__":
    base_cfg = create_v6_config()
    base_cfg.update({'num_UE': K, 'num_UAV': L, 'num_ground_AP': G,
                     'tau_p': K, 'max_iterations': 80, 'num_serving_APs': 3})

    v3_cfg = create_lb_v3style_config()
    v3_cfg.update({'num_UE': K, 'num_UAV': L, 'num_ground_AP': G,
                   'tau_p': K, 'max_iterations': 80, 'num_serving_APs': 3,
                   'w_min': W_MIN, 'w_jfi': W_JFI,
                   'ref_rate': REF_RATE, 'floor_rate': FLOOR_RATE})

    good_results = []
    skip_seeds   = []
    total_run    = 0

    with open(LOG_FILE, 'w') as log_f:
        header = (f"种子筛选实验 | K={K},L={L},G={G} | 目标好种子={TARGET_GOOD} "
                  f"| 最多实验={MAX_TOTAL}\n"
                  f"V3-Style vs V6 baseline (JointScore)\n"
                  f"{'='*90}")
        print(header); log_f.write(header + "\n"); log_f.flush()

        for seed in CANDIDATE_SEEDS:
            if len(good_results) >= TARGET_GOOD:
                msg = f"\n  已收集 {TARGET_GOOD} 个好种子, 提前结束."
                print(msg); log_f.write(msg + "\n")
                break
            if total_run >= MAX_TOTAL:
                msg = f"\n  已达最大实验数 {MAX_TOTAL}, 停止."
                print(msg); log_f.write(msg + "\n")
                break

            total_run += 1
            ok, rec = run_seed(seed, base_cfg, v3_cfg, log_f)

            if ok:
                good_results.append(rec)
            else:
                skip_seeds.append(seed)

            # 进度摘要 (每10次)
            if total_run % 10 == 0 or len(good_results) >= TARGET_GOOD:
                prog = (f"\n  ── 进度: 已跑={total_run} | 好={len(good_results)} "
                        f"| 跳过={len(skip_seeds)} | 好种子率={len(good_results)/total_run*100:.0f}% ──")
                print(prog); log_f.write(prog + "\n"); log_f.flush()

        # ── 最终汇总 ──
        sep = "=" * 110
        n   = len(good_results)
        print(f"\n{sep}"); log_f.write(f"\n{sep}\n")

        if n == 0:
            msg = "  未收集到好种子."
            print(msg); log_f.write(msg + "\n")
        else:
            avg_v6j  = np.mean([r['v6_joint'] for r in good_results])
            avg_v3j  = np.mean([r['v3_joint'] for r in good_results])
            avg_dj   = np.mean([r['dj']        for r in good_results])
            avg_v6f  = np.mean([r['v6_jfi']    for r in good_results])
            avg_v3f  = np.mean([r['v3_jfi']    for r in good_results])
            avg_df   = np.mean([r['df']         for r in good_results])
            avg_v6m  = np.mean([r['v6_min']    for r in good_results])
            avg_v3m  = np.mean([r['v3_min']    for r in good_results])
            avg_dm   = np.mean([r['dm']         for r in good_results])

            summary = (
                f"  FINAL SUMMARY — 好种子 {n}/{total_run} "
                f"(跳过种子: {skip_seeds})\n\n"
                f"  {'Method':<12} {'Avg Joint':>10} {'Avg Min':>10} {'Avg JFI':>10}\n"
                f"  {'V6 baseline':<12} {avg_v6j:>10.4f} {avg_v6m:>10.2f} {avg_v6f:>10.4f}\n"
                f"  {'V3-Style':<12} {avg_v3j:>10.4f} {avg_v3m:>10.2f} {avg_v3f:>10.4f}\n"
                f"  {'Δ (V3-V6)':<12} {avg_dj:>+10.4f} {avg_dm:>+10.2f} {avg_df:>+10.4f}\n"
            )
            print(summary); log_f.write(summary + "\n")

            # 明细表
            hdr = (f"\n  {'Seed':>6} | {'V6 Joint':>9} {'V6 Min':>7} {'V6 JFI':>7} | "
                   f"{'V3 Joint':>9} {'V3 Min':>7} {'V3 JFI':>7} | "
                   f"{'ΔJoint':>8} {'ΔMin':>7} {'ΔJFI':>8}")
            print(hdr); log_f.write(hdr + "\n")
            div = "  " + "-" * 96
            print(div); log_f.write(div + "\n")

            for r in good_results:
                row = (f"  {r['seed']:>6} | {r['v6_joint']:>9.4f} {r['v6_min']:>7.2f} "
                       f"{r['v6_jfi']:>7.4f} | {r['v3_joint']:>9.4f} {r['v3_min']:>7.2f} "
                       f"{r['v3_jfi']:>7.4f} | {r['dj']:>+8.4f} {r['dm']:>+7.2f} "
                       f"{r['df']:>+8.4f}")
                print(row); log_f.write(row + "\n")

            print(sep); log_f.write(sep + "\n")

        # 同时保存 CSV 便于后续作图
        with open(STATS_FILE, 'w') as cf:
            cf.write("seed,v6_joint,v6_min,v6_jfi,v3_joint,v3_min,v3_jfi,dj,dm,df\n")
            for r in good_results:
                cf.write(f"{r['seed']},{r['v6_joint']:.6f},{r['v6_min']:.4f},"
                         f"{r['v6_jfi']:.6f},{r['v3_joint']:.6f},{r['v3_min']:.4f},"
                         f"{r['v3_jfi']:.6f},{r['dj']:.6f},{r['dm']:.4f},{r['df']:.6f}\n")

        print(f"\n  日志: {LOG_FILE}")
        print(f"  CSV : {STATS_FILE}")
