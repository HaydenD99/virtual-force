"""
LB-BVF 三方对比测试:
  V6 (baseline) vs LB-BVF V3-Style (force field) vs LB-BVF (MC search)
评估指标: JointScore = w_min*(min/ref) + w_jfi*JFI_eff
"""
import numpy as np
import time

from balanced_virtual_force_optimizer_v6 import BalancedVirtualForceOptimizerV6, create_v6_config
from load_balanced_bvf_v3style import LoadBalancedBVF_V3Style, create_lb_v3style_config
from load_balanced_bvf_v6 import LoadBalancedBVF_V6, create_lb_v6_config

K, L, G = 40, 6, 9
EVAL_SEED = 99999
BACKHAUL = 500.0
LOAD_THRESH = 1.3
W_MIN = 0.35    # JFI优先: 与 V3-Pro 内部目标一致
W_JFI = 0.65
REF_RATE = 60.0
FLOOR_RATE = 48.0  # 软下限 (0.8 × ref)


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
    UE_xy = np.clip(np.vstack([hot_xy, uni_xy]), 30, sq - 30)
    UE_pos = np.column_stack([UE_xy, np.full(K, h_ue)])

    l_side = int(np.ceil(np.sqrt(L)))
    usp = sq / (l_side + 1)
    uavs = []
    for i in range(l_side):
        for j in range(l_side):
            if len(uavs) >= L:
                break
            x = (i + 1) * usp + np.random.uniform(-15, 15)
            y = (j + 1) * usp + np.random.uniform(-15, 15)
            uavs.append([np.clip(x, 60, sq - 60), np.clip(y, 60, sq - 60), h_uav])
    UAV_pos = np.array(uavs[:L])

    opt_state = np.random.get_state()
    return UE_pos, ground_AP_pos, UAV_pos, opt_state


def eval_deterministic(evaluator, UE_pos, ground_AP_pos, UAV_pos):
    """地面感知确定性评估 + JointScore"""
    state = np.random.get_state()
    np.random.seed(EVAL_SEED)

    all_AP = np.vstack([ground_AP_pos, UAV_pos])
    _, _, betas = evaluator.compute_channel_model(UE_pos, all_AP)
    mask = evaluator.compute_AP_selection_mask(betas)
    rates, sum_rate = evaluator.compute_user_rates(UE_pos, all_AP, mask)

    np.random.set_state(state)

    mask_uav = mask[:, G:]
    mask_ground = mask[:, :G]
    uav_user_count = mask_uav.sum(axis=0).astype(float)

    effective_load = np.zeros(L)
    for l in range(L):
        served = np.where(mask_uav[:, l])[0]
        for k in served:
            uav_beta = betas[k, G + l]
            sg = np.where(mask_ground[k])[0]
            gb = betas[k, sg].sum() if len(sg) > 0 else 0.0
            effective_load[l] += uav_beta / (gb + uav_beta + 1e-12)

    s = effective_load.sum()
    jfi_eff = float(s ** 2 / (L * (effective_load ** 2).sum() + 1e-12)) \
        if s > 1e-10 else 1.0

    s_uc = uav_user_count.sum()
    jfi_uc = float(s_uc ** 2 / (L * (uav_user_count ** 2).sum() + 1e-12)) \
        if s_uc > 1e-10 else 1.0

    raw   = W_MIN * (float(rates.min()) / REF_RATE) + W_JFI * jfi_eff
    if float(rates.min()) < FLOOR_RATE:
        raw = raw * (float(rates.min()) / FLOOR_RATE) ** 2
    joint = raw

    return {
        'rates': rates, 'min_rate': float(rates.min()),
        'sum_rate': float(sum_rate),
        'jfi_eff': jfi_eff, 'jfi_uc': jfi_uc,
        'joint_score': joint,
        'user_count': [int(x) for x in uav_user_count],
    }


def fmt(name, ev):
    return (f"  {name:>14}: Min={ev['min_rate']:>6.2f} | "
            f"JFI_eff={ev['jfi_eff']:.4f} | JFI_uc={ev['jfi_uc']:.4f} | "
            f"Joint={ev['joint_score']:.4f} | Users={ev['user_count']}")


if __name__ == "__main__":
    seeds = [42, 51, 62, 71, 75, 33, 88, 99, 107, 123]
    print("=" * 110)
    print(f"  3-Way Comparison | K={K}, L={L}, G={G} | Hotspot 75%")
    print(f"  JointScore = {W_MIN}*(min/{REF_RATE}) + {W_JFI}*JFI_eff")
    print(f"  V6 vs V3-Style (force field) vs LB-BVF (MC search)")
    print("=" * 110)

    results = []

    for seed in seeds:
        print(f"\n{'─'*110}")
        print(f"  Seed = {seed}")
        print(f"{'─'*110}")

        UE_pos, ground_AP_pos, init_UAV, opt_state = generate_hotspot(seed)

        base_cfg = create_v6_config()
        base_cfg.update({'num_UE': K, 'num_UAV': L, 'num_ground_AP': G,
                         'tau_p': K, 'max_iterations': 80, 'num_serving_APs': 3})

        v3_cfg = create_lb_v3style_config()
        v3_cfg.update({'num_UE': K, 'num_UAV': L, 'num_ground_AP': G,
                       'tau_p': K, 'max_iterations': 80, 'num_serving_APs': 3,
                       'w_min': W_MIN, 'w_jfi': W_JFI,
                       'ref_rate': REF_RATE, 'floor_rate': FLOOR_RATE})

        lb_cfg = create_lb_v6_config()
        lb_cfg.update({'num_UE': K, 'num_UAV': L, 'num_ground_AP': G,
                       'tau_p': K, 'max_iterations': 80, 'num_serving_APs': 3,
                       'w_min': W_MIN, 'w_jfi': W_JFI, 'ref_rate': REF_RATE})

        evaluator = BalancedVirtualForceOptimizerV6(base_cfg)
        init_ev = eval_deterministic(evaluator, UE_pos, ground_AP_pos, init_UAV)
        print(fmt("Initial", init_ev))

        # ── V6 ──
        print("\n  --- BVF V6 ---")
        np.random.set_state(opt_state)
        t0 = time.time()
        v6 = BalancedVirtualForceOptimizerV6(base_cfg)
        v6_res = v6.optimize(UE_pos, ground_AP_pos, init_UAV.copy())
        t_v6 = time.time() - t0
        v6_ev = eval_deterministic(evaluator, UE_pos, ground_AP_pos,
                                   v6_res['optimized_UAV_pos'])
        print(fmt("V6", v6_ev))
        print(f"  {'':>14}  Time={t_v6:.1f}s")

        # ── V3-Style ──
        print("\n  --- LB-BVF V3-Style ---")
        np.random.set_state(opt_state)
        t0 = time.time()
        v3 = LoadBalancedBVF_V3Style(v3_cfg)
        v3_res = v3.optimize(UE_pos, ground_AP_pos, init_UAV.copy())
        t_v3 = time.time() - t0
        v3_ev = eval_deterministic(evaluator, UE_pos, ground_AP_pos,
                                   v3_res['optimized_UAV_pos'])
        print(fmt("V3-Style", v3_ev))
        print(f"  {'':>14}  Time={t_v3:.1f}s")

        # ── LB-BVF (MC) ──
        print("\n  --- LB-BVF (MC) ---")
        np.random.set_state(opt_state)
        t0 = time.time()
        lb = LoadBalancedBVF_V6(lb_cfg)
        lb_res = lb.optimize(UE_pos, ground_AP_pos, init_UAV.copy())
        t_lb = time.time() - t0
        lb_ev = eval_deterministic(evaluator, UE_pos, ground_AP_pos,
                                   lb_res['optimized_UAV_pos'])
        print(fmt("LB-BVF-MC", lb_ev))
        print(f"  {'':>14}  Time={t_lb:.1f}s")

        # ── Δ vs V6 ──
        d3_joint = v3_ev['joint_score'] - v6_ev['joint_score']
        d3_min   = v3_ev['min_rate'] - v6_ev['min_rate']
        d3_jfi   = v3_ev['jfi_eff'] - v6_ev['jfi_eff']
        dm_joint = lb_ev['joint_score'] - v6_ev['joint_score']
        dm_min   = lb_ev['min_rate'] - v6_ev['min_rate']
        dm_jfi   = lb_ev['jfi_eff'] - v6_ev['jfi_eff']

        print(f"\n  Δ vs V6  [V3-Style]:  Joint={d3_joint:+.4f}  "
              f"min={d3_min:+.2f}  JFI={d3_jfi:+.4f}")
        print(f"  Δ vs V6  [LB-MC]:     Joint={dm_joint:+.4f}  "
              f"min={dm_min:+.2f}  JFI={dm_jfi:+.4f}")

        def verdict(dj): return "OK" if dj > 0 else ("--" if dj == 0 else "!!")
        print(f"  V3-Style: [{verdict(d3_joint)}]  LB-MC: [{verdict(dm_joint)}]")

        results.append({
            'seed': seed,
            'v6_joint': v6_ev['joint_score'],
            'v6_min': v6_ev['min_rate'], 'v6_jfi': v6_ev['jfi_eff'],
            'v3_joint': v3_ev['joint_score'],
            'v3_min': v3_ev['min_rate'], 'v3_jfi': v3_ev['jfi_eff'],
            'lb_joint': lb_ev['joint_score'],
            'lb_min': lb_ev['min_rate'], 'lb_jfi': lb_ev['jfi_eff'],
            'd3_joint': d3_joint, 'd3_min': d3_min, 'd3_jfi': d3_jfi,
            'dm_joint': dm_joint, 'dm_min': dm_min, 'dm_jfi': dm_jfi,
        })

    # ── Summary ──
    n3_ok = sum(1 for r in results if r['d3_joint'] > 0)
    nm_ok = sum(1 for r in results if r['dm_joint'] > 0)
    avg3_j = np.mean([r['d3_joint'] for r in results])
    avgm_j = np.mean([r['dm_joint'] for r in results])
    avg3_m = np.mean([r['d3_min'] for r in results])
    avgm_m = np.mean([r['dm_min'] for r in results])
    avg3_f = np.mean([r['d3_jfi'] for r in results])
    avgm_f = np.mean([r['dm_jfi'] for r in results])

    print(f"\n{'='*110}")
    print(f"  SUMMARY (Δ vs V6 baseline)  [{len(seeds)} seeds]")
    print(f"  {'Method':<16} {'SUCCESS':>8} {'Avg ΔJoint':>12} "
          f"{'Avg Δmin':>10} {'Avg ΔJFI_eff':>13}")
    print(f"  {'V3-Style':<16} {n3_ok:>8}/{len(seeds)} {avg3_j:>+12.4f} "
          f"{avg3_m:>+10.2f} {avg3_f:>+13.4f}")
    print(f"  {'LB-BVF-MC':<16} {nm_ok:>8}/{len(seeds)} {avgm_j:>+12.4f} "
          f"{avgm_m:>+10.2f} {avgm_f:>+13.4f}")
    print(f"{'='*110}")
    print(f"\n  {'':>4} {'Seed':>4} | "
          f"{'V6 Joint/min/JFIe':>20} | "
          f"{'V3 Joint/min/JFIe':>20} | "
          f"{'MC Joint/min/JFIe':>20} | "
          f"{'ΔV3':>8} {'ΔMC':>8}")
    for r in results:
        t3 = '+' if r['d3_joint'] > 0 else ('-' if r['d3_joint'] < 0 else '=')
        tm = '+' if r['dm_joint'] > 0 else ('-' if r['dm_joint'] < 0 else '=')
        print(f"  {t3+tm} {r['seed']:>4} | "
              f"{r['v6_joint']:>6.4f}/{r['v6_min']:>5.1f}/{r['v6_jfi']:.3f} | "
              f"{r['v3_joint']:>6.4f}/{r['v3_min']:>5.1f}/{r['v3_jfi']:.3f} | "
              f"{r['lb_joint']:>6.4f}/{r['lb_min']:>5.1f}/{r['lb_jfi']:.3f} | "
              f"{r['d3_joint']:>+8.4f} {r['dm_joint']:>+8.4f}")
    print(f"{'='*110}")
