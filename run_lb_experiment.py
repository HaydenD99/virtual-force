"""
负载均衡 BVF 完整对比实验

场景:
  A. 均匀随机分布用户
  B. 热点高密度分布用户 (70% 集中在 2 个热点, 30% 均匀)

系统配置:
  K = 40 UEs, L = 6 UAVs, G = 9 ground APs (3x3), M = 4
  → 平均 ~6.7 users/UAV, 热点场景下部分 UAV 严重过载

对比算法:
  1. LB-BVF (V6 + 负载均衡力) — 本文算法
  2. BVF V6 (无负载均衡) — 消融对比
  3. K-means 部署 (用户聚类 → UAV 放在簇中心)
  4. GA
  5. PSO
  6. NewSSA

关键修复:
  - 最终评估使用固定种子, 消除信道模型随机性导致的评估偏差
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json, os, time, sys
from typing import Dict

from load_balanced_bvf_v6 import LoadBalancedBVF_V6, create_lb_v6_config
from balanced_virtual_force_optimizer_v6 import BalancedVirtualForceOptimizerV6, create_v6_config
from genetic_algorithm_optimizer_discrete import DiscreteGeneticAlgorithmOptimizer, create_discrete_ga_config
from distributed_pso_optimizer import DistributedPSOOptimizer, create_distributed_pso_config
from newssa_optimizer import NewSSAOptimizer

# ── 系统参数 ──
K, L, G = 40, 6, 9
EVAL_SEED = 99999  # 固定种子, 用于所有最终评估的信道计算


# ======================================================================
# 场景生成
# ======================================================================
def generate_positions(scenario: str, config: Dict, seed: int):
    np.random.seed(seed)
    sq = config['square_length']
    h_ue = config.get('UE_height', 1.65)
    h_ap = config.get('ground_AP_height', 15.0)
    h_uav = config.get('UAV_height', 50.0)

    # 地面 AP: 3x3 均匀网格
    g_side = int(np.ceil(np.sqrt(G)))
    spacing = sq / (g_side + 1)
    ground_AP_pos = []
    for i in range(g_side):
        for j in range(g_side):
            if len(ground_AP_pos) >= G:
                break
            ground_AP_pos.append([(i + 1) * spacing, (j + 1) * spacing, h_ap])
    ground_AP_pos = np.array(ground_AP_pos[:G])

    # 用户分布
    if scenario == 'uniform':
        UE_xy = np.random.uniform(50, sq - 50, (K, 2))
    elif scenario == 'hotspot':
        n_hot = int(K * 0.70)
        n_uni = K - n_hot
        centers = [[sq * 0.25, sq * 0.30], [sq * 0.70, sq * 0.75]]
        per_center = n_hot // len(centers)
        hot_pts = []
        for cx, cy in centers:
            pts = np.random.normal(loc=[cx, cy], scale=sq * 0.055, size=(per_center, 2))
            hot_pts.append(pts)
        hot_xy = np.vstack(hot_pts)[:n_hot]
        uni_xy = np.random.uniform(50, sq - 50, (n_uni, 2))
        UE_xy = np.vstack([hot_xy, uni_xy])
        UE_xy = np.clip(UE_xy, 30, sq - 30)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    UE_pos = np.column_stack([UE_xy, np.full(K, h_ue)])

    # UAV 初始: 均匀网格 + 小扰动
    l_side = int(np.ceil(np.sqrt(L)))
    uav_sp = sq / (l_side + 1)
    UAV_pos = []
    for i in range(l_side):
        for j in range(l_side):
            if len(UAV_pos) >= L:
                break
            x = (i + 1) * uav_sp + np.random.uniform(-20, 20)
            y = (j + 1) * uav_sp + np.random.uniform(-20, 20)
            UAV_pos.append([np.clip(x, 60, sq - 60),
                            np.clip(y, 60, sq - 60), h_uav])
    UAV_pos = np.array(UAV_pos[:L])

    return UE_pos, ground_AP_pos, UAV_pos


# ======================================================================
# K-means 部署
# ======================================================================
def kmeans_deploy(UE_pos, num_uav, uav_height, sq_len,
                  max_iter=80, n_init=10):
    """手写 K-means"""
    pts = UE_pos[:, :2]
    n = len(pts)
    best_centers = None
    best_inertia = np.inf

    for _ in range(n_init):
        idx = np.random.choice(n, num_uav, replace=False)
        centers = pts[idx].copy()
        for _ in range(max_iter):
            dists = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centers = np.zeros_like(centers)
            for c in range(num_uav):
                members = pts[labels == c]
                new_centers[c] = members.mean(axis=0) if len(members) > 0 else pts[np.random.randint(n)]
            if np.allclose(centers, new_centers, atol=1e-4):
                break
            centers = new_centers
        inertia = sum(np.linalg.norm(pts[labels == c] - centers[c], axis=1).sum()
                      for c in range(num_uav))
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.copy()

    best_centers = np.clip(best_centers, 50, sq_len - 50)
    return np.column_stack([best_centers, np.full(num_uav, uav_height)])


# ======================================================================
# 确定性评估 (固定种子消除信道随机性)
# ======================================================================
def evaluate_deterministic(evaluator, UE_pos, ground_AP_pos, UAV_pos):
    """
    用固定种子做信道计算, 保证所有算法的最终评估在同一信道实现下比较.
    """
    saved_state = np.random.get_state()
    np.random.seed(EVAL_SEED)

    all_AP = np.vstack([ground_AP_pos, UAV_pos])
    _, _, betas = evaluator.compute_channel_model(UE_pos, all_AP)
    mask = evaluator.compute_AP_selection_mask(betas)
    rates, sum_rate = evaluator.compute_user_rates(UE_pos, all_AP, mask)

    np.random.set_state(saved_state)

    # 负载指标
    mask_uav = mask[:, G:]
    user_count = mask_uav.sum(axis=0).astype(float)
    throughput = np.zeros(L)
    for l in range(L):
        served = mask_uav[:, l]
        if served.any():
            throughput[l] = rates[served].sum()

    user_load = user_count / max(K, 1)
    backhaul_load = throughput / 500.0
    load_index = np.maximum(user_load, backhaul_load)
    avg_load = load_index.mean() if load_index.mean() > 0 else 1e-6
    overloaded = load_index > avg_load * 1.25

    s = load_index.sum()
    jfi = float(s ** 2 / (L * (load_index ** 2).sum() + 1e-12)) if s > 1e-10 else 1.0

    return {
        'rates': rates,
        'min_rate': float(rates.min()),
        'sum_rate': float(sum_rate),
        'jfi': jfi,
        'overload_count': int(overloaded.sum()),
        'load_index': load_index.tolist(),
        'user_count': user_count.tolist(),
        'max_avg_ratio': float(load_index.max() / (avg_load + 1e-12)),
    }


# ======================================================================
# 主实验
# ======================================================================
def run_experiment(scenario: str, seed: int, output_dir: str):
    # 配置
    base_cfg = create_v6_config()
    base_cfg.update({
        'num_UE': K, 'num_UAV': L, 'num_ground_AP': G,
        'tau_p': K, 'max_iterations': 80,
        'num_serving_APs': 3, 'step_size': 26,
    })

    lb_cfg = create_lb_v6_config()
    lb_cfg.update({
        'num_UE': K, 'num_UAV': L, 'num_ground_AP': G,
        'tau_p': K, 'max_iterations': 80,
        'num_serving_APs': 3, 'step_size': 26,
    })

    # 生成位置
    UE_pos, ground_AP_pos, init_UAV_pos = generate_positions(scenario, base_cfg, seed)

    print(f"\n{'='*90}")
    print(f"  {scenario.upper()} | Seed={seed} | K={K}, L={L}, G={G}")
    print(f"{'='*90}")

    # 统一评估器
    evaluator = BalancedVirtualForceOptimizerV6(base_cfg)

    # 初始评估 (固定种子)
    init_ev = evaluate_deterministic(evaluator, UE_pos, ground_AP_pos, init_UAV_pos)
    print(f"[Initial] Min={init_ev['min_rate']:.2f} | Sum={init_ev['sum_rate']:.1f} | "
          f"JFI={init_ev['jfi']:.4f} | OL={init_ev['overload_count']}/{L}")
    print(f"          User count per UAV: {[int(x) for x in init_ev['user_count']]}")

    results = {
        'scenario': scenario, 'seed': seed,
        'config': {'K': K, 'L': L, 'G': G},
        'initial': {k: v for k, v in init_ev.items() if k != 'rates'},
        'initial_rates': init_ev['rates'].tolist(),
        'algorithms': {}
    }

    def save_alg(name, uav_pos, opt_time, extra=None):
        ev = evaluate_deterministic(evaluator, UE_pos, ground_AP_pos, uav_pos)
        rec = {k: v for k, v in ev.items() if k != 'rates'}
        rec['time'] = opt_time
        rec['rates'] = ev['rates'].tolist()
        if extra:
            rec.update(extra)
        results['algorithms'][name] = rec
        print(f"  => Min={ev['min_rate']:.2f} | Sum={ev['sum_rate']:.1f} | "
              f"JFI={ev['jfi']:.4f} | OL={ev['overload_count']}/{L} | {opt_time:.1f}s")
        return ev

    # === 1. LB-BVF ===
    print(f"\n{'─'*90}\n[1/6] LB-BVF V6 (7 forces + load balance)\n{'─'*90}")
    t0 = time.time()
    lb_opt = LoadBalancedBVF_V6(lb_cfg)
    lb_res = lb_opt.optimize(UE_pos, ground_AP_pos, init_UAV_pos.copy())
    t1 = time.time() - t0
    lb_ev = save_alg('LB-BVF', lb_res['optimized_UAV_pos'], t1, {
        'history_jfis': lb_res['history']['jfis'],
        'history_min_rates': lb_res['history']['min_rates'],
    })

    # === 2. BVF V6 ===
    print(f"\n{'─'*90}\n[2/6] BVF V6 (6 forces, no LB)\n{'─'*90}")
    t0 = time.time()
    v6_opt = BalancedVirtualForceOptimizerV6(base_cfg)
    v6_res = v6_opt.optimize(UE_pos, ground_AP_pos, init_UAV_pos.copy())
    t2 = time.time() - t0
    save_alg('BVF-V6', v6_res['optimized_UAV_pos'], t2)

    # === 3. K-means ===
    print(f"\n{'─'*90}\n[3/6] K-means Deployment\n{'─'*90}")
    t0 = time.time()
    km_uav = kmeans_deploy(UE_pos, L, base_cfg['UAV_height'], base_cfg['square_length'])
    t3 = time.time() - t0
    save_alg('K-means', km_uav, t3)

    # === 4. GA ===
    print(f"\n{'─'*90}\n[4/6] Discrete GA\n{'─'*90}")
    ga_cfg = create_discrete_ga_config()
    ga_cfg.update({'num_UE': K, 'num_UAV': L, 'num_ground_AP': G,
                   'tau_p': K, 'num_serving_APs': 3})
    t0 = time.time()
    ga_opt = DiscreteGeneticAlgorithmOptimizer(ga_cfg)
    ga_res = ga_opt.optimize(UE_pos, ground_AP_pos)
    t4 = time.time() - t0
    save_alg('GA', ga_res['optimized_UAV_pos'], t4)

    # === 5. PSO ===
    print(f"\n{'─'*90}\n[5/6] Distributed PSO\n{'─'*90}")
    pso_cfg = create_distributed_pso_config()
    pso_cfg.update({'num_UE': K, 'num_UAV': L, 'num_ground_AP': G,
                    'tau_p': K, 'num_serving_APs': 3})
    t0 = time.time()
    pso_opt = DistributedPSOOptimizer(pso_cfg)
    pso_res = pso_opt.optimize(UE_pos, ground_AP_pos, init_UAV_pos.copy())
    t5 = time.time() - t0
    save_alg('PSO', pso_res['optimized_UAV_pos'], t5)

    # === 6. NewSSA ===
    print(f"\n{'─'*90}\n[6/6] NewSSA\n{'─'*90}")
    t0 = time.time()
    ssa_opt = NewSSAOptimizer(base_cfg)
    ssa_res = ssa_opt.optimize(UE_pos, ground_AP_pos, init_UAV_pos.copy())
    t6 = time.time() - t0
    save_alg('NewSSA', ssa_res['optimized_UAV_pos'], t6)

    # === 保存 ===
    os.makedirs(output_dir, exist_ok=True)
    json_path = f"{output_dir}/{scenario}_seed{seed}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    plot_results(results, output_dir, scenario, seed)
    return results


# ======================================================================
# 绘图
# ======================================================================
def plot_results(results, output_dir, scenario, seed):
    alg_keys = list(results['algorithms'].keys())
    colors = {
        'LB-BVF': '#1f77b4', 'BVF-V6': '#ff7f0e', 'K-means': '#2ca02c',
        'GA': '#d62728', 'PSO': '#9467bd', 'NewSSA': '#8c564b',
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{scenario.upper()} Scenario (K={K}, L={L}, G={G}, seed={seed})',
                 fontsize=14, fontweight='bold')

    # ──── (a) JFI 柱状图 ────
    ax = axes[0, 0]
    jfis = [results['algorithms'][k]['jfi'] for k in alg_keys]
    init_jfi = results['initial']['jfi']
    x = np.arange(len(alg_keys))
    bars = ax.bar(x, jfis, 0.6,
                  color=[colors.get(k, 'gray') for k in alg_keys],
                  edgecolor='black', linewidth=1.2)
    ax.axhline(y=init_jfi, color='gray', ls='--', lw=1.5,
               label=f'Initial ({init_jfi:.3f})')
    ax.set_xticks(x)
    ax.set_xticklabels(alg_keys, fontsize=9, rotation=15)
    ax.set_ylabel("Jain's Fairness Index", fontsize=11)
    ax.set_title('(a) Load Balance (JFI)', fontsize=13, fontweight='bold')
    ylo = min(min(jfis), init_jfi) - 0.05
    ax.set_ylim([max(0.4, ylo), 1.02])
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, jfis):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # ──── (b) JFI 收敛曲线 ────
    ax = axes[0, 1]
    lb_data = results['algorithms'].get('LB-BVF', {})
    if 'history_jfis' in lb_data:
        jfi_hist = lb_data['history_jfis']
        iters = range(len(jfi_hist))
        ax.plot(iters, jfi_hist, 'b-', linewidth=2.5, label='LB-BVF JFI')
        ax.axhline(y=0.9, color='red', ls=':', lw=1.5, label='Target JFI=0.9')
        ax.axhline(y=init_jfi, color='gray', ls='--', lw=1, alpha=0.7,
                   label=f'Initial JFI={init_jfi:.3f}')

        if 'history_min_rates' in lb_data:
            ax2 = ax.twinx()
            ax2.plot(range(len(lb_data['history_min_rates'])),
                     lb_data['history_min_rates'], 'r-', alpha=0.5, lw=1.5,
                     label='Min Rate (Mbps)')
            ax2.set_ylabel('Min Rate (Mbps)', fontsize=10, color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.legend(loc='center right', fontsize=8)

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('JFI', fontsize=11, color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.set_title('(b) LB-BVF Convergence', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ──── (c) 用户速率 CDF ────
    ax = axes[1, 0]
    for k in alg_keys:
        rates = np.sort(np.array(results['algorithms'][k]['rates']))
        cdf = np.arange(1, len(rates) + 1) / len(rates)
        ax.plot(rates, cdf, linewidth=2, color=colors.get(k, 'gray'), label=k)

    init_rates = np.sort(np.array(results.get('initial_rates', [])))
    if len(init_rates) > 0:
        ax.plot(init_rates, np.arange(1, len(init_rates) + 1) / len(init_rates),
                'k--', linewidth=1.5, alpha=0.5, label='Initial')

    ax.set_xlabel('User Rate (Mbps)', fontsize=11)
    ax.set_ylabel('CDF', fontsize=11)
    ax.set_title('(c) User Rate CDF', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)
    ax.axhline(y=0.1, color='gray', ls=':', lw=1, alpha=0.5)
    ax.text(ax.get_xlim()[0] + 0.5, 0.12, 'Bottom 10%', fontsize=8, color='gray')

    # ──── (d) Min Rate 柱状图 ────
    ax = axes[1, 1]
    min_rates = [results['algorithms'][k]['min_rate'] for k in alg_keys]
    init_min = results['initial']['min_rate']
    bars = ax.bar(x, min_rates, 0.6,
                  color=[colors.get(k, 'gray') for k in alg_keys],
                  edgecolor='black', linewidth=1.2)
    ax.axhline(y=init_min, color='gray', ls='--', lw=1.5,
               label=f'Initial ({init_min:.1f})')
    ax.set_xticks(x)
    ax.set_xticklabels(alg_keys, fontsize=9, rotation=15)
    ax.set_ylabel('Minimum User Rate (Mbps)', fontsize=11)
    ax.set_title('(d) Min Rate (Max-Min Fairness)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, min_rates):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plot_path = f"{output_dir}/{scenario}_seed{seed}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")
    plt.close()


# ======================================================================
if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    out = 'result/lb_experiment'

    for scenario in ['uniform', 'hotspot']:
        run_experiment(scenario, seed, out)

    print(f"\n{'='*90}\nAll experiments completed.\n{'='*90}")
