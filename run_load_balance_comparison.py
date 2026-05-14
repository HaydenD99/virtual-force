"""
负载均衡对比实验
对比算法：
1. LB-BVF (V5Pro + 负载均衡力)
2. BVF V5Pro (原版，无负载均衡)
3. GA (离散遗传算法)
4. PSO (分布式粒子群)
5. NewSSA (改进麻雀算法)

评估指标：
- 最小用户速率 (min_rate)
- 系统总速率 (sum_rate)
- 负载均衡度 (Jain's Fairness Index)
- 过载 UAV 数量
- 最大负载/平均负载比值
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time
from typing import Dict

from load_balanced_bvf_optimizer import LoadBalancedBVF
from balanced_virtual_force_optimizer_v5_sinr import BalancedVirtualForceOptimizerV5
from balanced_virtual_force_optimizer_v3 import (
    BalancedVirtualForceOptimizerV3, create_balanced_config
)
from genetic_algorithm_optimizer_discrete import (
    DiscreteGeneticAlgorithmOptimizer, create_discrete_ga_config
)
from distributed_pso_optimizer import (
    DistributedPSOOptimizer, create_distributed_pso_config
)
from newssa_optimizer import NewSSAOptimizer


def compute_load_metrics(optimizer, UE_pos, AP_pos, rates, mask,
                         num_ground_ap, num_uav, num_ue,
                         backhaul_capacity=500.0) -> Dict:
    """统一计算负载指标（适用于所有算法）"""
    mask_uav = mask[:, num_ground_ap:]
    
    user_count = mask_uav.sum(axis=0).astype(float)
    throughput = np.zeros(num_uav)
    for l in range(num_uav):
        served = mask_uav[:, l]
        if served.any():
            throughput[l] = rates[served].sum()
    
    user_load = user_count / num_ue
    backhaul_load = throughput / backhaul_capacity
    load_index = np.maximum(user_load, backhaul_load)
    
    avg_load = load_index.mean()
    overloaded = load_index > (avg_load * 1.3)
    
    # JFI
    if load_index.sum() < 1e-6:
        jfi = 1.0
    else:
        jfi = float((load_index.sum() ** 2) / (num_uav * (load_index ** 2).sum() + 1e-12))
    
    return {
        'user_count': user_count.tolist(),
        'throughput': throughput.tolist(),
        'load_index': load_index.tolist(),
        'avg_load': float(avg_load),
        'max_load': float(load_index.max()),
        'jfi': jfi,
        'overload_count': int(overloaded.sum()),
        'max_avg_ratio': float(load_index.max() / (avg_load + 1e-12))
    }


def evaluate_algorithm(name, eval_helper, UE_pos, ground_AP_pos, UAV_pos,
                       num_ground_ap, num_uav, num_ue):
    """统一评估一个算法的最终结果（使用 eval_helper 做信道计算）"""
    all_AP = np.vstack([ground_AP_pos, UAV_pos])
    _, _, betas = eval_helper.compute_channel_model(UE_pos, all_AP)
    mask = eval_helper.compute_AP_selection_mask(betas)
    rates, sum_rate = eval_helper.compute_user_rates(UE_pos, all_AP, mask)
    
    load_metrics = compute_load_metrics(
        eval_helper, UE_pos, all_AP, rates, mask,
        num_ground_ap, num_uav, num_ue)
    
    return rates, sum_rate, load_metrics


def run_load_balance_comparison(seed=42, num_ue=60, num_uav=9, num_ground_ap=4,
                                output_dir='result/load_balance_comparison'):
    np.random.seed(seed)
    
    print("\n" + "=" * 100)
    print(f"  Load Balance Comparison (Seed={seed}, K={num_ue}, L={num_uav}, G={num_ground_ap})".center(100))
    print("=" * 100)
    
    # 基础配置
    config_base = create_balanced_config()
    config_base.update({
        'num_UE': num_ue, 'num_UAV': num_uav, 'num_ground_AP': num_ground_ap,
        'random_seed': seed
    })
    
    # 初始化位置
    v3_helper = BalancedVirtualForceOptimizerV3(config_base)
    UE_pos, ground_AP_pos, initial_UAV_pos = v3_helper.initialize_positions()
    print(f"  Init: {num_ue} UEs, {num_ground_ap} ground APs, {num_uav} UAVs\n")
    
    # 初始评估
    rates_init, sum_rate_init, load_init = evaluate_algorithm(
        'Initial', v3_helper, UE_pos, ground_AP_pos, initial_UAV_pos,
        num_ground_ap, num_uav, num_ue)
    
    print(f"[Initial] Min={rates_init.min():.2f} | Sum={sum_rate_init:.1f} | "
          f"JFI={load_init['jfi']:.4f} | OL={load_init['overload_count']}/{num_uav}")
    
    results = {
        'seed': seed,
        'config': {'num_ue': num_ue, 'num_uav': num_uav, 'num_ground_ap': num_ground_ap},
        'initial': {
            'min_rate': float(rates_init.min()),
            'sum_rate': float(sum_rate_init),
            **load_init
        },
        'algorithms': {}
    }
    
    # === 1. Load-Balanced BVF ===
    print("\n" + "-" * 80)
    print("[1/5] Load-Balanced BVF (V5Pro + Load Balance Force)")
    print("-" * 80)
    
    config_lb = config_base.copy()
    config_lb.update({
        'enable_load_balance': True,
        'K_load': 2e4, 'w_load': 0.08,
        'load_threshold': 1.3,
    })
    
    t0 = time.time()
    lb_opt = LoadBalancedBVF(config_lb)
    lb_res = lb_opt.optimize(UE_pos, ground_AP_pos, initial_UAV_pos.copy())
    t_lb = time.time() - t0
    
    _, _, lb_load = evaluate_algorithm(
        'LB-BVF', lb_opt, UE_pos, ground_AP_pos, lb_res['optimized_UAV_pos'],
        num_ground_ap, num_uav, num_ue)
    
    results['algorithms']['LB-BVF'] = {
        'min_rate': lb_res['final_min_rate'],
        'sum_rate': lb_res['final_sum_rate'],
        **lb_load, 'time': t_lb,
        'history': lb_res['history']
    }
    print(f"  => Min={lb_res['final_min_rate']:.2f} | Sum={lb_res['final_sum_rate']:.1f} | "
          f"JFI={lb_load['jfi']:.4f} | OL={lb_load['overload_count']}/{num_uav} | {t_lb:.1f}s")
    
    # === 2. BVF V5Pro ===
    print("\n" + "-" * 80)
    print("[2/5] BVF V5Pro (Original, no load balance)")
    print("-" * 80)
    
    t0 = time.time()
    v5_opt = BalancedVirtualForceOptimizerV5(config_base)
    v5_res = v5_opt.optimize(UE_pos, ground_AP_pos, initial_UAV_pos.copy())
    t_v5 = time.time() - t0
    
    _, _, v5_load = evaluate_algorithm(
        'V5Pro', v5_opt, UE_pos, ground_AP_pos, v5_res['optimized_UAV_pos'],
        num_ground_ap, num_uav, num_ue)
    
    results['algorithms']['BVF_V5Pro'] = {
        'min_rate': v5_res['final_min_rate'],
        'sum_rate': v5_res['final_sum_rate'],
        **v5_load, 'time': t_v5
    }
    print(f"  => Min={v5_res['final_min_rate']:.2f} | Sum={v5_res['final_sum_rate']:.1f} | "
          f"JFI={v5_load['jfi']:.4f} | OL={v5_load['overload_count']}/{num_uav} | {t_v5:.1f}s")
    
    # === 3. GA ===
    print("\n" + "-" * 80)
    print("[3/5] Discrete GA")
    print("-" * 80)
    
    config_ga = create_discrete_ga_config()
    config_ga.update({
        'num_UE': num_ue, 'num_UAV': num_uav, 'num_ground_AP': num_ground_ap,
        'random_seed': seed
    })
    
    t0 = time.time()
    ga_opt = DiscreteGeneticAlgorithmOptimizer(config_ga)
    ga_res = ga_opt.optimize(UE_pos, ground_AP_pos)
    t_ga = time.time() - t0
    
    _, _, ga_load = evaluate_algorithm(
        'GA', v3_helper, UE_pos, ground_AP_pos, ga_res['optimized_UAV_pos'],
        num_ground_ap, num_uav, num_ue)
    
    results['algorithms']['GA'] = {
        'min_rate': ga_res['final_min_rate'],
        'sum_rate': ga_res['final_sum_rate'],
        **ga_load, 'time': t_ga
    }
    print(f"  => Min={ga_res['final_min_rate']:.2f} | Sum={ga_res['final_sum_rate']:.1f} | "
          f"JFI={ga_load['jfi']:.4f} | OL={ga_load['overload_count']}/{num_uav} | {t_ga:.1f}s")
    
    # === 4. PSO ===
    print("\n" + "-" * 80)
    print("[4/5] Distributed PSO")
    print("-" * 80)
    
    config_pso = create_distributed_pso_config()
    config_pso.update({
        'num_UE': num_ue, 'num_UAV': num_uav, 'num_ground_AP': num_ground_ap,
        'random_seed': seed
    })
    
    t0 = time.time()
    pso_opt = DistributedPSOOptimizer(config_pso)
    pso_res = pso_opt.optimize(UE_pos, ground_AP_pos, initial_UAV_pos.copy())
    t_pso = time.time() - t0
    
    _, _, pso_load = evaluate_algorithm(
        'PSO', v3_helper, UE_pos, ground_AP_pos, pso_res['optimized_UAV_pos'],
        num_ground_ap, num_uav, num_ue)
    
    results['algorithms']['PSO'] = {
        'min_rate': pso_res['final_min_rate'],
        'sum_rate': pso_res['final_sum_rate'],
        **pso_load, 'time': t_pso
    }
    print(f"  => Min={pso_res['final_min_rate']:.2f} | Sum={pso_res['final_sum_rate']:.1f} | "
          f"JFI={pso_load['jfi']:.4f} | OL={pso_load['overload_count']}/{num_uav} | {t_pso:.1f}s")
    
    # === 5. NewSSA ===
    print("\n" + "-" * 80)
    print("[5/5] NewSSA")
    print("-" * 80)
    
    t0 = time.time()
    ssa_opt = NewSSAOptimizer(config_base)
    ssa_res = ssa_opt.optimize(UE_pos, ground_AP_pos, initial_UAV_pos.copy())
    t_ssa = time.time() - t0
    
    _, _, ssa_load = evaluate_algorithm(
        'SSA', v3_helper, UE_pos, ground_AP_pos, ssa_res['optimized_UAV_pos'],
        num_ground_ap, num_uav, num_ue)
    
    results['algorithms']['NewSSA'] = {
        'min_rate': ssa_res['final_min_rate'],
        'sum_rate': ssa_res['final_sum_rate'],
        **ssa_load, 'time': t_ssa
    }
    print(f"  => Min={ssa_res['final_min_rate']:.2f} | Sum={ssa_res['final_sum_rate']:.1f} | "
          f"JFI={ssa_load['jfi']:.4f} | OL={ssa_load['overload_count']}/{num_uav} | {t_ssa:.1f}s")
    
    # === 保存与绘图 ===
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    json_path = f"{output_dir}/load_balance_seed{seed}.json"
    save_results = {k: v for k, v in results.items()}
    for alg_name in save_results.get('algorithms', {}):
        if 'history' in save_results['algorithms'][alg_name]:
            save_results['algorithms'][alg_name] = {
                k: v for k, v in save_results['algorithms'][alg_name].items()
                if k != 'history'
            }
    with open(json_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    plot_load_balance_comparison(results, output_dir, seed)
    
    return results


def plot_load_balance_comparison(results, output_dir, seed):
    """绘制负载均衡对比图"""
    
    alg_keys = ['LB-BVF', 'BVF_V5Pro', 'GA', 'PSO', 'NewSSA']
    labels = ['LB-BVF', 'BVF V5Pro', 'GA', 'PSO', 'NewSSA']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def get_vals(metric):
        return [results['algorithms'][k][metric] for k in alg_keys]
    
    min_rates = get_vals('min_rate')
    sum_rates = get_vals('sum_rate')
    jfis = get_vals('jfi')
    overloads = get_vals('overload_count')
    max_avg_ratios = get_vals('max_avg_ratio')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    x = np.arange(len(labels))
    bar_w = 0.6
    
    # (a) Min Rate
    ax = axes[0, 0]
    init_val = results['initial']['min_rate']
    bars = ax.bar(x, min_rates, bar_w, color=colors, edgecolor='black', linewidth=1.2)
    ax.axhline(y=init_val, color='gray', linestyle='--', linewidth=1.5, label=f'Initial ({init_val:.1f})')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Minimum User Rate (Mbps)', fontsize=11)
    ax.set_title('(a) Minimum User Rate', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, min_rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # (b) Sum Rate
    ax = axes[0, 1]
    init_val = results['initial']['sum_rate']
    bars = ax.bar(x, sum_rates, bar_w, color=colors, edgecolor='black', linewidth=1.2)
    ax.axhline(y=init_val, color='gray', linestyle='--', linewidth=1.5, label=f'Initial ({init_val:.0f})')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('System Sum Rate (Mbps)', fontsize=11)
    ax.set_title('(b) System Sum Rate', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, sum_rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    
    # (c) JFI
    ax = axes[0, 2]
    init_jfi = results['initial']['jfi']
    bars = ax.bar(x, jfis, bar_w, color=colors, edgecolor='black', linewidth=1.2)
    ax.axhline(y=init_jfi, color='gray', linestyle='--', linewidth=1.5, label=f'Initial ({init_jfi:.4f})')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Jain's Fairness Index (Load)", fontsize=11)
    ax.set_title('(c) Load Balance Index (JFI)', fontsize=12, fontweight='bold')
    ymin = min(min(jfis), init_jfi) - 0.02
    ax.set_ylim([max(0.85, ymin), 1.005])
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, jfis):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # (d) Overload Count
    ax = axes[1, 0]
    bars = ax.bar(x, overloads, bar_w, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Overloaded UAVs', fontsize=11)
    ax.set_title('(d) Overloaded UAV Count', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, overloads):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{int(val)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # (e) Max/Avg Load Ratio
    ax = axes[1, 1]
    bars = ax.bar(x, max_avg_ratios, bar_w, color=colors, edgecolor='black', linewidth=1.2)
    ax.axhline(y=1.3, color='red', linestyle='--', linewidth=2, label='Threshold (1.3x)')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Max Load / Avg Load', fontsize=11)
    ax.set_title('(e) Load Imbalance Ratio', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, max_avg_ratios):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # (f) LB-BVF Convergence
    ax = axes[1, 2]
    hist = results['algorithms']['LB-BVF'].get('history')
    if hist:
        iters = range(len(hist['min_rates']))
        ax.plot(iters, hist['min_rates'], 'b-', linewidth=2, label='Min Rate')
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Min Rate (Mbps)', fontsize=11, color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.grid(alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.plot(iters, hist['jfis'], 'r-', linewidth=2, alpha=0.7, label='JFI')
        ax2.set_ylabel("JFI", fontsize=11, color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax.set_title('(f) LB-BVF Convergence', fontsize=12, fontweight='bold')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plot_path = f"{output_dir}/load_balance_seed{seed}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    import sys
    
    seeds = [42, 62, 71] if len(sys.argv) == 1 else [int(s) for s in sys.argv[1:]]
    
    print("=" * 100)
    print(f"  Seeds: {seeds}")
    print(f"  Algorithms: LB-BVF, BVF V5Pro, GA, PSO, NewSSA")
    print("=" * 100)
    
    for seed in seeds:
        run_load_balance_comparison(seed=seed)
    
    print("\nAll experiments completed.")
