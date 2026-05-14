"""
对比三个优化器的性能
- Balanced Virtual Force Optimizer V3
- Discrete Genetic Algorithm Optimizer
- Distributed PSO Optimizer
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, Tuple

# 导入三个优化器
from balanced_virtual_force_optimizer_v3 import BalancedVirtualForceOptimizerV3, create_balanced_config
from genetic_algorithm_optimizer_discrete import DiscreteGeneticAlgorithmOptimizer, create_discrete_ga_config
from distributed_pso_optimizer import DistributedPSOOptimizer, create_distributed_pso_config

# 导入信道模型
import functionRlocalscattering
import SpectralEfficiencyDownlink


def compute_initial_performance(optimizer, UE_pos, ground_AP_pos, UAV_pos):
    """计算初始状态的性能"""
    all_AP_pos = np.vstack([ground_AP_pos, UAV_pos])
    
    # 计算信道模型
    H, Hhat, betas = optimizer.compute_channel_model(UE_pos, all_AP_pos)
    
    # 计算AP选择
    mask = optimizer.compute_AP_selection_mask(betas)
    
    # 计算速率
    rates, sum_rate = optimizer.compute_user_rates(UE_pos, all_AP_pos, mask)
    min_rate = rates.min()
    
    return min_rate, sum_rate, rates


def run_comparison(num_iterations=50):
    """运行三个优化器的对比实验"""
    
    print("="*80)
    print("  Optimizer Comparison Experiment  ".center(80))
    print("="*80)
    print(f"\nNumber of iterations/generations: {num_iterations}")
    print("\n")
    
    # 设置随机种子以确保公平比较
    np.random.seed(44)
    
    # ============================================================
    # 1. 初始化相同的场景
    # ============================================================
    print("[1/4] Initializing common scenario...")
    
    # 使用VF优化器生成初始位置
    config_vf = create_balanced_config()
    config_vf['max_iterations'] = num_iterations
    optimizer_vf = BalancedVirtualForceOptimizerV3(config_vf)
    UE_pos, ground_AP_pos, UAV_pos_init = optimizer_vf.initialize_positions()
    
    print(f"   ✓ Users: {len(UE_pos)}")
    print(f"   ✓ Ground APs: {len(ground_AP_pos)}")
    print(f"   ✓ UAVs: {len(UAV_pos_init)}")
    
    # 计算初始性能
    print("\n[Initial Performance]")
    initial_min_rate, initial_sum_rate, initial_rates = compute_initial_performance(
        optimizer_vf, UE_pos, ground_AP_pos, UAV_pos_init
    )
    print(f"   Initial Min Rate: {initial_min_rate:.4f} Mbps")
    print(f"   Initial Sum Rate: {initial_sum_rate:.2f} Mbps")
    print(f"   Initial Mean Rate: {initial_rates.mean():.4f} Mbps")
    print(f"   Initial Std Rate: {initial_rates.std():.4f} Mbps")
    
    # 存储结果
    results = {
        'initial': {
            'min_rate': initial_min_rate,
            'sum_rate': initial_sum_rate,
            'mean_rate': initial_rates.mean(),
            'std_rate': initial_rates.std(),
            'time': 0
        }
    }
    
    # ============================================================
    # 2. Virtual Force Optimizer V3
    # ============================================================
    print("\n" + "="*80)
    print("[2/4] Running Balanced Virtual Force Optimizer V3...")
    print("="*80)
    
    UAV_pos_vf = UAV_pos_init.copy()
    start_time = time.time()
    results_vf = optimizer_vf.optimize(UE_pos, ground_AP_pos, UAV_pos_vf)
    vf_time = time.time() - start_time
    
    results['VF'] = {
        'min_rate': results_vf['final_min_rate'],
        'sum_rate': results_vf['final_sum_rate'],
        'mean_rate': results_vf['final_rates'].mean(),
        'std_rate': results_vf['final_rates'].std(),
        'time': vf_time,
        'history': results_vf['history'],
        'UAV_pos': results_vf['optimized_UAV_pos']
    }
    
    print(f"\n✓ VF Optimization Complete")
    print(f"   Final Min Rate: {results['VF']['min_rate']:.4f} Mbps")
    print(f"   Final Sum Rate: {results['VF']['sum_rate']:.2f} Mbps")
    print(f"   Time: {vf_time:.2f} s")
    
    # ============================================================
    # 3. Discrete Genetic Algorithm
    # ============================================================
    print("\n" + "="*80)
    print("[3/4] Running Discrete Genetic Algorithm Optimizer...")
    print("="*80)
    
    config_ga = create_discrete_ga_config()
    config_ga['max_generations'] = num_iterations
    config_ga['nbrOfRealizations'] = 20  # 保持一致
    optimizer_ga = DiscreteGeneticAlgorithmOptimizer(config_ga)
    
    # 使用相同的UE和ground_AP位置
    optimizer_ga.K = len(UE_pos)
    optimizer_ga.G = len(ground_AP_pos)
    
    start_time = time.time()
    results_ga = optimizer_ga.optimize(UE_pos, ground_AP_pos)
    ga_time = time.time() - start_time
    
    results['GA'] = {
        'min_rate': results_ga['final_min_rate'],
        'sum_rate': results_ga['final_sum_rate'],
        'mean_rate': results_ga['final_rates'].mean(),
        'std_rate': results_ga['final_rates'].std(),
        'time': ga_time,
        'history': results_ga['history'],
        'UAV_pos': results_ga['optimized_UAV_pos']
    }
    
    print(f"\n✓ GA Optimization Complete")
    print(f"   Final Min Rate: {results['GA']['min_rate']:.4f} Mbps")
    print(f"   Final Sum Rate: {results['GA']['sum_rate']:.2f} Mbps")
    print(f"   Time: {ga_time:.2f} s")
    
    # ============================================================
    # 4. Distributed PSO
    # ============================================================
    print("\n" + "="*80)
    print("[4/4] Running Distributed PSO Optimizer...")
    print("="*80)
    
    config_pso = create_distributed_pso_config()
    config_pso['max_iterations'] = num_iterations
    optimizer_pso = DistributedPSOOptimizer(config_pso)
    
    UAV_pos_pso = UAV_pos_init.copy()
    start_time = time.time()
    results_pso = optimizer_pso.optimize(UE_pos, ground_AP_pos, UAV_pos_pso)
    pso_time = time.time() - start_time
    
    results['PSO'] = {
        'min_rate': results_pso['final_min_rate'],
        'sum_rate': results_pso['final_sum_rate'],
        'mean_rate': results_pso['final_mean_rate'],
        'std_rate': results_pso['final_std_rate'],
        'time': pso_time,
        'history': results_pso['history'],
        'UAV_pos': results_pso['optimized_UAV_pos']
    }
    
    print(f"\n✓ PSO Optimization Complete")
    print(f"   Final Min Rate: {results['PSO']['min_rate']:.4f} Mbps")
    print(f"   Final Sum Rate: {results['PSO']['sum_rate']:.2f} Mbps")
    print(f"   Time: {pso_time:.2f} s")
    
    return results, UE_pos, ground_AP_pos


def plot_comparison(results, save_path='optimizer_comparison.png'):
    """绘制对比图"""
    
    # 设置绘图风格 - 使用更通用的样式
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            # 如果seaborn样式都不可用，使用默认样式并手动设置网格
            plt.style.use('default')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 颜色方案
    colors = {
        'Initial': '#95a5a6',
        'VF': '#e74c3c',
        'GA': '#3498db',
        'PSO': '#2ecc71'
    }
    
    # ============================================================
    # 子图1: 最小速率对比（柱状图）
    # ============================================================
    ax1 = axes[0, 0]
    
    methods = ['Initial', 'VF', 'GA', 'PSO']
    min_rates = [
        results['initial']['min_rate'],
        results['VF']['min_rate'],
        results['GA']['min_rate'],
        results['PSO']['min_rate']
    ]
    
    bars1 = ax1.bar(methods, min_rates, color=[colors[m] for m in methods], 
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, val in zip(bars1, min_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Minimum User Rate (Mbps)', fontsize=13, fontweight='bold')
    ax1.set_title('Minimum Rate Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(min_rates) * 1.15)
    
    # ============================================================
    # 子图2: 总速率对比（柱状图）
    # ============================================================
    ax2 = axes[0, 1]
    
    sum_rates = [
        results['initial']['sum_rate'],
        results['VF']['sum_rate'],
        results['GA']['sum_rate'],
        results['PSO']['sum_rate']
    ]
    
    bars2 = ax2.bar(methods, sum_rates, color=[colors[m] for m in methods], 
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, val in zip(bars2, sum_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('System Sum Rate (Mbps)', fontsize=13, fontweight='bold')
    ax2.set_title('Sum Rate Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, max(sum_rates) * 1.15)
    
    # ============================================================
    # 子图3: 最小速率收敛曲线
    # ============================================================
    ax3 = axes[1, 0]
    
    # 绘制初始值水平线
    ax3.axhline(y=results['initial']['min_rate'], color=colors['Initial'], 
                linestyle='--', linewidth=2, label='Initial', alpha=0.7)
    
    # VF
    if 'history' in results['VF']:
        iterations_vf = results['VF']['history']['iterations']
        min_rates_vf = results['VF']['history']['min_rates']
        ax3.plot(iterations_vf, min_rates_vf, color=colors['VF'], 
                linewidth=2.5, label='Virtual Force', marker='o', 
                markevery=max(1, len(iterations_vf)//10), markersize=6)
    
    # GA
    if 'history' in results['GA']:
        generations_ga = results['GA']['history']['generations']
        min_rates_ga = results['GA']['history']['best_min_rates']
        ax3.plot(generations_ga, min_rates_ga, color=colors['GA'], 
                linewidth=2.5, label='Genetic Algorithm', marker='s', 
                markevery=max(1, len(generations_ga)//10), markersize=6)
    
    # PSO
    if 'history' in results['PSO']:
        iterations_pso = results['PSO']['history']['iterations']
        min_rates_pso = results['PSO']['history']['min_rates']
        ax3.plot(iterations_pso, min_rates_pso, color=colors['PSO'], 
                linewidth=2.5, label='PSO', marker='^', 
                markevery=max(1, len(iterations_pso)//10), markersize=6)
    
    ax3.set_xlabel('Iteration/Generation', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Minimum Rate (Mbps)', fontsize=13, fontweight='bold')
    ax3.set_title('Minimum Rate Convergence', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11, loc='lower right')
    ax3.grid(alpha=0.3)
    
    # ============================================================
    # 子图4: 总速率收敛曲线
    # ============================================================
    ax4 = axes[1, 1]
    
    # 绘制初始值水平线
    ax4.axhline(y=results['initial']['sum_rate'], color=colors['Initial'], 
                linestyle='--', linewidth=2, label='Initial', alpha=0.7)
    
    # VF
    if 'history' in results['VF']:
        iterations_vf = results['VF']['history']['iterations']
        sum_rates_vf = results['VF']['history']['sum_rates']
        ax4.plot(iterations_vf, sum_rates_vf, color=colors['VF'], 
                linewidth=2.5, label='Virtual Force', marker='o', 
                markevery=max(1, len(iterations_vf)//10), markersize=6)
    
    # GA
    if 'history' in results['GA']:
        generations_ga = results['GA']['history']['generations']
        sum_rates_ga = results['GA']['history']['best_sum_rates']
        ax4.plot(generations_ga, sum_rates_ga, color=colors['GA'], 
                linewidth=2.5, label='Genetic Algorithm', marker='s', 
                markevery=max(1, len(generations_ga)//10), markersize=6)
    
    # PSO
    if 'history' in results['PSO']:
        iterations_pso = results['PSO']['history']['iterations']
        sum_rates_pso = results['PSO']['history']['sum_rates']
        ax4.plot(iterations_pso, sum_rates_pso, color=colors['PSO'], 
                linewidth=2.5, label='PSO', marker='^', 
                markevery=max(1, len(iterations_pso)//10), markersize=6)
    
    ax4.set_xlabel('Iteration/Generation', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Sum Rate (Mbps)', fontsize=13, fontweight='bold')
    ax4.set_title('Sum Rate Convergence', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11, loc='lower right')
    ax4.grid(alpha=0.3)
    
    # ============================================================
    # 调整布局并保存
    # ============================================================
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {save_path}")
    
    return fig


def print_summary_table(results):
    """打印结果汇总表"""
    
    print("\n" + "="*80)
    print("  FINAL RESULTS SUMMARY  ".center(80))
    print("="*80)
    
    print(f"\n{'Method':<20} {'Min Rate':<15} {'Sum Rate':<15} {'Mean Rate':<15} {'Time(s)':<10}")
    print("-" * 80)
    
    methods = ['initial', 'VF', 'GA', 'PSO']
    method_names = {
        'initial': 'Initial',
        'VF': 'Virtual Force',
        'GA': 'Genetic Algorithm',
        'PSO': 'PSO'
    }
    
    for method in methods:
        name = method_names[method]
        min_r = results[method]['min_rate']
        sum_r = results[method]['sum_rate']
        mean_r = results[method]['mean_rate']
        t = results[method]['time']
        
        print(f"{name:<20} {min_r:<15.4f} {sum_r:<15.2f} {mean_r:<15.4f} {t:<10.2f}")
    
    print("-" * 80)
    
    # 计算改进百分比
    print("\n" + "="*80)
    print("  IMPROVEMENT OVER INITIAL (%)  ".center(80))
    print("="*80)
    
    print(f"\n{'Method':<20} {'Min Rate':<20} {'Sum Rate':<20}")
    print("-" * 80)
    
    for method in ['VF', 'GA', 'PSO']:
        name = method_names[method]
        
        min_improve = ((results[method]['min_rate'] - results['initial']['min_rate']) / 
                      results['initial']['min_rate'] * 100)
        sum_improve = ((results[method]['sum_rate'] - results['initial']['sum_rate']) / 
                      results['initial']['sum_rate'] * 100)
        
        print(f"{name:<20} {min_improve:>+18.2f}% {sum_improve:>+18.2f}%")
    
    print("-" * 80)


if __name__ == "__main__":
    # 运行对比实验
    results, UE_pos, ground_AP_pos = run_comparison(num_iterations=50)
    
    # 打印汇总表
    print_summary_table(results)
    
    # 绘制对比图
    plot_comparison(results, save_path='/home/hzl/hyd/virtualForce/optimizer_comparison.png')
    
    print("\n" + "="*80)
    print("  COMPARISON COMPLETE  ".center(80))
    print("="*80)
    print("\n✅ All optimizations completed successfully!")
    print("✅ Comparison plot generated!")

