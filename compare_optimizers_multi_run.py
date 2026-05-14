"""
多次运行优化器对比实验
运行5次实验，保存每次的log，并生成高级可视化结果
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from datetime import datetime
from typing import Dict, Tuple, List

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


def run_single_comparison(run_id, num_iterations=50, output_dir='results'):
    """运行单次对比实验"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f'run_{run_id}_{timestamp}.log')
    
    # 重定向输出到log文件
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w', encoding='utf-8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    logger = Logger(log_file)
    original_stdout = sys.stdout
    sys.stdout = logger
    
    print("="*80)
    print(f"  Optimizer Comparison Experiment - Run {run_id}  ".center(80))
    print("="*80)
    print(f"Timestamp: {timestamp}")
    print(f"Number of iterations/generations: {num_iterations}")
    print(f"nbrOfRealizations: 30")
    print("\n")
    
    # 设置随机种子（每次运行使用不同的种子）
    seed = 42 + run_id
    np.random.seed(seed)
    print(f"Random seed: {seed}\n")
    
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
            'time': 0,
            'UAV_pos': UAV_pos_init.copy()
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
    config_ga['nbrOfRealizations'] = 30  # 保持一致
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
    
    # 打印总结
    print_summary_table(results)
    
    # 恢复输出
    sys.stdout = original_stdout
    logger.log.close()
    
    print(f"✓ Run {run_id} completed. Log saved to: {log_file}")
    
    return results, UE_pos, ground_AP_pos


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


def plot_advanced_comparison(all_results, output_dir='results'):
    """绘制高级对比图（包含提升百分比和UAV轨迹）"""
    
    # 计算5次实验的平均结果
    avg_results = {}
    for method in ['Initial', 'VF', 'GA', 'PSO']:
        method_key = method.lower() if method != 'Initial' else 'initial'
        avg_results[method] = {
            'min_rate': np.mean([r[method_key]['min_rate'] for r in all_results]),
            'sum_rate': np.mean([r[method_key]['sum_rate'] for r in all_results]),
            'min_rate_std': np.std([r[method_key]['min_rate'] for r in all_results]),
            'sum_rate_std': np.std([r[method_key]['sum_rate'] for r in all_results]),
        }
    
    # 设置绘图风格
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 颜色方案
    colors = {
        'Initial': '#95a5a6',
        'VF': '#e74c3c',
        'GA': '#3498db',
        'PSO': '#2ecc71'
    }
    
    methods = ['Initial', 'VF', 'GA', 'PSO']
    
    # ============================================================
    # 子图1: 最小速率对比（柱状图 + 误差棒）
    # ============================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    min_rates = [avg_results[m]['min_rate'] for m in methods]
    min_rates_std = [avg_results[m]['min_rate_std'] for m in methods]
    
    bars1 = ax1.bar(methods, min_rates, yerr=min_rates_std, 
                    color=[colors[m] for m in methods], 
                    alpha=0.8, edgecolor='black', linewidth=1.5,
                    capsize=5)
    
    for bar, val in zip(bars1, min_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('Minimum User Rate (Mbps)', fontsize=12, fontweight='bold')
    ax1.set_title('Minimum Rate Comparison (Avg ± Std)', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # ============================================================
    # 子图2: 总速率对比（柱状图 + 误差棒）
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    sum_rates = [avg_results[m]['sum_rate'] for m in methods]
    sum_rates_std = [avg_results[m]['sum_rate_std'] for m in methods]
    
    bars2 = ax2.bar(methods, sum_rates, yerr=sum_rates_std,
                    color=[colors[m] for m in methods], 
                    alpha=0.8, edgecolor='black', linewidth=1.5,
                    capsize=5)
    
    for bar, val in zip(bars2, sum_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('System Sum Rate (Mbps)', fontsize=12, fontweight='bold')
    ax2.set_title('Sum Rate Comparison (Avg ± Std)', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # ============================================================
    # 子图3: 性能提升百分比
    # ============================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    initial_min = avg_results['Initial']['min_rate']
    initial_sum = avg_results['Initial']['sum_rate']
    
    improvement_min = [(avg_results[m]['min_rate'] - initial_min) / initial_min * 100 
                       for m in ['VF', 'GA', 'PSO']]
    improvement_sum = [(avg_results[m]['sum_rate'] - initial_sum) / initial_sum * 100 
                       for m in ['VF', 'GA', 'PSO']]
    
    x = np.arange(3)
    width = 0.35
    
    bars_min = ax3.bar(x - width/2, improvement_min, width, 
                       label='Min Rate Improvement', color='#e74c3c', alpha=0.8)
    bars_sum = ax3.bar(x + width/2, improvement_sum, width, 
                       label='Sum Rate Improvement', color='#3498db', alpha=0.8)
    
    ax3.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Performance Improvement over Initial', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['VF', 'GA', 'PSO'])
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    # 添加数值标签
    for bars in [bars_min, bars_sum]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom' if height >= 0 else 'top', 
                    fontsize=9, fontweight='bold')
    
    # ============================================================
    # 子图4: 最小速率收敛曲线
    # ============================================================
    ax4 = fig.add_subplot(gs[1, :2])
    
    # 使用第一次运行的数据作为代表
    results_sample = all_results[0]
    
    ax4.axhline(y=results_sample['initial']['min_rate'], color=colors['Initial'], 
                linestyle='--', linewidth=2, label='Initial', alpha=0.7)
    
    # VF
    if 'history' in results_sample['VF']:
        iterations_vf = results_sample['VF']['history']['iterations']
        min_rates_vf = results_sample['VF']['history']['min_rates']
        ax4.plot(iterations_vf, min_rates_vf, color=colors['VF'], 
                linewidth=2.5, label='Virtual Force', marker='o', 
                markevery=max(1, len(iterations_vf)//10), markersize=6)
    
    # GA
    if 'history' in results_sample['GA']:
        generations_ga = results_sample['GA']['history']['generations']
        min_rates_ga = results_sample['GA']['history']['best_min_rates']
        ax4.plot(generations_ga, min_rates_ga, color=colors['GA'], 
                linewidth=2.5, label='Genetic Algorithm', marker='s', 
                markevery=max(1, len(generations_ga)//10), markersize=6)
    
    # PSO
    if 'history' in results_sample['PSO']:
        iterations_pso = results_sample['PSO']['history']['iterations']
        min_rates_pso = results_sample['PSO']['history']['min_rates']
        ax4.plot(iterations_pso, min_rates_pso, color=colors['PSO'], 
                linewidth=2.5, label='PSO', marker='^', 
                markevery=max(1, len(iterations_pso)//10), markersize=6)
    
    ax4.set_xlabel('Iteration/Generation', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Minimum Rate (Mbps)', fontsize=12, fontweight='bold')
    ax4.set_title('Minimum Rate Convergence (Sample Run)', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10, loc='lower right')
    ax4.grid(alpha=0.3)
    
    # ============================================================
    # 子图5: UAV轨迹图
    # ============================================================
    ax5 = fig.add_subplot(gs[1, 2])
    
    # 获取UE和ground AP位置
    UE_pos = all_results[0]['UE_pos']
    ground_AP_pos = all_results[0]['ground_AP_pos']
    
    # 绘制UE位置
    ax5.scatter(UE_pos[:, 0], UE_pos[:, 1], c='gray', s=30, alpha=0.5, 
                marker='o', label='Users')
    
    # 绘制ground AP位置
    ax5.scatter(ground_AP_pos[:, 0], ground_AP_pos[:, 1], c='black', s=200, 
                marker='^', label='Ground APs', edgecolors='white', linewidths=2)
    
    # 绘制初始和最终UAV位置
    UAV_init = results_sample['initial']['UAV_pos']
    UAV_final_vf = results_sample['VF']['UAV_pos']
    
    # 初始位置
    ax5.scatter(UAV_init[:, 0], UAV_init[:, 1], c='blue', s=150, 
                marker='s', label='Initial UAVs', alpha=0.5, edgecolors='black', linewidths=1.5)
    
    # 最终位置（VF）
    ax5.scatter(UAV_final_vf[:, 0], UAV_final_vf[:, 1], c='red', s=150, 
                marker='s', label='Optimized UAVs (VF)', alpha=0.8, edgecolors='black', linewidths=1.5)
    
    # 绘制移动轨迹
    for i in range(len(UAV_init)):
        ax5.annotate('', xy=(UAV_final_vf[i, 0], UAV_final_vf[i, 1]), 
                    xytext=(UAV_init[i, 0], UAV_init[i, 1]),
                    arrowprops=dict(arrowstyle='->', color='purple', lw=2, alpha=0.6))
        ax5.text(UAV_init[i, 0], UAV_init[i, 1], f'{i+1}', 
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    ax5.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    ax5.set_title('UAV Movement Trajectory', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9, loc='upper right')
    ax5.grid(alpha=0.3)
    ax5.set_xlim(0, 1000)
    ax5.set_ylim(0, 1000)
    ax5.set_aspect('equal')
    
    # ============================================================
    # 子图6: 总速率收敛曲线
    # ============================================================
    ax6 = fig.add_subplot(gs[2, :2])
    
    ax6.axhline(y=results_sample['initial']['sum_rate'], color=colors['Initial'], 
                linestyle='--', linewidth=2, label='Initial', alpha=0.7)
    
    # VF
    if 'history' in results_sample['VF']:
        iterations_vf = results_sample['VF']['history']['iterations']
        sum_rates_vf = results_sample['VF']['history']['sum_rates']
        ax6.plot(iterations_vf, sum_rates_vf, color=colors['VF'], 
                linewidth=2.5, label='Virtual Force', marker='o', 
                markevery=max(1, len(iterations_vf)//10), markersize=6)
    
    # GA
    if 'history' in results_sample['GA']:
        generations_ga = results_sample['GA']['history']['generations']
        sum_rates_ga = results_sample['GA']['history']['best_sum_rates']
        ax6.plot(generations_ga, sum_rates_ga, color=colors['GA'], 
                linewidth=2.5, label='Genetic Algorithm', marker='s', 
                markevery=max(1, len(generations_ga)//10), markersize=6)
    
    # PSO
    if 'history' in results_sample['PSO']:
        iterations_pso = results_sample['PSO']['history']['iterations']
        sum_rates_pso = results_sample['PSO']['history']['sum_rates']
        ax6.plot(iterations_pso, sum_rates_pso, color=colors['PSO'], 
                linewidth=2.5, label='PSO', marker='^', 
                markevery=max(1, len(iterations_pso)//10), markersize=6)
    
    ax6.set_xlabel('Iteration/Generation', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Sum Rate (Mbps)', fontsize=12, fontweight='bold')
    ax6.set_title('Sum Rate Convergence (Sample Run)', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=10, loc='lower right')
    ax6.grid(alpha=0.3)
    
    # ============================================================
    # 子图7: 多次实验的箱线图
    # ============================================================
    ax7 = fig.add_subplot(gs[2, 2])
    
    data_min_rates = [
        [r['VF']['min_rate'] for r in all_results],
        [r['GA']['min_rate'] for r in all_results],
        [r['PSO']['min_rate'] for r in all_results]
    ]
    
    bp = ax7.boxplot(data_min_rates, labels=['VF', 'GA', 'PSO'],
                     patch_artist=True, widths=0.6)
    
    # 设置箱线图颜色
    for patch, method in zip(bp['boxes'], ['VF', 'GA', 'PSO']):
        patch.set_facecolor(colors[method])
        patch.set_alpha(0.7)
    
    ax7.set_ylabel('Minimum Rate (Mbps)', fontsize=12, fontweight='bold')
    ax7.set_title('Min Rate Distribution (5 Runs)', fontsize=13, fontweight='bold')
    ax7.grid(axis='y', alpha=0.3)
    
    # 保存图像
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f'advanced_comparison_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Advanced comparison plot saved to: {save_path}")
    
    plt.close()


def main():
    """主函数：运行5次对比实验"""
    
    # 创建results文件夹
    output_dir = '/home/hzl/hyd/virtualForce/results'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("  MULTI-RUN OPTIMIZER COMPARISON  ".center(80))
    print("="*80)
    print(f"\nTotal runs: 5")
    print(f"Iterations per run: 50")
    print(f"nbrOfRealizations: 30")
    print(f"Output directory: {output_dir}")
    print("\n")
    
    all_results = []
    
    for run_id in range(1, 6):
        print(f"\n{'='*80}")
        print(f"  Starting Run {run_id}/5  ".center(80))
        print(f"{'='*80}\n")
        
        results, UE_pos, ground_AP_pos = run_single_comparison(
            run_id=run_id,
            num_iterations=50,
            output_dir=output_dir
        )
        
        # 保存位置信息（第一次运行）
        if run_id == 1:
            results['UE_pos'] = UE_pos
            results['ground_AP_pos'] = ground_AP_pos
        
        all_results.append(results)
        
        print(f"\n{'='*80}")
        print(f"  Run {run_id}/5 Completed  ".center(80))
        print(f"{'='*80}\n")
    
    print("\n" + "="*80)
    print("  ALL RUNS COMPLETED  ".center(80))
    print("="*80)
    
    # 绘制综合对比图
    print("\nGenerating advanced comparison plots...")
    plot_advanced_comparison(all_results, output_dir=output_dir)
    
    print("\n" + "="*80)
    print("  EXPERIMENT COMPLETE  ".center(80))
    print("="*80)
    print(f"\n✅ All logs and plots saved to: {output_dir}")
    print("✅ 5 runs completed successfully!")


if __name__ == "__main__":
    main()

