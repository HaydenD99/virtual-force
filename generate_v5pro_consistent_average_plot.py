"""
根据 v5pro_final_consistent 数据绘制所有 seed 平均后的对比图
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = 'result/v5pro_final_consistent'
OUTPUT_DIR = 'result/v5pro_final_consistent/plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_all_data():
    """加载所有数据并按 UAV 配置分组"""
    data_by_uav = defaultdict(list)
    
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith('.json'):
            continue
        
        parts = filename.replace('.json', '').split('_')
        uav_str = parts[2]
        num_uav = int(uav_str.replace('uav', ''))
        
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        data_by_uav[num_uav].append(data)
    
    return data_by_uav


def compute_averages(data_by_uav):
    """计算每个 UAV 配置下各算法的平均值和标准差"""
    algorithms = ['initial', 'V6', 'GA', 'PSO', 'NewSSA']
    results = {}
    
    for num_uav in sorted(data_by_uav.keys()):
        data_list = data_by_uav[num_uav]
        n_samples = len(data_list)
        
        results[num_uav] = {
            'n_samples': n_samples,
            'min_rate': {},
            'sum_rate': {}
        }
        
        for algo in algorithms:
            min_rates = [d[algo]['min'] for d in data_list]
            sum_rates = [d[algo]['sum'] for d in data_list]
            
            results[num_uav]['min_rate'][algo] = {
                'mean': np.mean(min_rates),
                'std': np.std(min_rates),
                'values': min_rates
            }
            results[num_uav]['sum_rate'][algo] = {
                'mean': np.mean(sum_rates),
                'std': np.std(sum_rates),
                'values': sum_rates
            }
    
    return results


def plot_average_comparison(avg_results):
    """绘制平均结果对比图"""
    
    uav_counts = sorted(avg_results.keys())
    algorithms = ['Initial', 'BVF', 'GA', 'PSO', 'NewSSA']
    algo_keys = ['initial', 'V6', 'GA', 'PSO', 'NewSSA']
    
    colors = {
        'Initial': '#95a5a6',
        'BVF': '#e74c3c',
        'GA': '#3498db',
        'PSO': '#2ecc71',
        'NewSSA': '#f39c12'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    x = np.arange(len(uav_counts))
    width = 0.15
    
    # ============================================================
    # 子图1: 平均 Min Rate 柱状图 (带误差棒)
    # ============================================================
    ax1 = axes[0, 0]
    
    for i, (algo, key) in enumerate(zip(algorithms, algo_keys)):
        means = [avg_results[uav]['min_rate'][key]['mean'] for uav in uav_counts]
        stds = [avg_results[uav]['min_rate'][key]['std'] for uav in uav_counts]
        offset = (i - 2) * width
        
        bars = ax1.bar(x + offset, means, width, label=algo, color=colors[algo],
                      alpha=0.85, edgecolor='black', linewidth=0.8, yerr=stds, capsize=3)
        
        for bar, val in zip(bars, means):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('UAV Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Minimum Rate (Mbps)', fontsize=12, fontweight='bold')
    ax1.set_title('Average Minimum Rate Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{uav} UAVs' for uav in uav_counts])
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # ============================================================
    # 子图2: 平均 Sum Rate 柱状图 (带误差棒)
    # ============================================================
    ax2 = axes[0, 1]
    
    for i, (algo, key) in enumerate(zip(algorithms, algo_keys)):
        means = [avg_results[uav]['sum_rate'][key]['mean'] for uav in uav_counts]
        stds = [avg_results[uav]['sum_rate'][key]['std'] for uav in uav_counts]
        offset = (i - 2) * width
        
        bars = ax2.bar(x + offset, means, width, label=algo, color=colors[algo],
                      alpha=0.85, edgecolor='black', linewidth=0.8, yerr=stds, capsize=3)
        
        for bar, val in zip(bars, means):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 30,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax2.set_xlabel('UAV Configuration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Sum Rate (Mbps)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Sum Rate Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{uav} UAVs' for uav in uav_counts])
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # ============================================================
    # 子图3: 平均 Min Rate 改进百分比
    # ============================================================
    ax3 = axes[1, 0]
    opt_algos = ['BVF', 'GA', 'PSO', 'NewSSA']
    opt_keys = ['V6', 'GA', 'PSO', 'NewSSA']
    width_opt = 0.2
    x_opt = np.arange(len(uav_counts))
    
    for i, (algo, key) in enumerate(zip(opt_algos, opt_keys)):
        improvements = []
        stds = []
        for uav in uav_counts:
            init_vals = avg_results[uav]['min_rate']['initial']['values']
            algo_vals = avg_results[uav]['min_rate'][key]['values']
            
            # 计算每个样本的改进百分比，然后取平均
            imp_pcts = [((a - ini) / ini) * 100 for a, ini in zip(algo_vals, init_vals)]
            improvements.append(np.mean(imp_pcts))
            stds.append(np.std(imp_pcts))
        
        offset = (i - 1.5) * width_opt
        bars = ax3.bar(x_opt + offset, improvements, width_opt, label=algo,
                      color=colors[algo], alpha=0.85, edgecolor='black', linewidth=0.8,
                      yerr=stds, capsize=3)
        
        for bar, val in zip(bars, improvements):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                    f'{val:+.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('UAV Configuration', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Avg Min Rate Improvement (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Average Minimum Rate Improvement over Initial', fontsize=13, fontweight='bold')
    ax3.set_xticks(x_opt)
    ax3.set_xticklabels([f'{uav} UAVs' for uav in uav_counts])
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # ============================================================
    # 子图4: 平均 Sum Rate 改进百分比
    # ============================================================
    ax4 = axes[1, 1]
    
    for i, (algo, key) in enumerate(zip(opt_algos, opt_keys)):
        improvements = []
        stds = []
        for uav in uav_counts:
            init_vals = avg_results[uav]['sum_rate']['initial']['values']
            algo_vals = avg_results[uav]['sum_rate'][key]['values']
            
            imp_pcts = [((a - ini) / ini) * 100 for a, ini in zip(algo_vals, init_vals)]
            improvements.append(np.mean(imp_pcts))
            stds.append(np.std(imp_pcts))
        
        offset = (i - 1.5) * width_opt
        bars = ax4.bar(x_opt + offset, improvements, width_opt, label=algo,
                      color=colors[algo], alpha=0.85, edgecolor='black', linewidth=0.8,
                      yerr=stds, capsize=3)
        
        for bar, val in zip(bars, improvements):
            y_pos = bar.get_height() + 2 if val >= 0 else bar.get_height() - 5
            ax4.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{val:+.1f}%', ha='center', va='bottom' if val >= 0 else 'top',
                    fontsize=9, fontweight='bold')
    
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('UAV Configuration', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Avg Sum Rate Improvement (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Average Sum Rate Improvement over Initial', fontsize=13, fontweight='bold')
    ax4.set_xticks(x_opt)
    ax4.set_xticklabels([f'{uav} UAVs' for uav in uav_counts])
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_path = os.path.join(OUTPUT_DIR, 'average_comparison.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存平均对比图: {save_path}")
    
    return save_path


def print_average_summary(avg_results):
    """打印平均结果汇总"""
    print("\n" + "="*100)
    print("  AVERAGE PERFORMANCE SUMMARY (All Seeds Combined)  ".center(100))
    print("="*100)
    
    algorithms = ['initial', 'V6', 'GA', 'PSO', 'NewSSA']
    
    print("\n📊 Minimum Rate (Mbps)")
    print("-" * 90)
    print(f"{'UAV':<8} {'N':<6} {'Initial':<14} {'BVF':<14} {'GA':<14} {'PSO':<14} {'NewSSA':<14}")
    print("-" * 90)
    
    for uav in sorted(avg_results.keys()):
        n = avg_results[uav]['n_samples']
        vals = []
        for algo in algorithms:
            mean = avg_results[uav]['min_rate'][algo]['mean']
            std = avg_results[uav]['min_rate'][algo]['std']
            vals.append(f"{mean:.2f}±{std:.1f}")
        print(f"{uav:<8} {n:<6} {vals[0]:<14} {vals[1]:<14} {vals[2]:<14} {vals[3]:<14} {vals[4]:<14}")
    
    print("\n📊 Sum Rate (Mbps)")
    print("-" * 90)
    print(f"{'UAV':<8} {'N':<6} {'Initial':<14} {'BVF':<14} {'GA':<14} {'PSO':<14} {'NewSSA':<14}")
    print("-" * 90)
    
    for uav in sorted(avg_results.keys()):
        n = avg_results[uav]['n_samples']
        vals = []
        for algo in algorithms:
            mean = avg_results[uav]['sum_rate'][algo]['mean']
            std = avg_results[uav]['sum_rate'][algo]['std']
            vals.append(f"{mean:.0f}±{std:.0f}")
        print(f"{uav:<8} {n:<6} {vals[0]:<14} {vals[1]:<14} {vals[2]:<14} {vals[3]:<14} {vals[4]:<14}")
    
    print("\n📊 BVF Improvement over Initial (%)")
    print("-" * 60)
    print(f"{'UAV':<8} {'Min Rate Imp':<20} {'Sum Rate Imp':<20}")
    print("-" * 60)
    
    for uav in sorted(avg_results.keys()):
        init_min = avg_results[uav]['min_rate']['initial']['mean']
        bvf_min = avg_results[uav]['min_rate']['V6']['mean']
        min_imp = ((bvf_min - init_min) / init_min) * 100
        
        init_sum = avg_results[uav]['sum_rate']['initial']['mean']
        bvf_sum = avg_results[uav]['sum_rate']['V6']['mean']
        sum_imp = ((bvf_sum - init_sum) / init_sum) * 100
        
        print(f"{uav:<8} {min_imp:+.1f}%{'':<15} {sum_imp:+.1f}%")
    
    print("-" * 60)


def main():
    print("📊 计算 V5Pro Consistent 数据的平均值...")
    
    data_by_uav = load_all_data()
    print(f"✓ 数据加载完成:")
    for uav in sorted(data_by_uav.keys()):
        print(f"   {uav} UAV: {len(data_by_uav[uav])} 个样本")
    
    avg_results = compute_averages(data_by_uav)
    
    plot_average_comparison(avg_results)
    
    print_average_summary(avg_results)


if __name__ == "__main__":
    main()
