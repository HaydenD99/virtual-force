"""
对比固定AP选择 vs 阈值AP选择的性能
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_results(filename):
    """加载结果文件"""
    with open(f'result/{filename}', 'r') as f:
        return json.load(f)

def compare_results():
    """对比固定选择和阈值选择的结果"""
    
    print("\n" + "="*100)
    print(" 固定AP选择 vs 阈值AP选择 性能对比 ".center(100))
    print("="*100)
    
    uav_configs = [6, 9, 12]
    seed = 71
    
    for num_uav in uav_configs:
        total_aps = num_uav + 4
        
        print(f"\n{'='*100}")
        print(f" {num_uav} UAVs ({total_aps} 总AP) ".center(100))
        print('='*100)
        
        # 加载固定选择结果
        try:
            fixed_results = load_results(f'original_{num_uav}uav_seed{seed}.json')
            threshold_results = load_results(f'threshold_based_{num_uav}uav_seed{seed}.json')
            
            print(f"\n{'算法':<15} {'固定选择':<20} {'阈值选择':<20} {'改进':<15} {'AP连接数(阈值)'}")
            print("-" * 100)
            
            initial_fixed = fixed_results['initial']['min_rate']
            initial_threshold = threshold_results['initial']['min_rate']
            
            print(f"{'Initial':<15} {initial_fixed:<20.2f} {initial_threshold:<20.2f} {'-':<15} "
                  f"{threshold_results['initial'].get('ap_stats', {}).get('mean', 3):.2f}")
            
            for method in ['VF', 'GA', 'PSO', 'NewSSA']:
                min_rate_fixed = fixed_results[method]['min_rate']
                min_rate_threshold = threshold_results[method]['min_rate']
                improvement = (min_rate_threshold - min_rate_fixed) / min_rate_fixed * 100
                ap_mean = threshold_results[method].get('ap_stats', {}).get('mean', 3)
                ap_std = threshold_results[method].get('ap_stats', {}).get('std', 0)
                
                print(f"{method:<15} {min_rate_fixed:<20.2f} {min_rate_threshold:<20.2f} "
                      f"{improvement:>+13.2f}%  {ap_mean:.2f}±{ap_std:.2f}")
            
            # 计算平均改进
            improvements = []
            for method in ['VF', 'GA', 'PSO', 'NewSSA']:
                imp = (threshold_results[method]['min_rate'] - fixed_results[method]['min_rate']) / fixed_results[method]['min_rate'] * 100
                improvements.append(imp)
            
            print(f"\n平均改进: {np.mean(improvements):+.2f}% (std: {np.std(improvements):.2f}%)")
            
            # AP选择统计
            if 'ap_stats' in threshold_results['initial']:
                stats = threshold_results['VF']['ap_stats']
                print(f"\n阈值选择 AP连接统计:")
                print(f"  平均: {stats['mean']:.2f} ± {stats['std']:.2f}")
                print(f"  范围: [{stats['min']}, {stats['max']}]")
                print(f"  利用率: {stats['avg_utilization']:.1f}%")
                print(f"  固定选择利用率: {3/total_aps*100:.1f}%")
            
        except FileNotFoundError as e:
            print(f"  文件未找到: {e}")
            print(f"  等待实验完成...")
    
    print("\n" + "="*100)


def create_comparison_plot():
    """创建对比图表"""
    
    uav_configs = [6, 9, 12]
    seed = 71
    methods = ['VF', 'GA', 'PSO', 'NewSSA']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'固定AP选择 vs 阈值AP选择性能对比 (种子{seed})', fontsize=16, fontweight='bold')
    
    for idx, num_uav in enumerate(uav_configs):
        try:
            fixed_results = load_results(f'original_{num_uav}uav_seed{seed}.json')
            threshold_results = load_results(f'threshold_based_{num_uav}uav_seed{seed}.json')
            
            # 最小速率对比
            ax1 = axes[0, idx]
            x = np.arange(len(methods))
            width = 0.35
            
            fixed_rates = [fixed_results[m]['min_rate'] for m in methods]
            threshold_rates = [threshold_results[m]['min_rate'] for m in methods]
            
            ax1.bar(x - width/2, fixed_rates, width, label='固定选择(L=3)', alpha=0.8, color='#95B8D1')
            ax1.bar(x + width/2, threshold_rates, width, label='阈值选择', alpha=0.8, color='#E76F51')
            
            ax1.set_xlabel('算法', fontsize=11)
            ax1.set_ylabel('最小用户速率 (Mbps)', fontsize=11)
            ax1.set_title(f'{num_uav} UAVs - 最小速率对比', fontsize=12, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(methods)
            ax1.legend(fontsize=9)
            ax1.grid(axis='y', alpha=0.3)
            
            # 改进百分比
            ax2 = axes[1, idx]
            improvements = [(threshold_results[m]['min_rate'] - fixed_results[m]['min_rate']) / 
                           fixed_results[m]['min_rate'] * 100 for m in methods]
            
            bars = ax2.bar(methods, improvements, color=colors, alpha=0.8)
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            ax2.set_xlabel('算法', fontsize=11)
            ax2.set_ylabel('性能改进 (%)', fontsize=11)
            ax2.set_title(f'{num_uav} UAVs - 阈值选择相对固定选择的改进', fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            # 在柱状图上标注数值
            for bar, imp in zip(bars, improvements):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{imp:+.1f}%',
                        ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        except FileNotFoundError:
            axes[0, idx].text(0.5, 0.5, f'{num_uav} UAVs\n数据未准备好', 
                            ha='center', va='center', transform=axes[0, idx].transAxes)
            axes[1, idx].text(0.5, 0.5, f'{num_uav} UAVs\n数据未准备好', 
                            ha='center', va='center', transform=axes[1, idx].transAxes)
    
    plt.tight_layout()
    plt.savefig('result/fixed_vs_threshold_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ 对比图已保存到: result/fixed_vs_threshold_comparison.png")
    plt.close()


if __name__ == "__main__":
    compare_results()
    create_comparison_plot()
