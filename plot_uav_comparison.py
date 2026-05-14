"""
对比不同UAV数量配置的性能
4个地面AP，分别使用6/9/12个UAV
随机种子51
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 读取数据
def load_data():
    # 9 UAVs数据（从batch2_partial_11.json）
    data_9uav = {
        "initial": {
            "min_rate": 30.593229480255477,
            "sum_rate": 2613.6417776916123,
        },
        "VF": {
            "min_rate": 38.45973938484597,
            "sum_rate": 2747.3818989444912,
        },
        "GA": {
            "min_rate": 34.61927199968264,
            "sum_rate": 2607.085592381084,
        },
        "PSO": {
            "min_rate": 34.884425617079174,
            "sum_rate": 2651.1635258693295,
        },
        "NewSSA": {
            "min_rate": 31.01316688365497,
            "sum_rate": 2502.491888999317,
        }
    }
    
    # 6 UAVs数据
    with open('result/original_6uav_seed51.json', 'r') as f:
        data_6uav = json.load(f)
    
    # 12 UAVs数据
    with open('result/original_12uav_seed51.json', 'r') as f:
        data_12uav = json.load(f)
    
    return {
        '6 UAVs': data_6uav,
        '9 UAVs': data_9uav,
        '12 UAVs': data_12uav
    }

def plot_comparison():
    """绘制对比图"""
    data_all = load_data()
    
    # 准备数据
    uav_configs = ['6 UAVs', '9 UAVs', '12 UAVs']
    methods = ['Initial', 'VF', 'GA', 'PSO', 'NewSSA']
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 颜色方案
    colors = {
        'Initial': '#95a5a6',
        'VF': '#e74c3c',
        'GA': '#3498db',
        'PSO': '#2ecc71',
        'NewSSA': '#f39c12'
    }
    
    # ============================================================
    # 子图1: 最小速率对比（柱状图）
    # ============================================================
    ax1 = axes[0, 0]
    
    x = np.arange(len(uav_configs))
    width = 0.15
    
    for i, method in enumerate(methods):
        if method == 'Initial':
            method_key = 'initial'
        else:
            method_key = method
        
        min_rates = [data_all[config][method_key]['min_rate'] for config in uav_configs]
        offset = (i - 2) * width
        bars = ax1.bar(x + offset, min_rates, width, label=method, 
                      color=colors[method], alpha=0.8, edgecolor='black', linewidth=1)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8)
    
    ax1.set_ylabel('Minimum User Rate (Mbps)', fontsize=13, fontweight='bold')
    ax1.set_title('Minimum Rate Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(uav_configs, fontsize=11)
    ax1.legend(fontsize=10, ncol=2)
    ax1.grid(axis='y', alpha=0.3)
    
    # ============================================================
    # 子图2: 系统和速率对比（柱状图）
    # ============================================================
    ax2 = axes[0, 1]
    
    for i, method in enumerate(methods):
        if method == 'Initial':
            method_key = 'initial'
        else:
            method_key = method
        
        sum_rates = [data_all[config][method_key]['sum_rate'] for config in uav_configs]
        offset = (i - 2) * width
        bars = ax2.bar(x + offset, sum_rates, width, label=method,
                      color=colors[method], alpha=0.8, edgecolor='black', linewidth=1)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=8)
    
    ax2.set_ylabel('System Sum Rate (Mbps)', fontsize=13, fontweight='bold')
    ax2.set_title('Sum Rate Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(uav_configs, fontsize=11)
    ax2.legend(fontsize=10, ncol=2)
    ax2.grid(axis='y', alpha=0.3)
    
    # ============================================================
    # 子图3: 最小速率提升百分比
    # ============================================================
    ax3 = axes[0, 2]
    
    for i, method in enumerate(['VF', 'GA', 'PSO', 'NewSSA']):
        improvements = []
        for config in uav_configs:
            initial = data_all[config]['initial']['min_rate']
            final = data_all[config][method]['min_rate']
            improvement = (final - initial) / initial * 100
            improvements.append(improvement)
        
        offset = (i - 1.5) * width * 1.2
        bars = ax3.bar(x + offset, improvements, width * 1.2, label=method,
                      color=colors[method], alpha=0.8, edgecolor='black', linewidth=1)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    ax3.set_ylabel('Improvement over Initial (%)', fontsize=13, fontweight='bold')
    ax3.set_title('Min Rate Improvement', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(uav_configs, fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # ============================================================
    # 子图4: 最小速率绝对增量
    # ============================================================
    ax4 = axes[1, 0]
    
    for i, method in enumerate(['VF', 'GA', 'PSO', 'NewSSA']):
        increments = []
        for config in uav_configs:
            initial = data_all[config]['initial']['min_rate']
            final = data_all[config][method]['min_rate']
            increment = final - initial
            increments.append(increment)
        
        offset = (i - 1.5) * width * 1.2
        bars = ax4.bar(x + offset, increments, width * 1.2, label=method,
                      color=colors[method], alpha=0.8, edgecolor='black', linewidth=1)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    ax4.set_ylabel('Absolute Improvement (Mbps)', fontsize=13, fontweight='bold')
    ax4.set_title('Min Rate Absolute Gain', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(uav_configs, fontsize=11)
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # ============================================================
    # 子图5: 系统和速率提升百分比
    # ============================================================
    ax5 = axes[1, 1]
    
    for i, method in enumerate(['VF', 'GA', 'PSO', 'NewSSA']):
        improvements = []
        for config in uav_configs:
            initial = data_all[config]['initial']['sum_rate']
            final = data_all[config][method]['sum_rate']
            improvement = (final - initial) / initial * 100
            improvements.append(improvement)
        
        offset = (i - 1.5) * width * 1.2
        bars = ax5.bar(x + offset, improvements, width * 1.2, label=method,
                      color=colors[method], alpha=0.8, edgecolor='black', linewidth=1)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    ax5.set_ylabel('Improvement over Initial (%)', fontsize=13, fontweight='bold')
    ax5.set_title('Sum Rate Improvement', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(uav_configs, fontsize=11)
    ax5.legend(fontsize=10)
    ax5.grid(axis='y', alpha=0.3)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # ============================================================
    # 子图6: 系统和速率绝对增量
    # ============================================================
    ax6 = axes[1, 2]
    
    for i, method in enumerate(['VF', 'GA', 'PSO', 'NewSSA']):
        increments = []
        for config in uav_configs:
            initial = data_all[config]['initial']['sum_rate']
            final = data_all[config][method]['sum_rate']
            increment = final - initial
            increments.append(increment)
        
        offset = (i - 1.5) * width * 1.2
        bars = ax6.bar(x + offset, increments, width * 1.2, label=method,
                      color=colors[method], alpha=0.8, edgecolor='black', linewidth=1)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    ax6.set_ylabel('Absolute Improvement (Mbps)', fontsize=13, fontweight='bold')
    ax6.set_title('Sum Rate Absolute Gain', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(uav_configs, fontsize=11)
    ax6.legend(fontsize=10)
    ax6.grid(axis='y', alpha=0.3)
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('result/uav_configuration_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ 图表已保存到: result/uav_configuration_comparison.png")
    
    return fig

def print_summary():
    """打印数据总结"""
    data_all = load_data()
    
    print("\n" + "="*90)
    print(" UAV配置性能对比总结 ".center(90))
    print("="*90)
    
    print("\n📊 配置说明:")
    print("  • 地面AP数量: 4个（固定）")
    print("  • UAV数量: 6个、9个、12个")
    print("  • 随机种子: 51")
    print("  • Fitness: 原始版本")
    
    for config in ['6 UAVs', '9 UAVs', '12 UAVs']:
        print(f"\n{'='*90}")
        print(f" {config} ".center(90))
        print('='*90)
        
        data = data_all[config]
        
        print(f"\n{'方法':<12} {'最小速率':<15} {'系统和速率':<15} {'Min提升':<12} {'Sum提升':<12}")
        print("-" * 90)
        
        initial_min = data['initial']['min_rate']
        initial_sum = data['initial']['sum_rate']
        
        print(f"{'Initial':<12} {initial_min:<15.2f} {initial_sum:<15.2f} {'-':<12} {'-':<12}")
        
        for method in ['VF', 'GA', 'PSO', 'NewSSA']:
            min_rate = data[method]['min_rate']
            sum_rate = data[method]['sum_rate']
            min_improve = (min_rate - initial_min) / initial_min * 100
            sum_improve = (sum_rate - initial_sum) / initial_sum * 100
            
            print(f"{method:<12} {min_rate:<15.2f} {sum_rate:<15.2f} "
                  f"{min_improve:>+10.2f}% {sum_improve:>+10.2f}%")
    
    # 跨配置对比
    print(f"\n{'='*90}")
    print(" 跨配置对比 ".center(90))
    print('='*90)
    
    print("\n🏆 最小速率排名（按算法）:")
    for method in ['VF', 'GA', 'PSO', 'NewSSA']:
        rates = [(config, data_all[config][method]['min_rate']) 
                 for config in ['6 UAVs', '9 UAVs', '12 UAVs']]
        rates.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n  {method}:")
        for rank, (config, rate) in enumerate(rates, 1):
            print(f"    {rank}. {config}: {rate:.2f} Mbps")
    
    print("\n💡 结论:")
    print("  • UAV数量越多，初始性能和优化后性能都更好")
    print("  • VF算法在所有配置下表现最佳")
    print("  • 12 UAVs配置下，所有算法都能达到更高的绝对速率")

if __name__ == "__main__":
    print("\n" + "="*90)
    print(" UAV配置对比分析 ".center(90))
    print("="*90)
    
    # 打印总结
    print_summary()
    
    # 绘制图表
    print("\n[生成可视化图表...]")
    plot_comparison()
    
    print("\n✅ 分析完成！")
    print("="*90)
