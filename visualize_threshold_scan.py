"""
可视化阈值百分位扫描结果
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_results():
    """加载所有结果"""
    with open('result/threshold_percentile_scan_seed71.json', 'r') as f:
        threshold_results = json.load(f)
    
    fixed_6 = json.load(open('result/original_6uav_seed71.json'))
    fixed_9_all = json.load(open('result/seeds_66_76_partial_11.json'))
    fixed_9 = fixed_9_all['seed_71']
    fixed_12 = json.load(open('result/original_12uav_seed71.json'))
    
    return threshold_results, {
        '6UAV': fixed_6['VF']['min_rate'],
        '9UAV': fixed_9['VF']['min_rate'],
        '12UAV': fixed_12['VF']['min_rate']
    }

def create_plots():
    """创建可视化图表"""
    
    threshold_results, fixed_results = load_results()
    
    percentiles = [50, 60, 75, 80]
    uav_configs = ['6UAV', '9UAV', '12UAV']
    colors = ['#2E86AB', '#A23B72', '#06A77D']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('阈值百分位参数扫描结果 (种子71)', fontsize=16, fontweight='bold')
    
    # 图1: Min Rate vs 阈值
    ax1 = axes[0, 0]
    for idx, uav_key in enumerate(uav_configs):
        rates = [threshold_results[uav_key][f'p{p}']['VF']['min_rate'] for p in percentiles]
        ax1.plot(percentiles, rates, 'o-', color=colors[idx], label=uav_key, 
                linewidth=2, markersize=8, alpha=0.8)
        # 固定选择基准线
        ax1.axhline(y=fixed_results[uav_key], color=colors[idx], linestyle='--', 
                   alpha=0.5, linewidth=1.5)
    
    ax1.set_xlabel('阈值百分位 (%)', fontsize=12)
    ax1.set_ylabel('VF最小速率 (Mbps)', fontsize=12)
    ax1.set_title('最小速率 vs 阈值百分位', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(percentiles)
    
    # 图2: 相对固定选择的改进
    ax2 = axes[0, 1]
    x = np.arange(len(percentiles))
    width = 0.25
    
    for idx, uav_key in enumerate(uav_configs):
        improvements = []
        for p in percentiles:
            rate = threshold_results[uav_key][f'p{p}']['VF']['min_rate']
            improvement = (rate - fixed_results[uav_key]) / fixed_results[uav_key] * 100
            improvements.append(improvement)
        
        ax2.bar(x + idx*width, improvements, width, label=uav_key, 
               color=colors[idx], alpha=0.8)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.set_xlabel('阈值百分位 (%)', fontsize=12)
    ax2.set_ylabel('相对固定选择的改进 (%)', fontsize=12)
    ax2.set_title('阈值选择 vs 固定选择 (L=3)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f'{p}%' for p in percentiles])
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # 图3: AP连接数 vs 阈值
    ax3 = axes[1, 0]
    for idx, uav_key in enumerate(uav_configs):
        ap_counts = [threshold_results[uav_key][f'p{p}']['VF']['ap_stats']['mean'] 
                    for p in percentiles]
        ax3.plot(percentiles, ap_counts, 's-', color=colors[idx], label=uav_key,
                linewidth=2, markersize=8, alpha=0.8)
        # 固定选择基准线
        ax3.axhline(y=3, color=colors[idx], linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax3.set_xlabel('阈值百分位 (%)', fontsize=12)
    ax3.set_ylabel('平均AP连接数', fontsize=12)
    ax3.set_title('AP连接数 vs 阈值百分位', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(percentiles)
    
    # 图4: 优化提升幅度
    ax4 = axes[1, 1]
    for idx, uav_key in enumerate(uav_configs):
        opt_improvements = []
        for p in percentiles:
            result = threshold_results[uav_key][f'p{p}']
            init = result['initial']['min_rate']
            final = result['VF']['min_rate']
            improvement = (final - init) / init * 100
            opt_improvements.append(improvement)
        
        ax4.plot(percentiles, opt_improvements, '^-', color=colors[idx], label=uav_key,
                linewidth=2, markersize=8, alpha=0.8)
    
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax4.set_xlabel('阈值百分位 (%)', fontsize=12)
    ax4.set_ylabel('VF优化提升 (%)', fontsize=12)
    ax4.set_title('优化提升幅度 vs 阈值百分位', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(percentiles)
    
    plt.tight_layout()
    plt.savefig('result/threshold_percentile_scan_visualization.png', dpi=300, bbox_inches='tight')
    print("\n✓ 可视化图表已保存到: result/threshold_percentile_scan_visualization.png")
    plt.close()
    
    # 打印详细分析
    print("\n" + "="*100)
    print(" 详细分析 ".center(100))
    print("="*100)
    
    for uav_key in uav_configs:
        print(f"\n{uav_key}:")
        print(f"{'阈值':<10} {'Min Rate':<12} {'vs固定':<12} {'AP数':<10} {'优化提升':<12} {'总评'}")
        print("-" * 100)
        
        for p in percentiles:
            result = threshold_results[uav_key][f'p{p}']
            vf_min = result['VF']['min_rate']
            init_min = result['initial']['min_rate']
            vs_fixed = (vf_min - fixed_results[uav_key]) / fixed_results[uav_key] * 100
            opt_imp = (vf_min - init_min) / init_min * 100
            ap_count = result['VF']['ap_stats']['mean']
            
            rating = "✓优" if vs_fixed > 0 else ("~平" if vs_fixed > -1 else "✗差")
            
            print(f"{p}%{'':<7} {vf_min:<12.2f} {vs_fixed:>+10.2f}%  {ap_count:<10.2f} {opt_imp:>+10.2f}%  {rating}")


if __name__ == "__main__":
    create_plots()
