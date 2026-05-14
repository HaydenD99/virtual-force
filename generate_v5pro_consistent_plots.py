"""
根据 v5pro_final_consistent 数据按每个 seed 绘制不同 UAV 配置的对比图
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = 'result/v5pro_final_consistent'
OUTPUT_DIR = 'result/v5pro_final_consistent/plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_all_data():
    """加载所有数据并按 seed 分组"""
    data_by_seed = defaultdict(dict)
    
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith('.json'):
            continue
        
        # 解析文件名: consistent_comp_9uav_seed75.json
        parts = filename.replace('.json', '').split('_')
        uav_str = parts[2]  # e.g., "9uav"
        seed_str = parts[3]  # e.g., "seed75"
        
        num_uav = int(uav_str.replace('uav', ''))
        seed = int(seed_str.replace('seed', ''))
        
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        data_by_seed[seed][num_uav] = data
    
    return data_by_seed


def plot_seed_comparison(seed, uav_data, output_dir):
    """为单个 seed 绘制所有 UAV 配置的对比图"""
    
    uav_counts = sorted(uav_data.keys())
    if len(uav_counts) < 2:
        print(f"  跳过 seed {seed}: 只有 {len(uav_counts)} 个 UAV 配置")
        return
    
    algorithms = ['Initial', 'V5Pro', 'GA', 'PSO', 'NewSSA']
    colors = {
        'Initial': '#95a5a6',
        'V5Pro': '#e74c3c',
        'GA': '#3498db',
        'PSO': '#2ecc71',
        'NewSSA': '#f39c12'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Seed {seed} - Performance Comparison Across UAV Configurations (L=3)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # ============================================================
    # 子图1: Min Rate 柱状图
    # ============================================================
    ax1 = axes[0, 0]
    x = np.arange(len(uav_counts))
    width = 0.15
    
    for i, algo in enumerate(algorithms):
        key = 'initial' if algo == 'Initial' else algo
        min_rates = [uav_data[uav][key]['min'] for uav in uav_counts]
        offset = (i - 2) * width
        bars = ax1.bar(x + offset, min_rates, width, label=algo, color=colors[algo], 
                      alpha=0.85, edgecolor='black', linewidth=0.8)
        
        # 添加数值标签
        for bar, val in zip(bars, min_rates):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax1.set_xlabel('UAV Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Minimum User Rate (Mbps)', fontsize=12, fontweight='bold')
    ax1.set_title('Minimum Rate Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{uav} UAVs' for uav in uav_counts])
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # ============================================================
    # 子图2: Sum Rate 柱状图
    # ============================================================
    ax2 = axes[0, 1]
    
    for i, algo in enumerate(algorithms):
        key = 'initial' if algo == 'Initial' else algo
        sum_rates = [uav_data[uav][key]['sum'] for uav in uav_counts]
        offset = (i - 2) * width
        bars = ax2.bar(x + offset, sum_rates, width, label=algo, color=colors[algo], 
                      alpha=0.85, edgecolor='black', linewidth=0.8)
        
        for bar, val in zip(bars, sum_rates):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.0f}', ha='center', va='bottom', fontsize=7, rotation=45)
    
    ax2.set_xlabel('UAV Configuration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('System Sum Rate (Mbps)', fontsize=12, fontweight='bold')
    ax2.set_title('Sum Rate Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{uav} UAVs' for uav in uav_counts])
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # ============================================================
    # 子图3: Min Rate 改进百分比
    # ============================================================
    ax3 = axes[1, 0]
    
    opt_algos = ['V5Pro', 'GA', 'PSO', 'NewSSA']
    width_opt = 0.2
    x_opt = np.arange(len(uav_counts))
    
    for i, algo in enumerate(opt_algos):
        improvements = []
        for uav in uav_counts:
            init_val = uav_data[uav]['initial']['min']
            algo_val = uav_data[uav][algo]['min']
            imp = ((algo_val - init_val) / init_val) * 100
            improvements.append(imp)
        
        offset = (i - 1.5) * width_opt
        bars = ax3.bar(x_opt + offset, improvements, width_opt, label=algo, 
                      color=colors[algo], alpha=0.85, edgecolor='black', linewidth=0.8)
        
        for bar, val in zip(bars, improvements):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:+.1f}%', ha='center', va='bottom' if val >= 0 else 'top', 
                    fontsize=8, fontweight='bold')
    
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('UAV Configuration', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Min Rate Improvement (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Minimum Rate Improvement over Initial', fontsize=13, fontweight='bold')
    ax3.set_xticks(x_opt)
    ax3.set_xticklabels([f'{uav} UAVs' for uav in uav_counts])
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    
    # ============================================================
    # 子图4: Sum Rate 改进百分比
    # ============================================================
    ax4 = axes[1, 1]
    
    for i, algo in enumerate(opt_algos):
        improvements = []
        for uav in uav_counts:
            init_val = uav_data[uav]['initial']['sum']
            algo_val = uav_data[uav][algo]['sum']
            imp = ((algo_val - init_val) / init_val) * 100
            improvements.append(imp)
        
        offset = (i - 1.5) * width_opt
        bars = ax4.bar(x_opt + offset, improvements, width_opt, label=algo, 
                      color=colors[algo], alpha=0.85, edgecolor='black', linewidth=0.8)
        
        for bar, val in zip(bars, improvements):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:+.1f}%', ha='center', va='bottom' if val >= 0 else 'top', 
                    fontsize=8, fontweight='bold')
    
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('UAV Configuration', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Sum Rate Improvement (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Sum Rate Improvement over Initial', fontsize=13, fontweight='bold')
    ax4.set_xticks(x_opt)
    ax4.set_xticklabels([f'{uav} UAVs' for uav in uav_counts])
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(output_dir, f'seed{seed}_comparison.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 保存: {save_path}")


def generate_summary_table(data_by_seed):
    """生成汇总表格"""
    print("\n" + "="*100)
    print("  V5Pro CONSISTENT COMPARISON SUMMARY  ".center(100))
    print("="*100)
    
    for seed in sorted(data_by_seed.keys()):
        uav_data = data_by_seed[seed]
        uav_counts = sorted(uav_data.keys())
        
        print(f"\n--- Seed {seed} ---")
        print(f"{'UAV':<8} {'Initial':<12} {'V5Pro':<12} {'GA':<12} {'PSO':<12} {'NewSSA':<12} | {'V5Pro Imp':<10}")
        print("-" * 90)
        
        for uav in uav_counts:
            d = uav_data[uav]
            init_min = d['initial']['min']
            v5_min = d['V5Pro']['min']
            ga_min = d['GA']['min']
            pso_min = d['PSO']['min']
            ssa_min = d['NewSSA']['min']
            v5_imp = ((v5_min - init_min) / init_min) * 100
            
            print(f"{uav:<8} {init_min:<12.2f} {v5_min:<12.2f} {ga_min:<12.2f} {pso_min:<12.2f} {ssa_min:<12.2f} | {v5_imp:+.1f}%")


def main():
    print("📊 开始生成 V5Pro Consistent 对比图...")
    
    data_by_seed = load_all_data()
    print(f"✓ 已加载 {len(data_by_seed)} 个 seed 的数据")
    
    for seed in sorted(data_by_seed.keys()):
        print(f"\n📍 处理 Seed {seed}...")
        plot_seed_comparison(seed, data_by_seed[seed], OUTPUT_DIR)
    
    generate_summary_table(data_by_seed)
    
    print(f"\n✅ 所有图表已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
