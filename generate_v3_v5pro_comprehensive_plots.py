import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# 路径设置
RESULT_DIR = 'result/v3_v5pro_batch'
SAVE_DIR = 'result/plots_comprehensive'
os.makedirs(SAVE_DIR, exist_ok=True)

# 实验配置
UAV_CONFIGS = ["6UAV-L3", "9UAV-L4", "12UAV-L5"]
CONFIG_KEYS = ["6uav_L3", "9uav_L4", "12uav_L5"]
SEEDS = list(range(41, 51))
ALGORITHMS = ["V3", "V5Pro", "GA", "PSO", "NewSSA"]
COLORS = ['#7f7f7f', '#d62728', '#1f77b4', '#ff7f0e', '#2ca02c'] # V3灰色, V5Pro红色突出

def load_all_data():
    # 结构: data[config][alg]['min'] -> list of values across seeds
    all_min = {c: {alg: [] for alg in ALGORITHMS + ["initial"]} for c in CONFIG_KEYS}
    all_sum = {c: {alg: [] for alg in ALGORITHMS + ["initial"]} for c in CONFIG_KEYS}
    
    for conf_idx, conf_key in enumerate(CONFIG_KEYS):
        for seed in SEEDS:
            file_path = os.path.join(RESULT_DIR, f"comp_{conf_key}_seed{seed}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for alg in ALGORITHMS + ["initial"]:
                        all_min[conf_key][alg].append(data[alg]['min'])
                        all_sum[conf_key][alg].append(data[alg]['sum'])
            else:
                print(f"⚠️ 缺失文件: {file_path}")
    
    return all_min, all_sum

def plot_comprehensive():
    all_min, all_sum = load_all_data()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 13))
    
    x = np.arange(len(UAV_CONFIGS))
    width = 0.15
    
    # --- 1. Min Rate 统计图 (均值 + 误差棒) ---
    for i, alg in enumerate(ALGORITHMS):
        means = [np.mean(all_min[ck][alg]) for ck in CONFIG_KEYS]
        stds = [np.std(all_min[ck][alg]) for ck in CONFIG_KEYS]
        
        bars = ax1.bar(x + (i - 2) * width, means, width, label=alg, color=COLORS[i], 
                       edgecolor='black', alpha=0.85, yerr=stds, capsize=4)
        
        # 数值标注 (均值)
        for j, m in enumerate(means):
            ax1.text(j + (i - 2) * width, m + stds[j] + 0.2, f'{m:.1f}', 
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_ylabel('Average Min User Rate (Mbps)', fontsize=12, fontweight='bold')
    ax1.set_title('Comprehensive Min Rate Performance (Average over 10 Seeds)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(UAV_CONFIGS)
    ax1.legend(loc='upper left', frameon=True, shadow=True, ncol=3)

    # --- 2. Sum Rate 统计图 ---
    for i, alg in enumerate(ALGORITHMS):
        means = [np.mean(all_sum[ck][alg]) for ck in CONFIG_KEYS]
        stds = [np.std(all_sum[ck][alg]) for ck in CONFIG_KEYS]
        
        bars = ax2.bar(x + (i - 2) * width, means, width, label=alg, color=COLORS[i], 
                       edgecolor='black', alpha=0.85, yerr=stds, capsize=4)
        
        for j, m in enumerate(means):
            ax2.text(j + (i - 2) * width, m + stds[j] + 50, f'{int(m)}', 
                     ha='center', va='bottom', fontsize=9)

    ax2.set_ylabel('Average System Sum Rate (Mbps)', fontsize=12, fontweight='bold')
    ax2.set_title('Comprehensive Sum Rate Performance (Average over 10 Seeds)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(UAV_CONFIGS)
    ax2.legend(loc='upper left', frameon=True, shadow=True, ncol=3)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'comprehensive_v3_v5pro_comparison.png')
    plt.savefig(save_path, dpi=300)
    print(f"✅ 全量统计图已生成: {save_path}")

    # --- 性能分析总结表格 ---
    print("\n" + "="*95)
    print(f"{'Config':<12} | {'Metric':<10} | {'V3 (Old)':<10} | {'GA':<10} | {'PSO':<10} | {'NewSSA':<10} | {'V5Pro (IACF)':<12}")
    print("-" * 95)
    
    for ck, conf_name in zip(CONFIG_KEYS, UAV_CONFIGS):
        # 计算相比 Initial 的平均提升
        ini_m = np.mean(all_min[ck]['initial'])
        ini_s = np.mean(all_sum[ck]['initial'])
        
        m_line = f"{conf_name:<12} | {'MinRate Δ%':<10} "
        s_line = f"{'':<12} | {'SumRate Δ%':<10} "
        
        for alg in ALGORITHMS:
            m_imp = (np.mean(all_min[ck][alg]) - ini_m) / ini_m * 100
            s_imp = (np.mean(all_sum[ck][alg]) - ini_s) / ini_s * 100
            m_line += f"| {m_imp:>8.1f}% "
            s_line += f"| {s_imp:>8.1f}% "
            
        print(m_line)
        print(s_line)
        print("-" * 95)

if __name__ == "__main__":
    plot_comprehensive()
