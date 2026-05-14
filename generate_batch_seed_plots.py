import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 路径设置
RESULT_DIR = 'result/v3_v5pro_batch'
SAVE_DIR = 'result/plots_per_seed'
os.makedirs(SAVE_DIR, exist_ok=True)

# 实验配置
UAV_CONFIGS = ["6UAV-L3", "9UAV-L4", "12UAV-L5"]
CONFIG_KEYS = ["6uav_L3", "9uav_L4", "12uav_L5"]
SEEDS = list(range(41, 51))
ALGORITHMS = ["V5Pro", "GA", "PSO", "NewSSA"]
COLORS = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c'] # V5Pro红色突出

def generate_plot_for_seed(seed):
    min_rates = {alg: [] for alg in ALGORITHMS + ["initial"]}
    sum_rates = {alg: [] for alg in ALGORITHMS + ["initial"]}
    
    # 加载该种子在三种配置下的数据
    for conf_key in CONFIG_KEYS:
        file_name = f"comp_{conf_key}_seed{seed}.json"
        file_path = os.path.join(RESULT_DIR, file_name)
        
        if not os.path.exists(file_path):
            print(f"⚠️ 种子 {seed} 的文件缺失: {file_name}")
            return
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            for alg in ALGORITHMS + ["initial"]:
                min_rates[alg].append(data[alg]['min'])
                sum_rates[alg].append(data[alg]['sum'])

    # 开始绘图
    x = np.arange(len(UAV_CONFIGS))
    width = 0.18
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 13))
    
    # --- 图1: Min Rate ---
    ax1.plot(x, min_rates["initial"], 'k--', marker='o', label='Initial', alpha=0.7, linewidth=2)
    for i, alg in enumerate(ALGORITHMS):
        ax1.bar(x + (i - 1.5) * width, min_rates[alg], width, label=alg, color=COLORS[i], edgecolor='black', alpha=0.8)
        for j, val in enumerate(min_rates[alg]):
            ax1.text(j + (i - 1.5) * width, val + 0.3, f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    ax1.set_ylabel('Min User Rate (Mbps)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Seed {seed}: Minimum User Rate Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(UAV_CONFIGS)
    ax1.legend(loc='upper left', frameon=True, shadow=True)

    # --- 图2: Sum Rate ---
    ax2.plot(x, sum_rates["initial"], 'k--', marker='o', label='Initial', alpha=0.7, linewidth=2)
    for i, alg in enumerate(ALGORITHMS):
        ax2.bar(x + (i - 1.5) * width, sum_rates[alg], width, label=alg, color=COLORS[i], edgecolor='black', alpha=0.8)
        for j, val in enumerate(sum_rates[alg]):
            ax2.text(j + (i - 1.5) * width, val + 40, f'{int(val)}', ha='center', va='bottom', fontsize=9)

    ax2.set_ylabel('System Sum Rate (Mbps)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Seed {seed}: System Sum Rate Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(UAV_CONFIGS)
    ax2.legend(loc='upper left', frameon=True, shadow=True)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f'seed{seed}_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 种子 {seed} 的对比图已生成: {save_path}")

if __name__ == "__main__":
    for seed in SEEDS:
        generate_plot_for_seed(seed)
    print(f"\n✨ 所有种子的独立对比图已生成在: {SAVE_DIR}")
