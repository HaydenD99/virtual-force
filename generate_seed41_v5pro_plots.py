import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 路径设置
RESULT_DIR = 'result/v3_v5pro_batch'
SAVE_DIR = 'result/plots_seed41'
os.makedirs(SAVE_DIR, exist_ok=True)

# 实验配置
UAV_CONFIGS = ["6UAV-L3", "9UAV-L4", "12UAV-L5"]
FILES = [
    "comp_6uav_L3_seed41.json",
    "comp_9uav_L4_seed41.json",
    "comp_12uav_L5_seed41.json"
]
ALGORITHMS = ["V5Pro", "GA", "PSO", "NewSSA"]
COLORS = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c'] # V5Pro用红色突出

def load_data():
    min_rates = {alg: [] for alg in ALGORITHMS + ["initial"]}
    sum_rates = {alg: [] for alg in ALGORITHMS + ["initial"]}
    
    for file in FILES:
        with open(os.path.join(RESULT_DIR, file), 'r') as f:
            data = json.load(f)
            for alg in ALGORITHMS + ["initial"]:
                min_rates[alg].append(data[alg]['min'])
                sum_rates[alg].append(data[alg]['sum'])
    return min_rates, sum_rates

def plot_comparison():
    min_rates, sum_rates = load_data()
    
    x = np.arange(len(UAV_CONFIGS))
    width = 0.18
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # --- 图1: Min Rate 对比 ---
    # 画初始值作为基准线
    ax1.plot(x, min_rates["initial"], 'k--', marker='o', label='Initial (Unoptimized)', alpha=0.7)
    
    for i, alg in enumerate(ALGORITHMS):
        ax1.bar(x + (i - 1.5) * width, min_rates[alg], width, label=alg, color=COLORS[i], edgecolor='black', alpha=0.8)
        # 添加数值标注
        for j, val in enumerate(min_rates[alg]):
            ax1.text(j + (i - 1.5) * width, val + 0.5, f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    ax1.set_ylabel('Minimum User Rate (Mbps)', fontsize=12, fontweight='bold')
    ax1.set_title('Seed 41: Minimum User Rate Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(UAV_CONFIGS)
    ax1.legend(loc='upper left', frameon=True, shadow=True)
    ax1.set_ylim(0, max([max(v) for v in min_rates.values()]) * 1.2)

    # --- 图2: Sum Rate 对比 ---
    ax2.plot(x, sum_rates["initial"], 'k--', marker='o', label='Initial (Unoptimized)', alpha=0.7)
    
    for i, alg in enumerate(ALGORITHMS):
        ax2.bar(x + (i - 1.5) * width, sum_rates[alg], width, label=alg, color=COLORS[i], edgecolor='black', alpha=0.8)
        # 添加数值标注
        for j, val in enumerate(sum_rates[alg]):
            ax2.text(j + (i - 1.5) * width, val + 50, f'{int(val)}', ha='center', va='bottom', fontsize=9)

    ax2.set_ylabel('System Sum Rate (Mbps)', fontsize=12, fontweight='bold')
    ax2.set_title('Seed 41: System Sum Rate Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(UAV_CONFIGS)
    ax2.legend(loc='upper left', frameon=True, shadow=True)
    ax2.set_ylim(0, max([max(v) for v in sum_rates.values()]) * 1.2)

    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, 'seed41_v5pro_vs_heuristics.png')
    plt.savefig(plot_path, dpi=300)
    print(f"✅ 对比图已生成: {plot_path}")

    # --- 生成性能提升总结表格 ---
    print("\n" + "="*80)
    print(f"{'Config':<12} | {'Metric':<10} | {'GA Impr%':<10} | {'PSO Impr%':<10} | {'SSA Impr%':<10} | {'V5Pro Impr%':<10}")
    print("-" * 80)
    for i, conf in enumerate(UAV_CONFIGS):
        ini_m = min_rates["initial"][i]
        ini_s = sum_rates["initial"][i]
        
        m_line = f"{conf:<12} | {'Min Rate':<10} "
        s_line = f"{'':<12} | {'Sum Rate':<10} "
        
        for alg in ALGORITHMS:
            m_imp = (min_rates[alg][i] - ini_m) / ini_m * 100
            s_imp = (sum_rates[alg][i] - ini_s) / ini_s * 100
            m_line += f"| {m_imp:>8.1f}% "
            s_line += f"| {s_imp:>8.1f}% "
        
        print(m_line)
        print(s_line)
        print("-" * 80)

if __name__ == "__main__":
    plot_comparison()
