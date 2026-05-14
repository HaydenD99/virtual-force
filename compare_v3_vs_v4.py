"""
对比 Balanced Virtual Force V3 (静态权重) vs V4 (SA动态权重)
配置: 6, 9, 12 UAVs
种子: 71, 75
"""

import numpy as np
import json
import time
import os
import matplotlib.pyplot as plt

from balanced_virtual_force_optimizer_v3 import BalancedVirtualForceOptimizerV3, create_balanced_config
from balanced_virtual_force_optimizer_v4 import BalancedVirtualForceOptimizerV4

# 设置
SEEDS = [71, 75]
UAV_CONFIGS = [6, 9, 12]
OUTPUT_DIR = 'result'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_scenario(seed, num_uav):
    np.random.seed(seed)
    K, G = 60, 4
    UE_pos = np.column_stack([np.random.uniform(50, 950, (K, 2)), np.full((K, 1), 1.65)])
    ground_AP_pos = np.array([[250,250,15], [250,750,15], [750,250,15], [750,750,15]])
    
    # UAV 初始网格
    cols = 3 if num_uav <= 9 else 4
    rows = 2 if num_uav == 6 else 3
    gx, gy = np.meshgrid(np.linspace(250, 750, cols), np.linspace(250, 750, rows))
    UAV_pos = np.column_stack([gx.flatten()[:num_uav], gy.flatten()[:num_uav], np.full((num_uav, 1), 50.0)])
    
    return UE_pos, ground_AP_pos, UAV_pos

def run_comparison():
    all_results = {}
    
    for num_uav in UAV_CONFIGS:
        uav_results = {}
        for seed in SEEDS:
            print(f"\n🚀 正在测试: {num_uav} UAVs | 种子 {seed}")
            UE_pos, ground_AP_pos, UAV_pos_init = generate_scenario(seed, num_uav)
            
            # 配置
            config = create_balanced_config()
            config.update({'num_UAV': num_uav, 'random_seed': seed, 'max_iterations': 50})
            
            # --- 运行 V3 ---
            print("  运行 V3...")
            v3_opt = BalancedVirtualForceOptimizerV3(config)
            start = time.time()
            res_v3 = v3_opt.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
            time_v3 = time.time() - start
            
            # --- 运行 V4 ---
            print("  运行 V4...")
            v4_opt = BalancedVirtualForceOptimizerV4(config)
            start = time.time()
            res_v4 = v4_opt.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
            time_v4 = time.time() - start
            
            uav_results[f'seed_{seed}'] = {
                'v3': {'min_rate': res_v3['final_min_rate'], 'sum_rate': res_v3['final_sum_rate'], 'history': res_v3['history']['min_rates']},
                'v4': {'min_rate': res_v4['final_min_rate'], 'sum_rate': res_v4['final_sum_rate'], 'history': res_v4['history']['min_rates']}
            }
            
            print(f"  结果: V3={res_v3['final_min_rate']:.4f} | V4={res_v4['final_min_rate']:.4f} | 提升: {(res_v4['final_min_rate']-res_v3['final_min_rate'])/res_v3['final_min_rate']*100:+.2f}%")
            
        all_results[f'{num_uav}uav'] = uav_results

    # 保存
    with open(f'{OUTPUT_DIR}/v3_vs_v4_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

def plot_results(all_results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, num_uav in enumerate(UAV_CONFIGS):
        ax = axes[i]
        uav_key = f'{num_uav}uav'
        for seed in SEEDS:
            data = all_results[uav_key][f'seed_{seed}']
            ax.plot(data['v3']['history'], label=f'V3 (Seed {seed})', linestyle='--')
            ax.plot(data['v4']['history'], label=f'V4 (Seed {seed})', linewidth=2)
        
        ax.set_title(f'{num_uav} UAVs Comparison')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Min Rate (Mbps)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/v3_vs_v4_convergence.png')
    print(f"\n📈 对比图已保存至: {OUTPUT_DIR}/v3_vs_v4_convergence.png")

if __name__ == "__main__":
    results = run_comparison()
    # plot_results(results) # 暂时不画图，先看数据
