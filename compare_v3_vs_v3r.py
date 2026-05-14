"""
对比 Balanced Virtual Force V3 (原始细节) vs V3-Refined (精进细节)
配置: 6, 9, 12 UAVs
种子: 71, 75
"""

import numpy as np
import json
import time
import os
import matplotlib.pyplot as plt

from balanced_virtual_force_optimizer_v3 import BalancedVirtualForceOptimizerV3, create_balanced_config
from balanced_virtual_force_optimizer_v3_refined import BalancedVirtualForceOptimizerV3Refined

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
            print("  运行 V3 (Original)...")
            v3_opt = BalancedVirtualForceOptimizerV3(config)
            res_v3 = v3_opt.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
            
            # --- 运行 V3-Refined ---
            print("  运行 V3-Refined (Refined Physical Details)...")
            v3r_opt = BalancedVirtualForceOptimizerV3Refined(config)
            res_v3r = v3r_opt.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
            
            uav_results[f'seed_{seed}'] = {
                'v3': {'min_rate': res_v3['final_min_rate'], 'sum_rate': res_v3['final_sum_rate']},
                'v3r': {'min_rate': res_v3r['final_min_rate'], 'sum_rate': res_v3r['final_sum_rate']}
            }
            
            diff = (res_v3r['final_min_rate'] - res_v3['final_min_rate']) / res_v3['final_min_rate'] * 100
            print(f"  结果: V3={res_v3['final_min_rate']:.4f} | V3R={res_v3r['final_min_rate']:.4f} | 提升: {diff:+.2f}%")
            
        all_results[f'{num_uav}uav'] = uav_results

    # 保存结果汇总
    summary = "\n" + "="*60 + "\n"
    summary += " V3 vs V3-Refined 性能对比总结 ".center(60) + "\n"
    summary += "="*60 + "\n"
    summary += f"{'Config':<15} | {'V3 Min':<12} | {'V3R Min':<12} | {'Improve':<10}\n"
    summary += "-"*60 + "\n"
    
    for uav_key, seeds_data in all_results.items():
        for seed_key, data in seeds_data.items():
            v3 = data['v3']['min_rate']
            v3r = data['v3r']['min_rate']
            imp = (v3r - v3) / v3 * 100
            summary += f"{uav_key+' '+seed_key:<15} | {v3:<12.4f} | {v3r:<12.4f} | {imp:>+8.2f}%\n"
    
    print(summary)
    with open(f'{OUTPUT_DIR}/v3_vs_v3r_summary.txt', 'w') as f:
        f.write(summary)

    return all_results

if __name__ == "__main__":
    run_comparison()
