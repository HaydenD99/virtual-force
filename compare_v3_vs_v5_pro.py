"""
对比 Balanced Virtual Force V3 vs V5-Pro (Comm-Aware + Robust Memory)
同时监控 Min Rate 和 Sum Rate
"""

import numpy as np
import json
import time
import os
import matplotlib.pyplot as plt

from balanced_virtual_force_optimizer_v3 import BalancedVirtualForceOptimizerV3, create_balanced_config
from balanced_virtual_force_optimizer_v5_sinr import BalancedVirtualForceOptimizerV5

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
            config = create_balanced_config()
            config.update({'num_UAV': num_uav, 'random_seed': seed, 'max_iterations': 50})
            
            # --- V3 ---
            v3_opt = BalancedVirtualForceOptimizerV3(config)
            res_v3 = v3_opt.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
            
            # --- V5 Pro ---
            v5_opt = BalancedVirtualForceOptimizerV5(config)
            res_v5 = v5_opt.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
            
            uav_results[f'seed_{seed}'] = {
                'v3': {'min': res_v3['final_min_rate'], 'sum': res_v3['final_sum_rate']},
                'v5': {'min': res_v5['final_min_rate'], 'sum': res_v5['final_sum_rate']}
            }
            
            m_imp = (res_v5['final_min_rate'] - res_v3['final_min_rate']) / res_v3['final_min_rate'] * 100
            s_imp = (res_v5['final_sum_rate'] - res_v3['final_sum_rate']) / res_v3['final_sum_rate'] * 100
            print(f"  [Min Rate] V3: {res_v3['final_min_rate']:.2f} -> V5: {res_v5['final_min_rate']:.2f} ({m_imp:+.2f}%)")
            print(f"  [Sum Rate] V3: {res_v3['final_sum_rate']:.1f} -> V5: {res_v5['final_sum_rate']:.1f} ({s_imp:+.2f}%)")
            
        all_results[f'{num_uav}uav'] = uav_results

    # 打印最终总结表格
    header = f"\n{'Config':<15} | {'Metric':<10} | {'V3 (Baseline)':<15} | {'V5 (Pro)':<15} | {'Improve':<10}"
    print("\n" + "="*75 + "\n" + " V3 vs V5-Pro 综合性能对比 ".center(75) + "\n" + "="*75)
    print(header)
    print("-" * 75)
    
    for uav_key, seeds_data in all_results.items():
        for seed_key, data in seeds_data.items():
            conf = f"{uav_key} {seed_key}"
            # Min Rate 行
            v3_m, v5_m = data['v3']['min'], data['v5']['min']
            print(f"{conf:<15} | {'Min Rate':<10} | {v3_m:<15.4f} | {v5_m:<15.4f} | {(v5_m-v3_m)/v3_m*100:>+8.2f}%")
            # Sum Rate 行
            v3_s, v5_s = data['v3']['sum'], data['v5']['sum']
            print(f"{'':<15} | {'Sum Rate':<10} | {v3_s:<15.1f} | {v5_s:<15.1f} | {(v5_s-v3_s)/v3_s*100:>+8.2f}%")
            print("-" * 75)
            
    return all_results

if __name__ == "__main__":
    run_comparison()
