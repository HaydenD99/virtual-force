"""
对比 Balanced Virtual Force V3 (原始几何力) vs V5 (通信感知梯度力)
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
            
            print("  运行 V3 (Baseline)...")
            v3_opt = BalancedVirtualForceOptimizerV3(config)
            res_v3 = v3_opt.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
            
            print("  运行 V5 (Comm-Aware SINR Force)...")
            v5_opt = BalancedVirtualForceOptimizerV5(config)
            res_v5 = v5_opt.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
            
            uav_results[f'seed_{seed}'] = {
                'v3': {'min_rate': res_v3['final_min_rate'], 'sum_rate': res_v3['final_sum_rate']},
                'v5': {'min_rate': res_v5['final_min_rate'], 'sum_rate': res_v5['final_sum_rate']}
            }
            diff = (res_v5['final_min_rate'] - res_v3['final_min_rate']) / res_v3['final_min_rate'] * 100
            print(f"  结果: V3={res_v3['final_min_rate']:.4f} | V5={res_v5['final_min_rate']:.4f} | 提升: {diff:+.2f}%")
        all_results[f'{num_uav}uav'] = uav_results

    # 打印总结
    print("\n" + "="*65)
    print(" V3 (Geometric) vs V5 (Comm-Aware) 性能对比总结 ".center(65))
    print("="*65)
    print(f"{'Config':<15} | {'V3 Min':<12} | {'V5 Min':<12} | {'Improve':<10}")
    print("-" * 65)
    for uav_key, seeds_data in all_results.items():
        for seed_key, data in seeds_data.items():
            v3, v5 = data['v3']['min_rate'], data['v5']['min_rate']
            imp = (v5 - v3) / v3 * 100
            print(f"{uav_key+' '+seed_key:<15} | {v3:<12.4f} | {v5:<12.4f} | {imp:>+8.2f}%")
    return all_results

if __name__ == "__main__":
    run_comparison()
