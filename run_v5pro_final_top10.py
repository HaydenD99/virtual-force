"""
全算法对比 - V5Pro 最终版 (Top 10 Seeds)
初始化方案: 严格遵循 compare_optimizers_fair.py (UAV grid 200-800m)
环境参数: tau_p=60, Ground AP @ 250/750
对比算法: V5Pro, GA, PSO, NewSSA
场景: 6, 9, 12 UAVs | L=3
种子: 56, 61, 62, 64, 67, 69, 70, 73, 75, 76
"""

import numpy as np
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

from balanced_virtual_force_optimizer_v5_sinr import BalancedVirtualForceOptimizerV5
from genetic_algorithm_optimizer_discrete import DiscreteGeneticAlgorithmOptimizer, create_discrete_ga_config
from distributed_pso_optimizer import DistributedPSOOptimizer, create_distributed_pso_config
from newssa_optimizer import NewSSAOptimizer

# --- 配置 ---
TOP_SEEDS = [56, 61, 62, 64, 67, 69, 70, 73, 75, 76]
UAV_COUNTS = [6, 9, 12]
L_SERVING = 3
OUTPUT_DIR = 'result/v5pro_top10_comparison'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_original_scenario(seed, num_uav):
    np.random.seed(seed)
    K, G = 60, 4
    # 1. UE 位置
    UE_pos = np.random.uniform(low=[50, 50], high=[950, 950], size=(K, 2))
    UE_pos = np.column_stack([UE_pos, np.ones(K) * 1.65])
    
    # 2. 地面 AP：2x2 网格 (250, 750)
    gx, gy = np.meshgrid([250, 750], [250, 750])
    ground_AP_pos = np.column_stack([gx.flatten(), gy.flatten(), np.ones(G) * 15.0])
    
    # 3. UAV 初始位置：网格分布在 [200, 800]
    if num_uav == 6:
        ux, uy = np.meshgrid(np.linspace(200, 800, 3), np.linspace(300, 700, 2))
    elif num_uav == 9:
        ux, uy = np.meshgrid(np.linspace(200, 800, 3), np.linspace(200, 800, 3))
    elif num_uav == 12:
        ux, uy = np.meshgrid(np.linspace(200, 800, 4), np.linspace(200, 800, 3))
    else: # 默认
        ux, uy = np.meshgrid(np.linspace(200, 800, 3), np.linspace(200, 800, 3))
        
    UAV_pos_init = np.column_stack([ux.flatten()[:num_uav], uy.flatten()[:num_uav], np.ones(num_uav) * 50.0])
    return UE_pos, ground_AP_pos, UAV_pos_init

def run():
    print(f"🚀 开始 V5Pro 全算法 Top 10 种子性能对比...")
    
    for num_uav in UAV_COUNTS:
        for seed in TOP_SEEDS:
            filename = f"final_comp_{num_uav}uav_seed{seed}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            if os.path.exists(filepath):
                continue
                
            print(f"\n📍 场景: {num_uav}UAV | Seed {seed}")
            UE_pos, ground_AP_pos, UAV_pos_init = generate_original_scenario(seed, num_uav)
            
            base_cfg = {
                'num_UAV': num_uav, 'num_serving_APs': L_SERVING, 'tau_p': 60,
                'random_seed': seed, 'max_iterations': 50, 'num_UE': 60, 'M': 4
            }
            
            results = {}
            
            # --- 0. 初始性能 ---
            v5_temp = BalancedVirtualForceOptimizerV5(base_cfg)
            all_AP_init = np.vstack([ground_AP_pos, UAV_pos_init])
            _, _, betas_init = v5_temp.compute_channel_model(UE_pos, all_AP_init)
            mask_init = v5_temp.compute_AP_selection_mask(betas_init)
            rates_init, sum_r_init = v5_temp.compute_user_rates(UE_pos, all_AP_init, mask_init)
            results['initial'] = {'min': float(rates_init.min()), 'sum': float(sum_r_init)}
            
            # --- 优化器 ---
            optimizers = [
                ('V5Pro', BalancedVirtualForceOptimizerV5, base_cfg),
                ('GA', DiscreteGeneticAlgorithmOptimizer, create_discrete_ga_config()),
                ('PSO', DistributedPSOOptimizer, create_distributed_pso_config()),
                ('NewSSA', NewSSAOptimizer, base_cfg)
            ]
            
            for name, Cls, cfg in optimizers:
                np.random.seed(seed)
                current_cfg = cfg.copy()
                current_cfg.update(base_cfg)
                if name == 'PSO':
                    current_cfg['w_min_rate'], current_cfg['w_sum_rate'] = 1.0, 0.1
                
                opt_obj = Cls(current_cfg)
                if name == 'GA':
                    opt_obj.K, opt_obj.G = 60, 4
                    res = opt_obj.optimize(UE_pos, ground_AP_pos)
                else:
                    res = opt_obj.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
                
                results[name] = {'min': float(res['final_min_rate']), 'sum': float(res['final_sum_rate'])}
                print(f"    {name}: {res['final_min_rate']:.2f} Mbps")

            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ 保存结果: {filename}")

if __name__ == "__main__":
    run()
