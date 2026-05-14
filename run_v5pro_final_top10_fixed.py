"""
全算法对比 - V5Pro 最终修正版 (Top 10 Seeds)
修正点:
1. 显式指定各算法专属的迭代和种群参数键名。
2. 确保 GA/PSO/NewSSA 的初始种群包含 UAV_pos_init。
3. 严格遵循 compare_optimizers_fair.py 的物理环境逻辑。
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
OUTPUT_DIR = 'result/v5pro_top10_fixed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_original_scenario(seed, num_uav):
    np.random.seed(seed)
    K, G = 60, 4
    # 1. UE 位置 (50-950)
    UE_pos = np.random.uniform(low=[50, 50], high=[950, 950], size=(K, 2))
    UE_pos = np.column_stack([UE_pos, np.ones(K) * 1.65])
    
    # 2. 地面 AP (250, 750 Grid)
    gx, gy = np.meshgrid([250, 750], [250, 750])
    ground_AP_pos = np.column_stack([gx.flatten(), gy.flatten(), np.ones(G) * 15.0])
    
    # 3. UAV 初始位置 (200-800 Grid)
    if num_uav == 6:
        ux, uy = np.meshgrid(np.linspace(200, 800, 3), np.linspace(300, 700, 2))
    elif num_uav == 12:
        ux, uy = np.meshgrid(np.linspace(200, 800, 4), np.linspace(200, 800, 3))
    else: # 9 UAV
        ux, uy = np.meshgrid(np.linspace(200, 800, 3), np.linspace(200, 800, 3))
        
    UAV_pos_init = np.column_stack([ux.flatten()[:num_uav], uy.flatten()[:num_uav], np.ones(num_uav) * 50.0])
    return UE_pos, ground_AP_pos, UAV_pos_init

def run():
    print(f"🚀 开始 V5Pro 全算法 Top 10 修正对比实验...")
    
    for num_uav in UAV_COUNTS:
        for seed in TOP_SEEDS:
            filename = f"comp_{num_uav}uav_seed{seed}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            print(f"\n📍 场景: {num_uav}UAV | Seed {seed}")
            UE_pos, ground_AP_pos, UAV_pos_init = generate_original_scenario(seed, num_uav)
            
            results = {}
            
            # --- 0. 初始评估 (基准) ---
            base_cfg = {'num_UAV': num_uav, 'num_serving_APs': L_SERVING, 'tau_p': 60, 'nbrOfRealizations': 50}
            v5_temp = BalancedVirtualForceOptimizerV5(base_cfg)
            all_AP_init = np.vstack([ground_AP_pos, UAV_pos_init])
            _, _, betas_init = v5_temp.compute_channel_model(UE_pos, all_AP_init)
            mask_init = v5_temp.compute_AP_selection_mask(betas_init)
            rates_init, sum_r_init = v5_temp.compute_user_rates(UE_pos, all_AP_init, mask_init)
            results['initial'] = {'min': float(rates_init.min()), 'sum': float(sum_r_init)}
            print(f"    Initial: {results['initial']['min']:.4f}")

            # --- 1. V5Pro ---
            cfg_v5 = base_cfg.copy()
            cfg_v5['max_iterations'] = 50
            res_v5 = BalancedVirtualForceOptimizerV5(cfg_v5).optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
            results['V5Pro'] = {'min': res_v5['final_min_rate'], 'sum': res_v5['final_sum_rate']}
            print(f"    V5Pro: {results['V5Pro']['min']:.4f}")

            # --- 2. PSO (修正配置) ---
            cfg_pso = create_distributed_pso_config()
            cfg_pso.update(base_cfg)
            cfg_pso.update({'num_particles': 30, 'max_iterations': 50, 'w_min_rate': 1.0, 'w_sum_rate': 0.1})
            res_pso = DistributedPSOOptimizer(cfg_pso).optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
            results['PSO'] = {'min': res_pso['final_min_rate'], 'sum': res_pso['final_sum_rate']}
            print(f"    PSO:   {results['PSO']['min']:.4f}")

            # --- 3. NewSSA (修正配置) ---
            cfg_ssa = base_cfg.copy()
            cfg_ssa.update({'newssa_n_sparrows': 30, 'newssa_max_iter': 50})
            res_ssa = NewSSAOptimizer(cfg_ssa).optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
            results['NewSSA'] = {'min': res_ssa['final_min_rate'], 'sum': res_ssa['final_sum_rate']}
            print(f"    SSA:   {results['NewSSA']['min']:.4f}")

            # --- 4. GA (修正: 手动注入初始个体) ---
            cfg_ga = create_discrete_ga_config()
            cfg_ga.update(base_cfg)
            cfg_ga.update({'population_size': 30, 'max_generations': 50})
            ga_opt = DiscreteGeneticAlgorithmOptimizer(cfg_ga)
            ga_opt.K, ga_opt.G = 60, 4
            # 注入初始网格位置作为精英
            ga_opt.initial_UAV_pos = UAV_pos_init.copy() 
            res_ga = ga_opt.optimize(UE_pos, ground_AP_pos)
            results['GA'] = {'min': res_ga['final_min_rate'], 'sum': res_ga['final_sum_rate']}
            print(f"    GA:    {results['GA']['min']:.4f}")

            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)

if __name__ == "__main__":
    run()
