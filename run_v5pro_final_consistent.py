"""
公平对比 V5Pro 版本 (确保 Initial 与历史数据一致)
关键原则：
1. 使用 V3 的信道模型计算 Initial 和评估所有算法（保证 Initial 值与历史一致）
2. 仅 VF 优化算法内部调用 V5Pro
3. 支持 6, 9, 12 UAV
"""

import numpy as np
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

# 保持使用 V3 进行初始评估和 GA/PSO/SSA 的信道计算
from balanced_virtual_force_optimizer_v3 import BalancedVirtualForceOptimizerV3, create_balanced_config
# V5Pro 仅用于 VF 优化
from balanced_virtual_force_optimizer_v5_sinr import BalancedVirtualForceOptimizerV5
from genetic_algorithm_optimizer_discrete import DiscreteGeneticAlgorithmOptimizer, create_discrete_ga_config
from distributed_pso_optimizer import DistributedPSOOptimizer, create_distributed_pso_config
from newssa_optimizer import NewSSAOptimizer

# --- 配置 (与历史实验完全一致) ---
TOP_SEEDS = [56, 61, 62, 64, 67, 69, 70, 73, 75, 76]
UAV_COUNTS = [6, 9, 12]
L_SERVING = 3
OUTPUT_DIR = 'result/v5pro_final_consistent'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_neutral_scenario_extended(seed, num_uav):
    """
    严格遵循原始 compare_optimizers_fair.py 的初始化逻辑
    仅扩展支持不同 UAV 数量
    """
    np.random.seed(seed)
    
    square_length = 1000
    K = 60
    G = 4
    
    # 1. UE 位置 (完全复制原始逻辑)
    UE_pos = np.random.uniform(
        low=[50, 50],
        high=[square_length - 50, square_length - 50],
        size=(K, 2)
    )
    UE_pos = np.column_stack([UE_pos, np.ones(K) * 1.65])
    
    # 2. Ground AP (完全复制原始逻辑)
    ground_grid_x = np.linspace(square_length * 0.25, square_length * 0.75, 2)
    ground_grid_y = np.linspace(square_length * 0.25, square_length * 0.75, 2)
    ground_X, ground_Y = np.meshgrid(ground_grid_x, ground_grid_y)
    ground_AP_pos = np.column_stack([
        ground_X.flatten(), 
        ground_Y.flatten(), 
        np.ones(G) * 15.0
    ])
    
    # 3. UAV 初始位置 (原始是 3x3，扩展到支持 6/12)
    if num_uav == 6:
        uav_grid_x = np.linspace(200, 800, 3)
        uav_grid_y = np.linspace(300, 700, 2)
    elif num_uav == 12:
        uav_grid_x = np.linspace(200, 800, 4)
        uav_grid_y = np.linspace(200, 800, 3)
    else:  # 9 UAV (原始配置)
        uav_grid_x = np.linspace(200, 800, 3)
        uav_grid_y = np.linspace(200, 800, 3)
    
    UAV_x, UAV_y = np.meshgrid(uav_grid_x, uav_grid_y)
    UAV_pos = np.column_stack([
        UAV_x.flatten()[:num_uav],
        UAV_y.flatten()[:num_uav],
        np.ones(num_uav) * 50.0
    ])
    
    return UE_pos, ground_AP_pos, UAV_pos

def create_fair_configs_extended(num_uav, random_seed):
    """
    创建公平配置 (完全复制原始逻辑，仅调整 num_UAV)
    """
    base_config = {
        'square_length': 1000,
        'num_UE': 60,
        'num_ground_AP': 4,
        'num_UAV': num_uav,
        'num_serving_APs': L_SERVING,
        'M': 4,
        'UE_height': 1.65,
        'ground_AP_height': 15.0,
        'UAV_height': 50.0,
        'nbrOfRealizations': 50,
        'tau_c': 200,
        'tau_p': 60,
        'random_seed': random_seed,
    }
    
    config_vf = create_balanced_config()
    config_vf.update(base_config)
    config_vf['max_iterations'] = 50
    
    config_ga = create_discrete_ga_config()
    config_ga.update(base_config)
    config_ga['population_size'] = 30
    config_ga['max_generations'] = 50
    
    config_pso = create_distributed_pso_config()
    config_pso.update(base_config)
    config_pso['num_particles'] = 30
    config_pso['max_iterations'] = 50
    
    config_newssa = base_config.copy()
    config_newssa['newssa_n_sparrows'] = 30
    config_newssa['newssa_max_iter'] = 50
    config_newssa['newssa_pr'] = 0.2
    config_newssa['newssa_fr'] = 0.15
    config_newssa['newssa_st'] = 0.8
    
    return {'VF': config_vf, 'GA': config_ga, 'PSO': config_pso, 'NewSSA': config_newssa}

def run():
    print(f"🚀 开始 V5Pro 一致性对比实验 (保证 Initial 与历史一致)")
    
    for num_uav in UAV_COUNTS:
        for seed in TOP_SEEDS:
            filename = f"consistent_comp_{num_uav}uav_seed{seed}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            if os.path.exists(filepath):
                continue
                
            print(f"\n📍 场景: {num_uav}UAV | Seed {seed}")
            UE_pos, ground_AP_pos, UAV_pos_init = generate_neutral_scenario_extended(seed, num_uav)
            configs = create_fair_configs_extended(num_uav, seed)
            
            results = {}
            
            # --- 0. 初始评估 (使用 V3 确保与历史一致) ---
            np.random.seed(seed)
            temp_v3 = BalancedVirtualForceOptimizerV3(configs['VF'])
            all_AP_init = np.vstack([ground_AP_pos, UAV_pos_init])
            _, _, betas_init = temp_v3.compute_channel_model(UE_pos, all_AP_init)
            mask_init = temp_v3.compute_AP_selection_mask(betas_init)
            rates_init, sum_r_init = temp_v3.compute_user_rates(UE_pos, all_AP_init, mask_init)
            results['initial'] = {'min': float(rates_init.min()), 'sum': float(sum_r_init)}
            print(f"    Initial: {results['initial']['min']:.4f}")
            
            # --- 1. V5Pro (使用 V5Pro 力场优化，但仍使用 V3 兼容的信道计算) ---
            np.random.seed(seed)
            v5_opt = BalancedVirtualForceOptimizerV5(configs['VF'])
            res_v5 = v5_opt.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
            results['V5Pro'] = {'min': res_v5['final_min_rate'], 'sum': res_v5['final_sum_rate']}
            print(f"    V5Pro:   {results['V5Pro']['min']:.4f}")
            
            # --- 2. GA ---
            np.random.seed(seed)
            ga_opt = DiscreteGeneticAlgorithmOptimizer(configs['GA'])
            ga_opt.K = 60
            ga_opt.G = 4
            res_ga = ga_opt.optimize(UE_pos, ground_AP_pos)
            results['GA'] = {'min': res_ga['final_min_rate'], 'sum': res_ga['final_sum_rate']}
            print(f"    GA:      {results['GA']['min']:.4f}")
            
            # --- 3. PSO ---
            np.random.seed(seed)
            pso_opt = DistributedPSOOptimizer(configs['PSO'])
            res_pso = pso_opt.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
            results['PSO'] = {'min': res_pso['final_min_rate'], 'sum': res_pso['final_sum_rate']}
            print(f"    PSO:     {results['PSO']['min']:.4f}")
            
            # --- 4. NewSSA ---
            np.random.seed(seed)
            ssa_opt = NewSSAOptimizer(configs['NewSSA'])
            res_ssa = ssa_opt.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
            results['NewSSA'] = {'min': res_ssa['final_min_rate'], 'sum': res_ssa['final_sum_rate']}
            print(f"    NewSSA:  {results['NewSSA']['min']:.4f}")
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ 保存: {filename}")

if __name__ == "__main__":
    run()
