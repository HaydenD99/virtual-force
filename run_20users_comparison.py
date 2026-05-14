"""
20用户场景下的四算法公平对比
关键修改：将用户数量从60改为20
其他配置与 run_v5pro_final_consistent.py 保持一致
"""

import numpy as np
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

# 保持使用 V3 进行初始评估和 GA/PSO/SSA 的信道计算
from balanced_virtual_force_optimizer_v3 import BalancedVirtualForceOptimizerV3, create_balanced_config
# V6 用于 VF 优化
from balanced_virtual_force_optimizer_v6 import BalancedVirtualForceOptimizerV6
from genetic_algorithm_optimizer_discrete import DiscreteGeneticAlgorithmOptimizer, create_discrete_ga_config
from distributed_pso_optimizer import DistributedPSOOptimizer, create_distributed_pso_config
from newssa_optimizer import NewSSAOptimizer

# --- 配置 ---
TOP_SEEDS = [56, 61, 62, 64, 67, 69, 70, 73, 75, 76]
UAV_COUNTS = [6, 9, 12]
L_SERVING = 3
K = 20  # 用户数量从60改为20
OUTPUT_DIR = 'result/20users_comparison'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_scenario_20users(seed, num_uav):
    """
    生成20用户场景
    保持区域大小和其他配置不变，仅修改用户数量
    """
    np.random.seed(seed)
    
    square_length = 1000
    G = 4
    
    # 1. UE 位置 (20个用户)
    UE_pos = np.random.uniform(
        low=[50, 50],
        high=[square_length - 50, square_length - 50],
        size=(K, 2)
    )
    UE_pos = np.column_stack([UE_pos, np.ones(K) * 1.65])
    
    # 2. Ground AP (2x2 Grid，与原始保持一致)
    ground_grid_x = np.linspace(square_length * 0.25, square_length * 0.75, 2)
    ground_grid_y = np.linspace(square_length * 0.25, square_length * 0.75, 2)
    ground_X, ground_Y = np.meshgrid(ground_grid_x, ground_grid_y)
    ground_AP_pos = np.column_stack([
        ground_X.flatten(), 
        ground_Y.flatten(), 
        np.ones(G) * 15.0
    ])
    
    # 3. UAV 初始位置
    if num_uav == 6:
        uav_grid_x = np.linspace(200, 800, 3)
        uav_grid_y = np.linspace(300, 700, 2)
    elif num_uav == 12:
        uav_grid_x = np.linspace(200, 800, 4)
        uav_grid_y = np.linspace(200, 800, 3)
    else:  # 9 UAV
        uav_grid_x = np.linspace(200, 800, 3)
        uav_grid_y = np.linspace(200, 800, 3)
    
    UAV_x, UAV_y = np.meshgrid(uav_grid_x, uav_grid_y)
    UAV_pos = np.column_stack([
        UAV_x.flatten()[:num_uav],
        UAV_y.flatten()[:num_uav],
        np.ones(num_uav) * 50.0
    ])
    
    return UE_pos, ground_AP_pos, UAV_pos

def create_configs_20users(num_uav, random_seed):
    """
    创建20用户场景的配置
    """
    base_config = {
        'square_length': 1000,
        'num_UE': K,  # 20个用户
        'num_ground_AP': 4,
        'num_UAV': num_uav,
        'num_serving_APs': L_SERVING,
        'M': 4,
        'UE_height': 1.65,
        'ground_AP_height': 15.0,
        'UAV_height': 50.0,
        'nbrOfRealizations': 50,
        'tau_c': 200,
        'tau_p': 20,  # tau_p调整为20（与用户数匹配）
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
    print(f"🚀 开始 20用户场景四算法对比实验")
    print(f"   用户数: {K} | UAV配置: {UAV_COUNTS} | L={L_SERVING}")
    
    for num_uav in UAV_COUNTS:
        for seed in TOP_SEEDS:
            filename = f"20users_{num_uav}uav_seed{seed}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            if os.path.exists(filepath):
                print(f"⏭️  跳过已存在: {filename}")
                continue
                
            print(f"\n📍 场景: {num_uav}UAV | Seed {seed}")
            UE_pos, ground_AP_pos, UAV_pos_init = generate_scenario_20users(seed, num_uav)
            configs = create_configs_20users(num_uav, seed)
            
            results = {}
            
            # --- 0. 初始评估 (使用 V3 确保一致性) ---
            np.random.seed(seed)
            temp_v3 = BalancedVirtualForceOptimizerV3(configs['VF'])
            all_AP_init = np.vstack([ground_AP_pos, UAV_pos_init])
            _, _, betas_init = temp_v3.compute_channel_model(UE_pos, all_AP_init)
            mask_init = temp_v3.compute_AP_selection_mask(betas_init)
            rates_init, sum_r_init = temp_v3.compute_user_rates(UE_pos, all_AP_init, mask_init)
            results['initial'] = {'min': float(rates_init.min()), 'sum': float(sum_r_init)}
            print(f"    Initial: {results['initial']['min']:.4f} Mbps (sum: {results['initial']['sum']:.2f})")
            
            # --- 1. BVF (使用 V6) ---
            np.random.seed(seed)
            v6_opt = BalancedVirtualForceOptimizerV6(configs['VF'])
            res_v6 = v6_opt.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
            results['BVF'] = {'min': res_v6['final_min_rate'], 'sum': res_v6['final_sum_rate']}
            print(f"    BVF:     {results['BVF']['min']:.4f} Mbps (sum: {results['BVF']['sum']:.2f})")
            
            # --- 2. GA ---
            np.random.seed(seed)
            ga_opt = DiscreteGeneticAlgorithmOptimizer(configs['GA'])
            ga_opt.K = K
            ga_opt.G = 4
            res_ga = ga_opt.optimize(UE_pos, ground_AP_pos)
            results['GA'] = {'min': res_ga['final_min_rate'], 'sum': res_ga['final_sum_rate']}
            print(f"    GA:      {results['GA']['min']:.4f} Mbps (sum: {results['GA']['sum']:.2f})")
            
            # --- 3. PSO ---
            np.random.seed(seed)
            pso_opt = DistributedPSOOptimizer(configs['PSO'])
            res_pso = pso_opt.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
            results['PSO'] = {'min': res_pso['final_min_rate'], 'sum': res_pso['final_sum_rate']}
            print(f"    PSO:     {results['PSO']['min']:.4f} Mbps (sum: {results['PSO']['sum']:.2f})")
            
            # --- 4. NewSSA ---
            np.random.seed(seed)
            ssa_opt = NewSSAOptimizer(configs['NewSSA'])
            res_ssa = ssa_opt.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
            results['NewSSA'] = {'min': res_ssa['final_min_rate'], 'sum': res_ssa['final_sum_rate']}
            print(f"    NewSSA:  {results['NewSSA']['min']:.4f} Mbps (sum: {results['NewSSA']['sum']:.2f})")
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ 保存: {filename}")

if __name__ == "__main__":
    run()
