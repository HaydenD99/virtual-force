"""
对比 V3, V5Pro, V6 在相同条件下的性能
"""

import numpy as np
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

from balanced_virtual_force_optimizer_v3 import BalancedVirtualForceOptimizerV3, create_balanced_config
from balanced_virtual_force_optimizer_v5_sinr import BalancedVirtualForceOptimizerV5
from balanced_virtual_force_optimizer_v6 import BalancedVirtualForceOptimizerV6, create_v6_config

# 配置
SEEDS = [56, 62, 67, 75, 76]
UAV_COUNTS = [6, 9, 12]
L_SERVING = 3
OUTPUT_DIR = 'result/v3_v5_v6_comparison'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_scenario(seed, num_uav):
    """与 v5pro_final_consistent 完全一致的场景生成"""
    np.random.seed(seed)
    
    square_length = 1000
    K = 60
    G = 4
    
    # UE
    UE_pos = np.random.uniform(low=[50, 50], high=[950, 950], size=(K, 2))
    UE_pos = np.column_stack([UE_pos, np.ones(K) * 1.65])
    
    # Ground AP (2x2 grid)
    ground_grid_x = np.linspace(250, 750, 2)
    ground_grid_y = np.linspace(250, 750, 2)
    ground_X, ground_Y = np.meshgrid(ground_grid_x, ground_grid_y)
    ground_AP_pos = np.column_stack([ground_X.flatten(), ground_Y.flatten(), np.ones(G) * 15.0])
    
    # UAV 初始位置
    if num_uav == 6:
        uav_grid_x = np.linspace(200, 800, 3)
        uav_grid_y = np.linspace(300, 700, 2)
    elif num_uav == 12:
        uav_grid_x = np.linspace(200, 800, 4)
        uav_grid_y = np.linspace(200, 800, 3)
    else:  # 9
        uav_grid_x = np.linspace(200, 800, 3)
        uav_grid_y = np.linspace(200, 800, 3)
    
    UAV_x, UAV_y = np.meshgrid(uav_grid_x, uav_grid_y)
    UAV_pos = np.column_stack([
        UAV_x.flatten()[:num_uav],
        UAV_y.flatten()[:num_uav],
        np.ones(num_uav) * 50.0
    ])
    
    return UE_pos, ground_AP_pos, UAV_pos

def create_config(num_uav, seed):
    """创建统一配置"""
    return {
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
        'random_seed': seed,
        'max_iterations': 50,
        'step_size': 26,
        'alpha': 3.67,
        'constant_term': -30.5,
        'B': 20e6,
        'Pmax': 1000,
        'noise_figure': 7,
    }

def run_comparison():
    print("=" * 80)
    print("  V3 vs V5Pro vs V6 对比实验")
    print("=" * 80)
    
    all_results = {}
    
    for num_uav in UAV_COUNTS:
        for seed in SEEDS:
            key = f"{num_uav}uav_seed{seed}"
            filename = f"comp_{key}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            if os.path.exists(filepath):
                print(f"跳过已存在: {filename}")
                continue
            
            print(f"\n{'='*60}")
            print(f"配置: {num_uav} UAV | Seed {seed}")
            print(f"{'='*60}")
            
            UE_pos, ground_AP_pos, UAV_pos_init = generate_scenario(seed, num_uav)
            config = create_config(num_uav, seed)
            
            results = {}
            
            # --- Initial (用 V3 计算) ---
            np.random.seed(seed)
            temp_opt = BalancedVirtualForceOptimizerV3(config)
            all_AP_init = np.vstack([ground_AP_pos, UAV_pos_init])
            _, _, betas_init = temp_opt.compute_channel_model(UE_pos, all_AP_init)
            mask_init = temp_opt.compute_AP_selection_mask(betas_init)
            rates_init, sum_init = temp_opt.compute_user_rates(UE_pos, all_AP_init, mask_init)
            results['initial'] = {'min': float(rates_init.min()), 'sum': float(sum_init)}
            print(f"  Initial: Min={results['initial']['min']:.4f}, Sum={results['initial']['sum']:.1f}")
            
            # --- V3 ---
            np.random.seed(seed)
            v3_opt = BalancedVirtualForceOptimizerV3(config)
            start = time.time()
            res_v3 = v3_opt.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
            t_v3 = time.time() - start
            results['V3'] = {'min': res_v3['final_min_rate'], 'sum': res_v3['final_sum_rate'], 'time': t_v3}
            print(f"  V3:      Min={results['V3']['min']:.4f}, Sum={results['V3']['sum']:.1f}, Time={t_v3:.1f}s")
            
            # --- V5Pro ---
            np.random.seed(seed)
            v5_opt = BalancedVirtualForceOptimizerV5(config)
            start = time.time()
            res_v5 = v5_opt.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
            t_v5 = time.time() - start
            results['V5Pro'] = {'min': res_v5['final_min_rate'], 'sum': res_v5['final_sum_rate'], 'time': t_v5}
            print(f"  V5Pro:   Min={results['V5Pro']['min']:.4f}, Sum={results['V5Pro']['sum']:.1f}, Time={t_v5:.1f}s")
            
            # --- V6 ---
            np.random.seed(seed)
            v6_opt = BalancedVirtualForceOptimizerV6(config)
            start = time.time()
            res_v6 = v6_opt.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
            t_v6 = time.time() - start
            results['V6'] = {'min': res_v6['final_min_rate'], 'sum': res_v6['final_sum_rate'], 'time': t_v6}
            print(f"  V6:      Min={results['V6']['min']:.4f}, Sum={results['V6']['sum']:.1f}, Time={t_v6:.1f}s")
            
            # 保存
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  ✅ 保存: {filename}")
            
            all_results[key] = results
    
    # 汇总
    print("\n" + "=" * 80)
    print("  汇总对比")
    print("=" * 80)
    print(f"{'Config':<18} {'Initial':<10} {'V3':<10} {'V5Pro':<10} {'V6':<10} | {'Best':^10}")
    print("-" * 80)
    
    for key, res in all_results.items():
        init_min = res['initial']['min']
        v3_min = res['V3']['min']
        v5_min = res['V5Pro']['min']
        v6_min = res['V6']['min']
        
        best = 'V6' if v6_min >= max(v3_min, v5_min) else ('V5Pro' if v5_min >= v3_min else 'V3')
        
        print(f"{key:<18} {init_min:<10.2f} {v3_min:<10.2f} {v5_min:<10.2f} {v6_min:<10.2f} | {best:^10}")
    
    print("-" * 80)

if __name__ == "__main__":
    run_comparison()
