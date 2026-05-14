import numpy as np
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

from balanced_virtual_force_optimizer_v5_sinr import BalancedVirtualForceOptimizerV5

# --- 配置 ---
SEEDS = [41, 51, 62, 63, 71, 75, 76]
CONFIGS = [
    {'uav': 6, 'L': 3},
    {'uav': 9, 'L': 3},
    {'uav': 12, 'L': 3}
]
OUTPUT_DIR = 'result/v5pro_fair_low_init'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_fair_low_init_scenario(seed, num_uav):
    np.random.seed(seed)
    K = 60
    # 用户分布
    UE_pos = np.column_stack([np.random.uniform(50, 950, (K, 2)), np.full((K, 1), 1.65)])
    
    # 地面 AP：保持 2x2 均匀分布 (250, 750)
    grid_points = [250, 750]
    ground_AP_pos = []
    for x in grid_points:
        for y in grid_points:
            ground_AP_pos.append([x, y, 15.0])
    ground_AP_pos = np.array(ground_AP_pos)
    
    # UAV 初始位置：拉开到 [100, 900] 的宽网格，人为制造“较差”的初始覆盖
    cols = 3 if num_uav <= 9 else 4
    rows = 2 if num_uav == 6 else 3
    gx, gy = np.meshgrid(np.linspace(100, 900, cols), np.linspace(100, 900, rows))
    UAV_pos_init = np.column_stack([gx.flatten()[:num_uav], gy.flatten()[:num_uav], np.full((num_uav, 1), 50.0)])
    
    return UE_pos, ground_AP_pos, UAV_pos_init

def run():
    print(f"🚀 开始 V5Pro 公平环境对比 (tau_p=60, GroundAP@250/750, Wide-Init-UAV)")
    
    for config_item in CONFIGS:
        uav = config_item['uav']
        L = config_item['L']
        
        for seed in SEEDS:
            filename = f"v5pro_{uav}uav_L{L}_seed{seed}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            UE_pos, ground_AP_pos, UAV_pos_init = generate_fair_low_init_scenario(seed, uav)
            
            # 使用标准的 V5Pro (tau_p=60 由类内部逻辑根据 K 决定，或者我们在配置中指定)
            cfg = {
                'num_UAV': uav, 
                'num_serving_APs': L, 
                'tau_p': 60,  # 强制 tau_p = 60
                'random_seed': seed,
                'max_iterations': 50
            }
            
            optimizer = BalancedVirtualForceOptimizerV5(cfg)
            
            # 1. 评估 Initial
            all_AP_init = np.vstack([ground_AP_pos, UAV_pos_init])
            _, _, betas_init = optimizer.compute_channel_model(UE_pos, all_AP_init)
            mask_init = optimizer.compute_AP_selection_mask(betas_init)
            rates_init, sum_r_init = optimizer.compute_user_rates(UE_pos, all_AP_init, mask_init)
            initial_min = float(rates_init.min())
            
            # 2. 运行优化
            res = optimizer.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
            
            result_data = {
                'initial': {'min': initial_min, 'sum': float(sum_r_init)},
                'V5Pro': {'min': res['final_min_rate'], 'sum': res['final_sum_rate']}
            }
            
            with open(filepath, 'w') as f:
                json.dump(result_data, f, indent=2)
            print(f"✅ Seed {seed} | UAV {uav} | L {L} | Initial: {initial_min:.2f} -> Optimized: {res['final_min_rate']:.2f}")

if __name__ == "__main__":
    run()
