import numpy as np
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

from balanced_virtual_force_optimizer_v5_sinr import BalancedVirtualForceOptimizerV5

# --- 实验配置 ---
SEEDS = [41, 51, 62, 63, 71, 75, 76]
CONFIGS = [
    {'uav': 9, 'L': 3},
    {'uav': 12, 'L': 3}
]
OUTPUT_DIR = 'result/v5pro_l3_specific'
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

def run_specific_experiment():
    print(f"🚀 开始 V5Pro 特定配置测试 (L=3, 9/12 UAVs)")
    
    for config_item in CONFIGS:
        uav = config_item['uav']
        L = config_item['L']
        
        for seed in SEEDS:
            filename = f"v5pro_{uav}uav_L{L}_seed{seed}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            print(f"\n📍 运行中: {uav}UAV, L={L}, Seed={seed}")
            UE_pos, ground_AP_pos, UAV_pos_init = generate_scenario(seed, uav)
            
            # 配置参数
            cfg = {
                'num_UAV': uav, 
                'num_serving_APs': L, 
                'random_seed': seed,
                'nbrOfRealizations': 50, 
                'max_iterations': 50, 
                'num_UE': 60, 
                'M': 4,
                'square_length': 1000
            }
            
            results = {}
            optimizer = BalancedVirtualForceOptimizerV5(cfg)
            
            # 1. 初始状态评估
            all_AP_init = np.vstack([ground_AP_pos, UAV_pos_init])
            _, _, betas_init = optimizer.compute_channel_model(UE_pos, all_AP_init)
            mask_init = optimizer.compute_AP_selection_mask(betas_init)
            rates_init, sum_r_init = optimizer.compute_user_rates(UE_pos, all_AP_init, mask_init)
            results['initial'] = {
                'min': float(rates_init.min()),
                'sum': float(sum_r_init)
            }
            
            # 2. V5Pro 优化
            start_time = time.time()
            res_opt = optimizer.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
            duration = time.time() - start_time
            
            results['V5Pro'] = {
                'min': float(res_opt['final_min_rate']),
                'sum': float(res_opt['final_sum_rate']),
                'time': float(duration)
            }
            
            # 保存结果
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ 已记录至: {filename} | Min: {results['V5Pro']['min']:.4f}")

if __name__ == "__main__":
    run_specific_experiment()
