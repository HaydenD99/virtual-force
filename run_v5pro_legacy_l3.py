import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

from balanced_virtual_force_optimizer_v5_legacy_env import V5ProLegacyEnv

# --- 严苛环境配置 ---
SEEDS = [41, 51, 62, 63, 71, 75, 76]
CONFIGS = [
    {'uav': 6, 'L': 3},
    {'uav': 9, 'L': 3},
    {'uav': 12, 'L': 3}
]
OUTPUT_DIR = 'result/v5pro_legacy_L3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_legacy_scenario(seed, num_uav):
    np.random.seed(seed)
    K = 60
    # 用户保持不变
    UE_pos = np.column_stack([np.random.uniform(50, 950, (K, 2)), np.full((K, 1), 1.65)])
    
    # 1. 地面 AP 回退到四个角落 (0,0), (0,1000), (1000,0), (1000,1000)
    ground_AP_pos = np.array([
        [0, 0, 15.0],
        [0, 1000, 15.0],
        [1000, 0, 15.0],
        [1000, 1000, 15.0]
    ])
    
    # 2. UAV 初始位置拉得更开 (50-950)
    cols = 3 if num_uav <= 9 else 4
    rows = 2 if num_uav == 6 else 3
    gx, gy = np.meshgrid(np.linspace(100, 900, cols), np.linspace(100, 900, rows))
    UAV_pos = np.column_stack([gx.flatten()[:num_uav], gy.flatten()[:num_uav], np.full((num_uav, 1), 50.0)])
    
    return UE_pos, ground_AP_pos, UAV_pos

def run_legacy_experiment():
    print(f"🚀 开始 V5Pro 严苛环境测试 (tau_p=20, AP@Corners, Initial Worse)")
    
    for config_item in CONFIGS:
        uav = config_item['uav']
        L = config_item['L']
        
        for seed in SEEDS:
            filename = f"legacy_v5pro_{uav}uav_L{L}_seed{seed}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            UE_pos, ground_AP_pos, UAV_pos_init = generate_legacy_scenario(seed, uav)
            cfg = {'num_UAV': uav, 'num_serving_APs': L}
            
            optimizer = V5ProLegacyEnv(cfg)
            
            # 计算 Initial (环境此时已经由于 tau_p=20 和 AP@Corners 变得很差)
            all_AP_init = np.vstack([ground_AP_pos, UAV_pos_init])
            _, _, betas_init = optimizer.compute_channel_model(UE_pos, all_AP_init)
            mask_init = optimizer.compute_AP_selection_mask(betas_init)
            rates_init, _ = optimizer.compute_user_rates(UE_pos, all_AP_init, mask_init)
            initial_min = float(rates_init.min())
            
            # 运行优化
            res = optimizer.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
            
            result_data = {
                'initial': {'min': initial_min},
                'V5Pro': {'min': res['final_min_rate'], 'sum': res['final_sum_rate']}
            }
            
            with open(filepath, 'w') as f:
                json.dump(result_data, f, indent=2)
            print(f"✅ Seed {seed} | UAV {uav} | L {L} | Initial: {initial_min:.2f} -> Optimized: {res['final_min_rate']:.2f}")

if __name__ == "__main__":
    run_legacy_experiment()
