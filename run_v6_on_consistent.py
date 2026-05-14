"""
在 v5pro_final_consistent 的所有配置下运行 V6，并将结果添加到现有 JSON 文件
"""

import numpy as np
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

from balanced_virtual_force_optimizer_v6 import BalancedVirtualForceOptimizerV6

DATA_DIR = 'result/v5pro_final_consistent'
L_SERVING = 3

def generate_scenario(seed, num_uav):
    """与原脚本完全一致的场景生成"""
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

def run_v6_on_all():
    print("=" * 70)
    print("  在 v5pro_final_consistent 配置下运行 V6")
    print("=" * 70)
    
    # 获取所有 JSON 文件
    json_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    json_files.sort()
    
    total = len(json_files)
    completed = 0
    
    for filename in json_files:
        filepath = os.path.join(DATA_DIR, filename)
        
        # 读取现有数据
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # 检查是否已有 V6 结果
        if 'V6' in data:
            print(f"跳过 (已有V6): {filename}")
            completed += 1
            continue
        
        # 解析配置: consistent_comp_9uav_seed75.json
        parts = filename.replace('.json', '').split('_')
        uav_str = parts[2]  # "9uav"
        seed_str = parts[3]  # "seed75"
        num_uav = int(uav_str.replace('uav', ''))
        seed = int(seed_str.replace('seed', ''))
        
        print(f"\n[{completed+1}/{total}] 运行 V6: {num_uav} UAV, Seed {seed}")
        
        # 生成场景
        UE_pos, ground_AP_pos, UAV_pos_init = generate_scenario(seed, num_uav)
        config = create_config(num_uav, seed)
        
        # 运行 V6
        np.random.seed(seed)
        v6_opt = BalancedVirtualForceOptimizerV6(config)
        start = time.time()
        res_v6 = v6_opt.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
        t_v6 = time.time() - start
        
        # 添加 V6 结果
        data['V6'] = {
            'min': float(res_v6['final_min_rate']),
            'sum': float(res_v6['final_sum_rate'])
        }
        
        # 保存更新后的文件
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  V6: Min={data['V6']['min']:.4f}, Sum={data['V6']['sum']:.1f}, Time={t_v6:.1f}s")
        print(f"  ✅ 已更新: {filename}")
        
        completed += 1
    
    print("\n" + "=" * 70)
    print(f"  完成! 共处理 {completed} 个文件")
    print("=" * 70)

if __name__ == "__main__":
    run_v6_on_all()
