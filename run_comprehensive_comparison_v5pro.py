"""
全算法对比脚本 - 统一初始环境 (100-900m UAV 初始布局)
环境配置: tau_p=60, Ground AP @ 2x2 Grid (250/750)
对比算法: V5Pro, GA, PSO, NewSSA
种子范围: 71-81
场景配置: 6, 9, 12 UAVs | L=3
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

# --- 实验配置 ---
SEEDS = list(range(71, 82))
UAV_COUNTS = [6, 9, 12]
L_SERVING = 3
OUTPUT_DIR = 'result/final_comparison_l3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_unified_scenario(seed, num_uav):
    np.random.seed(seed)
    K, G = 60, 4
    # 1. UE 位置
    UE_pos = np.column_stack([np.random.uniform(50, 950, (K, 2)), np.full((K, 1), 1.65)])
    
    # 2. 地面 AP：2x2 均匀分布 (250, 750)
    grid_points = [250, 750]
    ground_AP_pos = []
    for x in grid_points:
        for y in grid_points:
            ground_AP_pos.append([x, y, 15.0])
    ground_AP_pos = np.array(ground_AP_pos)
    
    # 3. UAV 初始位置：回退到原始方案 (100-900m 宽网格)
    cols = 3 if num_uav <= 9 else 4
    rows = 2 if num_uav == 6 else 3
    gx, gy = np.meshgrid(np.linspace(100, 900, cols), np.linspace(100, 900, rows))
    UAV_pos_init = np.column_stack([gx.flatten()[:num_uav], gy.flatten()[:num_uav], np.full((num_uav, 1), 50.0)])
    
    return UE_pos, ground_AP_pos, UAV_pos_init

def run_experiment():
    print(f"🚀 开始全算法综合对比 (Seeds 71-81, L=3)")
    
    for num_uav in UAV_COUNTS:
        for seed in SEEDS:
            filename = f"comp_{num_uav}uav_seed{seed}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            if os.path.exists(filepath):
                print(f"⏩ 跳过已存在的结果: {filename}")
                continue
                
            print(f"\n📍 正在运行: {num_uav}UAV | Seed {seed}")
            UE_pos, ground_AP_pos, UAV_pos_init = generate_unified_scenario(seed, num_uav)
            
            # 基础配置
            base_cfg = {
                'num_UAV': num_uav, 'num_serving_APs': L_SERVING, 'tau_p': 60,
                'random_seed': seed, 'max_iterations': 50, 'num_UE': 60, 'M': 4
            }
            
            results = {}
            
            # --- 0. 初始状态评估 (使用 V5Pro 类的逻辑确保准确性) ---
            v5_temp = BalancedVirtualForceOptimizerV5(base_cfg)
            all_AP_init = np.vstack([ground_AP_pos, UAV_pos_init])
            _, _, betas_init = v5_temp.compute_channel_model(UE_pos, all_AP_init)
            mask_init = v5_temp.compute_AP_selection_mask(betas_init)
            rates_init, sum_r_init = v5_temp.compute_user_rates(UE_pos, all_AP_init, mask_init)
            results['initial'] = {'min': float(rates_init.min()), 'sum': float(sum_r_init)}
            
            # --- 优化器列表 ---
            optimizers = [
                ('V5Pro', BalancedVirtualForceOptimizerV5, base_cfg),
                ('GA', DiscreteGeneticAlgorithmOptimizer, create_discrete_ga_config()),
                ('PSO', DistributedPSOOptimizer, create_distributed_pso_config()),
                ('NewSSA', NewSSAOptimizer, base_cfg)
            ]
            
            for name, Cls, cfg in optimizers:
                print(f"  -> 正在运行 {name}...")
                np.random.seed(seed)
                # 针对不同算法调整配置字典
                current_cfg = cfg.copy()
                current_cfg.update(base_cfg)
                if name == 'PSO':
                    current_cfg['w_min_rate'], current_cfg['w_sum_rate'] = 1.0, 0.1
                
                opt_obj = Cls(current_cfg)
                
                if name == 'GA':
                    opt_obj.K, opt_obj.G = 60, 4 # GA 内部硬编码兼容
                    res = opt_obj.optimize(UE_pos, ground_AP_pos)
                else:
                    res = opt_obj.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
                
                results[name] = {
                    'min': float(res['final_min_rate']),
                    'sum': float(res['final_sum_rate'])
                }
                print(f"    {name} 完成: Min {res['final_min_rate']:.4f}")

            # 保存单次实验结果
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ 结果已保存: {filename}")

if __name__ == "__main__":
    run_experiment()
