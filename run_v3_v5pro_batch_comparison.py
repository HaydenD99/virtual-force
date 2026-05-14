import numpy as np
import json
import time
import os
import warnings
warnings.filterwarnings('ignore')

from balanced_virtual_force_optimizer_v3 import BalancedVirtualForceOptimizerV3, create_balanced_config
from balanced_virtual_force_optimizer_v5_sinr import BalancedVirtualForceOptimizerV5
from genetic_algorithm_optimizer_discrete import DiscreteGeneticAlgorithmOptimizer, create_discrete_ga_config
from distributed_pso_optimizer import DistributedPSOOptimizer, create_distributed_pso_config
from newssa_optimizer import NewSSAOptimizer

# --- 实验配置 ---
SEEDS = list(range(41, 51))
CONFIGS = [
    {'uav': 6, 'L': 3},
    {'uav': 9, 'L': 4},
    {'uav': 12, 'L': 5}
]
OUTPUT_DIR = 'result/v3_v5pro_batch'
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

def run_experiment():
    print(f"🚀 开始 V3 vs V5-Pro vs 其他算法全量对比 (Seeds 41-50)")
    
    for config_item in CONFIGS:
        uav = config_item['uav']
        L = config_item['L']
        
        for seed in SEEDS:
            filename = f"comp_{uav}uav_L{L}_seed{seed}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            if os.path.exists(filepath):
                continue
                
            print(f"\n📍 配置: {uav}UAV, L={L}, Seed={seed}")
            UE_pos, ground_AP_pos, UAV_pos_init = generate_scenario(seed, uav)
            
            # 通用参数
            base_cfg = {
                'num_UAV': uav, 'num_serving_APs': L, 'random_seed': seed,
                'nbrOfRealizations': 50, 'max_iterations': 50, 'num_UE': 60, 'M': 4
            }
            
            # 各算法配置
            pso_cfg = create_distributed_pso_config()
            pso_cfg.update(base_cfg)
            pso_cfg['w_min_rate'], pso_cfg['w_sum_rate'] = 1.0, 0.1
            
            ga_cfg = create_discrete_ga_config()
            ga_cfg.update(base_cfg)
            
            vf_cfg = create_balanced_config()
            vf_cfg.update(base_cfg)

            results = {}
            
            # 初始状态
            temp_opt = BalancedVirtualForceOptimizerV5(vf_cfg)
            all_AP = np.vstack([ground_AP_pos, UAV_pos_init])
            _, _, betas = temp_opt.compute_channel_model(UE_pos, all_AP)
            mask = temp_opt.compute_AP_selection_mask(betas)
            rates, sum_r = temp_opt.compute_user_rates(UE_pos, all_AP, mask)
            results['initial'] = {'min': float(rates.min()), 'sum': float(sum_r)}

            # 优化器列表
            optimizers = [
                ('V3', BalancedVirtualForceOptimizerV3, vf_cfg),
                ('V5Pro', BalancedVirtualForceOptimizerV5, vf_cfg),
                ('GA', DiscreteGeneticAlgorithmOptimizer, ga_cfg),
                ('PSO', DistributedPSOOptimizer, pso_cfg),
                ('NewSSA', NewSSAOptimizer, base_cfg)
            ]
            
            for name, Cls, cfg in optimizers:
                print(f"  -> 运行 {name}...")
                np.random.seed(seed)
                opt = Cls(cfg)
                if name == 'GA':
                    opt.K, opt.G = 60, 4
                    res = opt.optimize(UE_pos, ground_AP_pos)
                else:
                    res = opt.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
                
                results[name] = {
                    'min': float(res['final_min_rate']),
                    'sum': float(res['final_sum_rate'])
                }
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ 保存结果: {filename}")

if __name__ == "__main__":
    run_experiment()
