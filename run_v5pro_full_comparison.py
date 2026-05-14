"""
V5-Pro (IACF) 全方位性能对比脚本
对比算法：V5-Pro (IACF), GA, PSO, NewSSA
涵盖种子：51, 62, 63, 71, 75, 76, 77-87
涵盖配置：6UAV(L2/L3), 9UAV(L4), 12UAV(L5)
"""

import numpy as np
import json
import time
import os
from balanced_virtual_force_optimizer_v5_sinr import BalancedVirtualForceOptimizerV5
from genetic_algorithm_optimizer_discrete import DiscreteGeneticAlgorithmOptimizer, create_discrete_ga_config
from distributed_pso_optimizer import DistributedPSOOptimizer, create_distributed_pso_config
from newssa_optimizer import NewSSAOptimizer

# --- 实验矩阵定义 ---
EXPERIMENTS = [
    # 6 UAV L=3 基准
    {'uav': 6, 'L': 3, 'seeds': [71, 75, 76]},
    # 6 UAV L=2 专项
    {'uav': 6, 'L': 2, 'seeds': [51, 62, 63] + list(range(77, 88))},
    # 9 UAV L=4 增强
    {'uav': 9, 'L': 4, 'seeds': [71, 75, 76]},
    # 12 UAV L=5 增强
    {'uav': 12, 'L': 5, 'seeds': [71, 75, 76]}
]

OUTPUT_DIR = 'result/v5pro_comparison'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_scenario(seed, num_uav):
    np.random.seed(seed)
    K, G = 60, 4
    UE_pos = np.column_stack([np.random.uniform(50, 950, (K, 2)), np.full((K, 1), 1.65)])
    ground_AP_pos = np.array([[250,250,15], [250,750,15], [750,250,15], [750,750,15]])
    
    # 初始网格
    cols = 3 if num_uav <= 9 else 4
    rows = 2 if num_uav == 6 else 3
    gx, gy = np.meshgrid(np.linspace(250, 750, cols), np.linspace(250, 750, rows))
    UAV_pos = np.column_stack([gx.flatten()[:num_uav], gy.flatten()[:num_uav], np.full((num_uav, 1), 50.0)])
    return UE_pos, ground_AP_pos, UAV_pos

def get_configs(num_uav, L, seed):
    base = {
        'num_UAV': num_uav, 'num_serving_APs': L, 'random_seed': seed,
        'nbrOfRealizations': 50, 'max_iterations': 50, 'num_UE': 60, 'M': 4
    }
    
    # PSO 特殊权重
    pso_cfg = create_distributed_pso_config()
    pso_cfg.update(base)
    pso_cfg['w_min_rate'] = 1.0
    pso_cfg['w_sum_rate'] = 0.1
    
    # GA 特殊设置
    ga_cfg = create_discrete_ga_config()
    ga_cfg.update(base)
    ga_cfg['population_size'] = 30
    
    return {
        'VF': base,
        'GA': ga_cfg,
        'PSO': pso_cfg,
        'SSA': base
    }

def run_batch():
    print(f"开始全方位对比实验，结果将保存至 {OUTPUT_DIR}")
    
    for exp in EXPERIMENTS:
        num_uav = exp['uav']
        L = exp['L']
        for seed in exp['seeds']:
            filename = f"v5pro_comp_{num_uav}uav_L{L}_seed{seed}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            if os.path.exists(filepath):
                print(f"⏩ 跳过已存在的结果: {filename}")
                continue
                
            print(f"\n🔥 运行配置: {num_uav}UAV | L={L} | Seed={seed}")
            UE_pos, ground_AP_pos, UAV_pos_init = generate_scenario(seed, num_uav)
            cfgs = get_configs(num_uav, L, seed)
            
            results = {}
            
            # --- 初始状态 ---
            v5_temp = BalancedVirtualForceOptimizerV5(cfgs['VF'])
            all_AP = np.vstack([ground_AP_pos, UAV_pos_init])
            _, _, betas = v5_temp.compute_channel_model(UE_pos, all_AP)
            mask = v5_temp.compute_AP_selection_mask(betas)
            rates, sum_r = v5_temp.compute_user_rates(UE_pos, all_AP, mask)
            results['initial'] = {'min': float(rates.min()), 'sum': float(sum_r)}
            
            # --- 各优化器运行 ---
            opts = [
                ('V5Pro', BalancedVirtualForceOptimizerV5, cfgs['VF']),
                ('GA', DiscreteGeneticAlgorithmOptimizer, cfgs['GA']),
                ('PSO', DistributedPSOOptimizer, cfgs['PSO']),
                ('NewSSA', NewSSAOptimizer, cfgs['SSA'])
            ]
            
            for name, Cls, cfg in opts:
                print(f"  正在运行 {name}...")
                np.random.seed(seed)
                opt_obj = Cls(cfg)
                start = time.time()
                
                if name == 'GA':
                    # GA 接口略有不同
                    opt_obj.K, opt_obj.G = 60, 4
                    res = opt_obj.optimize(UE_pos, ground_AP_pos)
                else:
                    res = opt_obj.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
                
                duration = time.time() - start
                results[name] = {
                    'min': float(res['final_min_rate']),
                    'sum': float(res['final_sum_rate']),
                    'time': float(duration)
                }
                print(f"    {name} 完成: Min={res['final_min_rate']:.2f}, Sum={res['final_sum_rate']:.1f}")

            # 保存单次结果
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ 结果已保存: {filename}")

if __name__ == "__main__":
    run_batch()
