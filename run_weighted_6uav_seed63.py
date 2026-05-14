"""
四算法对比 - 加权Fitness版本 - 6 UAVs - 种子63
Fitness = min_rate + 0.01 × sum_rate
"""

import numpy as np
import json
import time
import os

SEED = 63
NUM_UAV = 6
NBR_OF_REALIZATIONS = 50
TOTAL_EVALUATIONS = 1500
OUTPUT_DIR = 'result'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_result(results, filename):
    """保存结果到JSON文件"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    serializable_results = {}
    for method, data in results.items():
        serializable_results[method] = {
            'min_rate': float(data['min_rate']),
            'sum_rate': float(data['sum_rate']),
            'mean_rate': float(data['mean_rate']),
            'std_rate': float(data['std_rate']),
            'time': float(data['time'])
        }
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✓ 结果已保存到: {filepath}")

if __name__ == "__main__":
    print("\n" + "="*90)
    print(f" 四算法对比 - 加权Fitness - {NUM_UAV} UAVs - 种子{SEED} ".center(90))
    print("="*90)
    
    print(f"\n📊 实验配置:")
    print(f"   • 随机种子: {SEED}")
    print(f"   • UAV数量: {NUM_UAV}")
    print(f"   • Fitness: 加权版本（min_rate + 0.01×sum_rate）")
    print(f"   • 信道实现: {NBR_OF_REALIZATIONS}")
    print(f"   • 总评估次数: {TOTAL_EVALUATIONS}")
    
    start_time = time.time()
    
    try:
        from compare_optimizers_weighted import create_fair_configs, generate_neutral_scenario
        from balanced_virtual_force_optimizer_v3 import BalancedVirtualForceOptimizerV3
        from genetic_algorithm_optimizer_weighted import WeightedGeneticAlgorithmOptimizer
        from distributed_pso_optimizer_weighted import WeightedPSOOptimizer
        from newssa_optimizer_weighted import WeightedNewSSAOptimizer
        
        # 生成场景
        print("\n[1/3] 生成中立场景...")
        UE_pos, ground_AP_pos, _ = generate_neutral_scenario(seed=SEED)
        
        # 重新生成UAV初始位置（6 UAVs: 2x3网格）
        uav_grid_x = np.linspace(250, 750, 3)
        uav_grid_y = np.linspace(250, 750, 2)
        
        UAV_x, UAV_y = np.meshgrid(uav_grid_x, uav_grid_y)
        UAV_pos_init = np.column_stack([
            UAV_x.flatten()[:NUM_UAV],
            UAV_y.flatten()[:NUM_UAV],
            np.ones(NUM_UAV) * 50.0
        ])
        
        # 创建配置
        print("[2/3] 创建配置...")
        configs = create_fair_configs(TOTAL_EVALUATIONS, NBR_OF_REALIZATIONS, SEED)
        
        # 修改UAV数量
        for cfg in configs.values():
            cfg['num_UAV'] = NUM_UAV
        
        # 计算初始性能
        temp_optimizer = BalancedVirtualForceOptimizerV3(configs['VF'])
        all_AP_pos = np.vstack([ground_AP_pos, UAV_pos_init])
        H, Hhat, betas = temp_optimizer.compute_channel_model(UE_pos, all_AP_pos)
        mask = temp_optimizer.compute_AP_selection_mask(betas)
        initial_rates, initial_sum_rate = temp_optimizer.compute_user_rates(UE_pos, all_AP_pos, mask)
        initial_min_rate = initial_rates.min()
        
        results = {
            'initial': {
                'min_rate': initial_min_rate,
                'sum_rate': initial_sum_rate,
                'mean_rate': initial_rates.mean(),
                'std_rate': initial_rates.std(),
                'time': 0
            }
        }
        
        print(f"  初始 Min Rate: {initial_min_rate:.4f} Mbps")
        
        # 运行优化器
        print("\n[3/3] 运行优化器...")
        
        optimizers_info = [
            ('VF', 'BVF', BalancedVirtualForceOptimizerV3),
            ('GA', 'GA (Weighted)', WeightedGeneticAlgorithmOptimizer),
            ('PSO', 'PSO (Weighted)', WeightedPSOOptimizer),
            ('NewSSA', 'NewSSA (Weighted)', WeightedNewSSAOptimizer)
        ]
        
        for method_key, method_name, OptimizerClass in optimizers_info:
            print(f"\n  运行 {method_name}...")
            np.random.seed(SEED)
            
            optimizer = OptimizerClass(configs[method_key])
            UAV_pos_copy = UAV_pos_init.copy()
            
            opt_start = time.time()
            
            if method_key == 'GA':
                optimizer.K = len(UE_pos)
                optimizer.G = len(ground_AP_pos)
                opt_results = optimizer.optimize(UE_pos, ground_AP_pos)
            else:
                opt_results = optimizer.optimize(UE_pos, ground_AP_pos, UAV_pos_copy)
            
            opt_time = time.time() - opt_start
            
            results[method_key] = {
                'min_rate': opt_results['final_min_rate'],
                'sum_rate': opt_results['final_sum_rate'],
                'mean_rate': opt_results['final_rates'].mean(),
                'std_rate': opt_results['final_rates'].std(),
                'time': opt_time
            }
            
            print(f"    Min Rate: {results[method_key]['min_rate']:.4f} Mbps ({opt_time:.1f}s)")
        
        # 保存结果
        save_result(results, f'weighted_{NUM_UAV}uav_seed{SEED}.json')
        
        total_time = time.time() - start_time
        
        print(f"\n✅ 实验完成！总时间: {total_time/60:.1f} 分钟")
        
        # 打印总结
        print("\n" + "="*90)
        print(" 结果总结 ".center(90))
        print("="*90)
        for method in ['initial', 'VF', 'GA', 'PSO', 'NewSSA']:
            improvement = ((results[method]['min_rate'] - results['initial']['min_rate']) / 
                          results['initial']['min_rate'] * 100) if method != 'initial' else 0
            print(f"  {method:<10}: {results[method]['min_rate']:6.2f} Mbps ({improvement:+6.2f}%)")
        
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
