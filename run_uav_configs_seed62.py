"""
四算法对比 - 原始Fitness版本 - 种子62
同时运行6 UAVs和12 UAVs配置
"""

import numpy as np
import json
import time
import os

SEED = 62
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
    
    print(f"✓ 结果已保存到: {filepath}")

def run_experiment(num_uav, seed):
    """运行单个实验"""
    from compare_optimizers_fair import create_fair_configs, generate_neutral_scenario
    from balanced_virtual_force_optimizer_v3 import BalancedVirtualForceOptimizerV3
    from genetic_algorithm_optimizer_discrete import DiscreteGeneticAlgorithmOptimizer
    from distributed_pso_optimizer import DistributedPSOOptimizer
    from newssa_optimizer import NewSSAOptimizer
    
    print("\n" + "="*90)
    print(f" 原始Fitness - {num_uav} UAVs - 种子{seed} ".center(90))
    print("="*90)
    
    # 生成场景
    print("\n[1/3] 生成中立场景...")
    UE_pos, ground_AP_pos, _ = generate_neutral_scenario(seed=seed)
    
    # 重新生成UAV初始位置（根据UAV数量）
    if num_uav == 6:
        uav_grid_x = np.linspace(250, 750, 3)
        uav_grid_y = np.linspace(250, 750, 2)
    else:  # 12 UAVs
        uav_grid_x = np.linspace(200, 800, 4)
        uav_grid_y = np.linspace(200, 800, 3)
    
    UAV_x, UAV_y = np.meshgrid(uav_grid_x, uav_grid_y)
    UAV_pos_init = np.column_stack([
        UAV_x.flatten()[:num_uav],
        UAV_y.flatten()[:num_uav],
        np.ones(num_uav) * 50.0
    ])
    
    # 创建配置
    print("[2/3] 创建配置...")
    configs = create_fair_configs(TOTAL_EVALUATIONS, NBR_OF_REALIZATIONS, seed)
    
    # 修改UAV数量
    for cfg in configs.values():
        cfg['num_UAV'] = num_uav
    
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
        ('GA', 'GA', DiscreteGeneticAlgorithmOptimizer),
        ('PSO', 'PSO', DistributedPSOOptimizer),
        ('NewSSA', 'NewSSA', NewSSAOptimizer)
    ]
    
    for method_key, method_name, OptimizerClass in optimizers_info:
        print(f"\n  运行 {method_name}...")
        np.random.seed(seed)
        
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
    save_result(results, f'original_{num_uav}uav_seed{seed}.json')
    
    # 打印总结
    print("\n" + "="*90)
    print(f" {num_uav} UAVs 结果总结 ".center(90))
    print("="*90)
    for method in ['initial', 'VF', 'GA', 'PSO', 'NewSSA']:
        improvement = ((results[method]['min_rate'] - results['initial']['min_rate']) / 
                      results['initial']['min_rate'] * 100) if method != 'initial' else 0
        print(f"  {method:<10}: {results[method]['min_rate']:6.2f} Mbps ({improvement:+6.2f}%)")
    
    return results

if __name__ == "__main__":
    print("\n" + "="*90)
    print(f" 种子{SEED} - 6 UAVs & 12 UAVs 配置对比 ".center(90))
    print("="*90)
    
    total_start = time.time()
    
    try:
        # 运行6 UAVs
        print("\n" + "🚀 开始实验 1/2: 6 UAVs")
        results_6uav = run_experiment(6, SEED)
        
        # 运行12 UAVs
        print("\n" + "🚀 开始实验 2/2: 12 UAVs")
        results_12uav = run_experiment(12, SEED)
        
        total_time = time.time() - total_start
        
        print("\n" + "="*90)
        print(" 所有实验完成 ".center(90))
        print("="*90)
        print(f"\n✅ 种子{SEED}的两个配置都已完成！")
        print(f"   总运行时间: {total_time/60:.1f} 分钟")
        print(f"\n📊 6 UAVs:")
        print(f"   初始: {results_6uav['initial']['min_rate']:.2f} Mbps")
        print(f"   VF:    {results_6uav['VF']['min_rate']:.2f} Mbps")
        print(f"   GA:    {results_6uav['GA']['min_rate']:.2f} Mbps")
        print(f"   PSO:   {results_6uav['PSO']['min_rate']:.2f} Mbps")
        print(f"   NewSSA:{results_6uav['NewSSA']['min_rate']:.2f} Mbps")
        print(f"\n📊 12 UAVs:")
        print(f"   初始: {results_12uav['initial']['min_rate']:.2f} Mbps")
        print(f"   VF:    {results_12uav['VF']['min_rate']:.2f} Mbps")
        print(f"   GA:    {results_12uav['GA']['min_rate']:.2f} Mbps")
        print(f"   PSO:   {results_12uav['PSO']['min_rate']:.2f} Mbps")
        print(f"   NewSSA:{results_12uav['NewSSA']['min_rate']:.2f} Mbps")
        
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
