"""
6 UAV场景下AP选择为2个的四算法对比
随机种子: 51, 62, 63, 77-87
"""

import numpy as np
import json
import time
import os
from balanced_virtual_force_optimizer_v3 import BalancedVirtualForceOptimizerV3, create_balanced_config
from genetic_algorithm_optimizer_discrete import DiscreteGeneticAlgorithmOptimizer, create_discrete_ga_config
from distributed_pso_optimizer import DistributedPSOOptimizer, create_distributed_pso_config
from newssa_optimizer import NewSSAOptimizer

# 配置
SEEDS = [51, 62, 63] + list(range(77, 88))  # 51, 62, 63, 77-87
NUM_UAV = 6
NUM_SERVING_APS = 2  # 关键：只选择2个AP
NBR_OF_REALIZATIONS = 50
TOTAL_EVALUATIONS = 1500
OUTPUT_DIR = 'result'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_neutral_scenario(seed=44, num_UAV=6):
    """生成中立场景"""
    np.random.seed(seed)
    
    square_length = 1000
    K = 60
    G = 4
    
    # UE位置
    UE_pos = np.random.uniform(
        low=[50, 50],
        high=[square_length - 50, square_length - 50],
        size=(K, 2)
    )
    UE_height = 1.65
    UE_pos = np.column_stack([UE_pos, np.ones(K) * UE_height])
    
    # Ground AP位置（均匀分布）
    ground_grid_x = np.linspace(square_length * 0.25, square_length * 0.75, 2)
    ground_grid_y = np.linspace(square_length * 0.25, square_length * 0.75, 2)
    ground_X, ground_Y = np.meshgrid(ground_grid_x, ground_grid_y)
    ground_AP_height = 15.0
    ground_AP_pos = np.column_stack([
        ground_X.flatten(),
        ground_Y.flatten(),
        np.ones(G) * ground_AP_height
    ])
    
    # UAV初始位置（6 UAVs: 2x3网格）
    uav_grid_x = np.linspace(250, 750, 3)
    uav_grid_y = np.linspace(250, 750, 2)
    UAV_x, UAV_y = np.meshgrid(uav_grid_x, uav_grid_y)
    UAV_pos = np.column_stack([
        UAV_x.flatten()[:num_UAV],
        UAV_y.flatten()[:num_UAV],
        np.ones(num_UAV) * 50.0
    ])
    
    return UE_pos, ground_AP_pos, UAV_pos


def create_configs(nbrOfRealizations=50, random_seed=44):
    """创建配置（L=2）"""
    
    base_config = {
        'square_length': 1000,
        'num_UE': 60,
        'num_ground_AP': 4,
        'num_UAV': NUM_UAV,
        'M': 4,
        'UE_height': 1.65,
        'ground_AP_height': 15.0,
        'UAV_height': 50.0,
        'UAV_height_min': 50.0,
        'UAV_height_max': 150.0,
        'nbrOfRealizations': nbrOfRealizations,
        'tau_c': 200,
        'tau_p': 60,
        'random_seed': random_seed,
        'num_serving_APs': NUM_SERVING_APS,  # L=2
    }
    
    # VF配置
    config_vf = create_balanced_config()
    config_vf.update(base_config)
    config_vf['max_iterations'] = 50
    
    # GA配置
    config_ga = create_discrete_ga_config()
    config_ga.update(base_config)
    config_ga['population_size'] = 30
    config_ga['max_generations'] = 50
    config_ga['crossover_rate'] = 0.8
    config_ga['mutation_rate'] = 0.15
    config_ga['elite_size'] = 3
    config_ga['tournament_size'] = 5
    
    # PSO配置
    config_pso = create_distributed_pso_config()
    config_pso.update(base_config)
    config_pso['num_particles'] = 30
    config_pso['max_iterations'] = 50
    config_pso['w_min_rate'] = 1.0
    config_pso['w_sum_rate'] = 0.1
    
    # NewSSA配置
    config_newssa = base_config.copy()
    config_newssa['newssa_n_sparrows'] = 30
    config_newssa['newssa_max_iter'] = 50
    config_newssa['newssa_pr'] = 0.2
    config_newssa['newssa_fr'] = 0.15
    config_newssa['newssa_st'] = 0.8
    
    configs = {
        'VF': config_vf,
        'GA': config_ga,
        'PSO': config_pso,
        'NewSSA': config_newssa
    }
    
    return configs


def run_single_experiment(seed):
    """运行单个种子的实验"""
    
    total_aps = NUM_UAV + 4
    utilization = NUM_SERVING_APS / total_aps * 100
    
    print(f"\n{'='*100}")
    print(f" 种子 {seed} - 6 UAVs, L={NUM_SERVING_APS} ({utilization:.1f}% 利用率) ".center(100))
    print(f"{'='*100}")
    
    # 生成场景
    UE_pos, ground_AP_pos, UAV_pos_init = generate_neutral_scenario(seed, NUM_UAV)
    
    # 创建配置
    configs = create_configs(NBR_OF_REALIZATIONS, seed)
    
    # 计算初始性能
    temp_optimizer = BalancedVirtualForceOptimizerV3(configs['VF'])
    all_AP_pos = np.vstack([ground_AP_pos, UAV_pos_init])
    H, Hhat, betas = temp_optimizer.compute_channel_model(UE_pos, all_AP_pos)
    mask = temp_optimizer.compute_AP_selection_mask(betas)
    initial_rates, initial_sum_rate = temp_optimizer.compute_user_rates(UE_pos, all_AP_pos, mask)
    initial_min_rate = initial_rates.min()
    
    print(f"初始: Min={initial_min_rate:.2f} Mbps, Sum={initial_sum_rate:.2f} Mbps")
    
    results = {
        'initial': {
            'min_rate': initial_min_rate,
            'sum_rate': initial_sum_rate,
            'mean_rate': initial_rates.mean(),
            'std_rate': initial_rates.std(),
            'time': 0
        }
    }
    
    # 运行优化器
    optimizers_info = [
        ('VF', 'BVF', BalancedVirtualForceOptimizerV3),
        ('GA', 'GA', DiscreteGeneticAlgorithmOptimizer),
        ('PSO', 'PSO', DistributedPSOOptimizer),
        ('NewSSA', 'NewSSA', NewSSAOptimizer)
    ]
    
    for method_key, method_name, OptimizerClass in optimizers_info:
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
        
        improvement = (results[method_key]['min_rate'] - initial_min_rate) / initial_min_rate * 100
        print(f"  {method_name:<8}: Min={results[method_key]['min_rate']:.2f} Mbps ({improvement:+.1f}%), "
              f"Time={opt_time:.1f}s")
    
    return results


def save_result(results, filename):
    """保存结果"""
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


def print_summary(all_results):
    """打印总结"""
    print("\n\n" + "="*100)
    print(" 6 UAVs, L=2 - 多种子总结 ".center(100))
    print("="*100)
    
    print(f"\n{'种子':<8} {'初始Min':<12} {'VF':<12} {'GA':<12} {'PSO':<12} {'NewSSA':<12}")
    print("-" * 100)
    
    for seed in SEEDS:
        if seed in all_results:
            res = all_results[seed]
            print(f"{seed:<8} {res['initial']['min_rate']:<12.2f} "
                  f"{res['VF']['min_rate']:<12.2f} "
                  f"{res['GA']['min_rate']:<12.2f} "
                  f"{res['PSO']['min_rate']:<12.2f} "
                  f"{res['NewSSA']['min_rate']:<12.2f}")
    
    # 计算统计
    print("-" * 100)
    methods = ['initial', 'VF', 'GA', 'PSO', 'NewSSA']
    stats = {}
    
    for method in methods:
        rates = [all_results[seed][method]['min_rate'] for seed in SEEDS if seed in all_results]
        stats[method] = {
            'mean': np.mean(rates),
            'std': np.std(rates),
            'min': np.min(rates),
            'max': np.max(rates)
        }
    
    print(f"{'平均':<8} {stats['initial']['mean']:<12.2f} "
          f"{stats['VF']['mean']:<12.2f} "
          f"{stats['GA']['mean']:<12.2f} "
          f"{stats['PSO']['mean']:<12.2f} "
          f"{stats['NewSSA']['mean']:<12.2f}")
    
    print(f"{'标准差':<8} {stats['initial']['std']:<12.2f} "
          f"{stats['VF']['std']:<12.2f} "
          f"{stats['GA']['std']:<12.2f} "
          f"{stats['PSO']['std']:<12.2f} "
          f"{stats['NewSSA']['std']:<12.2f}")
    
    # 平均提升
    print(f"\n平均提升:")
    for method in ['VF', 'GA', 'PSO', 'NewSSA']:
        improvement = (stats[method]['mean'] - stats['initial']['mean']) / stats['initial']['mean'] * 100
        print(f"  {method:<10}: {improvement:+.2f}%")
    
    # 排名
    print(f"\n算法排名（按平均Min Rate）:")
    ranking = sorted(['VF', 'GA', 'PSO', 'NewSSA'], 
                    key=lambda m: stats[m]['mean'], reverse=True)
    for rank, method in enumerate(ranking, 1):
        medal = ["🥇", "🥈", "🥉", ""][rank-1] if rank <= 3 else ""
        print(f"  {medal} {rank}. {method}: {stats[method]['mean']:.2f} Mbps")
    
    print("="*100)


def main():
    print("\n" + "="*100)
    print(" 6 UAV + L=2 多种子四算法对比 ".center(100))
    print("="*100)
    print(f"\n📊 实验配置:")
    print(f"   • 随机种子: {SEEDS}")
    print(f"   • 种子数量: {len(SEEDS)}")
    print(f"   • UAV数量: {NUM_UAV}")
    print(f"   • AP选择: L={NUM_SERVING_APS} (20% 利用率)")
    print(f"   • 算法: VF, GA, PSO, NewSSA")
    print(f"   • 信道实现: {NBR_OF_REALIZATIONS}")
    print(f"   • 总评估次数: {TOTAL_EVALUATIONS}")
    print(f"\n⏱️  预计时间: ~{len(SEEDS) * 15} 分钟 (每个种子约15分钟)")
    
    total_start = time.time()
    all_results = {}
    
    for idx, seed in enumerate(SEEDS, 1):
        print(f"\n{'🚀'*50}")
        print(f" 进度: {idx}/{len(SEEDS)} ".center(100))
        print(f"{'🚀'*50}")
        
        try:
            results = run_single_experiment(seed)
            all_results[seed] = results
            
            # 保存单个结果
            save_result(results, f'6uav_L2_seed{seed}.json')
            print(f"✓ 种子 {seed} 完成")
            
            # 保存中间结果
            combined_filepath = os.path.join(OUTPUT_DIR, f'6uav_L2_partial_{idx}.json')
            with open(combined_filepath, 'w') as f:
                def convert(obj):
                    if isinstance(obj, dict):
                        return {k: convert(v) for k, v in obj.items()}
                    elif isinstance(obj, (int, float, np.number)):
                        return float(obj)
                    else:
                        return obj
                json.dump({f'seed_{s}': convert(all_results[s]) for s in all_results}, f, indent=2)
            
        except Exception as e:
            print(f"\n❌ 种子 {seed} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印总结
    print_summary(all_results)
    
    # 保存最终结果
    final_filepath = os.path.join(OUTPUT_DIR, '6uav_L2_all_results.json')
    with open(final_filepath, 'w') as f:
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (int, float, np.number)):
                return float(obj)
            else:
                return obj
        json.dump({f'seed_{s}': convert(all_results[s]) for s in all_results}, f, indent=2)
    
    print(f"\n✓ 最终结果已保存到: {final_filepath}")
    
    total_time = time.time() - total_start
    print(f"\n✅ 所有实验完成！")
    print(f"   总时间: {total_time/60:.1f} 分钟")
    print(f"   成功: {len(all_results)}/{len(SEEDS)} 个种子")
    print("="*100)


if __name__ == "__main__":
    main()
