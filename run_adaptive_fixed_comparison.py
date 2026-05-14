"""
自适应固定AP选择方案 - 四算法对比
配置: 9 UAVs (L=4), 12 UAVs (L=5)
种子: 71, 75, 76
"""

import numpy as np
import json
import time
import os
from balanced_virtual_force_optimizer_v3 import BalancedVirtualForceOptimizerV3, create_balanced_config
from genetic_algorithm_optimizer_discrete import DiscreteGeneticAlgorithmOptimizer, create_discrete_ga_config
from distributed_pso_optimizer import DistributedPSOOptimizer, create_distributed_pso_config
from newssa_optimizer import NewSSAOptimizer

SEEDS = [71, 75, 76]
NBR_OF_REALIZATIONS = 50
TOTAL_EVALUATIONS = 1500
OUTPUT_DIR = 'result'

# 自适应配置
ADAPTIVE_AP_CONFIG = {
    6: 3,   # 6 UAVs: L=3 (已有结果)
    9: 4,   # 9 UAVs: L=4 (新配置)
    12: 5,  # 12 UAVs: L=5 (新配置)
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_neutral_scenario(seed=44, num_UAV=9):
    """生成中立场景"""
    np.random.seed(seed)
    
    square_length = 1000
    K = 60
    G = 4
    L = num_UAV
    
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
    
    # UAV初始位置
    if L == 6:
        uav_grid_x = np.linspace(250, 750, 3)
        uav_grid_y = np.linspace(250, 750, 2)
    elif L == 9:
        uav_grid_x = np.linspace(200, 800, 3)
        uav_grid_y = np.linspace(200, 800, 3)
    else:  # 12
        uav_grid_x = np.linspace(200, 800, 4)
        uav_grid_y = np.linspace(200, 800, 3)
    
    UAV_x, UAV_y = np.meshgrid(uav_grid_x, uav_grid_y)
    UAV_pos = np.column_stack([
        UAV_x.flatten()[:L],
        UAV_y.flatten()[:L],
        np.ones(L) * 50.0
    ])
    
    return UE_pos, ground_AP_pos, UAV_pos


def create_adaptive_configs(num_UAV, nbrOfRealizations=50, random_seed=44):
    """创建自适应固定AP选择配置"""
    
    # 根据UAV数量选择AP连接数
    num_serving_APs = ADAPTIVE_AP_CONFIG[num_UAV]
    
    base_config = {
        'square_length': 1000,
        'num_UE': 60,
        'num_ground_AP': 4,
        'num_UAV': num_UAV,
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
        'num_serving_APs': num_serving_APs,  # 自适应配置
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


def run_single_experiment(num_uav, seed):
    """运行单个实验配置"""
    
    num_serving_aps = ADAPTIVE_AP_CONFIG[num_uav]
    total_aps = num_uav + 4
    
    print(f"\n{'='*100}")
    print(f" {num_uav} UAVs, L={num_serving_aps}, 种子{seed} ".center(100))
    print(f"{'='*100}")
    print(f"配置: {num_uav} UAVs + 4 Ground APs = {total_aps} 总AP")
    print(f"AP选择: L={num_serving_aps} ({num_serving_aps/total_aps*100:.1f}% 利用率)")
    
    # 生成场景
    print("\n[1/4] 生成场景...")
    UE_pos, ground_AP_pos, UAV_pos_init = generate_neutral_scenario(seed, num_uav)
    
    # 创建配置
    print("[2/4] 创建配置...")
    configs = create_adaptive_configs(num_uav, NBR_OF_REALIZATIONS, seed)
    
    # 计算初始性能
    print("[3/4] 计算初始性能...")
    temp_optimizer = BalancedVirtualForceOptimizerV3(configs['VF'])
    all_AP_pos = np.vstack([ground_AP_pos, UAV_pos_init])
    H, Hhat, betas = temp_optimizer.compute_channel_model(UE_pos, all_AP_pos)
    mask = temp_optimizer.compute_AP_selection_mask(betas)
    initial_rates, initial_sum_rate = temp_optimizer.compute_user_rates(UE_pos, all_AP_pos, mask)
    initial_min_rate = initial_rates.min()
    
    print(f"  初始 Min Rate: {initial_min_rate:.4f} Mbps")
    print(f"  初始 Sum Rate: {initial_sum_rate:.2f} Mbps")
    
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
    print("\n[4/4] 运行优化器...")
    
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
        
        improvement = (results[method_key]['min_rate'] - initial_min_rate) / initial_min_rate * 100
        print(f"    Min Rate: {results[method_key]['min_rate']:.4f} Mbps ({improvement:+.2f}%)")
        print(f"    Sum Rate: {results[method_key]['sum_rate']:.2f} Mbps")
        print(f"    时间: {opt_time:.1f}s")
    
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
    
    print(f"\n✓ 结果已保存到: {filepath}")


def print_summary(all_results):
    """打印总结"""
    print("\n\n" + "="*100)
    print(" 自适应固定AP选择 - 性能总结 ".center(100))
    print("="*100)
    
    # 按UAV配置分组
    uav_configs = sorted(set([key.split('_')[0] for key in all_results.keys()]))
    
    for uav_str in uav_configs:
        num_uav = int(uav_str.replace('UAV', ''))
        num_serving = ADAPTIVE_AP_CONFIG[num_uav]
        total_aps = num_uav + 4
        
        print(f"\n{'='*100}")
        print(f" {num_uav} UAVs (L={num_serving}, {total_aps} 总AP, {num_serving/total_aps*100:.1f}% 利用率) ".center(100))
        print('='*100)
        
        # 收集该UAV配置下所有种子的结果
        config_results = {seed: all_results[f'{uav_str}_seed{seed}'] 
                         for seed in SEEDS if f'{uav_str}_seed{seed}' in all_results}
        
        if not config_results:
            print("  无结果")
            continue
        
        print(f"\n{'种子':<8} {'初始Min':<12} {'VF':<12} {'GA':<12} {'PSO':<12} {'NewSSA':<12}")
        print("-" * 100)
        
        for seed in SEEDS:
            key = f'{uav_str}_seed{seed}'
            if key in all_results:
                res = all_results[key]
                print(f"{seed:<8} {res['initial']['min_rate']:<12.2f} "
                      f"{res['VF']['min_rate']:<12.2f} "
                      f"{res['GA']['min_rate']:<12.2f} "
                      f"{res['PSO']['min_rate']:<12.2f} "
                      f"{res['NewSSA']['min_rate']:<12.2f}")
        
        # 计算平均值
        print("-" * 100)
        methods = ['initial', 'VF', 'GA', 'PSO', 'NewSSA']
        avg_rates = {}
        for method in methods:
            rates = [all_results[f'{uav_str}_seed{seed}'][method]['min_rate'] 
                    for seed in SEEDS if f'{uav_str}_seed{seed}' in all_results]
            avg_rates[method] = np.mean(rates)
        
        print(f"{'平均':<8} {avg_rates['initial']:<12.2f} "
              f"{avg_rates['VF']:<12.2f} "
              f"{avg_rates['GA']:<12.2f} "
              f"{avg_rates['PSO']:<12.2f} "
              f"{avg_rates['NewSSA']:<12.2f}")
        
        # 计算提升
        print(f"\n平均提升:")
        for method in ['VF', 'GA', 'PSO', 'NewSSA']:
            improvement = (avg_rates[method] - avg_rates['initial']) / avg_rates['initial'] * 100
            print(f"  {method:<10}: {improvement:+.2f}%")
    
    print("\n" + "="*100)


def main():
    print("\n" + "="*100)
    print(" 自适应固定AP选择 - 四算法对比实验 ".center(100))
    print("="*100)
    print(f"\n📊 实验配置:")
    print(f"   • 随机种子: {SEEDS}")
    print(f"   • UAV配置:")
    print(f"     - 6 UAVs: L=3 (已有结果，不重新运行)")
    print(f"     - 9 UAVs: L=4 (30.8% 利用率)")
    print(f"     - 12 UAVs: L=5 (31.2% 利用率)")
    print(f"   • 算法: VF, GA, PSO, NewSSA")
    print(f"   • 信道实现: {NBR_OF_REALIZATIONS}")
    print(f"   • 总评估次数: {TOTAL_EVALUATIONS}")
    print(f"\n⏱️  预计时间: ~{len(SEEDS) * 2 * 15} 分钟 (9和12 UAVs, 每个配置约15分钟)")
    
    total_start = time.time()
    all_results = {}
    
    # 只运行9和12 UAV配置
    uav_configs = [9, 12]
    
    for num_uav in uav_configs:
        for seed in SEEDS:
            try:
                results = run_single_experiment(num_uav, seed)
                key = f'{num_uav}UAV_seed{seed}'
                all_results[key] = results
                
                # 保存单个结果
                save_result(results, f'adaptive_{num_uav}uav_seed{seed}.json')
                
            except Exception as e:
                print(f"\n❌ {num_uav} UAVs, 种子{seed} 失败: {e}")
                import traceback
                traceback.print_exc()
    
    # 打印总结
    print_summary(all_results)
    
    # 保存综合结果
    combined_filepath = os.path.join(OUTPUT_DIR, 'adaptive_fixed_all_results.json')
    with open(combined_filepath, 'w') as f:
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (int, float, np.number)):
                return float(obj)
            else:
                return obj
        json.dump(convert_to_serializable(all_results), f, indent=2)
    
    print(f"\n✓ 综合结果已保存到: {combined_filepath}")
    
    total_time = time.time() - total_start
    print(f"\n✅ 所有实验完成！总时间: {total_time/60:.1f} 分钟")
    print("="*100)


if __name__ == "__main__":
    main()
