"""
基于信道质量阈值的AP选择 - 四算法对比实验
对比6、9、12 UAV配置下的性能

使用方案2：基于信道质量的阈值选择
"""

import os
# 设置matplotlib后端为非交互式，避免字体缓存问题
import matplotlib
matplotlib.use('Agg')

import numpy as np
import json
import time

# 导入基础优化器
from balanced_virtual_force_optimizer_v3 import BalancedVirtualForceOptimizerV3, create_balanced_config
from genetic_algorithm_optimizer_discrete import DiscreteGeneticAlgorithmOptimizer, create_discrete_ga_config
from distributed_pso_optimizer import DistributedPSOOptimizer, create_distributed_pso_config
from newssa_optimizer import NewSSAOptimizer

# 导入阈值选择包装器
from threshold_based_optimizer_wrapper import (
    create_threshold_based_vf_optimizer,
    create_threshold_based_ga_optimizer,
    create_threshold_based_pso_optimizer,
    create_threshold_based_newssa_optimizer,
    analyze_ap_selection_distribution
)

# 配置
SEED = 71  # 使用种子71（表现最好的种子）
UAV_CONFIGS = [6, 9, 12]
NBR_OF_REALIZATIONS = 50
TOTAL_EVALUATIONS = 1500
OUTPUT_DIR = 'result'
THRESHOLD_PERCENTILE = 70  # 70百分位阈值
MIN_SERVING = 3
MAX_SERVING = 8

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
    
    # Ground AP位置
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
    if L <= 9:
        grid_rows = 3
        grid_cols = 3
    else:
        grid_rows = 3
        grid_cols = 4
    
    uav_grid_x = np.linspace(200, 800, grid_cols)
    uav_grid_y = np.linspace(200, 800, grid_rows)
    UAV_x, UAV_y = np.meshgrid(uav_grid_x, uav_grid_y)
    UAV_pos = np.column_stack([
        UAV_x.flatten()[:L],
        UAV_y.flatten()[:L],
        np.ones(L) * 50.0
    ])
    
    return UE_pos, ground_AP_pos, UAV_pos


def create_threshold_configs(num_UAV, nbrOfRealizations=50, random_seed=44):
    """创建阈值选择配置"""
    
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
        # 阈值选择参数
        'threshold_percentile': THRESHOLD_PERCENTILE,
        'min_serving_APs': MIN_SERVING,
        'max_serving_APs': MAX_SERVING,
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
    
    # PSO配置
    config_pso = create_distributed_pso_config()
    config_pso.update(base_config)
    config_pso['num_particles'] = 30
    config_pso['max_iterations'] = 50
    
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


def run_single_experiment(num_UAV, seed):
    """运行单个UAV配置的实验"""
    
    print("\n" + "="*90)
    print(f" 阈值选择方案 - {num_UAV} UAVs - 种子{seed} ".center(90))
    print("="*90)
    print(f"阈值百分位: {THRESHOLD_PERCENTILE}%, 最少连接: {MIN_SERVING}, 最多连接: {MAX_SERVING}")
    
    # 生成场景
    print("\n[1/4] 生成场景...")
    UE_pos, ground_AP_pos, UAV_pos_init = generate_neutral_scenario(seed, num_UAV)
    
    # 创建配置
    print("[2/4] 创建阈值选择配置...")
    configs = create_threshold_configs(num_UAV, NBR_OF_REALIZATIONS, seed)
    
    # 创建阈值选择优化器类
    ThresholdVF = create_threshold_based_vf_optimizer(BalancedVirtualForceOptimizerV3)
    ThresholdGA = create_threshold_based_ga_optimizer(DiscreteGeneticAlgorithmOptimizer)
    ThresholdPSO = create_threshold_based_pso_optimizer(DistributedPSOOptimizer)
    ThresholdNewSSA = create_threshold_based_newssa_optimizer(NewSSAOptimizer)
    
    # 计算初始性能
    print("[3/4] 计算初始性能...")
    temp_optimizer = ThresholdVF(configs['VF'])
    all_AP_pos = np.vstack([ground_AP_pos, UAV_pos_init])
    H, Hhat, betas = temp_optimizer.compute_channel_model(UE_pos, all_AP_pos)
    mask = temp_optimizer.compute_AP_selection_mask(betas)
    initial_rates, initial_sum_rate = temp_optimizer.compute_user_rates(UE_pos, all_AP_pos, mask)
    initial_min_rate = initial_rates.min()
    
    # 分析AP选择分布
    ap_stats = analyze_ap_selection_distribution(betas, mask)
    print(f"  初始AP选择统计:")
    print(f"    总AP数: {ap_stats['total_APs']}")
    print(f"    每用户连接AP数: {ap_stats['mean']:.2f} ± {ap_stats['std']:.2f} "
          f"[{ap_stats['min']}, {ap_stats['max']}]")
    print(f"    平均利用率: {ap_stats['avg_utilization']:.1f}%")
    print(f"  初始 Min Rate: {initial_min_rate:.4f} Mbps")
    
    results = {
        'initial': {
            'min_rate': initial_min_rate,
            'sum_rate': initial_sum_rate,
            'mean_rate': initial_rates.mean(),
            'std_rate': initial_rates.std(),
            'time': 0,
            'ap_stats': ap_stats
        }
    }
    
    # 运行优化器
    print("\n[4/4] 运行优化器...")
    
    optimizers_info = [
        ('VF', 'BVF (Threshold)', ThresholdVF),
        ('GA', 'GA (Threshold)', ThresholdGA),
        ('PSO', 'PSO (Threshold)', ThresholdPSO),
        ('NewSSA', 'NewSSA (Threshold)', ThresholdNewSSA)
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
        
        # 分析最终AP选择
        final_UAV_pos = opt_results['optimized_UAV_pos']
        final_all_AP_pos = np.vstack([ground_AP_pos, final_UAV_pos])
        H_final, Hhat_final, betas_final = optimizer.compute_channel_model(UE_pos, final_all_AP_pos)
        mask_final = optimizer.compute_AP_selection_mask(betas_final)
        ap_stats_final = analyze_ap_selection_distribution(betas_final, mask_final)
        
        results[method_key] = {
            'min_rate': opt_results['final_min_rate'],
            'sum_rate': opt_results['final_sum_rate'],
            'mean_rate': opt_results['final_rates'].mean(),
            'std_rate': opt_results['final_rates'].std(),
            'time': opt_time,
            'ap_stats': ap_stats_final
        }
        
        print(f"    Min Rate: {results[method_key]['min_rate']:.4f} Mbps")
        print(f"    AP连接数: {ap_stats_final['mean']:.2f} ± {ap_stats_final['std']:.2f} "
              f"[{ap_stats_final['min']}, {ap_stats_final['max']}]")
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
            'time': float(data['time']),
            'ap_stats': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                        for k, v in data['ap_stats'].items()}
        }
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✓ 结果已保存到: {filepath}")


def print_summary(all_results):
    """打印总结"""
    print("\n\n" + "="*90)
    print(" 阈值选择方案 - 性能总结 ".center(90))
    print("="*90)
    
    for num_uav in UAV_CONFIGS:
        results = all_results[num_uav]
        total_APs = num_uav + 4
        
        print(f"\n{'='*90}")
        print(f" {num_uav} UAVs ({total_APs} 总AP) ".center(90))
        print('='*90)
        
        print(f"\n{'算法':<20} {'Min Rate':<12} {'提升':<12} {'AP连接数':<20}")
        print("-" * 90)
        
        initial_min = results['initial']['min_rate']
        initial_ap = results['initial']['ap_stats']['mean']
        
        print(f"{'Initial':<20} {initial_min:<12.2f} {'-':<12} "
              f"{initial_ap:.2f} ± {results['initial']['ap_stats']['std']:.2f}")
        
        for method in ['VF', 'GA', 'PSO', 'NewSSA']:
            min_rate = results[method]['min_rate']
            improvement = (min_rate - initial_min) / initial_min * 100
            ap_mean = results[method]['ap_stats']['mean']
            ap_std = results[method]['ap_stats']['std']
            
            print(f"{method:<20} {min_rate:<12.2f} {improvement:>+10.1f}%  "
                  f"{ap_mean:.2f} ± {ap_std:.2f}")


if __name__ == "__main__":
    print("\n" + "="*90)
    print(" 基于信道质量阈值的AP选择 - 四算法对比 ".center(90))
    print("="*90)
    print(f"\n📊 实验配置:")
    print(f"   • 随机种子: {SEED}")
    print(f"   • UAV配置: {UAV_CONFIGS}")
    print(f"   • AP选择策略: 阈值选择（百分位={THRESHOLD_PERCENTILE}%）")
    print(f"   • 连接范围: {MIN_SERVING}-{MAX_SERVING} APs")
    print(f"   • 信道实现: {NBR_OF_REALIZATIONS}")
    print(f"   • 总评估次数: {TOTAL_EVALUATIONS}")
    
    total_start = time.time()
    all_results = {}
    
    try:
        for num_uav in UAV_CONFIGS:
            print(f"\n{'🚀'*45}")
            print(f" 实验: {num_uav} UAVs ".center(90))
            print(f"{'🚀'*45}")
            
            results = run_single_experiment(num_uav, SEED)
            all_results[num_uav] = results
            
            # 保存单个结果
            save_result(results, f'threshold_based_{num_uav}uav_seed{SEED}.json')
        
        # 打印总结
        print_summary(all_results)
        
        # 保存综合结果
        combined_results = {
            f'{num_uav}UAV': all_results[num_uav]
            for num_uav in UAV_CONFIGS
        }
        
        filepath = os.path.join(OUTPUT_DIR, f'threshold_based_all_configs_seed{SEED}.json')
        with open(filepath, 'w') as f:
            def convert_to_serializable(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (int, float, np.number)):
                    return float(obj)
                else:
                    return obj
            json.dump(convert_to_serializable(combined_results), f, indent=2)
        
        print(f"\n✓ 综合结果已保存到: {filepath}")
        
        total_time = time.time() - total_start
        print(f"\n✅ 所有实验完成！总时间: {total_time/60:.1f} 分钟")
        
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
