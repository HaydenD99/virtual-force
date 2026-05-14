"""
运行所有三个UAV配置的阈值选择实验
6、9、12 UAVs
"""

import os
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import json
import time
from threshold_based_optimizer_wrapper import create_threshold_based_vf_optimizer
from balanced_virtual_force_optimizer_v3 import BalancedVirtualForceOptimizerV3, create_balanced_config

SEED = 71
UAV_CONFIGS = [6, 9, 12]
NBR_OF_REALIZATIONS = 50

def generate_scenario(seed, num_uav):
    """生成场景"""
    np.random.seed(seed)
    
    K = 60
    
    # UE位置
    UE_pos = np.random.uniform(low=[50, 50], high=[950, 950], size=(K, 2))
    UE_pos = np.column_stack([UE_pos, np.ones(K) * 1.65])
    
    # Ground AP位置
    ground_AP_pos = np.array([[250,250,15], [250,750,15], [750,250,15], [750,750,15]])
    
    # UAV位置
    if num_uav == 6:
        uav_grid_x = np.linspace(250, 750, 3)
        uav_grid_y = np.linspace(250, 750, 2)
    elif num_uav == 9:
        uav_grid_x = np.linspace(200, 800, 3)
        uav_grid_y = np.linspace(200, 800, 3)
    else:  # 12
        uav_grid_x = np.linspace(200, 800, 4)
        uav_grid_y = np.linspace(200, 800, 3)
    
    UAV_x, UAV_y = np.meshgrid(uav_grid_x, uav_grid_y)
    UAV_pos = np.column_stack([
        UAV_x.flatten()[:num_uav],
        UAV_y.flatten()[:num_uav],
        np.ones(num_uav) * 50.0
    ])
    
    return UE_pos, ground_AP_pos, UAV_pos


def run_vf_with_threshold(num_uav, seed):
    """运行VF优化器（阈值选择版本）"""
    
    print(f"\n{'='*90}")
    print(f" {num_uav} UAVs - VF优化器（阈值选择） ".center(90))
    print(f"{'='*90}")
    
    # 生成场景
    UE_pos, ground_AP_pos, UAV_pos = generate_scenario(seed, num_uav)
    
    # 创建配置
    config = create_balanced_config()
    config.update({
        'num_UAV': num_uav,
        'num_ground_AP': 4,
        'num_UE': 60,
        'nbrOfRealizations': NBR_OF_REALIZATIONS,
        'tau_p': 60,
        'random_seed': seed,
        'threshold_percentile': 70,
        'min_serving_APs': 3,
        'max_serving_APs': 8,
        'max_iterations': 50
    })
    
    # 创建阈值优化器
    ThresholdVF = create_threshold_based_vf_optimizer(BalancedVirtualForceOptimizerV3)
    optimizer = ThresholdVF(config)
    
    # 计算初始性能
    all_AP_pos = np.vstack([ground_AP_pos, UAV_pos])
    H, Hhat, betas = optimizer.compute_channel_model(UE_pos, all_AP_pos)
    mask = optimizer.compute_AP_selection_mask(betas)
    initial_rates, initial_sum_rate = optimizer.compute_user_rates(UE_pos, all_AP_pos, mask)
    
    # AP选择统计
    ap_connections = mask.sum(axis=1)
    ap_stats = {
        'mean': float(np.mean(ap_connections)),
        'std': float(np.std(ap_connections)),
        'min': int(np.min(ap_connections)),
        'max': int(np.max(ap_connections))
    }
    
    print(f"初始性能:")
    print(f"  Min Rate: {initial_rates.min():.4f} Mbps")
    print(f"  Sum Rate: {initial_sum_rate:.2f} Mbps")
    print(f"  AP连接数: {ap_stats['mean']:.2f}±{ap_stats['std']:.2f} [{ap_stats['min']}, {ap_stats['max']}]")
    
    # 运行优化
    print(f"\n运行VF优化...")
    start_time = time.time()
    results = optimizer.optimize(UE_pos, ground_AP_pos, UAV_pos.copy())
    opt_time = time.time() - start_time
    
    # 最终AP选择统计
    final_UAV_pos = results['optimized_UAV_pos']
    final_all_AP_pos = np.vstack([ground_AP_pos, final_UAV_pos])
    _, _, final_betas = optimizer.compute_channel_model(UE_pos, final_all_AP_pos)
    final_mask = optimizer.compute_AP_selection_mask(final_betas)
    final_ap_connections = final_mask.sum(axis=1)
    
    final_ap_stats = {
        'mean': float(np.mean(final_ap_connections)),
        'std': float(np.std(final_ap_connections)),
        'min': int(np.min(final_ap_connections)),
        'max': int(np.max(final_ap_connections))
    }
    
    print(f"\n最终性能:")
    print(f"  Min Rate: {results['final_min_rate']:.4f} Mbps "
          f"(+{(results['final_min_rate']-initial_rates.min())/initial_rates.min()*100:.2f}%)")
    print(f"  Sum Rate: {results['final_sum_rate']:.2f} Mbps "
          f"(+{(results['final_sum_rate']-initial_sum_rate)/initial_sum_rate*100:.2f}%)")
    print(f"  AP连接数: {final_ap_stats['mean']:.2f}±{final_ap_stats['std']:.2f} "
          f"[{final_ap_stats['min']}, {final_ap_stats['max']}]")
    print(f"  优化时间: {opt_time:.1f} 秒")
    
    return {
        'initial': {
            'min_rate': float(initial_rates.min()),
            'sum_rate': float(initial_sum_rate),
            'ap_stats': ap_stats
        },
        'VF': {
            'min_rate': float(results['final_min_rate']),
            'sum_rate': float(results['final_sum_rate']),
            'mean_rate': float(results['final_rates'].mean()),
            'std_rate': float(results['final_rates'].std()),
            'time': float(opt_time),
            'ap_stats': final_ap_stats
        }
    }


def main():
    print("\n" + "="*90)
    print(" 阈值选择方案 - VF优化器性能测试 ".center(90))
    print("="*90)
    print(f"\n配置:")
    print(f"  种子: {SEED}")
    print(f"  UAV配置: {UAV_CONFIGS}")
    print(f"  阈值选择: 70th percentile, range=[3,8]")
    print(f"  信道实现: {NBR_OF_REALIZATIONS}")
    
    all_results = {}
    total_start = time.time()
    
    for num_uav in UAV_CONFIGS:
        try:
            results = run_vf_with_threshold(num_uav, SEED)
            all_results[num_uav] = results
            
            # 保存单个结果
            with open(f'result/threshold_vf_{num_uav}uav_seed{SEED}.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ 结果已保存到: result/threshold_vf_{num_uav}uav_seed{SEED}.json")
            
        except Exception as e:
            print(f"\n❌ {num_uav} UAVs 实验失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印总结
    print("\n\n" + "="*90)
    print(" 总结 ".center(90))
    print("="*90)
    
    print(f"\n{'UAV数':<10} {'总AP':<10} {'初始Min':<12} {'VF Min':<12} {'提升':<12} {'AP连接数'}")
    print("-" * 90)
    
    for num_uav in UAV_CONFIGS:
        if num_uav in all_results:
            res = all_results[num_uav]
            init_min = res['initial']['min_rate']
            vf_min = res['VF']['min_rate']
            improvement = (vf_min - init_min) / init_min * 100
            ap_mean = res['VF']['ap_stats']['mean']
            ap_std = res['VF']['ap_stats']['std']
            total_aps = num_uav + 4
            
            print(f"{num_uav:<10} {total_aps:<10} {init_min:<12.2f} {vf_min:<12.2f} "
                  f"{improvement:>+10.1f}%  {ap_mean:.2f}±{ap_std:.2f}")
    
    total_time = time.time() - total_start
    print(f"\n总时间: {total_time/60:.1f} 分钟")
    print("="*90)


if __name__ == "__main__":
    main()
