"""
测试不同阈值百分位参数对性能的影响
测试: 50%, 60%, 75%, 80%
配置: 6, 9, 12 UAVs
"""

import os
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import json
import time
from threshold_based_optimizer_wrapper import create_threshold_based_vf_optimizer, analyze_ap_selection_distribution
from balanced_virtual_force_optimizer_v3 import BalancedVirtualForceOptimizerV3, create_balanced_config

SEED = 71
UAV_CONFIGS = [6, 9, 12]
THRESHOLD_PERCENTILES = [50, 60, 75, 80]  # 测试这些阈值
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


def test_single_config(num_uav, threshold_percentile, seed):
    """测试单个配置"""
    
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
        'threshold_percentile': threshold_percentile,
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
    ap_stats_init = analyze_ap_selection_distribution(betas, mask)
    
    # 运行优化
    start_time = time.time()
    results = optimizer.optimize(UE_pos, ground_AP_pos, UAV_pos.copy())
    opt_time = time.time() - start_time
    
    # 最终AP选择统计
    final_UAV_pos = results['optimized_UAV_pos']
    final_all_AP_pos = np.vstack([ground_AP_pos, final_UAV_pos])
    _, _, final_betas = optimizer.compute_channel_model(UE_pos, final_all_AP_pos)
    final_mask = optimizer.compute_AP_selection_mask(final_betas)
    ap_stats_final = analyze_ap_selection_distribution(final_betas, final_mask)
    
    return {
        'initial': {
            'min_rate': float(initial_rates.min()),
            'sum_rate': float(initial_sum_rate),
            'ap_stats': ap_stats_init
        },
        'VF': {
            'min_rate': float(results['final_min_rate']),
            'sum_rate': float(results['final_sum_rate']),
            'time': float(opt_time),
            'ap_stats': ap_stats_final
        }
    }


def main():
    print("\n" + "="*100)
    print(" 阈值百分位参数扫描实验 ".center(100))
    print("="*100)
    print(f"\n配置:")
    print(f"  种子: {SEED}")
    print(f"  UAV配置: {UAV_CONFIGS}")
    print(f"  测试阈值: {THRESHOLD_PERCENTILES}")
    print(f"  信道实现: {NBR_OF_REALIZATIONS}")
    
    all_results = {}
    total_start = time.time()
    
    for num_uav in UAV_CONFIGS:
        print(f"\n{'='*100}")
        print(f" {num_uav} UAVs ({num_uav + 4} 总AP) ".center(100))
        print(f"{'='*100}")
        
        uav_results = {}
        
        for percentile in THRESHOLD_PERCENTILES:
            print(f"\n  测试阈值 {percentile}%...")
            
            try:
                result = test_single_config(num_uav, percentile, SEED)
                uav_results[f'p{percentile}'] = result
                
                init_min = result['initial']['min_rate']
                vf_min = result['VF']['min_rate']
                improvement = (vf_min - init_min) / init_min * 100
                ap_mean = result['VF']['ap_stats']['mean']
                
                print(f"    初始: {init_min:.2f} Mbps, VF: {vf_min:.2f} Mbps "
                      f"(+{improvement:.2f}%), AP: {ap_mean:.2f}, 时间: {result['VF']['time']:.1f}s")
                
            except Exception as e:
                print(f"    ❌ 失败: {e}")
                import traceback
                traceback.print_exc()
        
        all_results[f'{num_uav}UAV'] = uav_results
    
    # 保存结果
    output_file = f'result/threshold_percentile_scan_seed{SEED}.json'
    with open(output_file, 'w') as f:
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (int, float, np.number)):
                return float(obj)
            else:
                return obj
        json.dump(convert_to_serializable(all_results), f, indent=2)
    
    print(f"\n✓ 结果已保存到: {output_file}")
    
    # 打印对比表
    print("\n\n" + "="*100)
    print(" 阈值百分位对比总结 ".center(100))
    print("="*100)
    
    # 加载固定选择结果作为基准
    try:
        fixed_6 = json.load(open('result/original_6uav_seed71.json'))
        fixed_9_all = json.load(open('result/seeds_66_76_partial_11.json'))
        fixed_9 = fixed_9_all['seed_71']
        fixed_12 = json.load(open('result/original_12uav_seed71.json'))
        
        fixed_results = {
            '6UAV': fixed_6['VF']['min_rate'],
            '9UAV': fixed_9['VF']['min_rate'],
            '12UAV': fixed_12['VF']['min_rate']
        }
        
        for num_uav in UAV_CONFIGS:
            key = f'{num_uav}UAV'
            print(f"\n{key} - 固定选择基准: {fixed_results[key]:.4f} Mbps")
            print(f"{'阈值':<10} {'VF Min':<12} {'vs固定':<12} {'AP数':<12} {'优化提升'}")
            print("-" * 100)
            
            for percentile in THRESHOLD_PERCENTILES:
                p_key = f'p{percentile}'
                if p_key in all_results[key]:
                    result = all_results[key][p_key]
                    vf_min = result['VF']['min_rate']
                    init_min = result['initial']['min_rate']
                    vs_fixed = (vf_min - fixed_results[key]) / fixed_results[key] * 100
                    opt_improvement = (vf_min - init_min) / init_min * 100
                    ap_mean = result['VF']['ap_stats']['mean']
                    
                    marker = "✓" if vs_fixed > 0 else "✗"
                    print(f"{percentile}%{marker:<7} {vf_min:<12.4f} {vs_fixed:>+10.2f}%  {ap_mean:<12.2f} {opt_improvement:>+10.2f}%")
        
        # 找出最佳阈值
        print(f"\n{'='*100}")
        print(" 最佳阈值推荐 ".center(100))
        print(f"{'='*100}\n")
        
        for num_uav in UAV_CONFIGS:
            key = f'{num_uav}UAV'
            best_percentile = None
            best_min_rate = 0
            
            for percentile in THRESHOLD_PERCENTILES:
                p_key = f'p{percentile}'
                if p_key in all_results[key]:
                    vf_min = all_results[key][p_key]['VF']['min_rate']
                    if vf_min > best_min_rate:
                        best_min_rate = vf_min
                        best_percentile = percentile
            
            if best_percentile:
                vs_fixed = (best_min_rate - fixed_results[key]) / fixed_results[key] * 100
                print(f"  {key}: 最佳阈值 = {best_percentile}%, Min Rate = {best_min_rate:.4f} Mbps "
                      f"(vs固定: {vs_fixed:+.2f}%)")
    
    except Exception as e:
        print(f"\n无法加载固定选择基准: {e}")
    
    total_time = time.time() - total_start
    print(f"\n总时间: {total_time/60:.1f} 分钟")
    print("="*100)


if __name__ == "__main__":
    main()
