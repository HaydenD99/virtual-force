"""
快速测试阈值选择功能 - 仅测试6 UAV配置
"""

import os
# 设置matplotlib后端
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import json
import time

# 直接导入，不通过包装器
from balanced_virtual_force_optimizer_v3 import BalancedVirtualForceOptimizerV3, create_balanced_config
from threshold_based_optimizer_wrapper import ThresholdBasedAPSelection, analyze_ap_selection_distribution

SEED = 71
NUM_UAV = 6
NBR_OF_REALIZATIONS = 50

def generate_scenario(seed, num_uav):
    """生成场景"""
    np.random.seed(seed)
    
    K = 60
    G = 4
    
    # UE位置
    UE_pos = np.random.uniform(
        low=[50, 50],
        high=[950, 950],
        size=(K, 2)
    )
    UE_pos = np.column_stack([UE_pos, np.ones(K) * 1.65])
    
    # Ground AP位置
    ground_AP_pos = np.array([
        [250, 250, 15],
        [250, 750, 15],
        [750, 250, 15],
        [750, 750, 15]
    ])
    
    # UAV位置
    if num_uav == 6:
        uav_grid_x = np.linspace(250, 750, 3)
        uav_grid_y = np.linspace(250, 750, 2)
    else:
        uav_grid_x = np.linspace(200, 800, 3)
        uav_grid_y = np.linspace(200, 800, 3)
    
    UAV_x, UAV_y = np.meshgrid(uav_grid_x, uav_grid_y)
    UAV_pos = np.column_stack([
        UAV_x.flatten()[:num_uav],
        UAV_y.flatten()[:num_uav],
        np.ones(num_uav) * 50.0
    ])
    
    return UE_pos, ground_AP_pos, UAV_pos


def test_threshold_selection():
    """测试阈值选择"""
    print("\n" + "="*90)
    print(f" 阈值选择快速测试 - {NUM_UAV} UAVs - 种子{SEED} ".center(90))
    print("="*90)
    
    # 生成场景
    print("\n[1/3] 生成场景...")
    UE_pos, ground_AP_pos, UAV_pos = generate_scenario(SEED, NUM_UAV)
    print(f"  UE数: {len(UE_pos)}, Ground AP数: {len(ground_AP_pos)}, UAV数: {len(UAV_pos)}")
    
    # 创建优化器
    print("\n[2/3] 创建优化器...")
    config = create_balanced_config()
    config.update({
        'num_UAV': NUM_UAV,
        'num_ground_AP': 4,
        'num_UE': 60,
        'nbrOfRealizations': NBR_OF_REALIZATIONS,
        'tau_p': 60,
        'random_seed': SEED
    })
    
    optimizer = BalancedVirtualForceOptimizerV3(config)
    
    # 计算信道
    print("\n[3/3] 测试AP选择策略...")
    all_AP_pos = np.vstack([ground_AP_pos, UAV_pos])
    H, Hhat, betas = optimizer.compute_channel_model(UE_pos, all_AP_pos)
    
    print(f"\nbetas shape: {betas.shape}")
    print(f"betas range: [{betas.min():.2e}, {betas.max():.2e}]")
    
    # 固定选择
    print(f"\n--- 固定选择 (L=3) ---")
    top_3 = np.argpartition(betas, -3, axis=1)[:, -3:]
    mask_fixed = np.zeros_like(betas, dtype=bool)
    for k in range(len(UE_pos)):
        mask_fixed[k, top_3[k]] = True
    
    stats_fixed = analyze_ap_selection_distribution(betas, mask_fixed)
    print(f"每用户连接AP数: {stats_fixed['mean']:.2f} ± {stats_fixed['std']:.2f}")
    print(f"范围: [{stats_fixed['min']}, {stats_fixed['max']}]")
    print(f"利用率: {stats_fixed['avg_utilization']:.1f}%")
    
    rates_fixed, sum_rate_fixed = optimizer.compute_user_rates(UE_pos, all_AP_pos, mask_fixed)
    print(f"Min Rate: {rates_fixed.min():.4f} Mbps")
    print(f"Sum Rate: {sum_rate_fixed:.2f} Mbps")
    
    # 阈值选择
    print(f"\n--- 阈值选择 (70th percentile, range=[3,8]) ---")
    selector = ThresholdBasedAPSelection(threshold_percentile=70, min_serving=3, max_serving=8)
    mask_threshold = selector.compute_AP_selection_mask(betas)
    
    stats_threshold = analyze_ap_selection_distribution(betas, mask_threshold)
    print(f"每用户连接AP数: {stats_threshold['mean']:.2f} ± {stats_threshold['std']:.2f}")
    print(f"范围: [{stats_threshold['min']}, {stats_threshold['max']}]")
    print(f"利用率: {stats_threshold['avg_utilization']:.1f}%")
    
    rates_threshold, sum_rate_threshold = optimizer.compute_user_rates(UE_pos, all_AP_pos, mask_threshold)
    print(f"Min Rate: {rates_threshold.min():.4f} Mbps")
    print(f"Sum Rate: {sum_rate_threshold:.2f} Mbps")
    
    # 对比
    print(f"\n--- 性能对比 ---")
    improvement_min = (rates_threshold.min() - rates_fixed.min()) / rates_fixed.min() * 100
    improvement_sum = (sum_rate_threshold - sum_rate_fixed) / sum_rate_fixed * 100
    
    print(f"Min Rate改进: {improvement_min:+.2f}%")
    print(f"Sum Rate改进: {improvement_sum:+.2f}%")
    print(f"AP连接数增加: {stats_threshold['mean'] - stats_fixed['mean']:.2f} "
          f"({(stats_threshold['mean'] - stats_fixed['mean'])/stats_fixed['mean']*100:+.1f}%)")
    
    # 保存结果
    result = {
        'config': f'{NUM_UAV} UAVs, seed {SEED}',
        'fixed': {
            'min_rate': float(rates_fixed.min()),
            'sum_rate': float(sum_rate_fixed),
            'ap_stats': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                        for k, v in stats_fixed.items()}
        },
        'threshold': {
            'min_rate': float(rates_threshold.min()),
            'sum_rate': float(sum_rate_threshold),
            'ap_stats': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                        for k, v in stats_threshold.items()}
        },
        'improvement': {
            'min_rate_pct': float(improvement_min),
            'sum_rate_pct': float(improvement_sum)
        }
    }
    
    with open('result/threshold_quick_test.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ 结果已保存到: result/threshold_quick_test.json")
    print("\n" + "="*90)


if __name__ == "__main__":
    start_time = time.time()
    test_threshold_selection()
    print(f"\n总时间: {time.time() - start_time:.2f} 秒")
