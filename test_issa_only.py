"""
单独运行ISSA算法
使用与四算法对比完全相同的配置和随机种子
"""

import numpy as np
import time

from issa_optimizer_bvf_channel import ISSAOptimizerBVFChannel

def generate_neutral_scenario(seed=44):
    """
    使用中立的方法生成初始场景（与compare_optimizers_fair.py完全相同）
    """
    np.random.seed(seed)
    
    # 场景参数
    square_length = 1000  # 米
    K = 60  # UE数量
    G = 4   # 地面AP数量
    L = 9   # UAV数量
    
    # 1. UE位置：在区域内均匀随机分布
    UE_pos = np.random.uniform(
        low=[50, 50],
        high=[square_length - 50, square_length - 50],
        size=(K, 2)
    )
    UE_height = 1.65
    UE_pos = np.column_stack([UE_pos, np.ones(K) * UE_height])
    
    # 2. Ground AP位置：在角落均匀分布（2x2网格）
    ground_AP_x = np.array([100, 900, 100, 900])
    ground_AP_y = np.array([100, 100, 900, 900])
    ground_AP_height = 15.0
    ground_AP_pos = np.column_stack([
        ground_AP_x, 
        ground_AP_y, 
        np.ones(G) * ground_AP_height
    ])
    
    # 3. UAV初始位置：3x3网格均匀分布
    uav_grid_x = np.linspace(200, 800, 3)
    uav_grid_y = np.linspace(200, 800, 3)
    UAV_x, UAV_y = np.meshgrid(uav_grid_x, uav_grid_y)
    UAV_pos = np.column_stack([
        UAV_x.flatten(),
        UAV_y.flatten(),
        np.ones(L) * 50.0  # 初始高度50米
    ])
    
    print(f"[Neutral Initialization - Same as full comparison]")
    print(f"  ✓ UE positions: {K} users randomly distributed")
    print(f"  ✓ Ground APs: {G} APs at corners")
    print(f"  ✓ UAVs: {L} UAVs in 3x3 grid")
    print(f"  ✓ Random seed: {seed}")
    
    return UE_pos, ground_AP_pos, UAV_pos


def create_issa_config():
    """创建ISSA配置（与compare_optimizers_fair.py完全相同）"""
    return {
        # 基本参数
        'square_length': 1000,
        'num_UE': 60,
        'num_ground_AP': 4,
        'num_UAV': 9,
        'M': 4,
        
        # 高度设置
        'UE_height': 1.65,
        'ground_AP_height': 15.0,
        'UAV_height': 50.0,
        'UAV_height_min': 50.0,
        'UAV_height_max': 150.0,
        
        # ISSA参数（与对比实验完全相同）
        'issa_n_sparrows': 30,
        'issa_max_iter': 50,
        'issa_pd': 0.2,
        'issa_sd': 0.2,
        'issa_st': 0.8,
        
        # 通信参数
        'nbrOfRealizations': 50,  # 与其他算法统一
        'tau_c': 200,
        'tau_p': 60,  # 导频正交
        'random_seed': 42,
        
        # 信道参数
        'alpha': 3.67,
        'constant_term': -30.5,
        'B': 20e6,
        'p': 100,
        'Pmax': 1000,
        'noise_figure': 7,
        'num_serving_APs': 3,
        'ASD_deg': 10,
    }


def run_issa_only():
    """只运行ISSA算法"""
    print("="*80)
    print(" ISSA算法单独运行（使用四算法对比的完全相同配置） ".center(80))
    print("="*80)
    print()
    
    # 1. 生成初始场景（与四算法对比完全相同）
    print("[1/4] Generating neutral scenario...")
    UE_pos, ground_AP_pos, UAV_pos = generate_neutral_scenario(seed=42)
    print()
    
    # 2. 创建ISSA配置
    print("[2/4] Creating ISSA configuration...")
    config = create_issa_config()
    print(f"  ✓ Sparrows: {config['issa_n_sparrows']}")
    print(f"  ✓ Max iterations: {config['issa_max_iter']}")
    print(f"  ✓ Channel realizations: {config['nbrOfRealizations']}")
    print(f"  ✓ Pilot length: {config['tau_p']}")
    print()
    
    # 3. 创建优化器
    print("[3/4] Creating ISSA optimizer...")
    np.random.seed(42)  # 重置随机种子
    optimizer = ISSAOptimizerBVFChannel(config)
    print("  ✓ Optimizer created")
    print()
    
    # 4. 运行优化
    print("[4/4] Running ISSA optimization...")
    print("-"*80)
    
    start_time = time.time()
    results = optimizer.optimize(UE_pos, ground_AP_pos, UAV_pos.copy())
    total_time = time.time() - start_time
    
    print()
    print("="*80)
    print(" ISSA OPTIMIZATION COMPLETE ".center(80))
    print("="*80)
    print()
    
    # 打印结果
    print("📊 Final Results:")
    print(f"  • Minimum Rate:  {results['final_min_rate']:.4f} Mbps")
    print(f"  • Sum Rate:      {results['final_sum_rate']:.2f} Mbps")
    print(f"  • Mean Rate:     {results['final_rates'].mean():.4f} Mbps")
    print(f"  • Std Rate:      {results['final_rates'].std():.4f} Mbps")
    print(f"  • Time:          {total_time:.2f} seconds")
    print(f"  • Iterations:    {results['total_iterations']}")
    print()
    
    # 与初始状态对比
    print("📈 Comparison with other algorithms (from previous full run):")
    print(f"  Initial:         28.70 Mbps")
    print(f"  Virtual Force:   34.40 Mbps (+19.85%)")
    print(f"  GA:              32.32 Mbps (+12.61%)")
    print(f"  PSO:             33.53 Mbps (+16.82%)")
    print(f"  ISSA (old):      19.76 Mbps (-31.16%) ❌")
    print(f"  ISSA (new):      {results['final_min_rate']:.2f} Mbps ({(results['final_min_rate']-28.70)/28.70*100:+.2f}%)")
    print()
    
    # 判断改进
    improvement = (results['final_min_rate'] - 28.70) / 28.70 * 100
    if improvement > 0:
        print(f"✅ ISSA现在能提升最小速率了！改进: {improvement:+.2f}%")
    elif improvement > -10:
        print(f"⚠️  ISSA性能接近初始值（{improvement:+.2f}%），还需进一步优化")
    else:
        print(f"❌ ISSA性能仍然较差（{improvement:+.2f}%），需要检查算法")
    
    return results


if __name__ == "__main__":
    results = run_issa_only()
    
    print()
    print("="*80)
    print(" Done! ".center(80))
    print("="*80)
