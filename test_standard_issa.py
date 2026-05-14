"""
测试标准ISSA优化器（论文版改进方法）
"""

import numpy as np
import time
from issa_optimizer_standard import StandardISSAOptimizer


def generate_neutral_scenario(seed=42):
    """生成中性场景（与四算法对比相同）"""
    np.random.seed(seed)
    
    square_length = 1000
    K = 60  # UE数量
    G = 4   # 地面AP数量
    L = 9   # UAV数量
    
    # UE位置：随机分布
    UE_pos = np.random.uniform([50, 50], [square_length - 50, square_length - 50], (K, 2))
    UE_pos = np.column_stack([UE_pos, np.ones(K) * 1.65])  # 添加高度
    
    # 地面AP：四个角
    ground_AP_pos = np.column_stack([
        [100, 900, 100, 900],
        [100, 100, 900, 900],
        [15.0] * 4
    ])
    
    # UAV初始位置：3x3网格
    uav_grid_x = np.linspace(200, 800, 3)
    uav_grid_y = np.linspace(200, 800, 3)
    UAV_x, UAV_y = np.meshgrid(uav_grid_x, uav_grid_y)
    UAV_pos = np.column_stack([UAV_x.flatten(), UAV_y.flatten(), [50.0] * L])
    
    return UE_pos, ground_AP_pos, UAV_pos


def create_standard_issa_config():
    """创建标准ISSA配置"""
    return {
        # 场景参数
        'square_length': 1000,
        'num_UE': 60,
        'num_ground_AP': 4,
        'num_UAV': 9,
        'M': 4,
        
        # 高度参数
        'UE_height': 1.65,
        'ground_AP_height': 15.0,
        'UAV_height': 50.0,
        'UAV_height_min': 50.0,
        'UAV_height_max': 150.0,
        
        # ISSA算法参数
        'issa_n_sparrows': 30,
        'issa_max_iter': 50,
        'issa_pd': 0.2,  # 生产者比例
        'issa_sd': 0.2,  # 警戒者比例
        'issa_st': 0.8,  # 安全阈值
        
        # 信道参数（与BVF相同）
        'nbrOfRealizations': 50,
        'tau_c': 200,
        'tau_p': 60,
        'random_seed': 42,
        
        # 传播参数
        'alpha': 3.67,
        'constant_term': -30.5,
        'B': 20e6,
        'p': 100,
        'Pmax': 1000,
        'noise_figure': 7,
        'num_serving_APs': 3,
        'ASD_deg': 10,
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" 标准ISSA优化器测试（论文版改进方法） ".center(80))
    print("="*80)
    print("\n📖 论文改进方法:")
    print("  1. Chaotic Strategy: Y_{i+1,j} = sin(0.7π/Y_{i,j})")
    print("  2. Cauchy-Gaussian Mutation: 公式(19)(20)")
    print("  3. 标准SSA三角色更新机制")
    print()
    
    # 生成场景
    print("[1/4] 生成测试场景...")
    UE_pos, ground_AP_pos, UAV_pos = generate_neutral_scenario(seed=42)
    print(f"  ✓ UE数量: {len(UE_pos)}")
    print(f"  ✓ 地面AP: {len(ground_AP_pos)}")
    print(f"  ✓ UAV数量: {len(UAV_pos)}")
    
    # 创建配置
    print("\n[2/4] 创建优化器配置...")
    config = create_standard_issa_config()
    print(f"  ✓ 麻雀数量: {config['issa_n_sparrows']}")
    print(f"  ✓ 最大迭代: {config['issa_max_iter']}")
    print(f"  ✓ 信道实现: {config['nbrOfRealizations']}")
    print(f"  ✓ 导频长度: {config['tau_p']}")
    
    # 创建优化器
    print("\n[3/4] 创建标准ISSA优化器...")
    optimizer = StandardISSAOptimizer(config)
    print(f"  ✓ 优化器已创建")
    
    # 运行优化
    print("\n[4/4] 开始优化...")
    np.random.seed(42)
    start_time = time.time()
    results = optimizer.optimize(UE_pos, ground_AP_pos, UAV_pos.copy())
    total_time = time.time() - start_time
    
    # 打印结果
    print("\n" + "="*80)
    print(" 最终结果 ".center(80))
    print("="*80)
    print(f"\n📊 性能指标:")
    print(f"  • 最小用户速率: {results['final_min_rate']:.4f} Mbps")
    print(f"  • 系统总速率:   {results['final_sum_rate']:.2f} Mbps")
    print(f"  • 平均用户速率: {results['final_rates'].mean():.4f} Mbps")
    print(f"  • 速率标准差:   {results['final_rates'].std():.4f} Mbps")
    print(f"\n⏱️  计算时间:")
    print(f"  • 优化时间:     {results['optimization_time']:.2f} 秒")
    print(f"  • 总运行时间:   {total_time:.2f} 秒")
    print(f"  • 总迭代次数:   {results['total_iterations']}")
    
    # 与之前结果对比
    print("\n" + "="*80)
    print(" 与其他算法对比 ".center(80))
    print("="*80)
    print(f"\n基于之前的完整测试结果:")
    print(f"  初始状态:       28.70 Mbps")
    print(f"  Virtual Force:  34.40 Mbps (+19.85%)")
    print(f"  GA:             32.32 Mbps (+12.61%)")
    print(f"  PSO:            33.53 Mbps (+16.82%)")
    print(f"  ISSA (旧版):    24.65 Mbps (-14.12%) ❌")
    print(f"  ISSA (标准版):  {results['final_min_rate']:.2f} Mbps ({(results['final_min_rate']-28.70)/28.70*100:+.2f}%)", end="")
    
    if results['final_min_rate'] > 28.70:
        print(" ✅ 成功超越初始值!")
        if results['final_min_rate'] > 32:
            print("\n🎉🎉🎉 优秀！接近或超过GA/PSO水平！")
    elif results['final_min_rate'] > 27:
        print(" 📈 非常接近!")
    elif results['final_min_rate'] > 25:
        print(" ⚠️ 有改善，但还需优化")
    else:
        print(" ❌ 仍需改进")
    
    print("\n" + "="*80)
    print(" 完成！ ".center(80))
    print("="*80)
