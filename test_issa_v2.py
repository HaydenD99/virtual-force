"""
测试严格遵循论文的ISSA优化器v2
- 优化2D位置(xy)，高度固定
- 优化目标：最大化最小用户速率
"""

import numpy as np
import time
from issa_optimizer_bvf_channel_v2 import ISSAOptimizerBVFChannel


def generate_neutral_scenario(seed=42):
    """生成中性场景"""
    np.random.seed(seed)
    
    square_length = 1000
    K = 60
    G = 4
    L = 9
    
    # UE位置
    UE_pos = np.random.uniform([50, 50], [square_length - 50, square_length - 50], (K, 2))
    UE_pos = np.column_stack([UE_pos, np.ones(K) * 1.65])
    
    # 地面AP
    ground_AP_pos = np.column_stack([
        [100, 900, 100, 900],
        [100, 100, 900, 900],
        [15.0] * 4
    ])
    
    # UAV初始位置（3x3网格）
    uav_grid_x = np.linspace(200, 800, 3)
    uav_grid_y = np.linspace(200, 800, 3)
    UAV_x, UAV_y = np.meshgrid(uav_grid_x, uav_grid_y)
    UAV_pos = np.column_stack([UAV_x.flatten(), UAV_y.flatten(), [50.0] * L])
    
    return UE_pos, ground_AP_pos, UAV_pos


def create_config():
    """创建配置"""
    return {
        'square_length': 1000,
        'num_UE': 60,
        'num_ground_AP': 4,
        'num_UAV': 9,
        'M': 4,
        
        'UE_height': 1.65,
        'ground_AP_height': 15.0,
        'UAV_height': 50.0,  # 固定高度
        
        # ISSA参数
        'issa_n_sparrows': 30,
        'issa_max_iter': 50,
        'issa_pd': 0.2,
        'issa_sd': 0.2,
        'issa_st': 0.8,
        
        # 信道参数
        'nbrOfRealizations': 50,
        'tau_c': 200,
        'tau_p': 60,
        'random_seed': 42,
        
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
    print(" ISSA优化器v2测试（严格遵循论文公式） ".center(80))
    print("="*80)
    print("\n📖 论文公式实现:")
    print("  • 公式14: 生产者更新")
    print("  • 公式15: 跟随者更新")
    print("  • 公式16: 警戒者更新")
    print("  • 公式17-18: 混沌初始化")
    print("  • 公式19-20: Cauchy-Gaussian变异")
    print("\n🎯 优化设置:")
    print("  • 优化目标: 最大化最小用户速率")
    print("  • 优化变量: UAV 2D位置(xy)")
    print("  • UAV高度: 固定50m")
    print()
    
    # 生成场景
    print("[1/3] 生成测试场景...")
    UE_pos, ground_AP_pos, UAV_pos = generate_neutral_scenario(seed=42)
    print(f"  ✓ UE数量: {len(UE_pos)}")
    print(f"  ✓ 地面AP: {len(ground_AP_pos)}")
    print(f"  ✓ UAV数量: {len(UAV_pos)}")
    
    # 创建优化器
    print("\n[2/3] 创建ISSA优化器...")
    config = create_config()
    optimizer = ISSAOptimizerBVFChannel(config)
    print(f"  ✓ 优化器已创建")
    
    # 运行优化
    print("\n[3/3] 开始优化...")
    np.random.seed(42)
    start_time = time.time()
    results = optimizer.optimize(UE_pos, ground_AP_pos, UAV_pos.copy())
    total_time = time.time() - start_time
    
    # 打印结果
    print("\n" + "="*80)
    print(" 最终结果 ".center(80))
    print("="*80)
    print(f"\n📊 性能指标:")
    print(f"  • 最小用户速率: {results['final_min_rate']:.4f} Mbps ⭐")
    print(f"  • 系统总速率:   {results['final_sum_rate']:.2f} Mbps")
    print(f"  • 平均用户速率: {results['final_rates'].mean():.4f} Mbps")
    print(f"  • 速率标准差:   {results['final_rates'].std():.4f} Mbps")
    print(f"\n⏱️  计算时间:")
    print(f"  • 优化时间:     {results['optimization_time']:.2f} 秒")
    print(f"  • 总运行时间:   {total_time:.2f} 秒")
    
    # 对比结果
    print("\n" + "="*80)
    print(" 与其他算法对比 ".center(80))
    print("="*80)
    
    initial_min_rate = 28.70
    vf_min_rate = 34.40
    ga_min_rate = 32.32
    pso_min_rate = 33.53
    issa_old = 24.65
    
    improvement = (results['final_min_rate'] - initial_min_rate) / initial_min_rate * 100
    
    print(f"\n算法性能对比:")
    print(f"  初始状态:       {initial_min_rate:.2f} Mbps")
    print(f"  Virtual Force:  {vf_min_rate:.2f} Mbps (+{(vf_min_rate-initial_min_rate)/initial_min_rate*100:.2f}%)")
    print(f"  GA:             {ga_min_rate:.2f} Mbps (+{(ga_min_rate-initial_min_rate)/initial_min_rate*100:.2f}%)")
    print(f"  PSO:            {pso_min_rate:.2f} Mbps (+{(pso_min_rate-initial_min_rate)/initial_min_rate*100:.2f}%)")
    print(f"  ISSA (旧版):    {issa_old:.2f} Mbps ({(issa_old-initial_min_rate)/initial_min_rate*100:.2f}%) ❌")
    print(f"  ISSA (论文版):  {results['final_min_rate']:.2f} Mbps ({improvement:+.2f}%)", end="")
    
    if results['final_min_rate'] > initial_min_rate:
        print(" ✅")
        if results['final_min_rate'] > 32:
            print("\n🎉🎉🎉 优秀！ISSA达到或接近GA/PSO水平！")
        elif results['final_min_rate'] > 30:
            print("\n🎉 很好！ISSA超越初始值显著！")
        else:
            print("\n✓ ISSA超越初始值！")
    elif results['final_min_rate'] > 27:
        print(" 📈 非常接近初始值！")
    else:
        print(" ⚠️ 仍需改进")
    
    # UAV位置变化
    print(f"\n📍 UAV位置优化:")
    print(f"  初始位置（部分）:")
    for i in range(min(3, len(UAV_pos))):
        print(f"    UAV{i}: [{UAV_pos[i,0]:.1f}, {UAV_pos[i,1]:.1f}, {UAV_pos[i,2]:.1f}]")
    print(f"  优化后位置（部分）:")
    for i in range(min(3, len(results['optimized_UAV_pos']))):
        uav = results['optimized_UAV_pos'][i]
        print(f"    UAV{i}: [{uav[0]:.1f}, {uav[1]:.1f}, {uav[2]:.1f}]")
    
    print("\n" + "="*80)
    print(" 完成！ ".center(80))
    print("="*80)
