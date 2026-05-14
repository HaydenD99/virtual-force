"""
ISSA算法单独测试脚本
用于验证算法能否正常运行，检查是否有bug
"""

import numpy as np
import time
import sys

# 导入ISSA优化器
from issa_optimizer_bvf_channel import ISSAOptimizerBVFChannel

def create_test_config():
    """创建测试配置"""
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
        
        # ISSA参数
        'issa_n_sparrows': 10,  # 减少到10个以便快速测试
        'issa_max_iter': 5,     # 只测试5次迭代
        'issa_pd': 0.2,
        'issa_sd': 0.2,
        'issa_st': 0.8,
        
        # 通信参数
        'nbrOfRealizations': 20,  # 减少到20以加快测试
        'tau_c': 200,
        'tau_p': 60,
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

def generate_test_positions(config):
    """生成测试位置"""
    np.random.seed(42)
    
    K = config['num_UE']
    G = config['num_ground_AP']
    L = config['num_UAV']
    square_length = config['square_length']
    
    # UE位置
    UE_pos = np.random.uniform(
        low=[50, 50],
        high=[square_length - 50, square_length - 50],
        size=(K, 2)
    )
    UE_pos = np.column_stack([UE_pos, np.ones(K) * config['UE_height']])
    
    # Ground AP位置（角落）
    ground_AP_x = np.array([100, 900, 100, 900])
    ground_AP_y = np.array([100, 100, 900, 900])
    ground_AP_pos = np.column_stack([
        ground_AP_x, 
        ground_AP_y, 
        np.ones(G) * config['ground_AP_height']
    ])
    
    # UAV初始位置（3x3网格）
    uav_grid_x = np.linspace(200, 800, 3)
    uav_grid_y = np.linspace(200, 800, 3)
    UAV_x, UAV_y = np.meshgrid(uav_grid_x, uav_grid_y)
    UAV_pos = np.column_stack([
        UAV_x.flatten(),
        UAV_y.flatten(),
        np.ones(L) * config['UAV_height']
    ])
    
    return UE_pos, ground_AP_pos, UAV_pos

def test_issa():
    """测试ISSA算法"""
    print("="*80)
    print(" ISSA算法单独测试 ".center(80))
    print("="*80)
    print()
    
    # 1. 创建配置
    print("[1/5] 创建配置...")
    config = create_test_config()
    print(f"  ✓ 麻雀数量: {config['issa_n_sparrows']}")
    print(f"  ✓ 迭代次数: {config['issa_max_iter']}")
    print(f"  ✓ 用户数量: {config['num_UE']}")
    print()
    
    # 2. 生成位置
    print("[2/5] 生成测试位置...")
    try:
        UE_pos, ground_AP_pos, UAV_pos = generate_test_positions(config)
        print(f"  ✓ UE位置: {UE_pos.shape}")
        print(f"  ✓ Ground AP位置: {ground_AP_pos.shape}")
        print(f"  ✓ UAV位置: {UAV_pos.shape}")
    except Exception as e:
        print(f"  ✗ 生成位置失败: {e}")
        return False
    print()
    
    # 3. 创建优化器
    print("[3/5] 创建ISSA优化器...")
    try:
        optimizer = ISSAOptimizerBVFChannel(config)
        print(f"  ✓ 优化器创建成功")
        print(f"  ✓ 种群大小: {optimizer.n_sparrows}")
        print(f"  ✓ 最大迭代: {optimizer.max_iter}")
    except Exception as e:
        print(f"  ✗ 创建优化器失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # 4. 测试初始化
    print("[4/5] 测试初始化...")
    try:
        dim = optimizer.L * 3
        population = optimizer._chaotic_initialization(optimizer.n_sparrows, dim)
        print(f"  ✓ 混沌初始化成功: {population.shape}")
        
        # 检查是否有NaN或Inf
        if np.any(np.isnan(population)):
            print(f"  ⚠️ 警告: 初始化包含NaN")
        if np.any(np.isinf(population)):
            print(f"  ⚠️ 警告: 初始化包含Inf")
        
        # 检查边界
        UAV_pos_decoded = optimizer._decode_population(population[0:1])[0]
        print(f"  ✓ 解码测试成功: {UAV_pos_decoded.shape}")
    except Exception as e:
        print(f"  ✗ 初始化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # 5. 运行优化
    print("[5/5] 运行ISSA优化...")
    print("-"*80)
    try:
        start_time = time.time()
        results = optimizer.optimize(UE_pos, ground_AP_pos, UAV_pos)
        elapsed_time = time.time() - start_time
        
        print()
        print("-"*80)
        print("✅ ISSA优化成功完成!")
        print()
        print("结果:")
        print(f"  • 最终最小速率: {results['final_min_rate']:.4f} Mbps")
        print(f"  • 最终总速率:   {results['final_sum_rate']:.2f} Mbps")
        print(f"  • 最终平均速率: {results['final_rates'].mean():.4f} Mbps")
        print(f"  • 速率标准差:   {results['final_rates'].std():.4f} Mbps")
        print(f"  • 优化时间:     {elapsed_time:.2f} 秒")
        print(f"  • 总迭代次数:   {results['total_iterations']}")
        print(f"  • 平均每次:     {elapsed_time/results['total_iterations']:.3f} 秒/迭代")
        
        # 验证结果有效性
        print()
        print("有效性检查:")
        if results['final_min_rate'] > 0:
            print(f"  ✓ 最小速率为正: {results['final_min_rate']:.4f} Mbps")
        else:
            print(f"  ✗ 最小速率异常: {results['final_min_rate']:.4f} Mbps")
            
        if results['final_sum_rate'] > 0:
            print(f"  ✓ 总速率为正: {results['final_sum_rate']:.2f} Mbps")
        else:
            print(f"  ✗ 总速率异常: {results['final_sum_rate']:.2f} Mbps")
            
        if np.all(results['final_rates'] > 0):
            print(f"  ✓ 所有用户速率为正")
        else:
            print(f"  ✗ 存在非正速率")
            
        # 检查收敛情况
        if 'history' in results:
            history = results['history']
            if len(history['min_rates']) > 0:
                initial_min = history['min_rates'][0]
                final_min = history['min_rates'][-1]
                improvement = (final_min - initial_min) / initial_min * 100
                print(f"  • 最小速率改进: {improvement:+.2f}%")
        
        return True
        
    except Exception as e:
        print()
        print(f"✗ ISSA优化失败: {e}")
        print()
        print("错误堆栈:")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    success = test_issa()
    
    print()
    print("="*80)
    if success:
        print(" 测试通过！ISSA算法可以正常运行 ".center(80, '✓'))
    else:
        print(" 测试失败！请检查错误信息 ".center(80, '✗'))
    print("="*80)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
