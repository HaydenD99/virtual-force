"""
性能对比脚本 - 对比原版和优化版的信道计算性能
"""

import numpy as np
import time
from balanced_virtual_force_optimizer import BalancedVirtualForceOptimizer, create_balanced_config
from balanced_virtual_force_optimizer_v2 import BalancedVirtualForceOptimizerV2

def test_channel_computation_speed(optimizer, UE_pos, AP_pos, num_tests=5):
    """测试信道计算速度"""
    times = []
    
    print(f"  测试信道计算速度（{num_tests}次）...")
    for i in range(num_tests):
        start = time.time()
        H, Hhat, betas = optimizer.compute_channel_model(UE_pos, AP_pos)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"    测试 {i+1}: {elapsed:.4f} 秒")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time

def main():
    print("="*70)
    print("信道计算性能对比测试")
    print("="*70)
    
    # 创建配置
    config = create_balanced_config()
    config['max_iterations'] = 10  # 只测试10次迭代即可看出差异
    
    # 初始化两个版本的优化器
    print("\n初始化优化器...")
    optimizer_original = BalancedVirtualForceOptimizer(config)
    optimizer_v2 = BalancedVirtualForceOptimizerV2(config)
    
    # 设置相同的随机种子以确保公平对比
    np.random.seed(42)
    UE_pos, ground_AP_pos, UAV_pos = optimizer_original.initialize_positions()
    all_AP_pos = np.vstack([ground_AP_pos, UAV_pos])
    
    print(f"\n测试参数:")
    print(f"  - UE数量: {len(UE_pos)}")
    print(f"  - 总AP数量: {len(all_AP_pos)} (地面AP: {len(ground_AP_pos)}, UAV: {len(UAV_pos)})")
    print(f"  - 天线数: {optimizer_original.M}")
    print(f"  - 实现次数: {optimizer_original.nbrOfRealizations}")
    
    # 测试原版
    print(f"\n{'='*70}")
    print("测试原版 (BalancedVirtualForceOptimizer)")
    print("="*70)
    avg_time_original, std_original = test_channel_computation_speed(
        optimizer_original, UE_pos, all_AP_pos, num_tests=5
    )
    print(f"\n  平均时间: {avg_time_original:.4f} ± {std_original:.4f} 秒")
    
    # 测试优化版
    print(f"\n{'='*70}")
    print("测试优化版V2 (BalancedVirtualForceOptimizerV2)")
    print("="*70)
    avg_time_v2, std_v2 = test_channel_computation_speed(
        optimizer_v2, UE_pos, all_AP_pos, num_tests=5
    )
    print(f"\n  平均时间: {avg_time_v2:.4f} ± {std_v2:.4f} 秒")
    
    # 性能对比
    speedup = avg_time_original / avg_time_v2
    time_saved = avg_time_original - avg_time_v2
    
    print(f"\n{'='*70}")
    print("性能对比结果")
    print("="*70)
    print(f"原版平均时间:     {avg_time_original:.4f} 秒")
    print(f"优化版V2平均时间: {avg_time_v2:.4f} 秒")
    print(f"加速比:           {speedup:.2f}x")
    print(f"每次节省时间:     {time_saved:.4f} 秒")
    print(f"\n100次迭代预计:")
    print(f"  原版总时间:     {avg_time_original * 100 * 2:.1f} 秒 (~{avg_time_original * 100 * 2/60:.1f} 分钟)")
    print(f"  优化版总时间:   {avg_time_v2 * 100 * 2:.1f} 秒 (~{avg_time_v2 * 100 * 2/60:.1f} 分钟)")
    print(f"  节省时间:       {time_saved * 100 * 2:.1f} 秒 (~{time_saved * 100 * 2/60:.1f} 分钟)")
    print(f"\n注: 每次迭代调用2次信道计算（计算性能 + 计算虚拟力）")
    
    # 完整迭代测试
    print(f"\n{'='*70}")
    print("完整优化测试 (10次迭代)")
    print("="*70)
    
    # 重新初始化位置
    np.random.seed(42)
    UE_pos, ground_AP_pos, UAV_pos = optimizer_original.initialize_positions()
    
    print("\n运行原版优化器...")
    start = time.time()
    results_original = optimizer_original.optimize(UE_pos, ground_AP_pos, UAV_pos)
    time_original = time.time() - start
    
    print("\n运行优化版V2优化器...")
    np.random.seed(42)  # 重置随机种子
    UE_pos, ground_AP_pos, UAV_pos = optimizer_v2.initialize_positions()
    start = time.time()
    results_v2 = optimizer_v2.optimize(UE_pos, ground_AP_pos, UAV_pos)
    time_v2 = time.time() - start
    
    print(f"\n{'='*70}")
    print("完整优化性能对比")
    print("="*70)
    print(f"原版总时间:       {time_original:.2f} 秒")
    print(f"优化版V2总时间:   {time_v2:.2f} 秒")
    print(f"加速比:           {time_original/time_v2:.2f}x")
    print(f"节省时间:         {time_original - time_v2:.2f} 秒")
    
    print(f"\n推算100次迭代:")
    iterations_original = results_original['total_iterations']
    iterations_v2 = results_v2['total_iterations']
    
    projected_time_original = (time_original / iterations_original) * 100
    projected_time_v2 = (time_v2 / iterations_v2) * 100
    
    print(f"  原版预计:       {projected_time_original:.1f} 秒 (~{projected_time_original/60:.1f} 分钟)")
    print(f"  优化版V2预计:   {projected_time_v2:.1f} 秒 (~{projected_time_v2/60:.1f} 分钟)")
    print(f"  预计节省:       {projected_time_original - projected_time_v2:.1f} 秒")
    
    if projected_time_v2 <= 100:
        print(f"\n✅ 优化版V2可以在100秒内完成100次迭代！")
    else:
        print(f"\n⚠️  优化版V2预计需要 {projected_time_v2/60:.1f} 分钟完成100次迭代")
        print(f"   考虑进一步优化或减少nbrOfRealizations参数")

if __name__ == "__main__":
    main()

