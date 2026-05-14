"""
调整ISSA超参数版本
避开有问题的更新策略
"""

import numpy as np
import time
from issa_optimizer_bvf_channel import ISSAOptimizerBVFChannel

def generate_neutral_scenario(seed=42):
    np.random.seed(seed)
    square_length = 1000
    K, G, L = 60, 4, 9
    
    UE_pos = np.random.uniform([50, 50], [square_length - 50, square_length - 50], (K, 2))
    UE_pos = np.column_stack([UE_pos, np.ones(K) * 1.65])
    
    ground_AP_pos = np.column_stack([
        [100, 900, 100, 900],
        [100, 100, 900, 900],
        [15.0] * 4
    ])
    
    uav_grid_x = np.linspace(200, 800, 3)
    uav_grid_y = np.linspace(200, 800, 3)
    UAV_x, UAV_y = np.meshgrid(uav_grid_x, uav_grid_y)
    UAV_pos = np.column_stack([UAV_x.flatten(), UAV_y.flatten(), [50.0] * L])
    
    return UE_pos, ground_AP_pos, UAV_pos

def create_tuned_config():
    """调整后的配置：增加生产者比例，减少警戒者比例"""
    return {
        'square_length': 1000,
        'num_UE': 60,
        'num_ground_AP': 4,
        'num_UAV': 9,
        'M': 4,
        'UE_height': 1.65,
        'ground_AP_height': 15.0,
        'UAV_height': 50.0,
        'UAV_height_min': 50.0,
        'UAV_height_max': 150.0,
        
        # 调整ISSA参数：减少有问题的警戒者
        'issa_n_sparrows': 30,
        'issa_max_iter': 50,
        'issa_pd': 0.4,  # 增加生产者比例（原0.2）
        'issa_sd': 0.1,  # 减少警戒者比例（原0.2）
        'issa_st': 0.6,  # 降低安全阈值（原0.8）
        
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

print("="*80)
print(" ISSA调优版本测试 ".center(80))
print("="*80)
print("\n调整内容：")
print("  1. 适应度函数：min_rate + 0.01*sum_rate（引导作用）")
print("  2. 生产者比例：0.2 → 0.4（增加稳定更新）")
print("  3. 警戒者比例：0.2 → 0.1（减少不稳定更新）")
print("  4. 安全阈值：0.8 → 0.6（更保守的探索）")
print()

UE_pos, ground_AP_pos, UAV_pos = generate_neutral_scenario(42)
config = create_tuned_config()

print("开始优化...")
np.random.seed(42)
optimizer = ISSAOptimizerBVFChannel(config)
start = time.time()
results = optimizer.optimize(UE_pos, ground_AP_pos, UAV_pos.copy())
elapsed = time.time() - start

print()
print("="*80)
print(" 结果对比 ".center(80))
print("="*80)
print(f"\n初始状态:       28.70 Mbps")
print(f"ISSA原版:        19.76 Mbps (-31.16%) ❌")
print(f"ISSA简化:        24.65 Mbps (-14.12%) ⚠️")
print(f"ISSA调优:        {results['final_min_rate']:.2f} Mbps ({(results['final_min_rate']-28.70)/28.70*100:+.2f}%)", end="")

if results['final_min_rate'] > 28.70:
    print(" ✅")
elif results['final_min_rate'] > 24.65:
    print(" 📈 有改善")
else:
    print(" ❌")

print(f"\n总速率:          {results['final_sum_rate']:.2f} Mbps")
print(f"平均速率:        {results['final_rates'].mean():.2f} Mbps")
print(f"优化时间:        {elapsed:.2f}秒")
print()

if results['final_min_rate'] > 28.70:
    print("🎉 成功！ISSA现在能超越初始状态了！")
elif results['final_min_rate'] > 26:
    print("📈 接近成功，可以继续调整超参数")
else:
    print("❌ 建议：ISSA算法的更新公式本身可能需要重新设计")
    print("   特别是跟随者和警戒者的公式存在数值稳定性问题")
