"""
多目标优化对比实验
优化目标: 最大化最小用户速率 + 最小化无人机能耗

包含三个算法:
1. Energy-Aware BVF (平衡虚拟力)
2. Energy-Aware GA (遗传算法)  
3. Energy-Aware PSO (粒子群优化)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
from datetime import datetime
from typing import Dict, Tuple

# 导入基础优化器
from balanced_virtual_force_optimizer_v3 import BalancedVirtualForceOptimizerV3, create_balanced_config
from genetic_algorithm_optimizer_discrete import DiscreteGeneticAlgorithmOptimizer, create_discrete_ga_config
from distributed_pso_optimizer import DistributedPSOOptimizer, create_distributed_pso_config


# ===============================================================================
# 能量模型 (所有算法共用)
# ===============================================================================

class UAVEnergyModel:
    """无人机能量模型"""
    
    def __init__(self, config: Dict):
        # 物理参数
        self.rho_air = config.get('rho_air', 1.225)  # 空气密度 kg/m³
        self.A_rotor = config.get('A_rotor', 0.503)  # 旋翼面积 m²
        self.m_uav = config.get('m_uav', 2.0)        # UAV质量 kg
        self.g = 9.81  # 重力加速度
        
        # 功率参数
        self.delta_blade = config.get('delta_blade', 0.012)
        self.Omega = config.get('Omega', 300)  # rad/s
        self.R_rotor = config.get('R_rotor', 0.4)  # m
        self.k_inc = config.get('k_inc', 0.1)
        self.A_drag = config.get('A_drag', 0.5)
        
        # 计算P0和P1
        self.P0 = (self.delta_blade / 8) * self.rho_air * self.A_rotor * \
                  (self.Omega * self.R_rotor)**3
        W = self.m_uav * self.g
        self.P1 = (1 + self.k_inc) * (W**3 / (2 * self.rho_air * self.A_rotor))**(1/2)
        self.U = self.Omega * self.R_rotor
        self.v_r = (W / (2 * self.rho_air * self.A_rotor))**(1/2)
        
        self.hover_power = self.P0 + self.P1  # 悬停功率
    
    def compute_propulsion_power(self, velocity: float) -> float:
        """计算推动功率(W)"""
        V = velocity
        if V == 0:
            return self.hover_power
        
        term1 = self.P0 * (1 + 3 * V**3 / self.U**2)
        inner_sqrt = np.sqrt(1 + V**4 / (4 * self.v_r**4)) - V**2 / (2 * self.v_r**2)
        outer_sqrt = np.sqrt(max(0, inner_sqrt))
        term2 = self.P1 * (outer_sqrt + 0.5 * self.A_drag * V**3)
        
        return term1 + term2
    
    def compute_energy(self, distance: float, time_duration: float) -> float:
        """计算能量消耗(J)"""
        velocity = distance / time_duration if distance > 0 else 0
        power = self.compute_propulsion_power(velocity)
        return power * time_duration


def compute_multi_objective_fitness(min_rate: float, total_energy: float, 
                                    w_rate: float = 0.7, w_energy: float = 0.3) -> float:
    """
    改进的多目标综合性能指标 (CPI - Comprehensive Performance Index)
    
    设计目标:
    1. 最大化最小速率 (性能指标)
    2. 最小化能量消耗 (能效指标)
    3. 结果在 [0, 100] 范围，可解释性强
    
    公式:
    - 性能得分 = (min_rate / 50) * 100，归一化到[0,100]
    - 能效得分 = (1 - energy/max_energy) * 100，能耗越低分数越高
    - CPI = w_rate * 性能得分 + w_energy * 能效得分
    
    返回值: [0, 100]，越大越好
    """
    # 参数设定
    MAX_EXPECTED_RATE = 50.0  # Mbps, 预期最大速率
    MAX_EXPECTED_ENERGY = 20000.0  # kJ, 预期最大能耗
    
    # 1. 性能得分: 最小速率越高越好，归一化到[0,100]
    performance_score = min(min_rate / MAX_EXPECTED_RATE, 1.0) * 100.0
    
    # 2. 能效得分: 能耗越低越好，归一化到[0,100]
    # 使用 1 - normalized_energy，使其与性能得分方向一致
    energy_kj = total_energy / 1000.0  # 转换为kJ
    normalized_energy = min(energy_kj / MAX_EXPECTED_ENERGY, 1.0)
    efficiency_score = (1.0 - normalized_energy) * 100.0
    
    # 3. 综合性能指标 (CPI)
    cpi = w_rate * performance_score + w_energy * efficiency_score
    
    return cpi


# ===============================================================================
# 多目标BVF优化器
# ===============================================================================

class MultiObjectiveBVF(BalancedVirtualForceOptimizerV3):
    """多目标BVF优化器"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.energy_model = UAVEnergyModel(config)
        self.w_rate = config.get('w_min_rate_objective', 0.7)
        self.w_energy = config.get('w_energy_objective', 0.3)
        self.total_energy_consumed = 0
        
        print(f"⚡ 能量模型: 悬停功率={self.energy_model.hover_power:.2f}W")
    
    def compute_fitness(self, min_rate: float) -> float:
        """计算当前适应度"""
        return compute_multi_objective_fitness(
            min_rate, self.total_energy_consumed, 
            self.w_rate, self.w_energy
        )
    
    def update_positions(self, UAV_pos: np.ndarray, forces: np.ndarray, 
                        iteration: int, current_min_rate: float) -> Tuple[np.ndarray, float]:
        """重写位置更新，添加能量计算"""
        new_UAV_pos, movement = super().update_positions(UAV_pos, forces, iteration, current_min_rate)
        
        # 计算本次移动能耗
        for l in range(self.L):
            distance = np.linalg.norm(new_UAV_pos[l] - UAV_pos[l])
            if distance > 0:
                time_duration = distance / 10.0  # 假设10m/s
                energy = self.energy_model.compute_energy(distance, time_duration)
                self.total_energy_consumed += energy
        
        return new_UAV_pos, movement
    
    def optimize(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray,
                UAV_pos: np.ndarray) -> Dict:
        """多目标优化"""
        print("开始多目标BVF优化...")
        self.total_energy_consumed = 0
        
        results = super().optimize(UE_pos, ground_AP_pos, UAV_pos)
        results['total_energy'] = self.total_energy_consumed
        results['multi_objective_fitness'] = self.compute_fitness(results['final_min_rate'])
        
        return results


# ===============================================================================
# 多目标GA优化器
# ===============================================================================

class MultiObjectiveGA(DiscreteGeneticAlgorithmOptimizer):
    """多目标GA优化器"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.energy_model = UAVEnergyModel(config)
        self.w_rate = config.get('w_min_rate_objective', 0.7)
        self.w_energy = config.get('w_energy_objective', 0.3)
        
        print(f"⚡ GA能量模型: 悬停功率={self.energy_model.hover_power:.2f}W")
    
    def compute_movement_energy(self, UAV_init: np.ndarray, UAV_final: np.ndarray) -> float:
        """计算UAV移动总能耗"""
        total_energy = 0
        for l in range(len(UAV_init)):
            distance = np.linalg.norm(UAV_final[l] - UAV_init[l])
            if distance > 0:
                time_duration = distance / 10.0
                energy = self.energy_model.compute_energy(distance, time_duration)
                total_energy += energy
        return total_energy
    
    def fitness_function(self, individual: np.ndarray, UE_pos: np.ndarray, 
                        ground_AP_pos: np.ndarray) -> float:
        """多目标适应度函数"""
        # 解码UAV位置
        UAV_pos = self.decode_individual(individual)
        all_AP_pos = np.vstack([ground_AP_pos, UAV_pos])
        
        try:
            _, _, betas = self.compute_channel_model(UE_pos, all_AP_pos)
            mask = self.compute_AP_selection_mask(betas)
            rates, sum_rate = self.compute_user_rates(UE_pos, all_AP_pos, mask)
            min_rate = rates.min()
            
            # 计算从初始位置到当前位置的能耗
            # 假设从网格中心开始
            UAV_init = np.array([[500, 500, 50]] * self.L)
            total_energy = self.compute_movement_energy(UAV_init, UAV_pos)
            
            # 多目标适应度
            fitness = compute_multi_objective_fitness(
                min_rate, total_energy, self.w_rate, self.w_energy
            )
            
            # 重复惩罚
            unique_indices = len(np.unique(individual))
            if unique_indices < self.L:
                fitness -= (self.L - unique_indices) * 0.5
            
            return fitness
            
        except:
            return -100
    
    def optimize(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray) -> Dict:
        """多目标GA优化"""
        print("开始多目标GA优化...")
        results = super().optimize(UE_pos, ground_AP_pos)
        
        # 计算最终能耗
        UAV_init = np.array([[500, 500, 50]] * self.L)
        UAV_final = results['optimized_UAV_pos']
        total_energy = self.compute_movement_energy(UAV_init, UAV_final)
        
        results['total_energy'] = total_energy
        results['multi_objective_fitness'] = compute_multi_objective_fitness(
            results['final_min_rate'], total_energy, self.w_rate, self.w_energy
        )
        
        return results


# ===============================================================================
# 多目标PSO优化器
# ===============================================================================

class MultiObjectivePSO(DistributedPSOOptimizer):
    """多目标PSO优化器"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.energy_model = UAVEnergyModel(config)
        self.w_rate = config.get('w_min_rate_objective', 0.7)
        self.w_energy = config.get('w_energy_objective', 0.3)
        
        print(f"⚡ PSO能量模型: 悬停功率={self.energy_model.hover_power:.2f}W")
    
    def compute_movement_energy(self, UAV_init: np.ndarray, UAV_final: np.ndarray) -> float:
        """计算UAV移动总能耗"""
        total_energy = 0
        for l in range(len(UAV_init)):
            distance = np.linalg.norm(UAV_final[l] - UAV_init[l])
            if distance > 0:
                time_duration = distance / 10.0
                energy = self.energy_model.compute_energy(distance, time_duration)
                total_energy += energy
        return total_energy
    
    def fitness_function(self, particle: np.ndarray, 
                        UE_pos: np.ndarray, 
                        ground_AP_pos: np.ndarray) -> float:
        """多目标适应度函数"""
        try:
            # 重构UAV位置
            UAV_pos = particle.reshape(self.L, 2)
            UAV_pos = np.column_stack([UAV_pos, np.full(self.L, self.heights['UAV'])])
            
            # 边界约束
            UAV_pos[:, 0] = np.clip(UAV_pos[:, 0], self.pos_min, self.pos_max)
            UAV_pos[:, 1] = np.clip(UAV_pos[:, 1], self.pos_min, self.pos_max)
            
            # 合并AP位置
            all_AP_pos = np.vstack([ground_AP_pos, UAV_pos])
            
            # 计算速率
            _, _, betas = self.compute_channel_model(UE_pos, all_AP_pos)
            mask = self.compute_AP_selection_mask(betas)
            rates, sum_rate = self.compute_user_rates(UE_pos, all_AP_pos, mask)
            min_rate = rates.min()
            
            # 计算能耗
            UAV_init = np.array([[500, 500, 50]] * self.L)
            total_energy = self.compute_movement_energy(UAV_init, UAV_pos)
            
            # 多目标适应度
            fitness = compute_multi_objective_fitness(
                min_rate, total_energy, self.w_rate, self.w_energy
            )
            
            return fitness
            
        except:
            return -1e10
    
    def optimize(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray,
                UAV_pos: np.ndarray) -> Dict:
        """多目标PSO优化"""
        print("开始多目标PSO优化...")
        results = super().optimize(UE_pos, ground_AP_pos, UAV_pos)
        
        # 计算最终能耗
        UAV_init = UAV_pos  # PSO有初始位置
        UAV_final = results['optimized_UAV_pos']
        total_energy = self.compute_movement_energy(UAV_init, UAV_final)
        
        results['total_energy'] = total_energy
        results['multi_objective_fitness'] = compute_multi_objective_fitness(
            results['final_min_rate'], total_energy, self.w_rate, self.w_energy
        )
        
        return results


# ===============================================================================
# 实验运行与对比
# ===============================================================================

def run_single_config_experiment(config_name, num_uav, num_ground_ap, run_id, output_dir):
    """运行单个配置的单次实验"""
    
    import sys
    
    # 创建log文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f'MultiObj_{config_name}_run{run_id}_{timestamp}.log')
    
    # 重定向输出
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w', encoding='utf-8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    logger = Logger(log_file)
    original_stdout = sys.stdout
    sys.stdout = logger
    
    print("="*80)
    print(f"  多目标优化实验 - {config_name} Run {run_id}  ".center(80))
    print("="*80)
    print(f"时间戳: {timestamp}")
    print(f"配置: {num_ground_ap} 地面AP, {num_uav} UAV")
    print(f"迭代次数: 50")
    print(f"nbrOfRealizations: 30")
    print(f"优化目标: 最大化最小速率 + 最小化能耗")
    print(f"权重: w_rate=0.7, w_energy=0.3")
    print(f"随机种子: {42 + run_id}")
    print("\n")
    
    seed = 42 + run_id
    np.random.seed(seed)
    
    # 创建配置
    config_base = create_balanced_config()
    config_base['num_UAV'] = num_uav
    config_base['num_ground_AP'] = num_ground_ap
    config_base['max_iterations'] = 50
    config_base['nbrOfRealizations'] = 30
    config_base['w_min_rate_objective'] = 0.7
    config_base['w_energy_objective'] = 0.3
    
    # 初始化场景
    print("[初始化场景]")
    optimizer_init = BalancedVirtualForceOptimizerV3(config_base)
    UE_pos, ground_AP_pos, UAV_pos_init = optimizer_init.initialize_positions()
    
    print(f"  ✓ UE数量: {len(UE_pos)}")
    print(f"  ✓ 地面AP数量: {len(ground_AP_pos)}")
    print(f"  ✓ UAV数量: {len(UAV_pos_init)}")
    
    # 打印初始位置
    print(f"\n初始UAV位置:")
    for i, pos in enumerate(UAV_pos_init):
        print(f"  UAV-{i+1}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    # 计算初始性能
    print(f"\n[初始性能评估]")
    all_AP_pos = np.vstack([ground_AP_pos, UAV_pos_init])
    _, _, betas = optimizer_init.compute_channel_model(UE_pos, all_AP_pos)
    mask = optimizer_init.compute_AP_selection_mask(betas)
    rates, sum_rate = optimizer_init.compute_user_rates(UE_pos, all_AP_pos, mask)
    initial_min_rate = rates.min()
    initial_mean_rate = rates.mean()
    
    print(f"  最小速率: {initial_min_rate:.4f} Mbps")
    print(f"  系统总速率: {sum_rate:.2f} Mbps")
    print(f"  平均速率: {initial_mean_rate:.4f} Mbps")
    print(f"  能耗: 0 J (初始状态)")
    
    results = {
        'config': config_name,
        'num_UAV': num_uav,
        'num_ground_AP': num_ground_ap,
        'run_id': run_id,
        'seed': seed,
        'initial': {
            'min_rate': float(initial_min_rate),
            'sum_rate': float(sum_rate),
            'mean_rate': float(initial_mean_rate),
            'energy': 0.0
        }
    }
    
    # ========== 1. 多目标BVF ==========
    print("\n" + "="*80)
    print("[1/3] 运行多目标BVF优化器")
    print("="*80)
    
    optimizer_bvf = MultiObjectiveBVF(config_base)
    optimizer_bvf.K = len(UE_pos)
    optimizer_bvf.G = len(ground_AP_pos)
    optimizer_bvf.L = len(UAV_pos_init)
    
    start_time = time.time()
    results_bvf = optimizer_bvf.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
    bvf_time = time.time() - start_time
    
    print(f"\n✓ BVF优化完成")
    print(f"  最终最小速率: {results_bvf['final_min_rate']:.4f} Mbps")
    print(f"  最终系统总速率: {results_bvf['final_sum_rate']:.2f} Mbps")
    print(f"  最终平均速率: {results_bvf['final_rates'].mean():.4f} Mbps")
    print(f"  总能耗: {results_bvf['total_energy']/1000:.2f} kJ")
    print(f"  多目标适应度: {results_bvf['multi_objective_fitness']:.4f}")
    print(f"  优化时间: {bvf_time:.2f} s")
    
    print(f"\nBVF优化后UAV位置:")
    for i, pos in enumerate(results_bvf['optimized_UAV_pos']):
        print(f"  UAV-{i+1}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    print(f"\nBVF UAV移动距离:")
    for i in range(len(UAV_pos_init)):
        distance = np.linalg.norm(results_bvf['optimized_UAV_pos'][i] - UAV_pos_init[i])
        print(f"  UAV-{i+1}: {distance:.2f} m")
    
    results['BVF'] = {
        'min_rate': float(results_bvf['final_min_rate']),
        'sum_rate': float(results_bvf['final_sum_rate']),
        'mean_rate': float(results_bvf['final_rates'].mean()),
        'energy': float(results_bvf['total_energy']),
        'fitness': float(results_bvf['multi_objective_fitness']),
        'time': float(bvf_time),
        'UAV_pos': results_bvf['optimized_UAV_pos'].tolist()
    }
    
    # ========== 2. 多目标GA ==========
    print("\n" + "="*80)
    print("[2/3] 运行多目标GA优化器")
    print("="*80)
    
    config_ga = create_discrete_ga_config()
    config_ga.update(config_base)
    config_ga['max_generations'] = 50
    
    optimizer_ga = MultiObjectiveGA(config_ga)
    optimizer_ga.K = len(UE_pos)
    optimizer_ga.G = len(ground_AP_pos)
    
    start_time = time.time()
    results_ga = optimizer_ga.optimize(UE_pos, ground_AP_pos)
    ga_time = time.time() - start_time
    
    print(f"\n✓ GA优化完成")
    print(f"  最终最小速率: {results_ga['final_min_rate']:.4f} Mbps")
    print(f"  最终系统总速率: {results_ga['final_sum_rate']:.2f} Mbps")
    print(f"  最终平均速率: {results_ga['final_rates'].mean():.4f} Mbps")
    print(f"  总能耗: {results_ga['total_energy']/1000:.2f} kJ")
    print(f"  多目标适应度: {results_ga['multi_objective_fitness']:.4f}")
    print(f"  优化时间: {ga_time:.2f} s")
    
    print(f"\nGA优化后UAV位置:")
    for i, pos in enumerate(results_ga['optimized_UAV_pos']):
        print(f"  UAV-{i+1}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    print(f"\nGA UAV移动距离:")
    for i in range(len(UAV_pos_init)):
        distance = np.linalg.norm(results_ga['optimized_UAV_pos'][i] - UAV_pos_init[i])
        print(f"  UAV-{i+1}: {distance:.2f} m")
    
    results['GA'] = {
        'min_rate': float(results_ga['final_min_rate']),
        'sum_rate': float(results_ga['final_sum_rate']),
        'mean_rate': float(results_ga['final_rates'].mean()),
        'energy': float(results_ga['total_energy']),
        'fitness': float(results_ga['multi_objective_fitness']),
        'time': float(ga_time),
        'UAV_pos': results_ga['optimized_UAV_pos'].tolist()
    }
    
    # ========== 3. 多目标PSO ==========
    print("\n" + "="*80)
    print("[3/3] 运行多目标PSO优化器")
    print("="*80)
    
    config_pso = create_distributed_pso_config()
    config_pso.update(config_base)
    config_pso['max_iterations'] = 50
    
    optimizer_pso = MultiObjectivePSO(config_pso)
    
    start_time = time.time()
    results_pso = optimizer_pso.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
    pso_time = time.time() - start_time
    
    print(f"\n✓ PSO优化完成")
    print(f"  最终最小速率: {results_pso['final_min_rate']:.4f} Mbps")
    print(f"  最终系统总速率: {results_pso['final_sum_rate']:.2f} Mbps")
    print(f"  最终平均速率: {results_pso['final_mean_rate']:.4f} Mbps")
    print(f"  总能耗: {results_pso['total_energy']/1000:.2f} kJ")
    print(f"  多目标适应度: {results_pso['multi_objective_fitness']:.4f}")
    print(f"  优化时间: {pso_time:.2f} s")
    
    print(f"\nPSO优化后UAV位置:")
    for i, pos in enumerate(results_pso['optimized_UAV_pos']):
        print(f"  UAV-{i+1}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    print(f"\nPSO UAV移动距离:")
    for i in range(len(UAV_pos_init)):
        distance = np.linalg.norm(results_pso['optimized_UAV_pos'][i] - UAV_pos_init[i])
        print(f"  UAV-{i+1}: {distance:.2f} m")
    
    results['PSO'] = {
        'min_rate': float(results_pso['final_min_rate']),
        'sum_rate': float(results_pso['final_sum_rate']),
        'mean_rate': float(results_pso['final_mean_rate']),
        'energy': float(results_pso['total_energy']),
        'fitness': float(results_pso['multi_objective_fitness']),
        'time': float(pso_time),
        'UAV_pos': results_pso['optimized_UAV_pos'].tolist()
    }
    
    # ========== 打印对比总结 ==========
    print("\n" + "="*80)
    print("  性能对比总结 (CPI: Comprehensive Performance Index)  ".center(80))
    print("="*80)
    
    print(f"\n{'算法':<8} {'CPI':<8} {'最小速率':<12} {'总速率':<12} {'能耗(kJ)':<10} {'时间(s)':<10}")
    print("-" * 80)
    print(f"{'Initial':<8} {'-':<8} {results['initial']['min_rate']:<12.4f} "
          f"{results['initial']['sum_rate']:<12.2f} {results['initial']['energy']:<10.2f} {'-':<10}")
    
    for method in ['BVF', 'GA', 'PSO']:
        cpi_val = results[method]['fitness']
        print(f"{method:<8} {cpi_val:<8.2f} {results[method]['min_rate']:<12.4f} "
              f"{results[method]['sum_rate']:<12.2f} {results[method]['energy']/1000:<10.2f} "
              f"{results[method]['time']:<10.2f}")
    print("-" * 80)
    
    # 详细CPI分解
    print(f"\n{'='*80}")
    print("  CPI 详细分解 (CPI = 0.7 * 性能得分 + 0.3 * 能效得分)  ".center(80))
    print(f"{'='*80}")
    print(f"\n{'算法':<8} {'性能得分':<12} {'能效得分':<12} {'CPI总分':<10} {'排名':<6}")
    print("-" * 80)
    
    # 计算各算法的详细得分
    cpi_scores = {}
    for method in ['BVF', 'GA', 'PSO']:
        min_rate = results[method]['min_rate']
        energy_kj = results[method]['energy'] / 1000.0
        
        # 性能得分 (0-100)
        performance_score = min(min_rate / 50.0, 1.0) * 100.0
        # 能效得分 (0-100)
        efficiency_score = (1.0 - min(energy_kj / 20000.0, 1.0)) * 100.0
        # CPI
        cpi = results[method]['fitness']
        
        cpi_scores[method] = {'cpi': cpi, 'perf': performance_score, 'eff': efficiency_score}
    
    # 按CPI排序
    ranked_methods = sorted(cpi_scores.items(), key=lambda x: x[1]['cpi'], reverse=True)
    
    for rank, (method, scores) in enumerate(ranked_methods, 1):
        medal = '🥇' if rank == 1 else '🥈' if rank == 2 else '🥉'
        print(f"{method:<8} {scores['perf']:<12.2f} {scores['eff']:<12.2f} {scores['cpi']:<10.2f} {medal}{rank}")
    print("-" * 80)
    
    print(f"\n{'='*80}")
    print("  性能提升对比  ".center(80))
    print(f"{'='*80}")
    print(f"\n{'算法':<10} {'最小速率提升%':<18} {'总速率提升%':<18}")
    print("-" * 60)
    for method in ['BVF', 'GA', 'PSO']:
        min_imp = (results[method]['min_rate'] - results['initial']['min_rate']) / results['initial']['min_rate'] * 100
        sum_imp = (results[method]['sum_rate'] - results['initial']['sum_rate']) / results['initial']['sum_rate'] * 100
        print(f"{method:<10} {min_imp:>+16.2f}% {sum_imp:>+16.2f}%")
    print("-" * 60)
    
    # 保存JSON
    json_file = log_file.replace('.log', '.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ JSON结果已保存: {json_file}")
    
    # 恢复输出
    sys.stdout = original_stdout
    logger.log.close()
    
    print(f"✓ {config_name} Run {run_id} 完成. Log: {log_file}")
    
    return results


def run_multi_objective_comparison():
    """运行多配置多目标优化对比实验"""
    
    output_dir = '/home/hzl/hyd/virtualForce/results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义多个配置
    configs = [
        {'name': 'Config_4AP_6UAV', 'num_ground_AP': 4, 'num_UAV': 6},
        {'name': 'Config_4AP_9UAV', 'num_ground_AP': 4, 'num_UAV': 9},
        {'name': 'Config_4AP_12UAV', 'num_ground_AP': 4, 'num_UAV': 12},
    ]
    
    print("="*80)
    print("  多目标优化对比实验 (多配置)  ".center(80))
    print("  目标: 最大化最小速率 + 最小化能耗  ".center(80))
    print("="*80)
    print(f"\n配置数量: {len(configs)}")
    print(f"每配置运行次数: 5")
    print(f"总实验次数: {len(configs) * 5}")
    print(f"迭代次数: 50")
    print(f"nbrOfRealizations: 30")
    print(f"权重: w_rate=0.7, w_energy=0.3")
    print(f"输出目录: {output_dir}")
    print("\n")
    
    all_results = {}
    
    for config in configs:
        config_name = config['name']
        num_ground_ap = config['num_ground_AP']
        num_uav = config['num_UAV']
        
        print(f"\n{'='*80}")
        print(f"  配置: {config_name}  ".center(80))
        print(f"  地面AP: {num_ground_ap}, UAV: {num_uav}  ".center(80))
        print(f"{'='*80}\n")
        
        config_results = []
        
        for run_id in range(1, 6):
            print(f"\n{'─'*80}")
            print(f"  {config_name} - 运行 {run_id}/5  ".center(80))
            print(f"{'─'*80}\n")
            
            result = run_single_config_experiment(
                config_name=config_name,
                num_uav=num_uav,
                num_ground_ap=num_ground_ap,
                run_id=run_id,
                output_dir=output_dir
            )
            
            config_results.append(result)
            
            print(f"\n{'─'*80}")
            print(f"  {config_name} - 运行 {run_id}/5 完成  ".center(80))
            print(f"{'─'*80}\n")
        
        all_results[config_name] = config_results
        
        print(f"\n{'='*80}")
        print(f"  配置 {config_name} 全部完成 (5次运行)  ".center(80))
        print(f"{'='*80}\n")
    
    # 保存总结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(output_dir, f'multi_objective_summary_{timestamp}.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("  所有实验完成  ".center(80))
    print("="*80)
    print(f"\n✓ 总结果已保存: {summary_file}")
    print(f"✓ 所有log和JSON文件保存在: {output_dir}")
    
    return all_results, output_dir


def plot_multi_objective_comparison(all_results, output_dir):
    """绘制多配置对比图"""
    
    import matplotlib.pyplot as plt
    
    for config_name, config_results in all_results.items():
        methods = ['BVF', 'GA', 'PSO']
        colors = {'BVF': '#e74c3c', 'GA': '#3498db', 'PSO': '#2ecc71'}
        
        # 计算平均值和标准差
        stats = {}
        for method in methods:
            stats[method] = {
                'fitness': [r[method]['fitness'] for r in config_results],
                'min_rate': [r[method]['min_rate'] for r in config_results],
                'sum_rate': [r[method]['sum_rate'] for r in config_results],
                'energy': [r[method]['energy']/1000 for r in config_results],  # 转换为kJ
            }
        
        # 创建图表
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 子图1: 多目标适应度
        ax1 = fig.add_subplot(gs[0, 0])
        fitness_means = [np.mean(stats[m]['fitness']) for m in methods]
        fitness_stds = [np.std(stats[m]['fitness']) for m in methods]
        bars = ax1.bar(methods, fitness_means, yerr=fitness_stds, 
                       color=[colors[m] for m in methods], alpha=0.8, 
                       edgecolor='black', linewidth=1.5, capsize=5)
        for bar, val in zip(bars, fitness_means):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
        ax1.set_ylabel('Multi-Objective Fitness', fontsize=12, fontweight='bold')
        ax1.set_title(f'{config_name}: Fitness (Higher is Better)', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # 子图2: 最小速率
        ax2 = fig.add_subplot(gs[0, 1])
        min_rate_means = [np.mean(stats[m]['min_rate']) for m in methods]
        min_rate_stds = [np.std(stats[m]['min_rate']) for m in methods]
        bars = ax2.bar(methods, min_rate_means, yerr=min_rate_stds,
                       color=[colors[m] for m in methods], alpha=0.8,
                       edgecolor='black', linewidth=1.5, capsize=5)
        for bar, val in zip(bars, min_rate_means):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        ax2.set_ylabel('Minimum Rate (Mbps)', fontsize=12, fontweight='bold')
        ax2.set_title('Minimum User Rate', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # 子图3: 能耗
        ax3 = fig.add_subplot(gs[0, 2])
        energy_means = [np.mean(stats[m]['energy']) for m in methods]
        energy_stds = [np.std(stats[m]['energy']) for m in methods]
        bars = ax3.bar(methods, energy_means, yerr=energy_stds,
                       color=[colors[m] for m in methods], alpha=0.8,
                       edgecolor='black', linewidth=1.5, capsize=5)
        for bar, val in zip(bars, energy_means):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        ax3.set_ylabel('Energy Consumption (kJ)', fontsize=12, fontweight='bold')
        ax3.set_title('Total Energy (Lower is Better)', fontsize=13, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # 子图4: 总速率
        ax4 = fig.add_subplot(gs[1, 0])
        sum_rate_means = [np.mean(stats[m]['sum_rate']) for m in methods]
        sum_rate_stds = [np.std(stats[m]['sum_rate']) for m in methods]
        bars = ax4.bar(methods, sum_rate_means, yerr=sum_rate_stds,
                       color=[colors[m] for m in methods], alpha=0.8,
                       edgecolor='black', linewidth=1.5, capsize=5)
        for bar, val in zip(bars, sum_rate_means):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        ax4.set_ylabel('System Sum Rate (Mbps)', fontsize=12, fontweight='bold')
        ax4.set_title('System Sum Rate', fontsize=13, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        # 子图5: 箱线图 - CPI分布
        ax5 = fig.add_subplot(gs[1, 1])
        data_fitness = [stats[m]['fitness'] for m in methods]
        bp = ax5.boxplot(data_fitness, labels=methods, patch_artist=True, widths=0.6)
        for patch, method in zip(bp['boxes'], methods):
            patch.set_facecolor(colors[method])
            patch.set_alpha(0.7)
        ax5.set_ylabel('CPI (Comprehensive Performance Index)', fontsize=12, fontweight='bold')
        ax5.set_title('CPI Distribution (0-100, Higher is Better)', fontsize=13, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        ax5.set_ylim(bottom=0)  # CPI从0开始
        
        # 子图6: 散点图 - 最小速率 vs 能耗
        ax6 = fig.add_subplot(gs[1, 2])
        for method in methods:
            ax6.scatter(stats[method]['energy'], stats[method]['min_rate'],
                       c=colors[method], s=100, alpha=0.7, edgecolors='black',
                       linewidths=1.5, label=method)
        ax6.set_xlabel('Energy Consumption (kJ)', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Minimum Rate (Mbps)', fontsize=12, fontweight='bold')
        ax6.set_title('Min Rate vs Energy (Pareto Front)', fontsize=13, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(alpha=0.3)
        
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, f'multi_obj_comparison_{config_name}_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ {config_name} 对比图已保存: {save_path}")
        
        plt.close()


if __name__ == "__main__":
    # 运行对比实验
    all_results, output_dir = run_multi_objective_comparison()
    
    # 绘制对比图
    print("\n生成对比图...")
    plot_multi_objective_comparison(all_results, output_dir)
    
    print("\n" + "="*80)
    print("  多目标优化对比实验完成  ".center(80))
    print("="*80)
