"""
动态场景：多目标适应度版本 (v2 - 统一 fitness)
=================================================
修改:
  1. GA / PSO / NewSSA 统一使用相同 fitness:
     fitness = log(min_rate + ε) - λ * E_flight / E_norm
  2. t=0 为初始状态 (无优化)，所有算法指标完全相同
  3. 从 t=5s 起用户布朗运动 + 四算法分别优化
"""

import numpy as np
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

from balanced_virtual_force_optimizer_v3 import BalancedVirtualForceOptimizerV3, create_balanced_config
from balanced_virtual_force_optimizer_v6 import BalancedVirtualForceOptimizerV6
from genetic_algorithm_optimizer_discrete import DiscreteGeneticAlgorithmOptimizer, create_discrete_ga_config
from distributed_pso_optimizer import DistributedPSOOptimizer, create_distributed_pso_config
from newssa_optimizer import NewSSAOptimizer

# 复用能耗模型和布朗运动
from run_dynamic_energy_comparison import UAVEnergyModel, brownian_motion_users, EnergyAwareBVF_V6


# =====================================================================
# 多目标 GA (能耗感知)
# =====================================================================

class EnergyAwareGA(DiscreteGeneticAlgorithmOptimizer):
    """GA 多目标版：fitness 加入移动能耗惩罚"""
    
    def __init__(self, config, energy_model, prev_UAV_pos, energy_lambda=0.1):
        super().__init__(config)
        self.energy_model = energy_model
        self.prev_UAV_pos = prev_UAV_pos  # 上一时间步的 UAV 位置
        self.energy_lambda = energy_lambda  # 能耗惩罚权重
        self.E_normalize = 20000.0  # 能耗归一化因子 20kJ
    
    def fitness_function(self, individual, UE_pos, ground_AP_pos):
        """多目标适应度: min_rate - λ * normalized_energy"""
        UAV_pos = self.decode_individual(individual)
        all_AP_pos = np.vstack([ground_AP_pos, UAV_pos])
        
        try:
            _, _, betas = self.compute_channel_model(UE_pos, all_AP_pos)
            mask = self.compute_AP_selection_mask(betas)
            rates, sum_rate = self.compute_user_rates(UE_pos, all_AP_pos, mask)
            min_rate = rates.min()
            
            # 能耗惩罚：从上一步位置移动到候选位置的飞行能耗
            total_energy = 0.0
            for l in range(self.L):
                dist = np.linalg.norm(UAV_pos[l, :2] - self.prev_UAV_pos[l, :2])
                total_energy += self.energy_model.flight_energy(dist, 10.0)
            
            energy_penalty = total_energy / self.E_normalize
            
            # 重复网格惩罚
            penalty = 0
            unique_indices = len(np.unique(individual))
            if unique_indices < self.L:
                penalty = (self.L - unique_indices) * 2.0
            
            # 多目标适应度
            fitness = np.log(min_rate + 1e-3) - self.energy_lambda * energy_penalty - penalty
            return max(fitness, -100)
            
        except:
            return -100


# =====================================================================
# 多目标 PSO (能耗感知)
# =====================================================================

class EnergyAwarePSO(DistributedPSOOptimizer):
    """PSO 多目标版：fitness 加入移动能耗惩罚"""
    
    def __init__(self, config, energy_model, prev_UAV_pos, energy_lambda=0.1):
        super().__init__(config)
        self.energy_model = energy_model
        self.prev_UAV_pos = prev_UAV_pos
        self.energy_lambda = energy_lambda
        self.E_normalize = 20000.0  # 20kJ 归一化
    
    def fitness_function(self, particle, UE_pos, ground_AP_pos):
        """多目标适应度 (与 GA 统一: log(min_rate) - λ * E/E_norm)"""
        try:
            UAV_pos = particle.reshape(self.L, 2)
            UAV_pos = np.column_stack([UAV_pos, np.full(self.L, self.heights['UAV'])])
            UAV_pos[:, 0] = np.clip(UAV_pos[:, 0], self.pos_min, self.pos_max)
            UAV_pos[:, 1] = np.clip(UAV_pos[:, 1], self.pos_min, self.pos_max)
            
            all_AP_pos = np.vstack([ground_AP_pos, UAV_pos])
            _, _, betas = self.compute_channel_model(UE_pos, all_AP_pos)
            mask = self.compute_AP_selection_mask(betas)
            rates, sum_rate = self.compute_user_rates(UE_pos, all_AP_pos, mask)
            min_rate = rates.min()
            
            # 能耗惩罚 (与 GA 完全一致)
            total_energy = 0.0
            for l in range(self.L):
                dist = np.linalg.norm(UAV_pos[l, :2] - self.prev_UAV_pos[l, :2])
                total_energy += self.energy_model.flight_energy(dist, 10.0)
            
            energy_penalty = total_energy / self.E_normalize
            
            # 统一 fitness: log(min_rate + ε) - λ * energy_penalty
            fitness = np.log(min_rate + 1e-3) - self.energy_lambda * energy_penalty
            return max(fitness, -100)
            
        except:
            return -100


# =====================================================================
# 多目标 NewSSA (能耗感知)
# =====================================================================

class EnergyAwareSSA(NewSSAOptimizer):
    """NewSSA 多目标版：fitness 加入移动能耗惩罚"""
    
    def __init__(self, config, energy_model, prev_UAV_pos, energy_lambda=0.1):
        super().__init__(config)
        self.energy_model_ssa = energy_model
        self.prev_UAV_pos = prev_UAV_pos
        self.energy_lambda = energy_lambda
        self.E_normalize = 20000.0
    
    def _compute_fitness(self, UAV_pos, UE_pos, ground_AP_pos):
        """多目标适应度 (与 GA 统一: log(min_rate) - λ * E/E_norm)"""
        try:
            all_AP_pos = np.vstack([ground_AP_pos, UAV_pos])
            _, _, betas = self.bvf_optimizer.compute_channel_model(UE_pos, all_AP_pos)
            mask = self.bvf_optimizer.compute_AP_selection_mask(betas)
            rates, sum_rate = self.bvf_optimizer.compute_user_rates(UE_pos, all_AP_pos, mask)
            min_rate = rates.min()
            
            # 能耗惩罚 (与 GA 完全一致)
            total_energy = 0.0
            for l in range(len(UAV_pos)):
                if l < len(self.prev_UAV_pos):
                    dist = np.linalg.norm(UAV_pos[l, :2] - self.prev_UAV_pos[l, :2])
                    total_energy += self.energy_model_ssa.flight_energy(dist, 10.0)
            
            energy_penalty = total_energy / self.E_normalize
            
            # 统一 fitness: log(min_rate + ε) - λ * energy_penalty
            # 注意: SSA 内部使用 fitness 值做比较，需要返回 fitness 而非 min_rate
            fitness = np.log(min_rate + 1e-3) - self.energy_lambda * energy_penalty
            
            # 返回 4 元组: (fitness用于优化, sum_rate, rates, min_rate用于对外汇报)
            return fitness, sum_rate, rates, float(min_rate)
            
        except Exception as e:
            return -100.0, 0.0, np.zeros(self.K), 0.0


# =====================================================================
# 单步优化包装
# =====================================================================

def run_energy_ga_one_step(UE_pos, ground_AP_pos, UAV_pos_prev, config, 
                           energy_model, max_gen=20, energy_lambda=0.1):
    """多目标 GA 单步"""
    config_ga = create_discrete_ga_config()
    config_ga.update(config)
    config_ga['max_generations'] = max_gen
    
    ga_opt = EnergyAwareGA(config_ga, energy_model, UAV_pos_prev, energy_lambda)
    ga_opt.K = config['num_UE']
    ga_opt.G = config['num_ground_AP']
    
    res = ga_opt.optimize(UE_pos, ground_AP_pos)
    optimized_pos = res['optimized_UAV_pos']
    
    energy, dist = energy_model.total_energy_for_repositioning(UAV_pos_prev, optimized_pos, 10.0)
    return optimized_pos, res['final_min_rate'], res['final_sum_rate'], energy, dist


def run_energy_pso_one_step(UE_pos, ground_AP_pos, UAV_pos_prev, config, 
                            energy_model, max_iter=20, energy_lambda=0.1):
    """多目标 PSO 单步"""
    config_pso = create_distributed_pso_config()
    config_pso.update(config)
    config_pso['max_iterations'] = max_iter
    
    pso_opt = EnergyAwarePSO(config_pso, energy_model, UAV_pos_prev, energy_lambda)
    res = pso_opt.optimize(UE_pos, ground_AP_pos, UAV_pos_prev.copy())
    optimized_pos = res['optimized_UAV_pos']
    
    energy, dist = energy_model.total_energy_for_repositioning(UAV_pos_prev, optimized_pos, 10.0)
    return optimized_pos, res['final_min_rate'], res['final_sum_rate'], energy, dist


def run_energy_ssa_one_step(UE_pos, ground_AP_pos, UAV_pos_prev, config, 
                            energy_model, max_iter=20, energy_lambda=0.1):
    """多目标 NewSSA 单步"""
    config_ssa = config.copy()
    config_ssa['newssa_n_sparrows'] = 30
    config_ssa['newssa_max_iter'] = max_iter
    config_ssa['newssa_pr'] = 0.2
    config_ssa['newssa_fr'] = 0.15
    config_ssa['newssa_st'] = 0.8
    
    ssa_opt = EnergyAwareSSA(config_ssa, energy_model, UAV_pos_prev, energy_lambda)
    res = ssa_opt.optimize(UE_pos, ground_AP_pos, UAV_pos_prev.copy())
    optimized_pos = res['optimized_UAV_pos']
    
    energy, dist = energy_model.total_energy_for_repositioning(UAV_pos_prev, optimized_pos, 10.0)
    return optimized_pos, res['final_min_rate'], res['final_sum_rate'], energy, dist


# =====================================================================
# 主动态仿真
# =====================================================================

def run_dynamic_multiobjective(seed=62, num_uav=9, num_steps=20,
                                time_step=5.0, iter_per_step=15,
                                user_sigma=8.0, energy_lambda=0.1):
    """
    运行多目标动态场景对比
    """
    np.random.seed(seed)
    
    square_length = 1000
    K = 60
    G = 4
    L_SERVING = 3
    
    base_config = {
        'square_length': square_length,
        'num_UE': K,
        'num_ground_AP': G,
        'num_UAV': num_uav,
        'num_serving_APs': L_SERVING,
        'M': 4,
        'UE_height': 1.65,
        'ground_AP_height': 15.0,
        'UAV_height': 50.0,
        'nbrOfRealizations': 50,
        'tau_c': 200,
        'tau_p': 60,
        'random_seed': seed,
    }
    
    energy_model = UAVEnergyModel()
    
    # 初始化位置 (与 run_dynamic_energy_comparison 完全一致)
    UE_pos = np.random.uniform(low=[50, 50], high=[square_length-50, square_length-50], size=(K, 2))
    UE_pos = np.column_stack([UE_pos, np.ones(K) * 1.65])
    
    gx = np.linspace(square_length * 0.25, square_length * 0.75, 2)
    gy = np.linspace(square_length * 0.25, square_length * 0.75, 2)
    GX, GY = np.meshgrid(gx, gy)
    ground_AP_pos = np.column_stack([GX.flatten(), GY.flatten(), np.ones(G) * 15.0])
    
    if num_uav == 6:
        ux = np.linspace(200, 800, 3); uy = np.linspace(300, 700, 2)
    elif num_uav == 12:
        ux = np.linspace(200, 800, 4); uy = np.linspace(200, 800, 3)
    else:
        ux = np.linspace(200, 800, 3); uy = np.linspace(200, 800, 3)
    UX, UY = np.meshgrid(ux, uy)
    UAV_pos_init = np.column_stack([
        UX.flatten()[:num_uav], UY.flatten()[:num_uav], np.ones(num_uav) * 50.0
    ])
    
    print("=" * 70)
    print("  动态多目标对比 (GA/PSO/SSA fitness 含能耗惩罚)")
    print("=" * 70)
    print(f"  Seed={seed} | UAV={num_uav} | K={K} | L={L_SERVING}")
    print(f"  步数={num_steps} × {time_step}s | 每步迭代={iter_per_step}")
    print(f"  能耗惩罚权重 λ={energy_lambda}")
    energy_model.summary()
    print("=" * 70)
    sys.stdout.flush()
    
    # 四算法的 UAV 位置
    uav_bvf = UAV_pos_init.copy()
    uav_ga  = UAV_pos_init.copy()
    uav_pso = UAV_pos_init.copy()
    uav_ssa = UAV_pos_init.copy()
    
    # BVF 优化器
    config_bvf = create_balanced_config()
    config_bvf.update(base_config)
    config_bvf['max_iterations'] = iter_per_step
    config_bvf['user_sigma'] = user_sigma       # 用户运动幅度 → 约束 UAV 位移
    config_bvf['max_displacement'] = 5.0 * user_sigma  # 位移硬上限 = 5× 用户移动
    # 与 GA/PSO/SSA 完全一致的多目标参数
    config_bvf['energy_lambda'] = energy_lambda
    config_bvf['E_normalize'] = 5000.0
    bvf_optimizer = EnergyAwareBVF_V6(config_bvf, energy_model)
    
    # 算法标签
    alg_keys = ['BVF', 'GA_MO', 'PSO_MO', 'SSA_MO']
    
    # 记录
    records = {'time': []}
    for k in alg_keys:
        records[k] = {'min_rate': [], 'sum_rate': [], 'energy_step': [], 'energy_cumul': [], 'distance': []}
    
    cumul = {k: 0.0 for k in alg_keys}
    # 悬停能耗: 每个时间步所有 UAV 都要悬停，这是真实能耗
    hover_E = energy_model.hover_energy(time_step) * num_uav
    
    current_UE_pos = UE_pos.copy()
    
    # =====================================================
    # Step 0 (t=0s): 初始状态，所有算法相同，不做优化
    # =====================================================
    records['time'].append(0.0)
    
    print(f"\n--- Step 0 (t=0s): 初始状态 (无优化) ---")
    np.random.seed(seed)
    init_v3 = BalancedVirtualForceOptimizerV3(create_balanced_config())
    init_v3.config.update(base_config)
    init_v3.setup_parameters()
    all_AP_init = np.vstack([ground_AP_pos, UAV_pos_init])
    _, _, betas_init = init_v3.compute_channel_model(current_UE_pos, all_AP_init)
    mask_init = init_v3.compute_AP_selection_mask(betas_init)
    rates_init, sr_init = init_v3.compute_user_rates(current_UE_pos, all_AP_init, mask_init)
    mr_init = float(rates_init.min())
    sr_init = float(sr_init)
    
    for k in alg_keys:
        records[k]['min_rate'].append(mr_init)
        records[k]['sum_rate'].append(sr_init)
        records[k]['energy_step'].append(0.0)
        records[k]['energy_cumul'].append(0.0)
        records[k]['distance'].append(0.0)
    
    print(f"  所有算法初始: min_rate={mr_init:.2f} Mbps | sum_rate={sr_init:.1f} Mbps")
    print(f"  能耗=0 | 距离=0")
    sys.stdout.flush()
    
    # =====================================================
    # Step 1 ~ num_steps: 用户运动 + 优化
    # =====================================================
    for step in range(1, num_steps):
        t = step * time_step
        records['time'].append(t)
        
        print(f"\n--- Step {step}/{num_steps-1} (t={t:.0f}s) ---")
        sys.stdout.flush()
        
        # 用户布朗运动
        current_UE_pos = brownian_motion_users(current_UE_pos, sigma=user_sigma,
                                                square_length=square_length)
        
        # 1. BVF (Energy-Aware V6)
        np.random.seed(seed + step * 100 + 1)
        pos_bvf, mr_bvf, sr_bvf, e_bvf, d_bvf = bvf_optimizer.optimize_one_step(
            current_UE_pos, ground_AP_pos, uav_bvf, max_iter=iter_per_step)
        e_bvf += hover_E  # 飞行能耗 + 悬停能耗
        cumul['BVF'] += e_bvf
        uav_bvf = pos_bvf
        records['BVF']['min_rate'].append(mr_bvf)
        records['BVF']['sum_rate'].append(sr_bvf)
        records['BVF']['energy_step'].append(e_bvf)
        records['BVF']['energy_cumul'].append(cumul['BVF'])
        records['BVF']['distance'].append(d_bvf)
        print(f"  BVF:    mr={mr_bvf:.2f} | E={e_bvf:.0f}J | d={d_bvf:.1f}m | cum={cumul['BVF']/1000:.2f}kJ")
        sys.stdout.flush()
        
        # 2. GA (多目标)
        np.random.seed(seed + step * 100 + 2)
        pos_ga, mr_ga, sr_ga, e_ga, d_ga = run_energy_ga_one_step(
            current_UE_pos, ground_AP_pos, uav_ga, base_config,
            energy_model, max_gen=iter_per_step, energy_lambda=energy_lambda)
        e_ga += hover_E
        cumul['GA_MO'] += e_ga
        uav_ga = pos_ga
        records['GA_MO']['min_rate'].append(mr_ga)
        records['GA_MO']['sum_rate'].append(sr_ga)
        records['GA_MO']['energy_step'].append(e_ga)
        records['GA_MO']['energy_cumul'].append(cumul['GA_MO'])
        records['GA_MO']['distance'].append(d_ga)
        print(f"  GA_MO:  mr={mr_ga:.2f} | E={e_ga:.0f}J | d={d_ga:.1f}m | cum={cumul['GA_MO']/1000:.2f}kJ")
        sys.stdout.flush()
        
        # 3. PSO (多目标)
        np.random.seed(seed + step * 100 + 3)
        pos_pso, mr_pso, sr_pso, e_pso, d_pso = run_energy_pso_one_step(
            current_UE_pos, ground_AP_pos, uav_pso, base_config,
            energy_model, max_iter=iter_per_step, energy_lambda=energy_lambda)
        e_pso += hover_E
        cumul['PSO_MO'] += e_pso
        uav_pso = pos_pso
        records['PSO_MO']['min_rate'].append(mr_pso)
        records['PSO_MO']['sum_rate'].append(sr_pso)
        records['PSO_MO']['energy_step'].append(e_pso)
        records['PSO_MO']['energy_cumul'].append(cumul['PSO_MO'])
        records['PSO_MO']['distance'].append(d_pso)
        print(f"  PSO_MO: mr={mr_pso:.2f} | E={e_pso:.0f}J | d={d_pso:.1f}m | cum={cumul['PSO_MO']/1000:.2f}kJ")
        sys.stdout.flush()
        
        # 4. NewSSA (多目标)
        np.random.seed(seed + step * 100 + 4)
        pos_ssa, mr_ssa, sr_ssa, e_ssa, d_ssa = run_energy_ssa_one_step(
            current_UE_pos, ground_AP_pos, uav_ssa, base_config,
            energy_model, max_iter=iter_per_step, energy_lambda=energy_lambda)
        e_ssa += hover_E
        cumul['SSA_MO'] += e_ssa
        uav_ssa = pos_ssa
        records['SSA_MO']['min_rate'].append(mr_ssa)
        records['SSA_MO']['sum_rate'].append(sr_ssa)
        records['SSA_MO']['energy_step'].append(e_ssa)
        records['SSA_MO']['energy_cumul'].append(cumul['SSA_MO'])
        records['SSA_MO']['distance'].append(d_ssa)
        print(f"  SSA_MO: mr={mr_ssa:.2f} | E={e_ssa:.0f}J | d={d_ssa:.1f}m | cum={cumul['SSA_MO']/1000:.2f}kJ")
        sys.stdout.flush()
    
    return records


# =====================================================================
# 绘图
# =====================================================================

def plot_multiobjective_results(records, num_uav, seed, output_dir):
    """绘制多目标对比图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    time_axis = records['time']
    methods = ['BVF', 'GA_MO', 'PSO_MO', 'SSA_MO']
    labels = {'BVF': 'BVF', 'GA_MO': 'GA (MO)', 'PSO_MO': 'PSO (MO)', 'SSA_MO': 'SSA (MO)'}
    colors = {'BVF': '#e74c3c', 'GA_MO': '#3498db', 'PSO_MO': '#2ecc71', 'SSA_MO': '#9b59b6'}
    markers = {'BVF': 'o', 'GA_MO': 's', 'PSO_MO': '^', 'SSA_MO': 'D'}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (a) Min Rate
    ax = axes[0, 0]
    for m in methods:
        ax.plot(time_axis, records[m]['min_rate'], color=colors[m], marker=markers[m],
                markersize=5, linewidth=2, label=labels[m])
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Min User Rate (Mbps)', fontsize=12)
    ax.set_title('(a) Minimum User Rate', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # (b) Cumulative Energy
    ax = axes[0, 1]
    for m in methods:
        ax.plot(time_axis, [e/1000 for e in records[m]['energy_cumul']],
                color=colors[m], marker=markers[m], markersize=5, linewidth=2, label=labels[m])
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Cumulative Energy (kJ)', fontsize=12)
    ax.set_title('(b) Cumulative Energy Consumption', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # (c) Per-Step Energy
    ax = axes[1, 0]
    for m in methods:
        ax.plot(time_axis, records[m]['energy_step'], color=colors[m], marker=markers[m],
                markersize=5, linewidth=2, label=labels[m])
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Energy per Step (J)', fontsize=12)
    ax.set_title('(c) Per-Step Energy', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # (d) Distance
    ax = axes[1, 1]
    for m in methods:
        ax.plot(time_axis, records[m]['distance'], color=colors[m], marker=markers[m],
                markersize=5, linewidth=2, label=labels[m])
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Movement Distance (m)', fontsize=12)
    ax.set_title('(d) UAV Movement per Step', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path1 = os.path.join(output_dir, f'dynamic_MO_{num_uav}uav_seed{seed}.png')
    plt.savefig(path1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {path1}")
    
    # 能效比 (跳过 t=0 避免除零)
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in methods:
        t_plot, eff_plot = [], []
        for i, (t, mr, ec) in enumerate(zip(time_axis, records[m]['min_rate'], records[m]['energy_cumul'])):
            if ec > 0:
                t_plot.append(t)
                eff_plot.append(mr / (ec / 1000))
        if len(t_plot) > 0:
            ax.plot(t_plot, eff_plot, color=colors[m], marker=markers[m],
                    markersize=5, linewidth=2, label=labels[m])
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Energy Efficiency (Mbps/kJ)', fontsize=12)
    ax.set_title(f'Energy Efficiency ({num_uav} UAVs, Multi-Objective)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(output_dir, f'dynamic_MO_efficiency_{num_uav}uav_seed{seed}.png')
    plt.savefig(path2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"能效图已保存: {path2}")
    
    return path1, path2


def print_summary(records):
    methods = ['BVF', 'GA_MO', 'PSO_MO', 'SSA_MO']
    labels = {'BVF': 'BVF', 'GA_MO': 'GA(MO)', 'PSO_MO': 'PSO(MO)', 'SSA_MO': 'SSA(MO)'}
    
    print(f"\n{'='*85}")
    print(f"  多目标动态场景对比总结")
    print(f"{'='*85}")
    print(f"{'算法':<10} {'平均MinRate':<13} {'平均SumRate':<13} {'累计能耗kJ':<13} "
          f"{'累计距离m':<13} {'能效Mbps/kJ':<13}")
    print(f"{'-'*85}")
    
    for m in methods:
        avg_mr = np.mean(records[m]['min_rate'])
        avg_sr = np.mean(records[m]['sum_rate'])
        total_e = records[m]['energy_cumul'][-1] / 1000
        total_d = sum(records[m]['distance'])
        eff = avg_mr / (total_e + 1e-6)
        print(f"{labels[m]:<10} {avg_mr:<13.2f} {avg_sr:<13.1f} {total_e:<13.2f} "
              f"{total_d:<13.1f} {eff:<13.4f}")
    print(f"{'-'*85}")
    sys.stdout.flush()


# =====================================================================
# 主入口
# =====================================================================

if __name__ == "__main__":
    
    OUTPUT_DIR = 'result/dynamic_multiobjective'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    SEED = 62
    NUM_UAV = 9
    NUM_STEPS = 20
    TIME_STEP = 5.0
    ITER_PER_STEP = 15
    USER_SIGMA = 8.0
    ENERGY_LAMBDA = 0.3  # 能耗惩罚权重
    
    records = run_dynamic_multiobjective(
        seed=SEED, num_uav=NUM_UAV, num_steps=NUM_STEPS,
        time_step=TIME_STEP, iter_per_step=ITER_PER_STEP,
        user_sigma=USER_SIGMA, energy_lambda=ENERGY_LAMBDA
    )
    
    print_summary(records)
    
    # 保存 JSON
    json_path = os.path.join(OUTPUT_DIR, f'dynamic_MO_{NUM_UAV}uav_seed{SEED}.json')
    save_data = {
        'config': {
            'seed': SEED, 'num_uav': NUM_UAV, 'num_steps': NUM_STEPS,
            'time_step': TIME_STEP, 'iter_per_step': ITER_PER_STEP,
            'user_sigma': USER_SIGMA, 'energy_lambda': ENERGY_LAMBDA
        },
        'time': records['time'],
    }
    for m in ['BVF', 'GA_MO', 'PSO_MO', 'SSA_MO']:
        save_data[m] = {
            'min_rate': records[m]['min_rate'],
            'sum_rate': records[m]['sum_rate'],
            'energy_step': records[m]['energy_step'],
            'energy_cumul': records[m]['energy_cumul'],
            'distance': records[m]['distance'],
        }
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nJSON 已保存: {json_path}")
    
    # 绘图
    plot_multiobjective_results(records, NUM_UAV, SEED, OUTPUT_DIR)
    
    print(f"\n{'='*70}")
    print(f"  多目标动态对比实验完成!")
    print(f"{'='*70}")
