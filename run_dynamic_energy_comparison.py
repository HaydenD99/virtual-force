"""
动态场景下的四算法能耗对比实验
=================================
场景：
  - 地面用户每隔 5s 进行布朗运动，改变位置
  - 无人机需要相应地重新优化位置
  - 对比 BVF (Energy-Aware V6)、GA、PSO、NewSSA 的能耗与性能

关键设计：
  1. BVF 在力场中加入 "能耗惩罚力"，让步长受到能量约束
  2. GA/PSO/NewSSA：上一时间步的优化结果作为下一时间步的初始位置
  3. 能耗模型：旋翼推进功率 P(V) = P0(1+3V³/U²) + P1[...] + 悬停功率
  4. 跟踪每个时间步的 min_rate、sum_rate、移动距离、累计能耗
"""

import numpy as np
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

# 导入优化器
from balanced_virtual_force_optimizer_v3 import BalancedVirtualForceOptimizerV3, create_balanced_config
from balanced_virtual_force_optimizer_v6 import BalancedVirtualForceOptimizerV6
from genetic_algorithm_optimizer_discrete import DiscreteGeneticAlgorithmOptimizer, create_discrete_ga_config
from distributed_pso_optimizer import DistributedPSOOptimizer, create_distributed_pso_config
from newssa_optimizer import NewSSAOptimizer


# =====================================================================
# 无人机能耗模型
# =====================================================================

class UAVEnergyModel:
    """
    无人机推进能耗模型
    参考文献公式 (4-7)/(4-8):
      P(V) = P0(1 + 3V³/U²) + P1[sqrt(sqrt(1+V⁴/4v_r⁴) - V²/2v_r²)] + ½ρAV³
      P_hover = P0 + P1
    """
    
    def __init__(self, config=None):
        config = config or {}
        # 物理参数
        self.rho_air = config.get('rho_air', 1.225)    # 空气密度 kg/m³
        self.A_rotor = config.get('A_rotor', 0.503)    # 旋翼面积 m²
        self.m_uav = config.get('m_uav', 2.0)          # UAV质量 kg
        self.g = 9.81
        
        # 旋翼参数
        self.delta_blade = config.get('delta_blade', 0.012)  # 叶片剖面阻力系数
        self.Omega = config.get('Omega', 300)                # 角速度 rad/s
        self.R_rotor = config.get('R_rotor', 0.4)            # 旋翼半径 m
        self.k_inc = config.get('k_inc', 0.1)                # 诱导功率修正因子
        self.A_drag = config.get('A_drag', 0.06)             # 机身等效阻力面积 m²
        
        # 计算关键功率参数
        self.U = self.Omega * self.R_rotor                   # 桨尖速度
        W = self.m_uav * self.g                              # 重力
        
        # P0: 叶片剖面功率
        self.P0 = (self.delta_blade / 8) * self.rho_air * self.A_rotor * self.U**3
        # P1: 诱导功率
        self.P1 = (1 + self.k_inc) * (W**3 / (2 * self.rho_air * self.A_rotor))**0.5
        # v_r: 悬停诱导速度
        self.v_r = (W / (2 * self.rho_air * self.A_rotor))**0.5
        # 悬停功率
        self.P_hover = self.P0 + self.P1
    
    def propulsion_power(self, V):
        """
        计算飞行速度 V (m/s) 下的推进功率 (W)
        V=0 时退化为悬停功率
        """
        if V < 1e-3:
            return self.P_hover
        
        # 第一项: P0(1 + 3V²/U²)  —— 注意：论文公式有时写 V² 而非 V³
        term1 = self.P0 * (1 + 3 * V**2 / self.U**2)
        
        # 第二项: P1 * sqrt( sqrt(1 + V⁴/(4v_r⁴)) - V²/(2v_r²) )
        inner = np.sqrt(1 + V**4 / (4 * self.v_r**4)) - V**2 / (2 * self.v_r**2)
        term2 = self.P1 * np.sqrt(max(0, inner))
        
        # 第三项: 寄生阻力  ½ρA_drag V³
        term3 = 0.5 * self.rho_air * self.A_drag * V**3
        
        return term1 + term2 + term3
    
    def flight_energy(self, distance, velocity=10.0):
        """
        计算飞行指定距离的能耗 (J)
        distance: 米, velocity: 飞行速度 m/s
        """
        if distance < 0.1:
            return 0.0
        t = distance / velocity
        P = self.propulsion_power(velocity)
        return P * t
    
    def hover_energy(self, duration):
        """悬停能耗 (J)"""
        return self.P_hover * duration
    
    def total_energy_for_repositioning(self, old_pos, new_pos, flight_speed=10.0, hover_time=0.0):
        """
        计算从 old_pos 到 new_pos 的总能耗 (J)
        包括飞行能耗 + 悬停能耗
        old_pos, new_pos: (L, 3) arrays
        """
        L = len(old_pos)
        total_E = 0.0
        total_dist = 0.0
        
        for l in range(L):
            dist = np.linalg.norm(old_pos[l, :2] - new_pos[l, :2])
            total_dist += dist
            total_E += self.flight_energy(dist, flight_speed)
        
        # 加上悬停能耗 (如果有)
        total_E += L * self.hover_energy(hover_time)
        
        return total_E, total_dist
    
    def summary(self):
        """打印能耗模型参数"""
        print(f"  P0 (叶片功率)   = {self.P0:.2f} W")
        print(f"  P1 (诱导功率)   = {self.P1:.2f} W")
        print(f"  P_hover         = {self.P_hover:.2f} W")
        print(f"  U  (桨尖速度)   = {self.U:.1f} m/s")
        print(f"  v_r (诱导速度)  = {self.v_r:.2f} m/s")
        print(f"  10m/s飞行功率   = {self.propulsion_power(10.0):.2f} W")
        print(f"  20m/s飞行功率   = {self.propulsion_power(20.0):.2f} W")


# =====================================================================
# 用户布朗运动模型
# =====================================================================

def brownian_motion_users(UE_pos, sigma=8.0, square_length=1000, margin=50):
    """
    用户布朗运动: 每个时间步用户位置加上高斯随机位移
    sigma: 每步标准差 (米)，对应 ~1.5 m/s 步行速度 × 5s
    """
    K = len(UE_pos)
    new_UE_pos = UE_pos.copy()
    displacement = np.random.normal(0, sigma, (K, 2))
    new_UE_pos[:, :2] += displacement
    
    # 边界反弹
    for k in range(K):
        for dim in range(2):
            if new_UE_pos[k, dim] < margin:
                new_UE_pos[k, dim] = 2 * margin - new_UE_pos[k, dim]
            elif new_UE_pos[k, dim] > square_length - margin:
                new_UE_pos[k, dim] = 2 * (square_length - margin) - new_UE_pos[k, dim]
    
    # 最终裁剪
    new_UE_pos[:, 0] = np.clip(new_UE_pos[:, 0], margin, square_length - margin)
    new_UE_pos[:, 1] = np.clip(new_UE_pos[:, 1], margin, square_length - margin)
    
    return new_UE_pos


# =====================================================================
# Energy-Aware BVF (based on V6)
# =====================================================================

class EnergyAwareBVF_V6(BalancedVirtualForceOptimizerV6):
    """
    Energy-Aware BVF (V6 force field + 7th energy conservation force)
    
    Design:
      1. 7th force - Energy Conservation Force:
         F_energy[l] = -K * P(V_l)/P_norm * dist_factor * budget_factor * unit(pos-prev)
         Pulls UAV back to previous position. Strength proportional to flight power P(V),
         which grows nonlinearly with speed, so large displacements face much stronger resistance.
      
      2. Selection criterion: min_rate (NOT multi-objective score).
         Energy efficiency is achieved through the force field itself, not post-hoc scoring.
      
      3. Hard displacement limit: max_displacement per step per UAV.
    """
    
    def __init__(self, config, energy_model):
        super().__init__(config)
        self.energy_model = energy_model
        self.cumulative_energy = 0.0
        self.flight_speed = config.get('uav_flight_speed', 10.0)
        
        # === Energy force parameters (7th force) ===
        self.K_energy = config.get('K_energy', 8e4)
        self.w_energy_base = config.get('w_energy', 0.30)
        self.energy_budget = config.get('energy_budget', 2000e3)  # 2000 kJ
        self.P_norm = self.energy_model.P_hover
        
        # === Displacement constraint ===
        self.user_sigma = config.get('user_sigma', 8.0)
        self.max_displacement = config.get('max_displacement', 40.0)
        
        # === Dynamic step size (smaller than static scenario) ===
        self.step_size = config.get('dynamic_step_size', 12)
        
        # Rebalance weights: 6 communication forces get (1-w_energy), energy force gets w_energy
        scale = 1.0 - self.w_energy_base
        self.force_params['w_critical']     = 0.35 * scale
        self.force_params['w_interference'] = 0.20 * scale
        self.force_params['w_universal']    = 0.18 * scale
        self.force_params['w_cooperation']  = 0.12 * scale
        self.force_params['w_separation']   = 0.08 * scale
        self.force_params['w_boundary']     = 0.07 * scale
    
    def _compute_energy_conservation_force(self, UAV_pos, prev_pos):
        """
        Energy Conservation Force (7th virtual force)
        
        F_energy proportional to P(V) / P_hover:
        - Small displacement (~5m): V~1m/s, P(V)~P_hover -> force ~ K * 1.0 (mild)
        - Medium displacement (~30m): V~6m/s, P(V)>P_hover -> force > K * 1.5 (strong)
        - Large displacement (~100m): V~20m/s, P(V)>>P_hover -> force >> K * 3.0 (very strong)
        
        Nonlinear growth: small adjustments are barely penalized, large displacements are strongly resisted.
        """
        forces = np.zeros((self.L, 3))
        
        for l in range(self.L):
            displacement = UAV_pos[l, :2] - prev_pos[l, :2]
            dist = np.linalg.norm(displacement)
            
            if dist < 0.5:
                continue
            
            unit_dir = displacement / (dist + 1e-6)
            
            # Flight power factor: assume 5s to fly, P(V) grows nonlinearly with speed
            velocity = min(dist / 5.0, 20.0)
            P_flight = self.energy_model.propulsion_power(velocity)
            power_factor = P_flight / self.P_norm
            
            # Distance factor: accelerates beyond max_displacement
            if dist <= self.max_displacement:
                dist_factor = dist / self.max_displacement
            else:
                dist_factor = 1.0 + 2.0 * (dist - self.max_displacement) / self.max_displacement
            
            # Budget factor: stronger as cumulative energy grows
            energy_ratio = self.cumulative_energy / self.energy_budget
            budget_factor = 1.0 + 3.0 * energy_ratio
            
            force_mag = self.K_energy * power_factor * dist_factor * budget_factor
            forces[l, :2] = -force_mag * unit_dir
        
        return forces
    
    def _current_energy_weight(self):
        """Dynamic energy force weight: increases with cumulative energy"""
        ratio = self.cumulative_energy / self.energy_budget
        return min(0.55, self.w_energy_base + 0.25 * ratio)
    
    def optimize_one_step(self, UE_pos, ground_AP_pos, UAV_pos_init, max_iter=20):
        """
        Single time-step optimization.
        
        Selection: min_rate (energy is handled by the force field naturally).
        The 7th force pulls UAVs toward previous positions, creating a natural
        trade-off between rate improvement and movement cost.
        """
        prev_pos = UAV_pos_init.copy()
        current_UAV_pos = UAV_pos_init.copy()
        best_min_rate = -np.inf
        best_sum_rate = -np.inf
        best_UAV_pos = UAV_pos_init.copy()
        
        for iteration in range(max_iter):
            # Evaluate current position
            all_AP_pos = np.vstack([ground_AP_pos, current_UAV_pos])
            _, _, betas = self.compute_channel_model(UE_pos, all_AP_pos)
            mask = self.compute_AP_selection_mask(betas)
            rates, sum_rate = self.compute_user_rates(UE_pos, all_AP_pos, mask)
            min_rate = rates.min()
            
            # Select best by min_rate (energy is controlled by force field)
            if min_rate > best_min_rate:
                best_min_rate = min_rate
                best_sum_rate = sum_rate
                best_UAV_pos = current_UAV_pos.copy()
            elif abs(min_rate - best_min_rate) < 0.1 and sum_rate > best_sum_rate:
                best_sum_rate = sum_rate
                best_UAV_pos = current_UAV_pos.copy()
            
            # === 6 communication forces (V6) ===
            forces = self.compute_balanced_virtual_forces(
                UE_pos, ground_AP_pos, current_UAV_pos, rates, mask, betas)
            
            # === 7th force: Energy Conservation Force ===
            f_energy = self._compute_energy_conservation_force(current_UAV_pos, prev_pos)
            w_energy = self._current_energy_weight()
            forces += w_energy * f_energy
            
            # === Position update ===
            progress = iteration / max_iter
            stage_factor = max(0.3, 1.0 - 0.5 * progress)
            adaptive_step = self.step_size * stage_factor
            
            # Energy-aware step size: decreases with cumulative energy
            energy_ratio = self.cumulative_energy / self.energy_budget
            energy_step_scale = max(0.4, 1.0 - 0.6 * energy_ratio)
            adaptive_step *= energy_step_scale
            adaptive_step = np.clip(adaptive_step, 1.5, 15)
            
            # Normalize forces + displacement
            force_norms = np.linalg.norm(forces[:, :2], axis=1)
            max_force = np.max(force_norms) if np.max(force_norms) > 0 else 1
            normalized = forces[:, :2] / max_force
            displacement = adaptive_step * normalized
            
            # Small perturbation
            perturb = max(0.3, 1.5 * (1 - progress) * energy_step_scale)
            displacement += np.random.normal(0, perturb, (self.L, 2))
            
            # Update position
            current_UAV_pos[:, :2] += displacement
            current_UAV_pos[:, :2] = np.clip(current_UAV_pos[:, :2], 50, self.square_length - 50)
            
            # Hard displacement limit per UAV
            for l in range(self.L):
                total_disp = np.linalg.norm(current_UAV_pos[l, :2] - prev_pos[l, :2])
                if total_disp > self.max_displacement:
                    direction = current_UAV_pos[l, :2] - prev_pos[l, :2]
                    current_UAV_pos[l, :2] = prev_pos[l, :2] + direction / total_disp * self.max_displacement
        
        # Compute flight energy for this step
        energy, dist = self.energy_model.total_energy_for_repositioning(
            prev_pos, best_UAV_pos, self.flight_speed)
        self.cumulative_energy += energy
        
        return best_UAV_pos, best_min_rate, best_sum_rate, energy, dist


# =====================================================================
# 对其他算法的包装（单步优化 + 能耗计算）
# =====================================================================

def run_ga_one_step(UE_pos, ground_AP_pos, UAV_pos_init, config, energy_model, max_gen=20):
    """GA 单步优化"""
    config_ga = create_discrete_ga_config()
    config_ga.update(config)
    config_ga['max_generations'] = max_gen
    
    ga_opt = DiscreteGeneticAlgorithmOptimizer(config_ga)
    ga_opt.K = config['num_UE']
    ga_opt.G = config['num_ground_AP']
    
    res = ga_opt.optimize(UE_pos, ground_AP_pos)
    optimized_pos = res['optimized_UAV_pos']
    
    # 计算能耗
    energy, dist = energy_model.total_energy_for_repositioning(
        UAV_pos_init, optimized_pos, 10.0)
    
    return optimized_pos, res['final_min_rate'], res['final_sum_rate'], energy, dist


def run_pso_one_step(UE_pos, ground_AP_pos, UAV_pos_init, config, energy_model, max_iter=20):
    """PSO 单步优化 (初始位置 = 上一步结果)"""
    config_pso = create_distributed_pso_config()
    config_pso.update(config)
    config_pso['max_iterations'] = max_iter
    
    pso_opt = DistributedPSOOptimizer(config_pso)
    res = pso_opt.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
    optimized_pos = res['optimized_UAV_pos']
    
    energy, dist = energy_model.total_energy_for_repositioning(
        UAV_pos_init, optimized_pos, 10.0)
    
    return optimized_pos, res['final_min_rate'], res['final_sum_rate'], energy, dist


def run_ssa_one_step(UE_pos, ground_AP_pos, UAV_pos_init, config, energy_model, max_iter=20):
    """NewSSA 单步优化 (初始位置 = 上一步结果)"""
    config_ssa = config.copy()
    config_ssa['newssa_n_sparrows'] = 30
    config_ssa['newssa_max_iter'] = max_iter
    config_ssa['newssa_pr'] = 0.2
    config_ssa['newssa_fr'] = 0.15
    config_ssa['newssa_st'] = 0.8
    
    ssa_opt = NewSSAOptimizer(config_ssa)
    res = ssa_opt.optimize(UE_pos, ground_AP_pos, UAV_pos_init.copy())
    optimized_pos = res['optimized_UAV_pos']
    
    energy, dist = energy_model.total_energy_for_repositioning(
        UAV_pos_init, optimized_pos, 10.0)
    
    return optimized_pos, res['final_min_rate'], res['final_sum_rate'], energy, dist


# =====================================================================
# 计算无优化时的初始性能
# =====================================================================

def evaluate_initial(UE_pos, ground_AP_pos, UAV_pos, config):
    """使用 V3 信道模型计算初始性能 (不优化)"""
    config_v3 = create_balanced_config()
    config_v3.update(config)
    v3 = BalancedVirtualForceOptimizerV3(config_v3)
    all_AP = np.vstack([ground_AP_pos, UAV_pos])
    _, _, betas = v3.compute_channel_model(UE_pos, all_AP)
    mask = v3.compute_AP_selection_mask(betas)
    rates, sum_rate = v3.compute_user_rates(UE_pos, all_AP, mask)
    return float(rates.min()), float(sum_rate)


# =====================================================================
# 动态场景主函数
# =====================================================================

def run_dynamic_comparison(seed=62, num_uav=9, num_steps=20, 
                           time_step=5.0, iter_per_step=20,
                           user_sigma=8.0):
    """
    运行动态场景对比实验
    
    参数:
      seed:           随机种子
      num_uav:        UAV 数量
      num_steps:      时间步数 (每步 5s)
      time_step:      时间步长 (秒)
      iter_per_step:  每时间步的优化迭代次数
      user_sigma:     用户布朗运动标准差 (米/步)
    """
    np.random.seed(seed)
    
    # --- 系统配置 ---
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
    
    # --- 初始化能耗模型 ---
    energy_model = UAVEnergyModel()
    print("="*70)
    print("  无人机能耗模型参数")
    print("="*70)
    energy_model.summary()
    
    # --- 初始化位置 ---
    # UE
    UE_pos = np.random.uniform(low=[50, 50], high=[square_length-50, square_length-50], size=(K, 2))
    UE_pos = np.column_stack([UE_pos, np.ones(K) * 1.65])
    
    # Ground AP (2x2 Grid)
    gx = np.linspace(square_length * 0.25, square_length * 0.75, 2)
    gy = np.linspace(square_length * 0.25, square_length * 0.75, 2)
    GX, GY = np.meshgrid(gx, gy)
    ground_AP_pos = np.column_stack([GX.flatten(), GY.flatten(), np.ones(G) * 15.0])
    
    # UAV 初始位置
    if num_uav == 6:
        ux = np.linspace(200, 800, 3); uy = np.linspace(300, 700, 2)
    elif num_uav == 12:
        ux = np.linspace(200, 800, 4); uy = np.linspace(200, 800, 3)
    else:
        ux = np.linspace(200, 800, 3); uy = np.linspace(200, 800, 3)
    UX, UY = np.meshgrid(ux, uy)
    UAV_pos_init = np.column_stack([
        UX.flatten()[:num_uav], UY.flatten()[:num_uav],
        np.ones(num_uav) * 50.0
    ])
    
    print(f"\n{'='*70}")
    print(f"  动态场景对比实验")
    print(f"{'='*70}")
    print(f"  Seed: {seed} | UAV: {num_uav} | 用户: {K} | L_serving: {L_SERVING}")
    print(f"  时间步: {num_steps} × {time_step}s = {num_steps*time_step}s")
    print(f"  每步优化迭代: {iter_per_step}")
    print(f"  用户运动 σ: {user_sigma} m/step")
    print(f"{'='*70}\n")
    
    # --- 初始化四算法的 UAV 位置 ---
    uav_bvf = UAV_pos_init.copy()
    uav_ga  = UAV_pos_init.copy()
    uav_pso = UAV_pos_init.copy()
    uav_ssa = UAV_pos_init.copy()
    
    # Energy-Aware BVF
    config_bvf = create_balanced_config()
    config_bvf.update(base_config)
    config_bvf['max_iterations'] = iter_per_step
    config_bvf['user_sigma'] = user_sigma
    config_bvf['max_displacement'] = 5.0 * user_sigma  # 位移硬上限
    bvf_optimizer = EnergyAwareBVF_V6(config_bvf, energy_model)
    
    # --- 记录数据 ---
    alg_keys = ['BVF', 'GA', 'PSO', 'NewSSA']
    records = {'time': []}
    for k in alg_keys:
        records[k] = {'min_rate': [], 'sum_rate': [], 'energy_step': [], 'energy_cumul': [], 'distance': []}
    
    cumul_energy = {k: 0.0 for k in alg_keys}
    hover_energy_per_step = energy_model.hover_energy(time_step) * num_uav
    
    current_UE_pos = UE_pos.copy()
    
    # =====================================================
    # Step 0 (t=0s): 初始状态，所有算法相同，不做优化
    # =====================================================
    records['time'].append(0.0)
    
    print(f"\n--- 时间步 0 (t=0s): 初始状态 (无优化) ---")
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
    
    # =====================================================
    # Step 1 ~ num_steps: 用户运动 + 优化
    # =====================================================
    for step in range(1, num_steps):
        t = step * time_step
        records['time'].append(t)
        
        print(f"\n--- 时间步 {step}/{num_steps-1} (t = {t:.0f}s) ---")
        
        # 用户布朗运动
        current_UE_pos = brownian_motion_users(current_UE_pos, sigma=user_sigma,
                                               square_length=square_length)
        
        # === 1. BVF (Energy-Aware V6) ===
        np.random.seed(seed + step * 100 + 1)
        pos_bvf, mr_bvf, sr_bvf, e_bvf, d_bvf = bvf_optimizer.optimize_one_step(
            current_UE_pos, ground_AP_pos, uav_bvf, max_iter=iter_per_step)
        e_bvf += hover_energy_per_step
        cumul_energy['BVF'] += e_bvf
        uav_bvf = pos_bvf
        records['BVF']['min_rate'].append(mr_bvf)
        records['BVF']['sum_rate'].append(sr_bvf)
        records['BVF']['energy_step'].append(e_bvf)
        records['BVF']['energy_cumul'].append(cumul_energy['BVF'])
        records['BVF']['distance'].append(d_bvf)
        print(f"  BVF:    min_rate={mr_bvf:.2f} Mbps | E_step={e_bvf:.0f}J | dist={d_bvf:.1f}m | E_total={cumul_energy['BVF']/1000:.2f}kJ")
        
        # === 2. GA ===
        np.random.seed(seed + step * 100 + 2)
        pos_ga, mr_ga, sr_ga, e_ga, d_ga = run_ga_one_step(
            current_UE_pos, ground_AP_pos, uav_ga, base_config, energy_model, max_gen=iter_per_step)
        e_ga += hover_energy_per_step
        cumul_energy['GA'] += e_ga
        uav_ga = pos_ga
        records['GA']['min_rate'].append(mr_ga)
        records['GA']['sum_rate'].append(sr_ga)
        records['GA']['energy_step'].append(e_ga)
        records['GA']['energy_cumul'].append(cumul_energy['GA'])
        records['GA']['distance'].append(d_ga)
        print(f"  GA:     min_rate={mr_ga:.2f} Mbps | E_step={e_ga:.0f}J | dist={d_ga:.1f}m | E_total={cumul_energy['GA']/1000:.2f}kJ")
        
        # === 3. PSO ===
        np.random.seed(seed + step * 100 + 3)
        pos_pso, mr_pso, sr_pso, e_pso, d_pso = run_pso_one_step(
            current_UE_pos, ground_AP_pos, uav_pso, base_config, energy_model, max_iter=iter_per_step)
        e_pso += hover_energy_per_step
        cumul_energy['PSO'] += e_pso
        uav_pso = pos_pso
        records['PSO']['min_rate'].append(mr_pso)
        records['PSO']['sum_rate'].append(sr_pso)
        records['PSO']['energy_step'].append(e_pso)
        records['PSO']['energy_cumul'].append(cumul_energy['PSO'])
        records['PSO']['distance'].append(d_pso)
        print(f"  PSO:    min_rate={mr_pso:.2f} Mbps | E_step={e_pso:.0f}J | dist={d_pso:.1f}m | E_total={cumul_energy['PSO']/1000:.2f}kJ")
        
        # === 4. NewSSA ===
        np.random.seed(seed + step * 100 + 4)
        pos_ssa, mr_ssa, sr_ssa, e_ssa, d_ssa = run_ssa_one_step(
            current_UE_pos, ground_AP_pos, uav_ssa, base_config, energy_model, max_iter=iter_per_step)
        e_ssa += hover_energy_per_step
        cumul_energy['NewSSA'] += e_ssa
        uav_ssa = pos_ssa
        records['NewSSA']['min_rate'].append(mr_ssa)
        records['NewSSA']['sum_rate'].append(sr_ssa)
        records['NewSSA']['energy_step'].append(e_ssa)
        records['NewSSA']['energy_cumul'].append(cumul_energy['NewSSA'])
        records['NewSSA']['distance'].append(d_ssa)
        print(f"  NewSSA: min_rate={mr_ssa:.2f} Mbps | E_step={e_ssa:.0f}J | dist={d_ssa:.1f}m | E_total={cumul_energy['NewSSA']/1000:.2f}kJ")
    
    return records


# =====================================================================
# 绘图
# =====================================================================

def plot_dynamic_results(records, num_uav, seed, output_dir):
    """生成动态场景对比图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    time_axis = records['time']
    methods = ['BVF', 'GA', 'PSO', 'NewSSA']
    colors = {'BVF': '#e74c3c', 'GA': '#3498db', 'PSO': '#2ecc71', 'NewSSA': '#9b59b6'}
    markers = {'BVF': 'o', 'GA': 's', 'PSO': '^', 'NewSSA': 'D'}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- 子图 1: Minimum User Rate ---
    ax = axes[0, 0]
    for m in methods:
        ax.plot(time_axis, records[m]['min_rate'], color=colors[m], marker=markers[m],
                markersize=5, linewidth=2, label=m)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Min User Rate (Mbps)', fontsize=12)
    ax.set_title('(a) Minimum User Rate over Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # --- 子图 2: Cumulative Energy Consumption ---
    ax = axes[0, 1]
    for m in methods:
        ax.plot(time_axis, [e/1000 for e in records[m]['energy_cumul']], 
                color=colors[m], marker=markers[m], markersize=5, linewidth=2, label=m)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Cumulative Energy (kJ)', fontsize=12)
    ax.set_title('(b) Cumulative Energy Consumption', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # --- 子图 3: Per-Step Energy ---
    ax = axes[1, 0]
    for m in methods:
        ax.plot(time_axis, records[m]['energy_step'], color=colors[m], marker=markers[m],
                markersize=5, linewidth=2, label=m)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Energy per Step (J)', fontsize=12)
    ax.set_title('(c) Energy Consumption per Time Step', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # --- 子图 4: Per-Step Movement Distance ---
    ax = axes[1, 1]
    for m in methods:
        ax.plot(time_axis, records[m]['distance'], color=colors[m], marker=markers[m],
                markersize=5, linewidth=2, label=m)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Total Movement Distance (m)', fontsize=12)
    ax.set_title('(d) UAV Movement Distance per Step', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'dynamic_energy_{num_uav}uav_seed{seed}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {save_path}")
    
    # --- 额外: 能效比图 (min_rate / cumulative_energy) ---
    # 跳过 t=0 (能耗为0导致除零), 从 t=5s 开始绘制
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in methods:
        # 只取 t>0 的数据
        t_plot = []
        eff_plot = []
        for i, (t, mr, ec) in enumerate(zip(time_axis, records[m]['min_rate'], records[m]['energy_cumul'])):
            if ec > 0:  # 只绘制有能耗的时间步
                t_plot.append(t)
                eff_plot.append(mr / (ec / 1000))  # Mbps / kJ
        if len(t_plot) > 0:
            ax.plot(t_plot, eff_plot, color=colors[m], marker=markers[m],
                    markersize=5, linewidth=2, label=m)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Energy Efficiency (Mbps/kJ)', fontsize=12)
    ax.set_title(f'Energy Efficiency: Min Rate / Cumulative Energy ({num_uav} UAVs)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path2 = os.path.join(output_dir, f'dynamic_efficiency_{num_uav}uav_seed{seed}.png')
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"能效图已保存: {save_path2}")
    
    return save_path, save_path2


def print_summary_table(records):
    """打印汇总表"""
    methods = ['BVF', 'GA', 'PSO', 'NewSSA']
    
    print(f"\n{'='*85}")
    print(f"  动态场景对比总结")
    print(f"{'='*85}")
    print(f"{'算法':<8} {'平均Min Rate':<14} {'平均Sum Rate':<14} {'累计能耗(kJ)':<14} "
          f"{'累计距离(m)':<14} {'能效(Mbps/kJ)':<14}")
    print(f"{'-'*85}")
    
    for m in methods:
        avg_mr = np.mean(records[m]['min_rate'])
        avg_sr = np.mean(records[m]['sum_rate'])
        total_e = records[m]['energy_cumul'][-1] / 1000  # kJ
        total_d = sum(records[m]['distance'])
        eff = avg_mr / (total_e + 1e-6)
        print(f"{m:<8} {avg_mr:<14.2f} {avg_sr:<14.1f} {total_e:<14.2f} {total_d:<14.1f} {eff:<14.4f}")
    
    print(f"{'-'*85}")


# =====================================================================
# 主入口
# =====================================================================

if __name__ == "__main__":
    
    OUTPUT_DIR = 'result/dynamic_energy'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 运行配置
    SEED = 62
    NUM_UAV = 9
    NUM_STEPS = 20    # 20 × 5s = 100s
    TIME_STEP = 5.0   # 秒
    ITER_PER_STEP = 15  # 每步优化迭代次数 (较少以适应实时需求)
    USER_SIGMA = 8.0   # 用户每步移动标准差 (米)
    
    # 运行对比
    records = run_dynamic_comparison(
        seed=SEED,
        num_uav=NUM_UAV,
        num_steps=NUM_STEPS,
        time_step=TIME_STEP,
        iter_per_step=ITER_PER_STEP,
        user_sigma=USER_SIGMA
    )
    
    # 汇总
    print_summary_table(records)
    
    # 保存 JSON
    json_path = os.path.join(OUTPUT_DIR, f'dynamic_{NUM_UAV}uav_seed{SEED}.json')
    save_data = {
        'config': {
            'seed': SEED, 'num_uav': NUM_UAV, 'num_steps': NUM_STEPS,
            'time_step': TIME_STEP, 'iter_per_step': ITER_PER_STEP,
            'user_sigma': USER_SIGMA
        },
        'time': records['time'],
    }
    for m in ['BVF', 'GA', 'PSO', 'NewSSA']:
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
    plot_dynamic_results(records, NUM_UAV, SEED, OUTPUT_DIR)
    
    print(f"\n{'='*70}")
    print(f"  实验完成!")
    print(f"{'='*70}")
