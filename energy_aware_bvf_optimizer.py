"""
能量感知的平衡虚拟力优化器 (Energy-Aware BVF)
在BVF基础上加入无人机能耗模型和能量约束的自适应步长

参考文献中的无人机能耗模型：
- 推动能量：E^P(V) - 公式(4-7)
- 悬停能量：E^H = P0 + P1 - 公式(4-8)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as sl
import time
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 导入必要的函数
import functionRlocalscattering
import SpectralEfficiencyDownlink


class EnergyAwareBVFOptimizer:
    """能量感知的平衡虚拟力优化器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_parameters()
        self.setup_energy_model()
        
    def setup_parameters(self):
        """设置系统参数"""
        # 基本参数
        self.square_length = self.config.get('square_length', 1000)
        self.K = self.config.get('num_UE', 60)
        self.L = self.config.get('num_UAV', 9)
        self.G = self.config.get('num_ground_AP', 4)
        self.M = self.config.get('M', 4)
        
        # 高度设置
        self.heights = {
            'UE': self.config.get('UE_height', 1.65),
            'ground_AP': self.config.get('ground_AP_height', 15.0),
            'UAV': self.config.get('UAV_height', 50.0)
        }
        
        # 信道参数
        self.alpha = self.config.get('alpha', 3.67)
        self.constant_term = self.config.get('constant_term', -30.5)
        self.sigma_sf = self.config.get('sigma_sf', 1)
        self.antenna_spacing = self.config.get('antenna_spacing', 0.5)
        self.ASD_deg = self.config.get('ASD_deg', 10)
        
        # 通信参数
        self.B = self.config.get('B', 20e6)
        self.p = self.config.get('p', 100)
        self.Pmax = self.config.get('Pmax', 1000)
        self.noise_figure = self.config.get('noise_figure', 7)
        self.distance_vertical = self.config.get('distance_vertical', 150)
        
        # 导频参数
        self.tau_p = self.config.get('tau_p', self.K)
        self.tau_c = self.config.get('tau_c', 200)
        self.prelogFactor = (self.tau_c - self.tau_p) / self.tau_c
        
        # 虚拟力参数
        self.force_params = {
            'K_min_rate': self.config.get('K_min_rate', 6e4),
            'K_universal': self.config.get('K_universal', 3e4),
            'K_cooperation': self.config.get('K_cooperation', 2e4),
            'K_separation': self.config.get('K_separation', 1.5e4),
            'K_boundary': self.config.get('K_boundary', 2.5e4),
            'w_min_rate': 0.4,
            'w_universal': 0.25,
            'w_cooperation': 0.15,
            'w_separation': 0.1,
            'w_boundary': 0.1,
            'separation_distance': 120,
            'boundary_margin': 60,
            'cooperation_distance': 200,
        }
        
        # 优化参数
        self.step_size = self.config.get('step_size', 25)
        self.max_iterations = self.config.get('max_iterations', 100)
        self.convergence_threshold = self.config.get('convergence_threshold', 1e-3)
        self.num_serving_APs = self.config.get('num_serving_APs', 3)
        self.nbrOfRealizations = self.config.get('nbrOfRealizations', 50)
        
        # 全局搜索参数
        self.restart_threshold = 25
        self.max_restarts = 3
        self.perturbation_strength = 80
        
        # 计算噪声功率
        self.noise_variance_dBm = -174 + 10*np.log10(self.B) + self.noise_figure
        self.eyeM = np.eye(self.M)
    
    def setup_energy_model(self):
        """
        设置无人机能耗模型参数
        基于论文公式(4-7)和(4-8)
        """
        # === 基本物理参数 ===
        self.rho_air = self.config.get('rho_air', 1.225)  # 空气密度 kg/m³
        self.A_rotor = self.config.get('A_rotor', 0.503)  # 旋翼面积 m²
        self.m_uav = self.config.get('m_uav', 2.0)        # UAV质量 kg
        self.g = 9.81  # 重力加速度 m/s²
        
        # === 能耗模型参数 (根据论文) ===
        # P0: 叶片剖面功率
        self.delta_blade = self.config.get('delta_blade', 0.012)  # 叶片剖面阻力系数
        self.Omega = self.config.get('Omega', 300)  # 旋翼角速度 rad/s
        self.R_rotor = self.config.get('R_rotor', 0.4)  # 旋翼半径 m
        self.k_inc = self.config.get('k_inc', 0.1)  # 诱导功率修正因子
        
        # 计算P0 (叶片剖面功率)
        self.P0 = (self.delta_blade / 8) * self.rho_air * self.A_rotor * \
                  (self.Omega * self.R_rotor)**3
        
        # P1: 诱导功率
        W = self.m_uav * self.g  # 无人机重量 N
        self.P1 = (1 + self.k_inc) * (W**3 / (2 * self.rho_air * self.A_rotor))**(1/2)
        
        # U: 转子尖端速度
        self.U = self.Omega * self.R_rotor
        
        # v_r: 转子诱导速度
        self.v_r = (W / (2 * self.rho_air * self.A_rotor))**(1/2)
        
        # A: 机身阻力系数
        self.A_drag = self.config.get('A_drag', 0.5)
        
        # === 能量预算 ===
        # 根据任务规模动态设置能量预算
        # 计算悬停功率用于估算
        hover_power = self.P0 + self.P1  # W
        
        # 更合理的能量预算估算:
        # - 优化通常在30-40次迭代内收敛（给予充足的优化空间）
        # - 每次迭代包含: 悬停0.4秒 + 移动15米(约1.5秒)
        # - 移动功率约为悬停功率的1.5倍
        avg_iterations = 40  # 预期迭代次数（给予充足优化空间）
        hover_time_per_iter = 0.4  # 秒
        move_time_per_iter = 1.5   # 秒（增加预算）
        move_power_factor = 1.5    # 移动时功率是悬停的1.5倍
        
        estimated_energy_per_iter = (
            hover_power * hover_time_per_iter + 
            hover_power * move_power_factor * move_time_per_iter
        )
        
        # 总能量 = 单次迭代能量 * 预期迭代次数 * 安全系数
        # 提高安全系数，确保有足够能量达到最佳性能
        safety_factor = 1.8  # 80%安全余量（优先保证性能）
        
        self.initial_energy = self.config.get('initial_energy', 
                                             estimated_energy_per_iter * avg_iterations * safety_factor)
        
        self.min_energy_reserve = self.config.get('min_energy_reserve', 0.05)  # 保留5%能量（放宽限制）
        
        # 初始化每个UAV的剩余能量
        self.remaining_energy = np.full(self.L, self.initial_energy, dtype=float)
        
        print(f"\n⚡ 能量模型初始化:")
        print(f"  P0 (叶片功率): {self.P0:.2f} W")
        print(f"  P1 (诱导功率): {self.P1:.2f} W")
        print(f"  悬停功率: {hover_power:.2f} W")
        print(f"  预期迭代次数: {avg_iterations}")
        print(f"  单次迭代能量: {estimated_energy_per_iter:.2f} J")
        print(f"  初始能量预算: {self.initial_energy/3600:.2f} Wh/UAV")
        print(f"  能量安全余量: {(safety_factor-1)*100:.0f}%")

    def compute_propulsion_power(self, velocity: float) -> float:
        """
        计算推动功率 (瓦特)
        根据论文公式(4-7): E^P = P0(1 + 3V³/U²) + P1[√(√(1 + V⁴/4v_r⁴) - V²/2v_r²) + (1/2)AV³]
        
        参数:
            velocity: UAV飞行速度 (m/s)
        返回:
            power: 推动功率 (W)
        """
        V = velocity
        
        if V == 0:
            # 悬停功率 - 公式(4-8)
            return self.P0 + self.P1
        
        # 公式(4-7)的各个部分
        # 第一项: P0(1 + 3V³/U²)
        term1 = self.P0 * (1 + 3 * V**3 / self.U**2)
        
        # 第二项: P1[√(√(1 + V⁴/4v_r⁴) - V²/2v_r²) + (1/2)AV³]
        inner_sqrt = np.sqrt(1 + V**4 / (4 * self.v_r**4)) - V**2 / (2 * self.v_r**2)
        outer_sqrt = np.sqrt(inner_sqrt) if inner_sqrt > 0 else 0
        term2 = self.P1 * (outer_sqrt + 0.5 * self.A_drag * V**3)
        
        power = term1 + term2
        
        return power
    
    def compute_energy_consumption(self, distance: float, time_duration: float) -> float:
        """
        计算飞行指定距离消耗的能量
        
        参数:
            distance: 飞行距离 (m)
            time_duration: 飞行时间 (s)
        返回:
            energy: 消耗的能量 (J)
        """
        if distance == 0:
            # 悬停
            velocity = 0
        else:
            velocity = distance / time_duration  # m/s
        
        power = self.compute_propulsion_power(velocity)
        energy = power * time_duration  # 焦耳
        
        return energy
    
    def estimate_movement_time(self, distance: float, base_velocity: float = 10.0) -> float:
        """
        估计移动时间
        
        参数:
            distance: 移动距离 (m)
            base_velocity: 基准速度 (m/s)，默认10m/s
        返回:
            time: 估计时间 (s)
        """
        return distance / base_velocity if distance > 0 else 1.0  # 最小1秒

    def compute_energy_aware_step_size(self, uav_idx: int, ideal_displacement: np.ndarray,
                                      base_step: float, iteration: int) -> Tuple[float, float]:
        """
        计算考虑能量约束的自适应步长
        
        参数:
            uav_idx: UAV索引
            ideal_displacement: 理想位移向量 (m)
            base_step: 基础步长 (m)
            iteration: 当前迭代次数
        返回:
            (adjusted_step, energy_consumed): 调整后的步长和消耗的能量
        """
        ideal_distance = np.linalg.norm(ideal_displacement)
        
        if ideal_distance == 0:
            return base_step, 0.0
        
        # 1. 计算理想移动所需能量
        movement_time = self.estimate_movement_time(ideal_distance)
        required_energy = self.compute_energy_consumption(ideal_distance, movement_time)
        
        # 2. 检查剩余能量
        available_energy = self.remaining_energy[uav_idx] * (1 - self.min_energy_reserve)
        
        # 3. 能量约束因子
        if required_energy > available_energy:
            # 能量不足，缩减步长
            energy_factor = available_energy / (required_energy + 1e-6)
            energy_factor = np.clip(energy_factor, 0.1, 1.0)  # 至少保留10%移动能力
            
            print(f"  ⚠️  UAV {uav_idx}: 能量限制 {energy_factor:.1%} (剩余: {self.remaining_energy[uav_idx]/3600:.2f}Wh)")
        else:
            energy_factor = 1.0
        
        # 4. 能量效率优化因子 - 温和的衰减策略（优先保证性能）
        # 使用更温和的衰减，让算法有足够空间优化
        progress_ratio = iteration / self.max_iterations
        
        # 前40%迭代: 几乎完整步长 (0.95-1.0)
        # 中间40%迭代: 中等步长 (0.7-0.95)  
        # 后20%迭代: 适度减小 (0.5-0.7)
        if progress_ratio < 0.4:
            conservation_factor = 1.0 - 0.05 * (progress_ratio / 0.4)  # 只减5%
        elif progress_ratio < 0.8:
            conservation_factor = 0.95 - 0.25 * ((progress_ratio - 0.4) / 0.4)  # 减到70%
        else:
            conservation_factor = 0.7 - 0.2 * ((progress_ratio - 0.8) / 0.2)  # 最后减到50%
        
        # 5. 能量预算因子 - 只在能量严重不足时才限制
        energy_ratio = self.remaining_energy[uav_idx] / self.initial_energy
        if energy_ratio < 0.15:  # 只在能量低于15%时才限制
            budget_factor = 0.6 * (energy_ratio / 0.15)  # 不要限制太严
            conservation_factor *= budget_factor
        
        # 6. 综合调整步长
        adjusted_step = base_step * energy_factor * conservation_factor
        
        # 7. 最小移动阈值 - 避免微小移动浪费能量（但不要限制太严）
        min_movement_threshold = 0.5  # 小于0.5米才不移动（放宽限制）
        if adjusted_step < min_movement_threshold:
            return 0.0, 0.0
        
        # 8. 计算实际消耗的能量
        actual_distance = adjusted_step
        actual_time = self.estimate_movement_time(actual_distance)
        energy_consumed = self.compute_energy_consumption(actual_distance, actual_time)
        
        return adjusted_step, energy_consumed

    # ==================== 从BVF复用的方法 ====================
    
    def initialize_positions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """初始化所有节点位置"""
        UE_xy = np.random.uniform(50, self.square_length-50, (self.K, 2))
        UE_pos = np.hstack([UE_xy, np.full((self.K, 1), self.heights['UE'])])
        ground_AP_pos = self._generate_ground_AP_positions()
        UAV_pos = self._smart_initialize_UAVs(UE_pos, ground_AP_pos)
        return UE_pos, ground_AP_pos, UAV_pos
    
    def _generate_ground_AP_positions(self) -> np.ndarray:
        """生成地面AP位置"""
        grid_size = int(np.sqrt(self.G))
        if grid_size * grid_size < self.G:
            grid_size += 1
        spacing = self.square_length / (grid_size + 1)
        positions = []
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= self.G:
                    break
                x = (i + 1) * spacing
                y = (j + 1) * spacing
                positions.append([x, y, self.heights['ground_AP']])
                count += 1
        return np.array(positions[:self.G])
    
    def _smart_initialize_UAVs(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray) -> np.ndarray:
        """智能初始化UAV位置"""
        user_centers = self._adaptive_clustering(UE_pos[:, :2], self.L)
        positions = []
        for i in range(self.L):
            base_pos = user_centers[i]
            distances_to_ground = [np.linalg.norm(base_pos - ground_AP_pos[g, :2]) 
                                 for g in range(self.G)]
            nearest_ground_idx = np.argmin(distances_to_ground)
            nearest_ground_pos = ground_AP_pos[nearest_ground_idx, :2]
            cooperation_factor = 0.3
            balanced_pos = (1 - cooperation_factor) * base_pos + cooperation_factor * nearest_ground_pos
            angle = np.random.uniform(0, 2*np.pi)
            radius = np.random.uniform(30, 100)
            x = balanced_pos[0] + radius * np.cos(angle)
            y = balanced_pos[1] + radius * np.sin(angle)
            x = np.clip(x, 100, self.square_length - 100)
            y = np.clip(y, 100, self.square_length - 100)
            positions.append([x, y, self.heights['UAV']])
        return np.array(positions)
    
    def _adaptive_clustering(self, points: np.ndarray, k: int) -> np.ndarray:
        """K-means++聚类"""
        n_points = len(points)
        centers = np.zeros((k, 2))
        centers[0] = points[np.random.choice(n_points)]
        for i in range(1, k):
            distances = np.min([np.sum((points - c)**2, axis=1) for c in centers[:i]], axis=0)
            probabilities = distances / (distances.sum() + 1e-12)
            centers[i] = points[np.random.choice(n_points, p=probabilities)]
        for _ in range(15):
            distances = np.sqrt(((points[:, np.newaxis] - centers) ** 2).sum(axis=2))
            assignments = np.argmin(distances, axis=1)
            for i in range(k):
                cluster_points = points[assignments == i]
                if len(cluster_points) > 0:
                    centers[i] = cluster_points.mean(axis=0)
        return centers
    
    def compute_channel_model(self, UE_pos: np.ndarray, AP_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算信道模型（简化版，复用BVF逻辑）"""
        L_total = len(AP_pos)
        diff = UE_pos[:, None, :2] - AP_pos[None, :, :2]
        distances = np.sqrt(np.sum(diff**2, axis=-1) + self.distance_vertical**2)
        angles = np.arctan2(diff[..., 1], diff[..., 0])
        channel_gain_dB = self.constant_term - self.alpha * 10 * np.log10(distances)
        channel_gain_over_noise = channel_gain_dB - self.noise_variance_dBm
        betas = 10 ** (channel_gain_over_noise / 10)
        
        R = np.zeros((self.M, self.M, self.K, L_total), dtype=complex)
        for k in range(self.K):
            for l in range(L_total):
                R[:, :, k, l] = functionRlocalscattering.R(self.M, angles[k, l], self.ASD_deg)
        CorrR = betas[None, None, :, :] * R
        
        CH = np.sqrt(0.5) * (np.random.randn(self.M, self.nbrOfRealizations, self.K, L_total) +
                            1j * np.random.randn(self.M, self.nbrOfRealizations, self.K, L_total))
        H = np.empty_like(CH, dtype=complex)
        
        for k in range(self.K):
            for l in range(L_total):
                try:
                    corr_matrix = CorrR[:, :, k, l]
                    if not np.isfinite(corr_matrix).all():
                        H[:, :, k, l] = CH[:, :, k, l]
                        continue
                    Rsqrt = sl.sqrtm(corr_matrix)
                    if not np.isfinite(Rsqrt).all():
                        Rsqrt = np.linalg.cholesky(corr_matrix + 1e-6 * np.eye(self.M))
                    H[:, :, k, l] = Rsqrt @ CH[:, :, k, l]
                except:
                    scaling = np.sqrt(np.abs(betas[k, l]) + 1e-12)
                    H[:, :, k, l] = scaling * CH[:, :, k, l]
        
        pilotIndex = np.random.permutation(self.K) % self.tau_p
        Np = np.sqrt(0.5) * (np.random.randn(self.M, self.nbrOfRealizations, L_total, self.tau_p) +
                            1j * np.random.randn(self.M, self.nbrOfRealizations, L_total, self.tau_p))
        Hhat = np.zeros_like(H)
        
        for l in range(L_total):
            for t in range(self.tau_p):
                indices = np.where(pilotIndex == t)[0]
                if len(indices) == 0:
                    continue
                yp = np.sqrt(self.p * self.tau_p) * np.sum(H[:, :, indices, l], axis=2) + Np[:, :, l, t]
                PsiInv = (self.p * self.tau_p * np.sum(CorrR[:, :, indices, l], axis=2) + self.eyeM)
                PsiInvInv = np.linalg.inv(PsiInv)
                for k in indices:
                    RPsi = CorrR[:, :, k, l] @ PsiInvInv
                    Hhat[:, :, k, l] = np.sqrt(self.p * self.tau_p) * RPsi @ yp
        
        return H, Hhat, betas
    
    def compute_AP_selection_mask(self, betas: np.ndarray) -> np.ndarray:
        """计算AP选择掩码"""
        top_AP_indices = np.argpartition(betas, -self.num_serving_APs, axis=1)[:, -self.num_serving_APs:]
        mask = np.zeros((self.K, betas.shape[1]), dtype=bool)
        for k in range(self.K):
            mask[k, top_AP_indices[k]] = True
        return mask
    
    def compute_user_rates(self, UE_pos: np.ndarray, AP_pos: np.ndarray, 
                          mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """计算用户速率"""
        L_total = len(AP_pos)
        H, Hhat, betas = self.compute_channel_model(UE_pos, AP_pos)
        Hhat_uc = Hhat * mask[np.newaxis, np.newaxis, :, :]
        num_served_per_AP = mask.sum(axis=0)
        rho = np.zeros((self.K, L_total))
        for l in range(L_total):
            if num_served_per_AP[l] > 0:
                rho[mask[:, l], l] = self.Pmax / num_served_per_AP[l]
        gamma = np.sqrt(rho)
        a_MR, B_MR = self._compute_a_B_MR(H, Hhat_uc, gamma)
        SE_MR = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(a_MR, B_MR, self.B, gamma, self.Pmax)
        rates = SE_MR * self.prelogFactor / 1e6
        return rates, np.sum(rates)
    
    def _compute_a_B_MR(self, H: np.ndarray, Hhat_uc: np.ndarray, 
                       gamma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算MR处理的信号和干扰项"""
        M, N, K, L = H.shape
        w_MR = Hhat_uc / (np.linalg.norm(Hhat_uc, axis=0, keepdims=True) + 1e-12)
        a_MR = np.einsum('mnkl,mnkl->lk', np.conj(H), w_MR) / N
        interf_MR = np.einsum('mnkl,mnil->kiln', np.conj(H), w_MR).mean(axis=-1)
        B_MR = np.zeros((L, L, K, K), dtype=np.float64)
        for k in range(K):
            for i in range(K):
                B_MR[:, :, k, i] = np.outer(interf_MR[k, i, :], interf_MR[k, i, :].conj()).real
        interf2_MR = np.abs(interf_MR) ** 2
        for l in range(L):
            B_MR[l, l, :, :] = interf2_MR[:, :, l]
        a_MR = np.abs(a_MR)
        return a_MR, B_MR
    
    def compute_balanced_virtual_forces(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray,
                                      UAV_pos: np.ndarray, rates: np.ndarray, 
                                      mask: np.ndarray, betas: np.ndarray) -> np.ndarray:
        """计算平衡虚拟力（复用BVF逻辑，这里简化）"""
        forces = np.zeros((self.L, 3))
        weights = self.force_params
        
        # 简化的力计算（完整实现可参考BVF）
        # 这里只示例最小速率导向力
        min_rate = rates.min()
        threshold = min_rate * 1.15
        critical_users = np.where(rates <= threshold)[0]
        urgency = 1.0 / (rates + 0.001)
        urgency = urgency / urgency.max()
        
        for l in range(self.L):
            force_total = np.zeros(3)
            served_users = np.where(mask[:, l])[0]
            
            for k in served_users:
                if k in critical_users:
                    direction = UE_pos[k] - UAV_pos[l]
                    distance = np.linalg.norm(direction) + 1e-6
                    unit_direction = direction / distance
                    urgency_weight = urgency[k]
                    channel_weight = np.sqrt(betas[k, self.G + l] + 1e-12)
                    distance_weight = 1.0 / (1 + distance / 180)
                    critical_boost = 2.5 if rates[k] == min_rate else 1.5
                    total_weight = urgency_weight * channel_weight * distance_weight * critical_boost
                    force_magnitude = self.force_params['K_min_rate'] * total_weight
                    force_total += force_magnitude * unit_direction
            
            forces[l, :2] = force_total[:2]
        
        return forces

    def update_positions_with_energy(self, UAV_pos: np.ndarray, forces: np.ndarray, 
                                    iteration: int, current_min_rate: float) -> Tuple[np.ndarray, float, Dict]:
        """
        带能量约束的位置更新（核心改进！）
        
        返回:
            (new_UAV_pos, movement, energy_info): 新位置、总移动距离、能量信息
        """
        new_UAV_pos = UAV_pos.copy()
        
        # 1. 基础步长计算
        base_step = self.step_size
        
        # 迭代阶段调整
        if iteration < 20:
            stage_factor = 1.2
        elif iteration < 50:
            stage_factor = 1.0
        elif iteration < 80:
            stage_factor = 0.7
        else:
            stage_factor = 0.4
        
        # 性能反馈调整
        if hasattr(self, 'last_min_rate'):
            if current_min_rate > self.last_min_rate * 1.02:
                performance_factor = 1.1
            elif current_min_rate < self.last_min_rate * 0.98:
                performance_factor = 0.8
            else:
                performance_factor = 1.0
        else:
            performance_factor = 1.0
        
        adaptive_step = base_step * stage_factor * performance_factor
        adaptive_step = np.clip(adaptive_step, 3, 40)
        
        # 2. 力的归一化
        force_norms = np.linalg.norm(forces[:, :2], axis=1)
        max_force = np.max(force_norms) if len(force_norms) > 0 and np.max(force_norms) > 0 else 1
        normalized_forces = forces[:, :2] / max_force
        
        # 3. 能量感知的步长调整（逐UAV计算）
        total_energy_consumed = 0
        total_movement = 0
        energy_info = {
            'consumed': np.zeros(self.L),
            'remaining': np.zeros(self.L),
            'step_factors': np.zeros(self.L)
        }
        
        for l in range(self.L):
            # 计算理想位移
            ideal_displacement = adaptive_step * normalized_forces[l]
            
            # === 能量感知步长调整 ===
            energy_aware_step, energy_consumed = self.compute_energy_aware_step_size(
                l, ideal_displacement, adaptive_step, iteration)
            
            # === 收益-成本评估 ===（仅在后期且步长极小时才考虑跳过）
            # 如果步长太小（被能量约束限制），评估是否值得移动
            if iteration > 35 and energy_aware_step > 0 and energy_aware_step < adaptive_step * 0.2:
                # 步长被严重削减，检查收益
                force_norm_l = np.linalg.norm(forces[l, :2])
                if force_norm_l < 30:  # 力很小，说明收益有限
                    # 不值得移动，跳过此UAV
                    energy_info['consumed'][l] = 0
                    energy_info['remaining'][l] = self.remaining_energy[l]
                    energy_info['step_factors'][l] = 0
                    continue
            
            # 计算实际位移
            force_norm_l = np.linalg.norm(forces[l, :2])
            if force_norm_l > 0:
                actual_displacement = energy_aware_step * normalized_forces[l]
            else:
                actual_displacement = np.zeros(2)
            
            # 添加随机扰动（前期探索，后期减弱）
            if iteration < 40:  # 前40次迭代都保持一定扰动
                perturbation_strength = max(1, 10 * (1 - iteration / self.max_iterations))
                random_perturbation = np.random.normal(0, perturbation_strength, 2)
                actual_displacement += random_perturbation
            
            # 更新位置
            new_UAV_pos[l, :2] += actual_displacement
            
            # 边界约束
            new_UAV_pos[l, 0] = np.clip(new_UAV_pos[l, 0], 50, self.square_length-50)
            new_UAV_pos[l, 1] = np.clip(new_UAV_pos[l, 1], 50, self.square_length-50)
            
            # 更新能量
            self.remaining_energy[l] -= energy_consumed
            self.remaining_energy[l] = max(0, self.remaining_energy[l])
            
            # 记录信息
            movement_l = np.linalg.norm(actual_displacement)
            total_movement += movement_l
            total_energy_consumed += energy_consumed
            energy_info['consumed'][l] = energy_consumed
            energy_info['remaining'][l] = self.remaining_energy[l]
            energy_info['step_factors'][l] = energy_aware_step / adaptive_step
        
        self.last_min_rate = current_min_rate
        
        return new_UAV_pos, total_movement, energy_info

    def compute_multi_objective_fitness(self, min_rate: float, total_energy_consumed: float) -> float:
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
        # 权重参数
        w_rate = self.config.get('w_min_rate_objective', 0.7)
        w_energy = self.config.get('w_energy_objective', 0.3)
        
        # 参数设定
        MAX_EXPECTED_RATE = 50.0  # Mbps, 预期最大速率
        MAX_EXPECTED_ENERGY = 20000.0  # kJ, 预期最大能耗
        
        # 1. 性能得分: 最小速率越高越好，归一化到[0,100]
        performance_score = min(min_rate / MAX_EXPECTED_RATE, 1.0) * 100.0
        
        # 2. 能效得分: 能耗越低越好，归一化到[0,100]
        energy_kj = total_energy_consumed / 1000.0  # 转换为kJ
        normalized_energy = min(energy_kj / MAX_EXPECTED_ENERGY, 1.0)
        efficiency_score = (1.0 - normalized_energy) * 100.0
        
        # 3. 综合性能指标 (CPI)
        cpi = w_rate * performance_score + w_energy * efficiency_score
        
        return cpi
    
    def optimize(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray,
                UAV_pos: np.ndarray) -> Dict:
        """执行能量感知的BVF优化（多目标版本）"""
        print("\n" + "="*70)
        print("多目标优化: 最小速率 + 能耗 (Energy-Aware BVF)")
        print("="*70)
        
        history = {
            'iterations': [], 'sum_rates': [], 'min_rates': [], 'movements': [],
            'UAV_positions': [], 'force_magnitudes': [], 'restarts': [],
            'energy_consumed': [], 'remaining_energy': [], 'fitness': []
        }
        
        current_UAV_pos = UAV_pos.copy()
        best_fitness = -np.inf
        best_min_rate = -np.inf
        best_sum_rate = -np.inf
        best_UAV_pos = None
        best_iteration = 0
        no_improvement_count = 0
        restart_count = 0
        
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            # 合并所有AP位置
            all_AP_pos = np.vstack([ground_AP_pos, current_UAV_pos])
            
            # 计算性能
            _, _, betas = self.compute_channel_model(UE_pos, all_AP_pos)
            mask = self.compute_AP_selection_mask(betas)
            rates, sum_rate = self.compute_user_rates(UE_pos, all_AP_pos, mask)
            min_rate = rates.min()
            
            # 更新最佳结果
            improved = False
            if min_rate > best_min_rate:
                best_min_rate = min_rate
                best_sum_rate = sum_rate
                best_UAV_pos = current_UAV_pos.copy()
                best_iteration = iteration
                improved = True
                no_improvement_count = 0
            elif min_rate == best_min_rate and sum_rate > best_sum_rate:
                best_sum_rate = sum_rate
                best_UAV_pos = current_UAV_pos.copy()
                best_iteration = iteration
                improved = True
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # 计算虚拟力
            forces = self.compute_balanced_virtual_forces(UE_pos, ground_AP_pos, 
                                                        current_UAV_pos, rates, mask, betas)
            force_magnitude = np.mean(np.linalg.norm(forces[:, :2], axis=1))
            
            # === 能量感知的位置更新 ===
            new_UAV_pos, movement, energy_info = self.update_positions_with_energy(
                current_UAV_pos, forces, iteration, min_rate)
            
            # 记录历史
            history['iterations'].append(iteration)
            history['sum_rates'].append(sum_rate)
            history['min_rates'].append(min_rate)
            history['movements'].append(movement)
            history['force_magnitudes'].append(force_magnitude)
            history['UAV_positions'].append(current_UAV_pos.copy())
            history['energy_consumed'].append(energy_info['consumed'].sum())
            history['remaining_energy'].append(energy_info['remaining'].copy())
            
            # 输出进度
            if iteration % 10 == 0:
                avg_energy = energy_info['remaining'].mean()
                energy_used_pct = (1 - avg_energy / self.initial_energy) * 100
                active_uavs = np.sum(energy_info['consumed'] > 0)
                print(f"Iter {iteration}: Sum={sum_rate:.2f}Mbps, Min={min_rate:.4f}Mbps, "
                      f"Move={movement:.2f}m, ActiveUAVs={active_uavs}/{self.L}, "
                      f"Energy={avg_energy/3600:.1f}Wh ({energy_used_pct:.1f}% used)")
            
            # 能量耗尽检查
            if np.any(self.remaining_energy < self.initial_energy * 0.05):
                low_energy_uavs = np.where(self.remaining_energy < self.initial_energy * 0.05)[0]
                print(f"\n⚠️  警告: UAV {low_energy_uavs} 能量不足5%，建议停止优化")
            
            # 早停机制 - 平衡性能和能量（优先保证性能）
            # 1. 性能长时间不再改善
            if no_improvement_count >= 20:  # 给予充足的优化机会
                print(f"\n✅ 早停: 连续{no_improvement_count}次迭代无改善 (Iter {iteration})")
                break
            
            # 2. 收敛检查（更严格的条件）
            if iteration > 35:  # 给予足够的迭代次数
                if movement < self.convergence_threshold * 0.5 and force_magnitude < 50:
                    print(f"\n✅ 收敛: 移动和力都很小 (Iter {iteration})")
                    break
            
            # 3. 能量预算耗尽（只在真正耗尽时停止）
            avg_energy_ratio = self.remaining_energy.mean() / self.initial_energy
            if avg_energy_ratio < 0.05:  # 平均剩余能量低于5%才停止
                print(f"\n⚠️  早停: 能量预算不足 ({avg_energy_ratio:.1%}) (Iter {iteration})")
                break
            
            # 4. 移动太小且性能已经很好（确保性能达标再考虑早停）
            if iteration > 25 and movement < 3.0 and min_rate > best_min_rate * 0.95:
                consecutive_small_moves = sum(1 for m in history['movements'][-8:] if m < 3.0)
                if consecutive_small_moves >= 8:  # 更多次数确认收敛
                    print(f"\n✅ 早停: 连续多次移动过小且性能良好 (Iter {iteration})")
                    break
            
            current_UAV_pos = new_UAV_pos
        
        optimization_time = time.time() - start_time
        
        # 最终评估
        final_all_AP_pos = np.vstack([ground_AP_pos, best_UAV_pos])
        _, _, final_betas = self.compute_channel_model(UE_pos, final_all_AP_pos)
        final_mask = self.compute_AP_selection_mask(final_betas)
        final_rates, final_sum_rate = self.compute_user_rates(UE_pos, final_all_AP_pos, final_mask)
        final_min_rate = final_rates.min()
        
        results = {
            'optimized_UAV_pos': best_UAV_pos,
            'final_sum_rate': final_sum_rate,
            'final_min_rate': final_min_rate,
            'final_rates': final_rates,
            'optimization_time': optimization_time,
            'total_iterations': iteration + 1,
            'best_iteration': best_iteration,
            'history': history,
            'final_mask': final_mask,
            'improvement_achieved': final_min_rate > 0.1,
            'restart_count': restart_count,
            'remaining_energy': self.remaining_energy.copy(),
            'total_energy_consumed': self.initial_energy * self.L - self.remaining_energy.sum()
        }
        
        # 计算能量统计
        total_energy_budget = self.initial_energy * self.L
        total_energy_used = results['total_energy_consumed']
        energy_efficiency = (total_energy_used / 3600) / self.L  # Wh per UAV
        
        print(f"\n" + "="*70)
        print("🎉 能量感知BVF优化完成!")
        print("="*70)
        print(f"✨ 最佳性能出现在第 {best_iteration} 次迭代")
        print(f"📊 最终总速率: {final_sum_rate:.2f} Mbps")
        print(f"🎯 最终最小速率: {final_min_rate:.4f} Mbps")
        print(f"⏱️  优化时间: {optimization_time:.2f} 秒")
        print(f"🔄 实际迭代次数: {iteration + 1}/{self.max_iterations}")
        print(f"\n⚡ 能量统计:")
        print(f"  总能量预算: {total_energy_budget/3600:.2f} Wh")
        print(f"  总能量消耗: {total_energy_used/3600:.2f} Wh ({total_energy_used/total_energy_budget*100:.1f}%)")
        print(f"  平均每UAV消耗: {energy_efficiency:.2f} Wh")
        print(f"  平均剩余能量: {self.remaining_energy.mean()/3600:.2f} Wh")
        
        return results


def create_energy_aware_config() -> Dict:
    """创建能量感知BVF配置"""
    return {
        # 基本参数
        'square_length': 1000,
        'num_UE': 60,
        'num_UAV': 9,
        'num_ground_AP': 4,
        'M': 4,
        
        # 高度设置
        'UE_height': 1.65,
        'ground_AP_height': 15.0,
        'UAV_height': 50.0,
        
        # 虚拟力参数
        'K_min_rate': 6e4,
        'K_universal': 3e4,
        'K_cooperation': 2e4,
        'K_separation': 1.5e4,
        'K_boundary': 2.5e4,
        
        # 优化参数
        'step_size': 25,
        'max_iterations': 100,
        'convergence_threshold': 1e-3,
        'num_serving_APs': 3,
        'nbrOfRealizations': 50,
        
        # 信道参数
        'alpha': 3.67,
        'constant_term': -30.5,
        'B': 20e6,
        'Pmax': 1000,
        'noise_figure': 7,
        'distance_vertical': 150,
        'tau_p': 60,
        'tau_c': 200,
        'ASD_deg': 10,
        
        # === 能量模型参数 ===
        'rho_air': 1.225,        # 空气密度 kg/m³
        'A_rotor': 0.503,        # 旋翼面积 m²
        'm_uav': 2.0,            # UAV质量 kg
        'delta_blade': 0.012,    # 叶片剖面阻力系数
        'Omega': 300,            # 旋翼角速度 rad/s
        'R_rotor': 0.4,          # 旋翼半径 m
        'k_inc': 0.1,            # 诱导功率修正因子
        'A_drag': 0.5,           # 机身阻力系数
        'initial_energy': 500000,  # 初始能量 500kJ (约139Wh)
        'min_energy_reserve': 0.2, # 保留20%能量
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  能量感知的平衡虚拟力优化器测试  ".center(70))
    print("="*70)
    
    # 创建优化器
    config = create_energy_aware_config()
    optimizer = EnergyAwareBVFOptimizer(config)
    
    # 初始化位置
    UE_pos, ground_AP_pos, UAV_pos = optimizer.initialize_positions()
    
    print(f"\n系统初始化:")
    print(f"  UE数量: {len(UE_pos)}")
    print(f"  地面AP数量: {len(ground_AP_pos)}")
    print(f"  UAV数量: {len(UAV_pos)}")
    
    # 执行优化
    results = optimizer.optimize(UE_pos, ground_AP_pos, UAV_pos)
    
    print(f"\n最终结果:")
    print(f"  总速率: {results['final_sum_rate']:.2f} Mbps")
    print(f"  最小速率: {results['final_min_rate']:.4f} Mbps")
    print(f"  优化时间: {results['optimization_time']:.2f} 秒")
    print(f"  能量消耗: {results['total_energy_consumed']/3600:.2f} Wh")
    
    # 绘制能量消耗图
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['history']['iterations'], 
            [e/3600 for e in results['history']['energy_consumed']], 
            'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Energy Consumed per Iteration (Wh)')
    plt.title('Energy Consumption Profile')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    remaining_avg = [e.mean()/3600 for e in results['history']['remaining_energy']]
    plt.plot(results['history']['iterations'], remaining_avg, 'r-', linewidth=2)
    plt.axhline(y=config['initial_energy']/3600 * 0.2, color='orange', 
               linestyle='--', label='20% Reserve')
    plt.xlabel('Iteration')
    plt.ylabel('Average Remaining Energy (Wh)')
    plt.title('Remaining Energy Profile')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('energy_aware_bvf_results.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ 能量分析图已保存: energy_aware_bvf_results.png")

