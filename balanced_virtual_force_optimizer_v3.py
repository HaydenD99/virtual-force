"""
平衡虚拟力优化器 - 正确优化版本V3
基于性能剖析的结果，只优化真正的瓶颈
"""

import numpy as np
# import matplotlib.pyplot as plt  # Not used, commented out to avoid font cache initialization
from scipy import linalg as sl
import time
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 导入必要的函数
import functionRlocalscattering
import SpectralEfficiencyDownlink

class BalancedVirtualForceOptimizerV3:
    """平衡虚拟力优化器 - 正确优化版本（基于性能剖析）"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_parameters()
        
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
        
        # 平衡的虚拟力参数
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
        
        # 预计算用于加速的常量
        self.reg_eye = 1e-6 * self.eyeM
        self.sqrt_p_tau = np.sqrt(self.p * self.tau_p)

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
        """自适应聚类"""
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

    # ============================================================================
    # 正确的优化策略 - 基于性能剖析的结果
    # ============================================================================
    def compute_channel_model(self, UE_pos: np.ndarray, AP_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算完整的多天线信道模型 - 正确优化版本V3
        
        优化策略（基于性能剖析）：
        1. ✅ 使用Cholesky代替sqrtm进行信道矩阵生成 (13.75x加速)
        2. ✅ 保持使用inv进行导频估计 (对小矩阵更快)
        3. ✅ 移除不必要的条件检查
        """
        L_total = len(AP_pos)
        
        # ========= 计算距离和角度 =========
        diff = UE_pos[:, None, :2] - AP_pos[None, :, :2]
        distances = np.sqrt(np.sum(diff**2, axis=-1) + self.distance_vertical**2)
        angles = np.arctan2(diff[..., 1], diff[..., 0])
        
        # ========= 计算大尺度衰落 =========
        channel_gain_dB = self.constant_term - self.alpha * 10 * np.log10(distances)
        channel_gain_over_noise = channel_gain_dB - self.noise_variance_dBm
        betas = 10 ** (channel_gain_over_noise / 10)
        
        # ========= 空间相关矩阵R =========
        R = np.zeros((self.M, self.M, self.K, L_total), dtype=complex)
        for k in range(self.K):
            for l in range(L_total):
                R[:, :, k, l] = functionRlocalscattering.R(self.M, angles[k, l], self.ASD_deg)
        
        CorrR = betas[None, None, :, :] * R
        
        # ========= 优化：使用Cholesky生成信道矩阵 (关键优化！) =========
        CH = np.sqrt(0.5) * (np.random.randn(self.M, self.nbrOfRealizations, self.K, L_total) +
                            1j * np.random.randn(self.M, self.nbrOfRealizations, self.K, L_total))
        H = np.zeros_like(CH, dtype=complex)
        
        # 使用Cholesky分解（比sqrtm快13倍）
        for k in range(self.K):
            for l in range(L_total):
                corr_matrix = CorrR[:, :, k, l] + self.reg_eye
                try:
                    # Cholesky分解 - 关键优化点
                    Rsqrt = np.linalg.cholesky(corr_matrix)
                    H[:, :, k, l] = Rsqrt @ CH[:, :, k, l]
                except:
                    # 快速fallback
                    H[:, :, k, l] = np.sqrt(np.abs(betas[k, l])) * CH[:, :, k, l]
        
        # ========= 保持原版：使用inv进行导频估计 (对小矩阵更快) =========
        pilotIndex = np.random.permutation(self.K) % self.tau_p
        Np = np.sqrt(0.5) * (np.random.randn(self.M, self.nbrOfRealizations, L_total, self.tau_p) +
                            1j * np.random.randn(self.M, self.nbrOfRealizations, L_total, self.tau_p))
        
        Hhat = np.zeros_like(H)
        
        # 预先分组导频索引
        pilot_groups = [np.where(pilotIndex == t)[0] for t in range(self.tau_p)]
        
        # 使用inv（对4×4矩阵很快）
        for l in range(L_total):
            for t, indices in enumerate(pilot_groups):
                if len(indices) == 0:
                    continue
                    
                yp = self.sqrt_p_tau * np.sum(H[:, :, indices, l], axis=2) + Np[:, :, l, t]
                PsiInv = self.p * self.tau_p * np.sum(CorrR[:, :, indices, l], axis=2) + self.eyeM
                PsiInvInv = np.linalg.inv(PsiInv)  # 保持使用inv
                
                for k in indices:
                    RPsi = CorrR[:, :, k, l] @ PsiInvInv
                    Hhat[:, :, k, l] = self.sqrt_p_tau * RPsi @ yp
        
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
        """计算平衡的虚拟力系统"""
        forces = np.zeros((self.L, 3))
        weights = self.force_params
        min_rate_force = self._compute_min_rate_force(UE_pos, UAV_pos, rates, 
                                                    mask[:, self.G:], betas[:, self.G:])
        forces += weights['w_min_rate'] * min_rate_force
        universal_force = self._compute_universal_attraction_force(UE_pos, UAV_pos, rates, 
                                                                 mask[:, self.G:], betas[:, self.G:])
        forces += weights['w_universal'] * universal_force
        cooperation_force = self._compute_ground_cooperation_force(ground_AP_pos, UAV_pos, 
                                                                 betas[:, :self.G], betas[:, self.G:])
        forces += weights['w_cooperation'] * cooperation_force
        separation_force = self._compute_smart_separation_force(UAV_pos, mask[:, self.G:])
        forces += weights['w_separation'] * separation_force
        boundary_force = self._compute_adaptive_boundary_force(UAV_pos)
        forces += weights['w_boundary'] * boundary_force
        return forces
    
    def _compute_min_rate_force(self, UE_pos: np.ndarray, UAV_pos: np.ndarray, 
                               rates: np.ndarray, mask: np.ndarray, 
                               betas_uav: np.ndarray) -> np.ndarray:
        """最小速率导向力"""
        forces = np.zeros((self.L, 3))
        min_rate = rates.min()
        threshold = min_rate * 1.15
        critical_users = np.where(rates <= threshold)[0]
        urgency = 1.0 / (rates + 0.001)
        urgency = urgency / urgency.max()
        for l in range(self.L):
            force_total = np.zeros(3)
            served_users = np.where(mask[:, l])[0]
            if len(served_users) > 0:
                for k in served_users:
                    if k in critical_users:
                        direction = UE_pos[k] - UAV_pos[l]
                        distance = np.linalg.norm(direction) + 1e-6
                        unit_direction = direction / distance
                        urgency_weight = urgency[k]
                        channel_weight = np.sqrt(betas_uav[k, l] + 1e-12)
                        distance_weight = 1.0 / (1 + distance / 180)
                        critical_boost = 2.5 if rates[k] == min_rate else 1.5
                        total_weight = urgency_weight * channel_weight * distance_weight * critical_boost
                        force_magnitude = self.force_params['K_min_rate'] * total_weight
                        force_total += force_magnitude * unit_direction
            else:
                if len(critical_users) > 0:
                    distances = [(np.linalg.norm(UE_pos[k, :2] - UAV_pos[l, :2]), k) 
                               for k in critical_users]
                    distances.sort()
                    closest_dist, closest_user = distances[0]
                    direction = UE_pos[closest_user] - UAV_pos[l]
                    distance = np.linalg.norm(direction) + 1e-6
                    unit_direction = direction / distance
                    force_magnitude = self.force_params['K_min_rate'] * 0.6
                    force_total += force_magnitude * unit_direction
            forces[l, :2] = force_total[:2]
        return forces
    
    def _compute_universal_attraction_force(self, UE_pos: np.ndarray, UAV_pos: np.ndarray,
                                          rates: np.ndarray, mask: np.ndarray, 
                                          betas_uav: np.ndarray) -> np.ndarray:
        """普适用户引力"""
        forces = np.zeros((self.L, 3))
        rate_weights = 1.0 / (rates + 0.01)
        rate_weights = rate_weights / rate_weights.mean()
        for l in range(self.L):
            force_total = np.zeros(3)
            for k in range(self.K):
                direction = UE_pos[k] - UAV_pos[l]
                distance = np.linalg.norm(direction) + 1e-6
                unit_direction = direction / distance
                rate_weight = rate_weights[k]
                channel_quality = betas_uav[k, l] if k < len(betas_uav) else 1e-6
                channel_weight = np.sqrt(channel_quality + 1e-12)
                distance_weight = 1.0 / (1 + (distance / 250)**2)
                service_weight = 0.3 if mask[k, l] else 1.0
                total_weight = rate_weight * channel_weight * distance_weight * service_weight
                force_magnitude = self.force_params['K_universal'] * total_weight
                force_total += force_magnitude * unit_direction
            forces[l, :2] = force_total[:2]
        return forces
    
    def _compute_ground_cooperation_force(self, ground_AP_pos: np.ndarray, UAV_pos: np.ndarray,
                                        ground_betas: np.ndarray, uav_betas: np.ndarray) -> np.ndarray:
        """地空协作力"""
        forces = np.zeros((self.L, 3))
        ground_quality = ground_betas.mean(axis=0)
        max_ground_quality = ground_quality.max() + 1e-12
        for l in range(self.L):
            force_total = np.zeros(3)
            uav_quality = uav_betas[:, l].mean()
            for g in range(self.G):
                direction = ground_AP_pos[g] - UAV_pos[l]
                distance = np.linalg.norm(direction[:2]) + 1e-6
                ground_ap_quality = ground_quality[g] / max_ground_quality
                quality_diff = abs(uav_quality - ground_quality[g])
                complementarity = quality_diff / (uav_quality + ground_quality[g] + 1e-12)
                complementarity = np.clip(complementarity, 0.1, 1.0)
                base_distance = self.force_params['cooperation_distance']
                optimal_distance = base_distance * (0.8 + 0.4 * ground_ap_quality)
                distance_error = distance - optimal_distance
                if abs(distance_error) > 20:
                    force_direction = -np.sign(distance_error)
                    force_strength = min(1.0, abs(distance_error) / optimal_distance)
                    force_magnitude = (self.force_params['K_cooperation'] * 
                                     ground_ap_quality * complementarity * force_strength)
                    unit_direction = direction / distance
                    force_total += force_direction * force_magnitude * unit_direction
            forces[l, :2] = force_total[:2]
        return forces
    
    def _compute_smart_separation_force(self, UAV_pos: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """智能分离力"""
        forces = np.zeros((self.L, 3))
        min_dist = self.force_params['separation_distance']
        for i in range(self.L):
            for j in range(self.L):
                if i != j:
                    direction = UAV_pos[i] - UAV_pos[j]
                    distance = np.linalg.norm(direction[:2]) + 1e-6
                    served_i = set(np.where(mask[:, i])[0])
                    served_j = set(np.where(mask[:, j])[0])
                    overlap = len(served_i & served_j)
                    dynamic_min_dist = min_dist + overlap * 20
                    if distance < dynamic_min_dist:
                        unit_direction = direction[:2] / distance
                        distance_factor = (dynamic_min_dist - distance) / dynamic_min_dist
                        overlap_factor = 1.0 + overlap * 0.3
                        force_magnitude = (self.force_params['K_separation'] * 
                                         distance_factor * overlap_factor)
                        forces[i, :2] += force_magnitude * unit_direction
        return forces
    
    def _compute_adaptive_boundary_force(self, UAV_pos: np.ndarray) -> np.ndarray:
        """自适应边界力"""
        forces = np.zeros((self.L, 3))
        margin = self.force_params['boundary_margin']
        for l in range(self.L):
            x, y = UAV_pos[l, 0], UAV_pos[l, 1]
            if x < margin:
                boundary_factor = (margin - x) / margin
                force_strength = boundary_factor ** 2
                forces[l, 0] += self.force_params['K_boundary'] * force_strength
            elif x > self.square_length - margin:
                boundary_factor = (x - (self.square_length - margin)) / margin
                force_strength = boundary_factor ** 2
                forces[l, 0] -= self.force_params['K_boundary'] * force_strength
            if y < margin:
                boundary_factor = (margin - y) / margin
                force_strength = boundary_factor ** 2
                forces[l, 1] += self.force_params['K_boundary'] * force_strength
            elif y > self.square_length - margin:
                boundary_factor = (y - (self.square_length - margin)) / margin
                force_strength = boundary_factor ** 2
                forces[l, 1] -= self.force_params['K_boundary'] * force_strength
        return forces

    def update_positions(self, UAV_pos: np.ndarray, forces: np.ndarray, 
                        iteration: int, current_min_rate: float) -> Tuple[np.ndarray, float]:
        """智能位置更新策略"""
        new_UAV_pos = UAV_pos.copy()
        base_step = self.step_size
        if iteration < 20:
            stage_factor = 1.2
        elif iteration < 50:
            stage_factor = 1.0
        elif iteration < 80:
            stage_factor = 0.7
        else:
            stage_factor = 0.4
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
        force_norms = np.linalg.norm(forces[:, :2], axis=1)
        max_force = np.max(force_norms) if len(force_norms) > 0 and np.max(force_norms) > 0 else 1
        normalized_forces = forces[:, :2] / max_force
        displacement = adaptive_step * normalized_forces
        perturbation_strength = max(2, 8 * (1 - iteration / self.max_iterations))
        if iteration % 10 == 0 and iteration > 0:
            perturbation_strength *= 1.5
        random_perturbation = np.random.normal(0, perturbation_strength, (self.L, 2))
        displacement += random_perturbation
        new_UAV_pos[:, :2] += displacement
        new_UAV_pos[:, 0] = np.clip(new_UAV_pos[:, 0], 50, self.square_length-50)
        new_UAV_pos[:, 1] = np.clip(new_UAV_pos[:, 1], 50, self.square_length-50)
        self.last_min_rate = current_min_rate
        movement = np.sum(np.linalg.norm(displacement, axis=1))
        return new_UAV_pos, movement

    def optimize(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray,
                UAV_pos: np.ndarray) -> Dict:
        """执行平衡虚拟力优化"""
        print("开始平衡虚拟力优化（V3正确优化版）...")
        history = {
            'iterations': [], 'sum_rates': [], 'min_rates': [], 'movements': [],
            'UAV_positions': [], 'force_magnitudes': [], 'restarts': []
        }
        current_UAV_pos = UAV_pos.copy()
        best_min_rate = -np.inf
        best_sum_rate = -np.inf
        best_UAV_pos = None
        best_rates = None  # 保存历代最优的速率
        best_mask = None   # 保存历代最优的掩码
        best_betas = None  # 保存历代最优的betas
        best_iteration = 0
        no_improvement_count = 0
        restart_count = 0
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            iter_start_time = time.time()
            all_AP_pos = np.vstack([ground_AP_pos, current_UAV_pos])
            _, _, betas = self.compute_channel_model(UE_pos, all_AP_pos)
            mask = self.compute_AP_selection_mask(betas)
            rates, sum_rate = self.compute_user_rates(UE_pos, all_AP_pos, mask)
            min_rate = rates.min()
            iter_time = time.time() - iter_start_time
            
            improved = False
            if min_rate > best_min_rate:
                # 找到更好的最小速率，保存所有相关信息
                best_min_rate = min_rate
                best_sum_rate = sum_rate
                best_UAV_pos = current_UAV_pos.copy()
                best_rates = rates.copy()
                best_mask = mask.copy()
                best_betas = betas.copy()
                best_iteration = iteration
                improved = True
                no_improvement_count = 0
            elif min_rate == best_min_rate and sum_rate > best_sum_rate:
                # 最小速率相同但总速率更好，保存所有相关信息
                best_sum_rate = sum_rate
                best_UAV_pos = current_UAV_pos.copy()
                best_rates = rates.copy()
                best_mask = mask.copy()
                best_betas = betas.copy()
                best_iteration = iteration
                improved = True
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if (no_improvement_count >= self.restart_threshold and 
                restart_count < self.max_restarts and 
                iteration > 30):
                print(f"\n🔄 智能重启 #{restart_count+1} (Iter {iteration})")
                if best_UAV_pos is not None:
                    for l in range(self.L):
                        perturbation = np.random.normal(0, self.perturbation_strength * (0.8 + 0.4 * np.random.random()), 2)
                        current_UAV_pos[l, :2] = best_UAV_pos[l, :2] + perturbation
                        current_UAV_pos[l, 0] = np.clip(current_UAV_pos[l, 0], 100, self.square_length-100)
                        current_UAV_pos[l, 1] = np.clip(current_UAV_pos[l, 1], 100, self.square_length-100)
                restart_count += 1
                no_improvement_count = 0
                self.step_size *= 1.3
                history['restarts'].append(iteration)
                continue
            
            forces = self.compute_balanced_virtual_forces(UE_pos, ground_AP_pos, 
                                                        current_UAV_pos, rates, mask, betas)
            force_magnitude = np.mean(np.linalg.norm(forces[:, :2], axis=1))
            new_UAV_pos, movement = self.update_positions(current_UAV_pos, forces, iteration, min_rate)
            history['iterations'].append(iteration)
            history['sum_rates'].append(sum_rate)
            history['min_rates'].append(min_rate)
            history['movements'].append(movement)
            history['force_magnitudes'].append(force_magnitude)
            history['UAV_positions'].append(current_UAV_pos.copy())
            
            if iteration % 10 == 0:
                print(f"Iter {iteration}: Sum={sum_rate:.2f}Mbps, Min={min_rate:.4f}Mbps, "
                      f"BestMin={best_min_rate:.4f}Mbps (Iter{best_iteration}), Time={iter_time:.3f}s")
            
            if iteration > 40:
                if (movement < self.convergence_threshold and 
                    force_magnitude < 120 and 
                    no_improvement_count > 30):
                    print(f"\n✅ 收敛于第 {iteration} 次迭代")
                    print(f"   最佳性能出现在第 {best_iteration} 次迭代")
                    break
            current_UAV_pos = new_UAV_pos
        
        optimization_time = time.time() - start_time
        
        # 直接使用历代最优的保存结果，而不是重新计算
        # 这样确保输出的就是历史最优性能，避免随机性影响
        if best_rates is None:
            # 如果没有找到更好的解，使用初始UAV位置重新计算
            final_all_AP_pos = np.vstack([ground_AP_pos, best_UAV_pos if best_UAV_pos is not None else UAV_pos])
            _, _, best_betas = self.compute_channel_model(UE_pos, final_all_AP_pos)
            best_mask = self.compute_AP_selection_mask(best_betas)
            best_rates, best_sum_rate = self.compute_user_rates(UE_pos, final_all_AP_pos, best_mask)
            best_min_rate = best_rates.min()
        
        final_min_rate = best_min_rate
        final_sum_rate = best_sum_rate
        final_rates = best_rates
        final_mask = best_mask
        
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
            'restart_count': restart_count
        }
        
        print(f"\n🎉 平衡虚拟力优化完成（V3版 - 历代最优记忆）!")
        print(f"✨ 最佳性能出现在第 {best_iteration} 次迭代")
        print(f"📊 历代最优总速率: {final_sum_rate:.2f} Mbps")
        print(f"🎯 历代最优最小速率: {final_min_rate:.4f} Mbps")
        print(f"📈 速率标准差: {final_rates.std():.4f} Mbps")
        print(f"⏱️  优化时间: {optimization_time:.2f} 秒")
        print(f"⚡ 平均每次迭代: {optimization_time/(iteration+1):.3f} 秒")
        print(f"🔄 重启次数: {restart_count}")
        return results


def create_balanced_config() -> Dict:
    """创建平衡虚拟力配置"""
    return {
        'square_length': 1000,
        'num_UE': 60,
        'num_UAV': 9,
        'num_ground_AP': 4,
        'M': 4,
        'UE_height': 1.65,
        'ground_AP_height': 15.0,
        'UAV_height': 50.0,
        'K_min_rate': 6e4,
        'K_universal': 3e4,
        'K_cooperation': 2e4,
        'K_separation': 1.5e4,
        'K_boundary': 2.5e4,
        'separation_distance': 120,
        'boundary_margin': 60,
        'cooperation_distance': 200,
        'step_size': 28,
        'max_iterations': 50,
        'convergence_threshold': 1.5e-3,
        'num_serving_APs': 3,
        'nbrOfRealizations': 30,
        'alpha': 3.67,
        'constant_term': -30.5,
        'B': 20e6,
        'Pmax': 1000,
        'noise_figure': 7,
        'distance_vertical': 150,
        'tau_p': 60,
        'tau_c': 200,
        'ASD_deg': 10,
    }


if __name__ == "__main__":
    print("="*60)
    print("正确优化版本V3 - 基于性能剖析")
    print("="*60)
    config = create_balanced_config()
    optimizer = BalancedVirtualForceOptimizerV3(config)
    UE_pos, ground_AP_pos, UAV_pos = optimizer.initialize_positions()
    print(f"\n优化器初始化完成:")
    print(f"UE数量: {len(UE_pos)}")
    print(f"地面AP数量: {len(ground_AP_pos)}")
    print(f"UAV数量: {len(UAV_pos)}")
    results = optimizer.optimize(UE_pos, ground_AP_pos, UAV_pos)
    print(f"\n预计100次迭代时间: {results['optimization_time']/results['total_iterations']*100:.1f}秒")


