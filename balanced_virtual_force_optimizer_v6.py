"""
1.  5 类力框架
2. 通信感知思想（urgency, interference sensitivity）
3. 健性机制
4. 改进距离衰减函数和权重配置
"""

import numpy as np
from scipy import linalg as sl
import time
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import functionRlocalscattering
import SpectralEfficiencyDownlink


class BalancedVirtualForceOptimizerV6:
    """平衡虚拟力优化器 - V6 (科学力场终极版)"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_parameters()
        
    def setup_parameters(self):
        """设置系统参数"""
        self.square_length = self.config.get('square_length', 1000)
        self.K = self.config.get('num_UE', 60)
        self.L = self.config.get('num_UAV', 9)
        self.G = self.config.get('num_ground_AP', 4)
        self.M = self.config.get('M', 4)
        
        self.heights = {
            'UE': self.config.get('UE_height', 1.65),
            'ground_AP': self.config.get('ground_AP_height', 15.0),
            'UAV': self.config.get('UAV_height', 50.0)
        }
        
        # 信道参数
        self.alpha = self.config.get('alpha', 3.67)
        self.constant_term = self.config.get('constant_term', -30.5)
        self.B = self.config.get('B', 20e6)
        self.Pmax = self.config.get('Pmax', 1000)
        self.noise_figure = self.config.get('noise_figure', 7)
        self.tau_p = self.config.get('tau_p', self.K)
        self.tau_c = self.config.get('tau_c', 200)
        self.prelogFactor = (self.tau_c - self.tau_p) / self.tau_c
        
        # === V6 力场参数（恢复 V3 框架 + 优化权重）===
        self.force_params = {
            # 力的强度系数
            'K_critical': 5e4,      # 关键用户引力（V3: K_min_rate）
            'K_universal': 2.5e4,   # 普适引力（V3 保留）
            'K_cooperation': 2e4,   # 地空协作（V3 保留，V5 丢失）
            'K_separation': 1.8e4,  # 分离力
            'K_boundary': 2.5e4,    # 边界力
            'K_interference': 1.5e4, # 干扰抑制（V5 引入）
            
            # 权重分配（总和 = 1.0）
            'w_critical': 0.35,      # 关键用户（核心）
            'w_interference': 0.20,  # 干扰抑制（V5 特色）
            'w_universal': 0.18,     # 普适引力
            'w_cooperation': 0.12,   # 地空协作
            'w_separation': 0.08,    # UAV 间分离
            'w_boundary': 0.07,      # 边界约束
            
            # 几何参数
            'critical_threshold': 1.15,  # 关键用户阈值（V3: min_rate * 1.15）
            'separation_distance': 130,
            'boundary_margin': 60,
            'cooperation_distance': 200,
            'distance_decay_factor': 180,  # V3 风格的距离衰减
        }
        
        # 优化参数
        self.step_size = self.config.get('step_size', 26)
        self.max_iterations = self.config.get('max_iterations', 100)
        self.num_serving_APs = self.config.get('num_serving_APs', 3)
        self.nbrOfRealizations = self.config.get('nbrOfRealizations', 50)
        
        # 稳健性参数
        self.restart_threshold = 25
        self.max_restarts = 2
        self.perturbation_strength = 75
        
        # 预计算常量
        self.noise_variance_dBm = -174 + 10*np.log10(self.B) + self.noise_figure
        self.eyeM = np.eye(self.M)
        self.reg_eye = 1e-6 * self.eyeM
        self.sqrt_p_tau = np.sqrt(100 * self.tau_p)

    # ========== 信道模型 ==========
    def compute_channel_model(self, UE_pos: np.ndarray, AP_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        L_total = len(AP_pos)
        diff_xy = UE_pos[:, None, :2] - AP_pos[None, :, :2]
        diff_z = UE_pos[:, None, 2] - AP_pos[None, :, 2]
        distances = np.sqrt(np.sum(diff_xy**2, axis=-1) + diff_z**2)
        angles = np.arctan2(diff_xy[..., 1], diff_xy[..., 0])
        channel_gain_dB = self.constant_term - self.alpha * 10 * np.log10(distances)
        betas = 10 ** ((channel_gain_dB - self.noise_variance_dBm) / 10)
        
        R = np.zeros((self.M, self.M, self.K, L_total), dtype=complex)
        for k in range(self.K):
            for l in range(L_total):
                R[:, :, k, l] = functionRlocalscattering.R(self.M, angles[k, l], 10)
        
        CH = np.sqrt(0.5) * (np.random.randn(self.M, self.nbrOfRealizations, self.K, L_total) +
                            1j * np.random.randn(self.M, self.nbrOfRealizations, self.K, L_total))
        H = np.zeros_like(CH, dtype=complex)
        CorrR = betas[None, None, :, :] * R
        
        for k in range(self.K):
            for l in range(L_total):
                corr = CorrR[:, :, k, l] + self.reg_eye
                try:
                    H[:, :, k, l] = np.linalg.cholesky(corr) @ CH[:, :, k, l]
                except:
                    H[:, :, k, l] = np.sqrt(np.abs(betas[k, l])) * CH[:, :, k, l]
        
        pilotIndex = np.random.permutation(self.K) % self.tau_p
        Np = np.sqrt(0.5) * (np.random.randn(self.M, self.nbrOfRealizations, L_total, self.tau_p) +
                            1j * np.random.randn(self.M, self.nbrOfRealizations, L_total, self.tau_p))
        Hhat = np.zeros_like(H)
        for l in range(L_total):
            for t in range(self.tau_p):
                indices = np.where(pilotIndex == t)[0]
                if len(indices) == 0:
                    continue
                yp = self.sqrt_p_tau * np.sum(H[:, :, indices, l], axis=2) + Np[:, :, l, t]
                PsiInv = np.linalg.inv(100 * self.tau_p * np.sum(CorrR[:, :, indices, l], axis=2) + self.eyeM)
                for k in indices:
                    Hhat[:, :, k, l] = self.sqrt_p_tau * (CorrR[:, :, k, l] @ PsiInv) @ yp
        return H, Hhat, betas

    def compute_AP_selection_mask(self, betas: np.ndarray) -> np.ndarray:
        top_AP_indices = np.argpartition(betas, -self.num_serving_APs, axis=1)[:, -self.num_serving_APs:]
        mask = np.zeros((self.K, betas.shape[1]), dtype=bool)
        for k in range(self.K):
            mask[k, top_AP_indices[k]] = True
        return mask

    def compute_user_rates(self, UE_pos: np.ndarray, AP_pos: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
        H, Hhat, betas = self.compute_channel_model(UE_pos, AP_pos)
        Hhat_uc = Hhat * mask[np.newaxis, np.newaxis, :, :]
        num_served_per_AP = mask.sum(axis=0)
        rho = np.zeros((self.K, len(AP_pos)))
        for l in range(len(AP_pos)):
            if num_served_per_AP[l] > 0:
                rho[mask[:, l], l] = self.Pmax / num_served_per_AP[l]
        gamma = np.sqrt(rho)
        
        M, N, K, L = H.shape
        w_MR = Hhat_uc / (np.linalg.norm(Hhat_uc, axis=0, keepdims=True) + 1e-12)
        a_MR = np.abs(np.einsum('mnkl,mnkl->lk', np.conj(H), w_MR) / N)
        interf_MR = np.einsum('mnkl,mnil->kiln', np.conj(H), w_MR).mean(axis=-1)
        B_MR = np.zeros((L, L, K, K))
        for k in range(K):
            for i in range(K):
                B_MR[:, :, k, i] = np.outer(interf_MR[k, i, :], interf_MR[k, i, :].conj()).real
        for l in range(L):
            B_MR[l, l, :, :] = np.abs(interf_MR[:, :, l]) ** 2
        
        SE_MR = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(a_MR, B_MR, self.B, gamma, self.Pmax)
        rates = SE_MR * self.prelogFactor / 1e6
        return rates, np.sum(rates)

    # ========== 虚拟力计算 ==========
    def compute_balanced_virtual_forces(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray,
                                        UAV_pos: np.ndarray, rates: np.ndarray,
                                        mask: np.ndarray, betas: np.ndarray) -> np.ndarray:
        forces = np.zeros((self.L, 3))
        w = self.force_params
        
        mask_uav = mask[:, self.G:]
        betas_uav = betas[:, self.G:]
        betas_ground = betas[:, :self.G]
        
        # 1. 关键用户引力
        f_critical = self._compute_critical_user_force(UE_pos, UAV_pos, rates, mask_uav, betas_uav)
        forces += w['w_critical'] * f_critical
        
        # 2. 干扰抑制斥力
        f_interference = self._compute_interference_repulsion(UE_pos, UAV_pos, rates, mask_uav, betas_uav)
        forces += w['w_interference'] * f_interference
        
        # 3. 普适用户引力
        f_universal = self._compute_universal_attraction(UE_pos, UAV_pos, rates, mask_uav, betas_uav)
        forces += w['w_universal'] * f_universal
        
        # 4. 地空协作力
        f_cooperation = self._compute_ground_cooperation(ground_AP_pos, UAV_pos, betas_ground, betas_uav)
        forces += w['w_cooperation'] * f_cooperation
        
        # 5. 智能分离力
        f_separation = self._compute_smart_separation(UAV_pos, mask_uav)
        forces += w['w_separation'] * f_separation
        
        # 6. 自适应边界力
        f_boundary = self._compute_adaptive_boundary(UAV_pos)
        forces += w['w_boundary'] * f_boundary
        
        return forces

    def _compute_critical_user_force(self, UE_pos, UAV_pos, rates, mask_uav, betas_uav):
        """关键用户引力"""
        forces = np.zeros((self.L, 3))
        min_rate = rates.min()
        threshold = min_rate * self.force_params['critical_threshold']
        critical_users = np.where(rates <= threshold)[0]
        
        if len(critical_users) == 0:
            return forces
        
        # V5 的 urgency
        avg_rate = rates.mean()
        urgency = (avg_rate / (rates + 1e-3)) ** 1.5  # 稍微平滑一点
        urgency = urgency / urgency.max()
        
        decay = self.force_params['distance_decay_factor']
        
        for l in range(self.L):
            force_total = np.zeros(3)
            served = np.where(mask_uav[:, l])[0]
            
            # 对服务的关键用户施加引力
            for k in served:
                if k in critical_users:
                    direction = UE_pos[k] - UAV_pos[l]
                    dist = np.linalg.norm(direction[:2]) + 1e-6
                    unit_dir = direction / (np.linalg.norm(direction) + 1e-6)
                    
                    # 距离衰减
                    dist_weight = 1.0 / (1 + dist / decay)
                    # 信道权重
                    channel_weight = np.sqrt(betas_uav[k, l] + 1e-12)
                    # 最差用户 boost
                    critical_boost = 2.5 if rates[k] == min_rate else 1.5
                    
                    total_weight = urgency[k] * channel_weight * dist_weight * critical_boost
                    force_mag = self.force_params['K_critical'] * total_weight
                    force_total += force_mag * unit_dir
            
            # 如果没有服务关键用户，向最近的关键用户移动
            if len(served) == 0 or not any(k in critical_users for k in served):
                if len(critical_users) > 0:
                    dists_to_critical = [np.linalg.norm(UE_pos[k, :2] - UAV_pos[l, :2]) for k in critical_users]
                    nearest_idx = critical_users[np.argmin(dists_to_critical)]
                    direction = UE_pos[nearest_idx] - UAV_pos[l]
                    dist = np.linalg.norm(direction) + 1e-6
                    force_total += 0.5 * self.force_params['K_critical'] * (direction / dist)
            
            forces[l, :2] = force_total[:2]
        return forces

    def _compute_interference_repulsion(self, UE_pos, UAV_pos, rates, mask_uav, betas_uav):
        """干扰抑制斥力"""
        forces = np.zeros((self.L, 3))
        min_rate = rates.min()
        threshold = min_rate * self.force_params['critical_threshold']
        critical_users = np.where(rates <= threshold)[0]
        
        for l in range(self.L):
            force_total = np.zeros(3)
            unserved = np.where(~mask_uav[:, l])[0]
            
            for k in unserved:
                # 只对关键用户施加干扰斥力
                if k in critical_users:
                    direction = UAV_pos[l] - UE_pos[k]  # 推远
                    dist = np.linalg.norm(direction[:2]) + 1e-6
                    unit_dir = direction[:2] / dist
                    
                    # 干扰敏感度
                    served_aps = np.where(mask_uav[k, :])[0]
                    if len(served_aps) > 0:
                        avg_served_beta = np.mean(betas_uav[k, served_aps])
                        interference_sensitivity = betas_uav[k, l] / (avg_served_beta + 1e-12)
                    else:
                        interference_sensitivity = 1.0
                    
                    # 距离衰减
                    dist_weight = 1.0 / (1 + dist / 200)
                    
                    force_mag = self.force_params['K_interference'] * interference_sensitivity * dist_weight
                    force_total[:2] += force_mag * unit_dir
            
            forces[l, :2] = force_total[:2]
        return forces

    def _compute_universal_attraction(self, UE_pos, UAV_pos, rates, mask_uav, betas_uav):
        """普适用户引力"""
        forces = np.zeros((self.L, 3))
        rate_weights = 1.0 / (rates + 0.01)
        rate_weights = rate_weights / rate_weights.mean()
        decay = self.force_params['distance_decay_factor']
        
        for l in range(self.L):
            force_total = np.zeros(3)
            for k in range(self.K):
                direction = UE_pos[k] - UAV_pos[l]
                dist = np.linalg.norm(direction[:2]) + 1e-6
                unit_dir = direction / (np.linalg.norm(direction) + 1e-6)
            
                dist_weight = 1.0 / (1 + (dist / (decay * 1.4))**2)
                channel_weight = np.sqrt(betas_uav[k, l] + 1e-12)
                # 已服务用户权重降低
                service_weight = 0.3 if mask_uav[k, l] else 1.0
                
                total_weight = rate_weights[k] * channel_weight * dist_weight * service_weight
                force_mag = self.force_params['K_universal'] * total_weight
                force_total += force_mag * unit_dir
            
            forces[l, :2] = force_total[:2]
        return forces

    def _compute_ground_cooperation(self, ground_AP_pos, UAV_pos, betas_ground, betas_uav):
        """地空协作力"""
        forces = np.zeros((self.L, 3))
        ground_quality = betas_ground.mean(axis=0)
        max_ground_quality = ground_quality.max() + 1e-12
        optimal_dist = self.force_params['cooperation_distance']
        
        for l in range(self.L):
            force_total = np.zeros(3)
            uav_quality = betas_uav[:, l].mean()
            
            for g in range(self.G):
                direction = ground_AP_pos[g] - UAV_pos[l]
                dist = np.linalg.norm(direction[:2]) + 1e-6
                
                # 与地面 AP 的互补性
                g_quality = ground_quality[g] / max_ground_quality
                quality_diff = abs(uav_quality - ground_quality[g])
                complementarity = quality_diff / (uav_quality + ground_quality[g] + 1e-12)
                complementarity = np.clip(complementarity, 0.1, 1.0)
                
                # 最优距离调节
                distance_error = dist - optimal_dist * (0.8 + 0.4 * g_quality)
                
                if abs(distance_error) > 20:
                    force_direction = -np.sign(distance_error)  # 太近则推开，太远则拉近
                    force_strength = min(1.0, abs(distance_error) / optimal_dist)
                    force_mag = self.force_params['K_cooperation'] * g_quality * complementarity * force_strength
                    unit_dir = direction[:2] / dist
                    force_total[:2] += force_direction * force_mag * unit_dir
            
            forces[l, :2] = force_total[:2]
        return forces

    def _compute_smart_separation(self, UAV_pos, mask_uav):
        """智能分离力"""
        forces = np.zeros((self.L, 3))
        min_dist = self.force_params['separation_distance']
        
        for i in range(self.L):
            for j in range(self.L):
                if i != j:
                    direction = UAV_pos[i] - UAV_pos[j]
                    dist = np.linalg.norm(direction[:2]) + 1e-6
                    
                    # 服务重叠度
                    served_i = set(np.where(mask_uav[:, i])[0])
                    served_j = set(np.where(mask_uav[:, j])[0])
                    overlap = len(served_i & served_j)
                    
                    # 重叠越多，分离距离越大
                    dynamic_min_dist = min_dist + overlap * 25
                    
                    if dist < dynamic_min_dist:
                        unit_dir = direction[:2] / dist
                        dist_factor = (dynamic_min_dist - dist) / dynamic_min_dist
                        overlap_factor = 1.0 + overlap * 0.35
                        force_mag = self.force_params['K_separation'] * dist_factor * overlap_factor
                        forces[i, :2] += force_mag * unit_dir
        return forces

    def _compute_adaptive_boundary(self, UAV_pos):
        """自适应边界力"""
        forces = np.zeros((self.L, 3))
        margin = self.force_params['boundary_margin']
        K_boundary = self.force_params['K_boundary']
        
        for l in range(self.L):
            x, y = UAV_pos[l, 0], UAV_pos[l, 1]
            
            # 二次方递增
            if x < margin:
                factor = ((margin - x) / margin) ** 2
                forces[l, 0] += K_boundary * factor
            elif x > self.square_length - margin:
                factor = ((x - (self.square_length - margin)) / margin) ** 2
                forces[l, 0] -= K_boundary * factor
            
            if y < margin:
                factor = ((margin - y) / margin) ** 2
                forces[l, 1] += K_boundary * factor
            elif y > self.square_length - margin:
                factor = ((y - (self.square_length - margin)) / margin) ** 2
                forces[l, 1] -= K_boundary * factor
        return forces

    # ========== 更新策略==========
    def update_positions(self, UAV_pos, forces, iteration, current_min_rate):
        """位置更新"""
        new_UAV_pos = UAV_pos.copy()
        
        # 阶段性步长
        if iteration < 20:
            stage_factor = 1.2
        elif iteration < 50:
            stage_factor = 1.0
        elif iteration < 80:
            stage_factor = 0.7
        else:
            stage_factor = 0.4
        
        # 性能反馈
        if hasattr(self, 'last_min_rate'):
            if current_min_rate > self.last_min_rate * 1.02:
                perf_factor = 1.1
            elif current_min_rate < self.last_min_rate * 0.98:
                perf_factor = 0.8
            else:
                perf_factor = 1.0
        else:
            perf_factor = 1.0
        
        adaptive_step = self.step_size * stage_factor * perf_factor
        adaptive_step = np.clip(adaptive_step, 3, 38)
        
        #  归一化
        force_norms = np.linalg.norm(forces[:, :2], axis=1)
        max_force = np.max(force_norms) if np.max(force_norms) > 0 else 1
        normalized_forces = forces[:, :2] / max_force
        
        displacement = adaptive_step * normalized_forces
        
        # 周期性扰动
        perturbation_strength = max(2, 7 * (1 - iteration / self.max_iterations))
        if iteration % 10 == 0 and iteration > 0:
            perturbation_strength *= 1.3
        displacement += np.random.normal(0, perturbation_strength, (self.L, 2))
        
        new_UAV_pos[:, :2] += displacement
        new_UAV_pos[:, :2] = np.clip(new_UAV_pos[:, :2], 50, self.square_length - 50)
        
        self.last_min_rate = current_min_rate
        movement = np.sum(np.linalg.norm(displacement, axis=1))
        return new_UAV_pos, movement

    # ========== 主优化循环==========
    def optimize(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray, UAV_pos: np.ndarray) -> Dict:
        
        print("Start Optimizing...")
        current_UAV_pos = UAV_pos.copy()
        
        # 稳健性记忆变量
        best_min_rate = -np.inf
        best_sum_rate = -np.inf
        best_UAV_pos = UAV_pos.copy()
        best_rates = None
        best_iteration = 0
        no_improvement_count = 0
        restart_count = 0
        
        history = {'iterations': [], 'min_rates': [], 'sum_rates': [], 'movements': []}
        
        for iteration in range(self.max_iterations):
            # 1. 评估
            all_AP_pos = np.vstack([ground_AP_pos, current_UAV_pos])
            _, _, betas = self.compute_channel_model(UE_pos, all_AP_pos)
            mask = self.compute_AP_selection_mask(betas)
            rates, sum_rate = self.compute_user_rates(UE_pos, all_AP_pos, mask)
            min_rate = rates.min()
            
            # 2. 记忆最优
            if min_rate > best_min_rate:
                best_min_rate = min_rate
                best_sum_rate = sum_rate
                best_UAV_pos = current_UAV_pos.copy()
                best_rates = rates.copy()
                best_iteration = iteration
                no_improvement_count = 0
            elif abs(min_rate - best_min_rate) < 1e-4 and sum_rate > best_sum_rate:
                best_sum_rate = sum_rate
                best_UAV_pos = current_UAV_pos.copy()
                best_rates = rates.copy()
                best_iteration = iteration
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # 3. 智能重启
            if no_improvement_count >= self.restart_threshold and restart_count < self.max_restarts and iteration < self.max_iterations - 10:
                print(f" Renewing...")
                current_UAV_pos = best_UAV_pos.copy() + np.random.normal(0, self.perturbation_strength, (self.L, 3))
                current_UAV_pos[:, 2] = self.heights['UAV']
                current_UAV_pos[:, :2] = np.clip(current_UAV_pos[:, :2], 100, self.square_length - 100)
                no_improvement_count = 0
                restart_count += 1
                continue
            
            # 4. 计算力
            forces = self.compute_balanced_virtual_forces(UE_pos, ground_AP_pos, current_UAV_pos, rates, mask, betas)
            
            # 5. 更新位置
            current_UAV_pos, movement = self.update_positions(current_UAV_pos, forces, iteration, min_rate)
            
            history['iterations'].append(iteration)
            history['min_rates'].append(min_rate)
            history['sum_rates'].append(sum_rate)
            history['movements'].append(movement)
            
            if iteration % 10 == 0:
                print(f"Iter {iteration:2d} | Min: {min_rate:.4f} | Sum: {sum_rate:.1f} | BestMin: {best_min_rate:.4f}")
        
        return {
            'optimized_UAV_pos': best_UAV_pos,
            'final_min_rate': best_min_rate,
            'final_sum_rate': best_sum_rate,
            'final_rates': best_rates,
            'history': history,
            'best_iteration': best_iteration,
            'restart_count': restart_count
        }


def create_v6_config() -> Dict:
    return {
        'square_length': 1000,
        'num_UE': 60,
        'num_UAV': 9,
        'num_ground_AP': 4,
        'M': 4,
        'UE_height': 1.65,
        'ground_AP_height': 15.0,
        'UAV_height': 50.0,
        'step_size': 26,
        'max_iterations': 50,
        'num_serving_APs': 3,
        'nbrOfRealizations': 50,
        'alpha': 3.67,
        'constant_term': -30.5,
        'B': 20e6,
        'Pmax': 1000,
        'noise_figure': 7,
        'tau_p': 60,
        'tau_c': 200,
    }
