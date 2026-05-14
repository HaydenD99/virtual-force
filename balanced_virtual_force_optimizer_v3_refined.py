"""
平衡虚拟力优化器 - V3-Refined (精进版)
精进细节：
1. 物理层校准：修正垂直距离计算，区分地/空 AP 高度
2. 干扰感知：基于用户重叠度的动态分离力
3. 运动优化：引入动量更新，抑制受力振荡
"""

import numpy as np
from scipy import linalg as sl
import time
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import functionRlocalscattering
import SpectralEfficiencyDownlink

class BalancedVirtualForceOptimizerV3Refined:
    """平衡虚拟力优化器 - V3-Refined (物理细节精进版)"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_parameters()
        self.momentum = None # 用于存储动量
        
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
        
        self.alpha = self.config.get('alpha', 3.67)
        self.constant_term = self.config.get('constant_term', -30.5)
        self.sigma_sf = self.config.get('sigma_sf', 1)
        self.antenna_spacing = self.config.get('antenna_spacing', 0.5)
        self.ASD_deg = self.config.get('ASD_deg', 10)
        
        self.B = self.config.get('B', 20e6)
        self.p = self.config.get('p', 100)
        self.Pmax = self.config.get('Pmax', 1000)
        self.noise_figure = self.config.get('noise_figure', 7)
        
        self.tau_p = self.config.get('tau_p', self.K)
        self.tau_c = self.config.get('tau_c', 200)
        self.prelogFactor = (self.tau_c - self.tau_p) / self.tau_c
        
        # 权重设置 (基于 V3 最优值)
        self.force_params = {
            'w_min_rate': 0.4,
            'w_universal': 0.25,
            'w_cooperation': 0.15,
            'w_separation': 0.1,
            'w_boundary': 0.1,
            'K_min_rate': self.config.get('K_min_rate', 6e4),
            'K_universal': self.config.get('K_universal', 3e4),
            'K_cooperation': self.config.get('K_cooperation', 2e4),
            'K_separation': self.config.get('K_separation', 1.5e4),
            'K_boundary': self.config.get('K_boundary', 2.5e4),
            'separation_distance': 150, # 基础排斥距离
            'boundary_margin': 60,
            'cooperation_distance': 200,
        }
        
        self.step_size = self.config.get('step_size', 25)
        self.max_iterations = self.config.get('max_iterations', 100)
        self.num_serving_APs = self.config.get('num_serving_APs', 3)
        self.nbrOfRealizations = self.config.get('nbrOfRealizations', 50)
        
        self.noise_variance_dBm = -174 + 10*np.log10(self.B) + self.noise_figure
        self.eyeM = np.eye(self.M)
        self.reg_eye = 1e-6 * self.eyeM
        self.sqrt_p_tau = np.sqrt(self.p * self.tau_p)

    def compute_channel_model(self, UE_pos: np.ndarray, AP_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算信道模型 - 物理细节改进：使用真实垂直高度差"""
        L_total = len(AP_pos)
        
        # 1. 物理精度改进：使用 AP 的真实高度坐标
        # UE_pos[:, 2] 是 1.65, AP_pos[:, 2] 是 15 或 50
        diff_xy = UE_pos[:, None, :2] - AP_pos[None, :, :2]
        diff_z = UE_pos[:, None, 2] - AP_pos[None, :, 2]
        distances = np.sqrt(np.sum(diff_xy**2, axis=-1) + diff_z**2)
        
        angles = np.arctan2(diff_xy[..., 1], diff_xy[..., 0])
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
        H = np.zeros_like(CH, dtype=complex)
        
        for k in range(self.K):
            for l in range(L_total):
                corr_matrix = CorrR[:, :, k, l] + self.reg_eye
                try:
                    Rsqrt = np.linalg.cholesky(corr_matrix)
                    H[:, :, k, l] = Rsqrt @ CH[:, :, k, l]
                except:
                    H[:, :, k, l] = np.sqrt(np.abs(betas[k, l])) * CH[:, :, k, l]
        
        pilotIndex = np.random.permutation(self.K) % self.tau_p
        Np = np.sqrt(0.5) * (np.random.randn(self.M, self.nbrOfRealizations, L_total, self.tau_p) +
                            1j * np.random.randn(self.M, self.nbrOfRealizations, L_total, self.tau_p))
        Hhat = np.zeros_like(H)
        pilot_groups = [np.where(pilotIndex == t)[0] for t in range(self.tau_p)]
        
        for l in range(L_total):
            for indices in pilot_groups:
                if len(indices) == 0: continue
                yp = self.sqrt_p_tau * np.sum(H[:, :, indices, l], axis=2) + Np[:, :, l, 0]
                PsiInv = self.p * self.tau_p * np.sum(CorrR[:, :, indices, l], axis=2) + self.eyeM
                PsiInvInv = np.linalg.inv(PsiInv)
                for k in indices:
                    Hhat[:, :, k, l] = self.sqrt_p_tau * (CorrR[:, :, k, l] @ PsiInvInv) @ yp
        return H, Hhat, betas

    def compute_AP_selection_mask(self, betas: np.ndarray) -> np.ndarray:
        top_AP_indices = np.argpartition(betas, -self.num_serving_APs, axis=1)[:, -self.num_serving_APs:]
        mask = np.zeros((self.K, betas.shape[1]), dtype=bool)
        for k in range(self.K): mask[k, top_AP_indices[k]] = True
        return mask

    def compute_user_rates(self, UE_pos: np.ndarray, AP_pos: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
        H, Hhat, betas = self.compute_channel_model(UE_pos, AP_pos)
        Hhat_uc = Hhat * mask[np.newaxis, np.newaxis, :, :]
        num_served_per_AP = mask.sum(axis=0)
        rho = np.zeros((self.K, len(AP_pos)))
        for l in range(len(AP_pos)):
            if num_served_per_AP[l] > 0: rho[mask[:, l], l] = self.Pmax / num_served_per_AP[l]
        gamma = np.sqrt(rho)
        
        M, N, K, L = H.shape
        w_MR = Hhat_uc / (np.linalg.norm(Hhat_uc, axis=0, keepdims=True) + 1e-12)
        a_MR = np.abs(np.einsum('mnkl,mnkl->lk', np.conj(H), w_MR) / N)
        interf_MR = np.einsum('mnkl,mnil->kiln', np.conj(H), w_MR).mean(axis=-1)
        B_MR = np.zeros((L, L, K, K))
        for k in range(K):
            for i in range(K): B_MR[:, :, k, i] = np.outer(interf_MR[k, i, :], interf_MR[k, i, :].conj()).real
        for l in range(L): B_MR[l, l, :, :] = np.abs(interf_MR[:, :, l]) ** 2
        
        SE_MR = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(a_MR, B_MR, self.B, gamma, self.Pmax)
        rates = SE_MR * self.prelogFactor / 1e6
        return rates, np.sum(rates)

    def optimize(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray, UAV_pos: np.ndarray) -> Dict:
        """执行精进版 V3 优化"""
        print("🚀 开始平衡虚拟力优化 V3-Refined (Precision & Momentum)...")
        current_UAV_pos = UAV_pos.copy()
        self.momentum = np.zeros((self.L, 2))
        
        best_min_rate = -np.inf
        best_UAV_pos = UAV_pos.copy()
        best_rates = None
        best_sum_rate = -np.inf
        
        history = {'min_rates': []}
        
        for iteration in range(self.max_iterations):
            all_AP_pos = np.vstack([ground_AP_pos, current_UAV_pos])
            _, _, betas = self.compute_channel_model(UE_pos, all_AP_pos)
            mask = self.compute_AP_selection_mask(betas)
            rates, sum_rate = self.compute_user_rates(UE_pos, all_AP_pos, mask)
            min_rate = rates.min()
            
            if min_rate > best_min_rate:
                best_min_rate = min_rate
                best_UAV_pos = current_UAV_pos.copy()
                best_rates = rates.copy()
                best_sum_rate = sum_rate
            
            # 计算虚拟力
            forces = self._compute_refined_forces(UE_pos, ground_AP_pos, current_UAV_pos, rates, mask, betas)
            
            # 更新位置：细节精进：引入动量项
            current_UAV_pos = self._update_positions_momentum(current_UAV_pos, forces, iteration)
            
            history['min_rates'].append(min_rate)
            if iteration % 10 == 0:
                print(f"Iter {iteration:2d} | MinRate: {min_rate:6.4f} | Best: {best_min_rate:6.4f}")

        return {
            'optimized_UAV_pos': best_UAV_pos,
            'final_min_rate': best_min_rate,
            'final_sum_rate': best_sum_rate,
            'final_rates': best_rates,
            'history': history
        }

    def _compute_refined_forces(self, UE_pos, ground_AP_pos, UAV_pos, rates, mask, betas):
        forces = np.zeros((self.L, 3))
        w = self.force_params
        
        # 1. 最小速率力 (同 V3)
        forces += w['w_min_rate'] * self._calc_min_rate_force(UE_pos, UAV_pos, rates, mask[:, self.G:])
        # 2. 全局引力 (同 V3)
        forces += w['w_universal'] * self._calc_univ_force(UE_pos, UAV_pos)
        # 3. 协作力 (同 V3)
        forces += w['w_cooperation'] * self._calc_coop_force(ground_AP_pos, UAV_pos)
        # 4. 细节精进：干扰感知分离力
        forces += w['w_separation'] * self._calc_interference_aware_sep_force(UAV_pos, mask[:, self.G:])
        # 5. 边界力 (同 V3)
        forces += w['w_boundary'] * self._calc_boundary_force(UAV_pos)
        
        return forces

    def _calc_interference_aware_sep_force(self, UAV_pos, mask_uav):
        """细节精进：如果两个无人机服务的用户重叠度高，排斥力增强"""
        forces = np.zeros((self.L, 3))
        for i in range(self.L):
            for j in range(self.L):
                if i == j: continue
                direction = UAV_pos[i] - UAV_pos[j]
                dist = np.linalg.norm(direction[:2]) + 1e-6
                
                # 计算用户重叠度 (Jaccard Similarity 思想)
                users_i = mask_uav[:, i]
                users_j = mask_uav[:, j]
                overlap = np.sum(users_i & users_j)
                
                # 动态排斥距离：重叠越多，排斥越远
                dynamic_sep_dist = self.force_params['separation_distance'] * (1.0 + overlap / self.num_serving_APs)
                
                if dist < dynamic_sep_dist:
                    force_mag = self.force_params['K_separation'] * (dynamic_sep_dist - dist) / dynamic_sep_dist
                    forces[i, :2] += force_mag * (direction[:2] / dist)
        return forces

    def _update_positions_momentum(self, UAV_pos, forces, iteration):
        """细节精进：引入动量机制，平滑运动轨迹"""
        force_norms = np.linalg.norm(forces[:, :2], axis=1)
        max_f = np.max(force_norms) if np.max(force_norms) > 0 else 1
        
        # 当前位移增量
        adaptive_step = self.step_size * (0.5 + 0.5 * (1 - iteration / self.max_iterations))
        v_current = adaptive_step * (forces[:, :2] / max_f)
        
        # 动量更新：v = beta * v_old + (1-beta) * v_new
        beta = 0.3 # 30% 惯性
        self.momentum = beta * self.momentum + (1 - beta) * v_current
        
        new_pos = UAV_pos.copy()
        new_pos[:, :2] += self.momentum
        
        # 加入微小扰动跳出马鞍点
        jitter = np.random.normal(0, 1.5 * (1 - iteration / self.max_iterations), (self.L, 2))
        new_pos[:, :2] += jitter
        
        new_pos[:, 0] = np.clip(new_pos[:, 0], 50, self.square_length - 50)
        new_pos[:, 1] = np.clip(new_pos[:, 1], 50, self.square_length - 50)
        return new_pos

    # 其余基础力计算函数
    def _calc_min_rate_force(self, UE_pos, UAV_pos, rates, mask_uav):
        forces = np.zeros((self.L, 3))
        min_r = rates.min()
        critical = np.where(rates <= min_r * 1.15)[0]
        for l in range(self.L):
            for k in critical:
                if mask_uav[k, l]:
                    direction = UE_pos[k] - UAV_pos[l]
                    dist = np.linalg.norm(direction) + 1e-6
                    forces[l, :2] += (self.force_params['K_min_rate'] / (dist + 5)) * (direction[:2] / dist)
        return forces

    def _calc_univ_force(self, UE_pos, UAV_pos):
        forces = np.zeros((self.L, 3))
        for l in range(self.L):
            for k in range(self.K):
                direction = UE_pos[k] - UAV_pos[l]
                dist = np.linalg.norm(direction) + 1e-6
                forces[l, :2] += (self.force_params['K_universal'] / (dist + 50)) * (direction[:2] / dist)
        return forces

    def _calc_coop_force(self, ground_AP_pos, UAV_pos):
        forces = np.zeros((self.L, 3))
        for l in range(self.L):
            for g in range(self.G):
                direction = ground_AP_pos[g] - UAV_pos[l]
                dist = np.linalg.norm(direction[:2]) + 1e-6
                if dist > self.force_params['cooperation_distance']:
                    forces[l, :2] += self.force_params['K_cooperation'] * (direction[:2] / dist)
        return forces

    def _calc_boundary_force(self, UAV_pos):
        forces = np.zeros((self.L, 3))
        margin = self.force_params['boundary_margin']
        for l in range(self.L):
            x, y = UAV_pos[l, 0], UAV_pos[l, 1]
            if x < margin: forces[l, 0] += self.force_params['K_boundary']
            elif x > self.square_length - margin: forces[l, 0] -= self.force_params['K_boundary']
            if y < margin: forces[l, 1] += self.force_params['K_boundary']
            elif y > self.square_length - margin: forces[l, 1] -= self.force_params['K_boundary']
        return forces
