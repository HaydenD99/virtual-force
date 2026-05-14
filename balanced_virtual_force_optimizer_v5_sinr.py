"""
平衡虚拟力优化器 - V5-Pro (通信感知力场 + 稳健性记忆增强版)
创新与稳健性结合：
1. 通信感知力场 (IACF)：信号增强引力 + 干扰抑制斥力
2. 历代最优记忆：确保不丢失优化过程中的最佳配置
3. 智能重启机制：陷入局部最优时自动回跳并扰动
4. 物理精度校准：基于真实垂直高度差
"""

import numpy as np
from scipy import linalg as sl
import time
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import functionRlocalscattering
import SpectralEfficiencyDownlink

class BalancedVirtualForceOptimizerV5:
    """平衡虚拟力优化器 - V5-Pro (终极版)"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_parameters()
        self.momentum = None
        
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
        self.B = self.config.get('B', 20e6)
        self.Pmax = self.config.get('Pmax', 1000)
        self.noise_figure = self.config.get('noise_figure', 7)
        self.tau_p = self.config.get('tau_p', self.K)
        self.tau_c = self.config.get('tau_c', 200)
        self.prelogFactor = (self.tau_c - self.tau_p) / self.tau_c
        
        # 力场参数
        self.force_params = {
            'w_signal': 0.55,      # 信号引力
            'w_interference': 0.25, # 干扰斥力
            'w_separation': 0.10,   # 间距
            'w_boundary': 0.10,     
            'sep_dist': 150,
            'margin': 60
        }
        
        # 优化与稳健性参数 (同 V3)
        self.step_size = self.config.get('step_size', 28)
        self.max_iterations = self.config.get('max_iterations', 100)
        self.num_serving_APs = self.config.get('num_serving_APs', 3)
        self.nbrOfRealizations = self.config.get('nbrOfRealizations', 50)
        self.restart_threshold = 25
        self.max_restarts = 2
        self.perturbation_strength = 70
        
        self.noise_variance_dBm = -174 + 10*np.log10(self.B) + self.noise_figure
        self.eyeM = np.eye(self.M)
        self.reg_eye = 1e-6 * self.eyeM
        self.sqrt_p_tau = np.sqrt(100 * self.tau_p)

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
                try: H[:, :, k, l] = np.linalg.cholesky(corr) @ CH[:, :, k, l]
                except: H[:, :, k, l] = np.sqrt(np.abs(betas[k, l])) * CH[:, :, k, l]
        
        pilotIndex = np.random.permutation(self.K) % self.tau_p
        Np = np.sqrt(0.5) * (np.random.randn(self.M, self.nbrOfRealizations, L_total, self.tau_p) +
                            1j * np.random.randn(self.M, self.nbrOfRealizations, L_total, self.tau_p))
        Hhat = np.zeros_like(H)
        for l in range(L_total):
            for t in range(self.tau_p):
                indices = np.where(pilotIndex == t)[0]
                if len(indices) == 0: continue
                yp = self.sqrt_p_tau * np.sum(H[:, :, indices, l], axis=2) + Np[:, :, l, t]
                PsiInv = np.linalg.inv(100 * self.tau_p * np.sum(CorrR[:, :, indices, l], axis=2) + self.eyeM)
                for k in indices: Hhat[:, :, k, l] = self.sqrt_p_tau * (CorrR[:, :, k, l] @ PsiInv) @ yp
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
        """执行 V5-Pro 优化"""
        print("🚀 开始平衡虚拟力优化 V5-Pro (Comm-Aware + Memory Robustness)...")
        current_UAV_pos = UAV_pos.copy()
        self.momentum = np.zeros((self.L, 2))
        
        # 稳健性记忆变量
        best_min_rate = -np.inf
        best_sum_rate = -np.inf
        best_UAV_pos = UAV_pos.copy()
        best_rates = None
        best_iteration = 0
        no_improvement_count = 0
        restart_count = 0
        
        history = {'min_rates': [], 'sum_rates': []}
        
        for iteration in range(self.max_iterations):
            # 1. 评估
            all_AP_pos = np.vstack([ground_AP_pos, current_UAV_pos])
            _, _, betas = self.compute_channel_model(UE_pos, all_AP_pos)
            mask = self.compute_AP_selection_mask(betas)
            rates, sum_rate = self.compute_user_rates(UE_pos, all_AP_pos, mask)
            min_rate = rates.min()
            
            # 2. 记忆最优逻辑 (Min Rate 优先，Sum Rate 其次)
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
            
            # 3. 智能重启逻辑
            if no_improvement_count >= self.restart_threshold and restart_count < self.max_restarts and iteration < self.max_iterations - 10:
                print(f"  🔄 触发智能重启 #{restart_count+1} (代 {iteration}) | 尝试跳出局部最优...")
                # 跳回最优并施加中等抖动
                current_UAV_pos = best_UAV_pos.copy() + np.random.normal(0, self.perturbation_strength, (self.L, 3))
                current_UAV_pos[:, 2] = self.heights['UAV']
                self.momentum *= 0 # 重置动量
                no_improvement_count = 0
                restart_count += 1
                continue

            # 4. 计算力
            forces = self._compute_comm_aware_forces(UE_pos, ground_AP_pos, current_UAV_pos, rates, mask, betas)
            
            # 5. 更新位置
            current_UAV_pos = self._update_positions_v5(current_UAV_pos, forces, iteration)
            
            history['min_rates'].append(min_rate)
            history['sum_rates'].append(sum_rate)
            
            if iteration % 10 == 0:
                print(f"Iter {iteration:2d} | MinRate: {min_rate:6.4f} | SumRate: {sum_rate:7.1f} | BestMin: {best_min_rate:6.4f}")

        return {
            'optimized_UAV_pos': best_UAV_pos,
            'final_min_rate': best_min_rate,
            'final_sum_rate': best_sum_rate,
            'final_rates': best_rates,
            'history': history,
            'best_iteration': best_iteration,
            'restart_count': restart_count
        }

    def _compute_comm_aware_forces(self, UE_pos, ground_AP_pos, UAV_pos, rates, mask, betas):
        forces = np.zeros((self.L, 3))
        w = self.force_params
        mask_uav = mask[:, self.G:]
        betas_uav = betas[:, self.G:]
        avg_rate = rates.mean()
        
        for l in range(self.L):
            # 信号增强力
            served = np.where(mask_uav[:, l])[0]
            for k in served:
                direction = UE_pos[k] - UAV_pos[l]
                dist = np.linalg.norm(direction[:2]) + 1e-6
                urgency = (avg_rate / (rates[k] + 1e-3)) ** 1.8 # 增强紧迫度敏感
                force_mag = urgency * (betas_uav[k, l] / (dist + 5))
                forces[l, :2] += w['w_signal'] * force_mag * (direction[:2] / dist)
            
            # 干扰抑制力
            unserved = np.where(~mask_uav[:, l])[0]
            for k in unserved:
                if rates[k] < avg_rate * 1.3:
                    direction = UAV_pos[l] - UE_pos[k]
                    dist = np.linalg.norm(direction[:2]) + 1e-6
                    interference_sensitivity = (betas_uav[k, l] / (np.mean(betas_uav[k, mask_uav[k,:]]) + 1e-12)) 
                    force_mag = interference_sensitivity / (dist + 10)
                    forces[l, :2] += w['w_interference'] * force_mag * (direction[:2] / dist)
            
            # 分离与边界
            forces[l, :2] += w['w_separation'] * self._calc_sep_force(l, UAV_pos)
            forces[l, :2] += w['w_boundary'] * self._calc_bound_force(UAV_pos[l])
        return forces

    def _calc_sep_force(self, l, UAV_pos):
        f = np.zeros(2)
        for i in range(self.L):
            if i == l: continue
            d_vec = UAV_pos[l] - UAV_pos[i]
            dist = np.linalg.norm(d_vec[:2]) + 1e-6
            if dist < self.force_params['sep_dist']:
                f += (self.force_params['sep_dist'] - dist) / dist * d_vec[:2]
        return f

    def _calc_bound_force(self, pos):
        f = np.zeros(2)
        m = self.force_params['margin']
        if pos[0] < m: f[0] += 1
        elif pos[0] > self.square_length - m: f[0] -= 1
        if pos[1] < m: f[1] += 1
        elif pos[1] > self.square_length - m: f[1] -= 1
        return f

    def _update_positions_v5(self, UAV_pos, forces, iteration):
        force_norms = np.linalg.norm(forces[:, :2], axis=1)
        max_f = np.max(force_norms) if np.max(force_norms) > 0 else 1
        alpha = (1 - iteration / self.max_iterations) ** 0.6
        v_current = self.step_size * alpha * (forces[:, :2] / max_f)
        
        beta = 0.2 # 减小动量，增加灵活性
        self.momentum = beta * self.momentum + (1 - beta) * v_current
        
        new_pos = UAV_pos.copy()
        new_pos[:, :2] += self.momentum
        new_pos[:, :2] = np.clip(new_pos[:, :2], 50, self.square_length - 50)
        return new_pos
