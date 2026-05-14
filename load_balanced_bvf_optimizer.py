"""
Load-Balanced BVF Optimizer (负载均衡的平衡虚拟力优化器)
基于 V5Pro，新增第 5 类力：负载均衡力

设计原则：
1. 继承 V5Pro 全部优化能力（动量、智能重启、记忆最优）
2. 负载均衡力作为轻量附加力，不削减原有通信力权重
3. 选择标准：min_rate 为主，JFI(负载均衡度) 为 tiebreaker
"""

import numpy as np
from typing import Tuple, Dict
from balanced_virtual_force_optimizer_v5_sinr import BalancedVirtualForceOptimizerV5


class LoadBalancedBVF(BalancedVirtualForceOptimizerV5):
    """
    负载均衡的 BVF (基于 V5Pro + 第 5 类负载均衡力)
    
    力场组成：
      1. 信号增强引力 (w_signal=0.55)  — V5Pro 原有
      2. 干扰抑制斥力 (w_interference=0.25)  — V5Pro 原有
      3. 分离力 (w_separation=0.10)  — V5Pro 原有
      4. 边界力 (w_boundary=0.10)  — V5Pro 原有
      5. 负载均衡力 (附加，不削减上述权重)  — NEW
    
    优化目标：
      主目标 = max min_rate (纯 V5Pro 的最大化最小速率)
      Tiebreaker = JFI (负载均衡度，仅在 min_rate 近似相等时启用)
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # === 负载均衡参数 ===
        self.enable_load_balance = config.get('enable_load_balance', True)
        self.K_load = config.get('K_load', 2e4)
        self.w_load = config.get('w_load', 0.08)
        self.load_threshold = config.get('load_threshold', 1.3)
        self.backhaul_capacity = config.get('backhaul_capacity', 500.0)
        self.jfi_tiebreak_tol = config.get('jfi_tiebreak_tol', 0.3)
    
    def compute_uav_loads(self, rates: np.ndarray, mask: np.ndarray) -> Dict:
        """
        计算每个 UAV 的负载度
        
        负载度 = max(用户数归一化, 回程流量占用率)
        """
        mask_uav = mask[:, self.G:]
        
        user_count = mask_uav.sum(axis=0).astype(float)
        
        throughput = np.zeros(self.L)
        for l in range(self.L):
            served = mask_uav[:, l]
            if served.any():
                throughput[l] = rates[served].sum()
        
        user_load = user_count / self.K
        backhaul_load = throughput / self.backhaul_capacity
        load_index = np.maximum(user_load, backhaul_load)
        
        avg_load = load_index.mean()
        overloaded = load_index > (avg_load * self.load_threshold)
        
        return {
            'user_count': user_count,
            'throughput': throughput,
            'load_index': load_index,
            'avg_load': avg_load,
            'overloaded': overloaded
        }
    
    @staticmethod
    def jains_fairness_index(x: np.ndarray) -> float:
        """Jain's Fairness Index: (sum x)^2 / (n * sum x^2)"""
        if x.sum() < 1e-6:
            return 1.0
        n = len(x)
        return float((x.sum() ** 2) / (n * (x ** 2).sum() + 1e-12))
    
    def _compute_load_balance_force(self, UAV_pos: np.ndarray, UE_pos: np.ndarray,
                                    rates: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        负载均衡力（第 5 类虚拟力）
        
        过载 UAV → 从服务簇中心向外的斥力，促使边缘用户切换至邻近轻载 UAV
        力的强度与过载程度正相关，与离簇中心距离负相关
        """
        forces = np.zeros((self.L, 3))
        
        if not self.enable_load_balance:
            return forces
        
        load_info = self.compute_uav_loads(rates, mask)
        load_index = load_info['load_index']
        avg_load = load_info['avg_load']
        overloaded = load_info['overloaded']
        mask_uav = mask[:, self.G:]
        
        for l in range(self.L):
            if not overloaded[l]:
                continue
            
            served_users = np.where(mask_uav[:, l])[0]
            if len(served_users) == 0:
                continue
            
            # 加权簇中心（权重 = 1/rate，低速率用户权重更高）
            user_rates = rates[served_users]
            user_positions = UE_pos[served_users, :2]
            weights = 1.0 / (user_rates + 1e-3)
            weights /= weights.sum()
            cluster_center = (user_positions * weights[:, None]).sum(axis=0)
            
            # 斥力方向：从簇中心指向 UAV
            direction = UAV_pos[l, :2] - cluster_center
            dist = np.linalg.norm(direction) + 1e-6
            unit_dir = direction / dist
            
            # 力的大小 = K_load × 过载因子 × 距离衰减 × 密度因子
            overload_factor = np.clip((load_index[l] - avg_load) / (avg_load + 1e-6), 0, 2.0)
            dist_factor = 1.0 / (1.0 + dist / 120.0)
            density_factor = len(served_users) / self.K
            
            force_mag = self.K_load * overload_factor * dist_factor * density_factor
            forces[l, :2] = force_mag * unit_dir
        
        return forces
    
    def _compute_comm_aware_forces(self, UE_pos, ground_AP_pos, UAV_pos, rates, mask, betas):
        """
        重写 V5Pro 的力场计算：在 4 类通信力基础上附加第 5 类负载均衡力
        不修改原有力的权重，仅叠加
        """
        # V5Pro 原有 4 类力（信号引力 + 干扰斥力 + 分离 + 边界）
        forces = super()._compute_comm_aware_forces(UE_pos, ground_AP_pos, UAV_pos, rates, mask, betas)
        
        # 附加：负载均衡力
        if self.enable_load_balance:
            f_load = self._compute_load_balance_force(UAV_pos, UE_pos, rates, mask)
            forces[:, :2] += self.w_load * f_load[:, :2]
        
        return forces
    
    def optimize(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray, UAV_pos: np.ndarray) -> Dict:
        """
        负载均衡的 BVF 优化
        
        完整继承 V5Pro 的优化流程（动量、记忆、智能重启），
        力场计算中附加负载均衡力，选择逻辑中以 JFI 作为 tiebreaker。
        """
        print("=" * 80)
        print("  Load-Balanced BVF (V5Pro + Load Balancing Force)".center(80))
        print("=" * 80)
        if self.enable_load_balance:
            print(f"  Load balance: w={self.w_load}, threshold={self.load_threshold}x, "
                  f"K={self.K_load:.0e}")
            print(f"  JFI tiebreak tolerance: {self.jfi_tiebreak_tol} Mbps")
        
        current_UAV_pos = UAV_pos.copy()
        self.momentum = np.zeros((self.L, 2))
        
        best_min_rate = -np.inf
        best_sum_rate = -np.inf
        best_jfi = -np.inf
        best_UAV_pos = UAV_pos.copy()
        best_rates = None
        best_iteration = 0
        no_improvement_count = 0
        restart_count = 0
        
        history = {
            'min_rates': [], 'sum_rates': [],
            'jfis': [], 'overload_counts': []
        }
        
        for iteration in range(self.max_iterations):
            # 1. 评估
            all_AP_pos = np.vstack([ground_AP_pos, current_UAV_pos])
            _, _, betas = self.compute_channel_model(UE_pos, all_AP_pos)
            mask = self.compute_AP_selection_mask(betas)
            rates, sum_rate = self.compute_user_rates(UE_pos, all_AP_pos, mask)
            min_rate = rates.min()
            
            # 负载指标
            load_info = self.compute_uav_loads(rates, mask)
            jfi = self.jains_fairness_index(load_info['load_index'])
            overload_count = int(load_info['overloaded'].sum())
            
            # 2. 记忆最优（min_rate 为主，JFI 为 tiebreaker）
            improved = False
            if min_rate > best_min_rate + self.jfi_tiebreak_tol:
                improved = True
            elif abs(min_rate - best_min_rate) <= self.jfi_tiebreak_tol:
                if jfi > best_jfi + 0.005:
                    improved = True
                elif min_rate > best_min_rate and jfi >= best_jfi - 0.01:
                    improved = True
            
            if improved:
                best_min_rate = min_rate
                best_sum_rate = sum_rate
                best_jfi = jfi
                best_UAV_pos = current_UAV_pos.copy()
                best_rates = rates.copy()
                best_iteration = iteration
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # 3. 智能重启（继承 V5Pro）
            if (no_improvement_count >= self.restart_threshold and 
                restart_count < self.max_restarts and 
                iteration < self.max_iterations - 10):
                print(f"  Restart #{restart_count+1} (iter {iteration}) | "
                      f"BestMin={best_min_rate:.2f}, JFI={best_jfi:.4f}")
                current_UAV_pos = best_UAV_pos.copy() + np.random.normal(
                    0, self.perturbation_strength, (self.L, 3))
                current_UAV_pos[:, 2] = self.heights['UAV']
                current_UAV_pos[:, :2] = np.clip(current_UAV_pos[:, :2], 50, self.square_length - 50)
                self.momentum *= 0
                no_improvement_count = 0
                restart_count += 1
                continue
            
            # 4. 计算力（V5Pro 4 类 + 负载均衡力）
            forces = self._compute_comm_aware_forces(
                UE_pos, ground_AP_pos, current_UAV_pos, rates, mask, betas)
            
            # 5. 更新位置（V5Pro 的动量更新）
            current_UAV_pos = self._update_positions_v5(current_UAV_pos, forces, iteration)
            
            # 记录
            history['min_rates'].append(min_rate)
            history['sum_rates'].append(sum_rate)
            history['jfis'].append(jfi)
            history['overload_counts'].append(overload_count)
            
            if iteration % 10 == 0 or iteration == self.max_iterations - 1:
                print(f"Iter {iteration:>3} | Min={min_rate:>6.2f} | Sum={sum_rate:>7.1f} | "
                      f"JFI={jfi:.4f} | OL={overload_count}/{self.L} | "
                      f"Best={best_min_rate:.2f}")
        
        # 最终评估
        all_AP_final = np.vstack([ground_AP_pos, best_UAV_pos])
        _, _, betas_final = self.compute_channel_model(UE_pos, all_AP_final)
        mask_final = self.compute_AP_selection_mask(betas_final)
        rates_final, sum_rate_final = self.compute_user_rates(UE_pos, all_AP_final, mask_final)
        load_info_final = self.compute_uav_loads(rates_final, mask_final)
        jfi_final = self.jains_fairness_index(load_info_final['load_index'])
        
        print(f"\nOptimization complete (best iter={best_iteration}, restarts={restart_count})")
        print(f"  Min Rate: {rates_final.min():.2f} Mbps | Sum Rate: {sum_rate_final:.1f} Mbps")
        print(f"  JFI: {jfi_final:.4f} | Overload: {load_info_final['overloaded'].sum()}/{self.L}")
        
        return {
            'optimized_UAV_pos': best_UAV_pos,
            'final_min_rate': float(rates_final.min()),
            'final_sum_rate': float(sum_rate_final),
            'final_rates': rates_final,
            'final_load_info': load_info_final,
            'load_balance_index': jfi_final,
            'history': history,
            'best_iteration': best_iteration,
            'restart_count': restart_count
        }


if __name__ == "__main__":
    from balanced_virtual_force_optimizer_v3 import create_balanced_config, BalancedVirtualForceOptimizerV3
    
    config = create_balanced_config()
    config.update({
        'num_UE': 60, 'num_UAV': 9, 'num_ground_AP': 4,
        'enable_load_balance': True,
        'K_load': 2e4, 'w_load': 0.08,
        'load_threshold': 1.3,
    })
    
    v3_helper = BalancedVirtualForceOptimizerV3(config)
    UE_pos, ground_AP_pos, UAV_pos = v3_helper.initialize_positions()
    
    optimizer = LoadBalancedBVF(config)
    results = optimizer.optimize(UE_pos, ground_AP_pos, UAV_pos)
    
    print(f"\nMin Rate: {results['final_min_rate']:.2f}")
    print(f"JFI: {results['load_balance_index']:.4f}")
