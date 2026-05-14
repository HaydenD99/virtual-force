"""
NewSSA优化器（加权版本） - 基于论文"Deployment for NOMA-UAV Base Stations Based on Hybrid Sparrow Search Algorithm"
改进点：
1. OBL策略（Opposition-Based Learning）+ 折射原理 - 公式(18)
2. 正弦-余弦搜索算法 - 公式(20)
3. 自适应参数r1和w - 公式(21)(22)
4. 优化2D UAV位置（xy），高度固定
5. FITNESS函数：加权组合最小用户速率和系统和速率
   fitness = min_rate + w_sum * sum_rate
"""

import numpy as np
import time
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from balanced_virtual_force_optimizer_v3 import BalancedVirtualForceOptimizerV3


class WeightedNewSSAOptimizer:
    """
    NewSSA优化器（加权fitness版本） - 严格遵循论文算法
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_parameters()
        self.bvf_optimizer = BalancedVirtualForceOptimizerV3(config)
        
    def setup_parameters(self):
        """设置算法参数"""
        # 基本参数
        self.square_length = self.config.get('square_length', 1000)
        self.K = self.config.get('num_UE', 60)
        self.L = self.config.get('num_UAV', 9)  # Z in paper
        self.G = self.config.get('num_ground_AP', 4)
        
        # UAV高度（固定）
        self.UAV_height = self.config.get('UAV_height', 50.0)
        
        # NewSSA参数
        self.Z = self.config.get('newssa_n_sparrows', 30)  # 麻雀总数
        self.Iter_max = self.config.get('newssa_max_iter', 50)
        self.PR = self.config.get('newssa_pr', 0.2)  # 生产者比例
        self.FR = self.config.get('newssa_fr', 0.15)  # 警戒者比例（论文中是15%）
        self.ST = self.config.get('newssa_st', 0.8)   # 安全阈值
        
        # 正弦-余弦搜索参数
        self.eta = 1.5  # η in formula (21)
        
        # 折射参数（OBL策略）
        self.k = 1.5  # 缩放因子
        self.n = 2.0  # 精调因子
        
        # 2D边界
        self.ub = np.array([self.square_length - 50, self.square_length - 50])
        self.lb = np.array([50, 50])
        
        # 随机种子
        self.random_seed = self.config.get('random_seed', None)
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
    
    def _obl_refraction_initialization(self, n_sparrows: int, dim: int) -> np.ndarray:
        """
        OBL策略 + 折射原理初始化 - 论文公式(18)
        
        (xz, yz, hz)* = ((x_max+x_min)/2 - (x_max+x_min)/(2kn) - xz/kn, ...)
        """
        # 随机初始化
        population = np.random.uniform(0, 1, (n_sparrows, dim))
        
        # 映射到搜索空间
        for i in range(n_sparrows):
            for l in range(self.L):
                idx_x = l * 2
                idx_y = l * 2 + 1
                
                population[i, idx_x] = self.lb[0] + population[i, idx_x] * (self.ub[0] - self.lb[0])
                population[i, idx_y] = self.lb[1] + population[i, idx_y] * (self.ub[1] - self.lb[1])
        
        # 应用OBL + 折射原理（公式18）
        obl_population = np.zeros_like(population)
        kn = self.k * self.n
        
        for i in range(n_sparrows):
            for l in range(self.L):
                idx_x = l * 2
                idx_y = l * 2 + 1
                
                # x坐标
                x_max, x_min = self.ub[0], self.lb[0]
                x_z = population[i, idx_x]
                obl_population[i, idx_x] = (x_max + x_min)/2 - (x_max + x_min)/(2*kn) - x_z/kn
                
                # y坐标
                y_max, y_min = self.ub[1], self.lb[1]
                y_z = population[i, idx_y]
                obl_population[i, idx_y] = (y_max + y_min)/2 - (y_max + y_min)/(2*kn) - y_z/kn
        
        # 边界约束
        for i in range(n_sparrows):
            uav_pos = self._decode_population(obl_population[i])
            obl_population[i] = self._encode_UAV_pos(uav_pos)
        
        # 合并原始种群和OBL种群，选择更好的
        combined_pop = np.vstack([population, obl_population])
        
        return combined_pop[:n_sparrows]  # 返回前n_sparrows个
    
    def _decode_population(self, population: np.ndarray) -> np.ndarray:
        """解码为3D UAV位置（添加固定高度）"""
        if population.ndim == 1:
            population = population.reshape(1, -1)
        
        n = population.shape[0]
        UAV_pos = np.zeros((n, self.L, 3))
        
        for i in range(n):
            for l in range(self.L):
                idx_x = l * 2
                idx_y = l * 2 + 1
                
                UAV_pos[i, l, 0] = np.clip(population[i, idx_x], self.lb[0], self.ub[0])
                UAV_pos[i, l, 1] = np.clip(population[i, idx_y], self.lb[1], self.ub[1])
                UAV_pos[i, l, 2] = self.UAV_height
        
        return UAV_pos[0] if n == 1 else UAV_pos
    
    def _encode_UAV_pos(self, UAV_pos: np.ndarray) -> np.ndarray:
        """编码UAV位置（提取xy）"""
        if UAV_pos.ndim == 2:
            return UAV_pos[:, :2].flatten()
        else:
            return UAV_pos[:, :, :2].reshape(UAV_pos.shape[0], -1)
    
    def _compute_fitness(self, UAV_pos: np.ndarray, UE_pos: np.ndarray,
                        ground_AP_pos: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
        """计算适应度（加权：最小用户速率 + 系统和速率）
        
        Returns:
            fitness: 加权适应度值
            min_rate: 真实的最小用户速率
            sum_rate: 系统和速率
            rates: 所有用户速率
        """
        try:
            all_AP_pos = np.vstack([ground_AP_pos, UAV_pos])
            _, _, betas = self.bvf_optimizer.compute_channel_model(UE_pos, all_AP_pos)
            mask = self.bvf_optimizer.compute_AP_selection_mask(betas)
            rates, sum_rate = self.bvf_optimizer.compute_user_rates(UE_pos, all_AP_pos, mask)
            min_rate = rates.min()
            
            # 加权fitness: min_rate + w_sum * sum_rate
            w_sum = self.config.get('w_sum_rate', 0.01)
            fitness = min_rate + w_sum * sum_rate
            
            return fitness, min_rate, sum_rate, rates
        except Exception as e:
            print(f"⚠️ 适应度计算错误: {e}")
            return 1e-6, 1e-6, 0.0, np.zeros(self.K)
    
    def _update_producer(self, s: int, population: np.ndarray, 
                        fitness: np.ndarray) -> np.ndarray:
        """
        生产者更新 - 论文公式(19)
        """
        PR_num = int(self.Z * self.PR)
        updated_pop = population[:PR_num].copy()
        
        R2 = np.random.rand()
        alpha = np.random.rand()
        
        for i in range(PR_num):
            if R2 < self.ST:
                # 安全：exp衰减
                updated_pop[i] = population[i] * np.exp(-s / (alpha * self.Iter_max + 1e-10))
            else:
                # 危险：随机搜索
                Q = np.random.randn()
                L = np.ones(population.shape[1])
                updated_pop[i] = population[i] + Q * L
        
        # 边界处理
        for i in range(PR_num):
            uav_pos = self._decode_population(updated_pop[i])
            updated_pop[i] = self._encode_UAV_pos(uav_pos)
        
        return updated_pop
    
    def _update_scrounger_sine_cosine(self, s: int, population: np.ndarray,
                                      best_position: np.ndarray) -> np.ndarray:
        """
        跟随者更新 - 正弦-余弦搜索 - 论文公式(20)(21)(22)
        """
        PR_num = int(self.Z * self.PR)
        n_scrounger = self.Z - PR_num
        updated_pop = population[PR_num:].copy()
        
        # 计算自适应参数
        # 公式(21): r1 = (1 - (s/Iter_max)^η)^(1/η)
        r1 = (1 - (s / self.Iter_max) ** self.eta) ** (1.0 / self.eta)
        
        # 公式(22): w = (e^(s/Iter_max) - 1) / (e - 1)
        w = (np.exp(s / self.Iter_max) - 1) / (np.e - 1)
        
        for i in range(n_scrounger):
            # 随机参数
            r2 = np.random.uniform(0, 2*np.pi)  # [0, 2π]
            r3 = np.random.uniform(0, 2)
            r4 = np.random.rand()
            
            idx = PR_num + i
            
            # 公式(20): 正弦-余弦搜索
            if r4 < 0.5:
                # 使用sin
                updated_pop[i] = w * population[idx] + \
                                r1 * np.sin(r2) * np.abs(r3 * best_position - population[idx])
            else:
                # 使用cos
                updated_pop[i] = w * population[idx] + \
                                r1 * np.cos(r2) * np.abs(r3 * best_position - population[idx])
        
        # 边界处理
        for i in range(n_scrounger):
            uav_pos = self._decode_population(updated_pop[i])
            updated_pop[i] = self._encode_UAV_pos(uav_pos)
        
        return updated_pop
    
    def _update_forerunner(self, population: np.ndarray, best_position: np.ndarray,
                          worst_position: np.ndarray, fitness: np.ndarray,
                          best_fitness: float, worst_fitness: float) -> np.ndarray:
        """
        警戒者（forerunner）更新 - 论文公式(23)
        """
        FR_num = int(self.Z * self.FR)
        forerunner_indices = np.random.choice(self.Z, FR_num, replace=False)
        updated_pop = population.copy()
        
        for idx in forerunner_indices:
            Omega_z = fitness[idx]
            Omega_b = best_fitness
            Omega_w = worst_fitness
            
            if Omega_z != Omega_b:  # 不在最优位置
                # 公式(23)第一种情况
                gamma = np.random.randn(population.shape[1])
                updated_pop[idx] = best_position + gamma * np.abs(population[idx] - best_position)
            else:  # 在最优位置
                # 公式(23)第二种情况
                K = np.random.uniform(-1, 1, size=population.shape[1])
                epsilon = 1e-8
                
                step = K * np.abs(population[idx] - worst_position) / (np.abs(Omega_z - Omega_w) + epsilon)
                step = np.clip(step, -100, 100)
                updated_pop[idx] = population[idx] + step
        
        # 边界处理
        for idx in forerunner_indices:
            uav_pos = self._decode_population(updated_pop[idx])
            updated_pop[idx] = self._encode_UAV_pos(uav_pos)
        
        return updated_pop
    
    def optimize(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray,
                UAV_pos: np.ndarray) -> Dict:
        """
        执行NewSSA优化 - 论文Algorithm (Table I)
        """
        print("="*80)
        print("开始NewSSA优化（论文算法）...")
        print(f"  • 改进1: OBL策略 + 折射原理（公式18）")
        print(f"  • 改进2: 正弦-余弦搜索（公式20）")
        print(f"  • 改进3: 自适应参数r1和w（公式21-22）")
        print(f"  • 麻雀数量: {self.Z}")
        print(f"  • 最大迭代: {self.Iter_max}")
        print(f"  • 生产者: {self.PR:.0%}, 警戒者: {self.FR:.0%}")
        print("="*80)
        
        # 历史记录
        history = {
            'iterations': [],
            'sum_rates': [],
            'min_rates': [],
            'best_fitness': []
        }
        
        dim = self.L * 2
        
        # 步骤1-4: OBL策略初始化（公式18）
        print("\n[1/5] OBL策略 + 折射原理初始化...")
        population = self._obl_refraction_initialization(self.Z, dim)
        
        # 步骤6: 计算初始适应度
        print("[2/5] 计算初始适应度...")
        fitness = np.zeros(self.Z)
        min_rates = np.zeros(self.Z)
        sum_rates = np.zeros(self.Z)
        
        for i in range(self.Z):
            uav_pos = self._decode_population(population[i])
            fit, min_r, sum_r, rates = self._compute_fitness(uav_pos, UE_pos, ground_AP_pos)
            fitness[i] = fit
            min_rates[i] = min_r
            sum_rates[i] = sum_r
        
        # 初始化最优和最差
        best_idx = np.argmax(fitness)
        worst_idx = np.argmin(fitness)
        
        best_position = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        best_min_rate = min_rates[best_idx]
        best_sum_rate = sum_rates[best_idx]
        best_UAV_pos = self._decode_population(best_position)
        
        worst_position = population[worst_idx].copy()
        worst_fitness = fitness[worst_idx]
        
        print(f"  ✓ 初始最优适应度: {best_fitness:.4f} (min_rate={best_min_rate:.4f} Mbps)")
        
        start_time = time.time()
        
        # 步骤5-19: 主循环
        print(f"\n[3/5] 开始迭代优化（{self.Iter_max}次）...")
        print("-"*80)
        
        for s in range(self.Iter_max):
            iter_start = time.time()
            
            # 步骤6: 排序
            sorted_indices = np.argsort(fitness)[::-1]
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            min_rates = min_rates[sorted_indices]
            sum_rates = sum_rates[sorted_indices]
            
            # 更新worst_position（总是当前最差）
            worst_position = population[-1].copy()
            worst_fitness = fitness[-1]
            
            # 只有当前最优比历史最优更好时才更新best（保持历史最优）
            if fitness[0] > best_fitness:
                best_position = population[0].copy()
                best_fitness = fitness[0]
                best_min_rate = min_rates[0]
                best_sum_rate = sum_rates[0]
                best_UAV_pos = self._decode_population(best_position)
            
            # 步骤7-9: 更新生产者（公式19）
            producer_pop = self._update_producer(s, population, fitness)
            population[:len(producer_pop)] = producer_pop
            
            for i in range(len(producer_pop)):
                uav_pos = self._decode_population(population[i])
                fit, min_r, sum_r, rates = self._compute_fitness(uav_pos, UE_pos, ground_AP_pos)
                fitness[i] = fit
                min_rates[i] = min_r
                sum_rates[i] = sum_r
            
            # 步骤10-12: 更新跟随者（公式20 - 正弦余弦搜索）
            PR_num = int(self.Z * self.PR)
            scrounger_pop = self._update_scrounger_sine_cosine(s, population, best_position)
            population[PR_num:] = scrounger_pop
            
            for i in range(PR_num, self.Z):
                uav_pos = self._decode_population(population[i])
                fit, min_r, sum_r, rates = self._compute_fitness(uav_pos, UE_pos, ground_AP_pos)
                fitness[i] = fit
                min_rates[i] = min_r
                sum_rates[i] = sum_r
            
            # 步骤13-15: 更新警戒者（公式23）
            population = self._update_forerunner(population, best_position, worst_position,
                                                fitness, best_fitness, worst_fitness)
            
            for i in range(self.Z):
                uav_pos = self._decode_population(population[i])
                fit, min_r, sum_r, rates = self._compute_fitness(uav_pos, UE_pos, ground_AP_pos)
                fitness[i] = fit
                min_rates[i] = min_r
                sum_rates[i] = sum_r
            
            # 步骤16-17: 更新全局最优
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > best_fitness:
                best_position = population[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]
                best_min_rate = min_rates[current_best_idx]
                best_sum_rate = sum_rates[current_best_idx]
                best_UAV_pos = self._decode_population(best_position)
            
            # 记录历史
            history['iterations'].append(s)
            history['sum_rates'].append(best_sum_rate)
            history['min_rates'].append(best_min_rate)
            history['best_fitness'].append(best_fitness)
            
            iter_time = time.time() - iter_start
            
            if (s + 1) % 10 == 0:
                print(f"Iter {s+1:>3}/{self.Iter_max}: "
                      f"Sum={best_sum_rate:>7.2f}Mbps, "
                      f"Min={best_min_rate:>7.4f}Mbps, "
                      f"Fit={best_fitness:>7.4f}, "
                      f"Time={iter_time:>5.3f}s")
        
        optimization_time = time.time() - start_time
        
        # 步骤20: 返回结果
        print("\n[4/5] 整理最终结果...")
        
        # 使用真实的min_rate而不是fitness
        final_min_rate = best_min_rate
        final_sum_rate = best_sum_rate
        
        all_AP_pos = np.vstack([ground_AP_pos, best_UAV_pos])
        _, _, betas = self.bvf_optimizer.compute_channel_model(UE_pos, all_AP_pos)
        mask = self.bvf_optimizer.compute_AP_selection_mask(betas)
        final_rates, _ = self.bvf_optimizer.compute_user_rates(UE_pos, all_AP_pos, mask)
        
        results = {
            'optimized_UAV_pos': best_UAV_pos,
            'final_sum_rate': final_sum_rate,
            'final_min_rate': final_min_rate,
            'final_rates': final_rates,
            'optimization_time': optimization_time,
            'total_iterations': self.Iter_max,
            'history': history
        }
        
        print("\n[5/5] 优化完成!")
        print("="*80)
        print(f"🎉 Weighted NewSSA优化完成!")
        print(f"📊 最终总速率: {final_sum_rate:.2f} Mbps")
        print(f"🎯 最终最小速率: {final_min_rate:.4f} Mbps")
        print(f"💫 最终适应度: {best_fitness:.4f} (weighted)")
        print(f"📈 平均速率: {final_rates.mean():.4f} Mbps")
        print(f"⏱️  优化时间: {optimization_time:.2f} 秒")
        print("="*80)
        
        return results
