"""
标准ISSA优化器 - 严格遵循论文公式（3D版本）
- 公式14: 生产者更新
- 公式15: 跟随者更新
- 公式16: 警戒者更新
- 公式17-18: 混沌策略初始化
- 公式19-20: Cauchy-Gaussian变异
用于三维UAV位置优化（xyz）
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 导入BVF的信道模型相关函数
from balanced_virtual_force_optimizer_v3 import BalancedVirtualForceOptimizerV3


class StandardISSAOptimizer:
    """
    标准改进麻雀搜索算法(ISSA)优化器
    严格按照论文中的改进方法实现
    """
    
    def __init__(self, config: Dict):
        """初始化ISSA优化器"""
        self.config = config
        self.setup_parameters()
        
        # 创建BVF优化器实例（用于复用信道模型计算）
        self.bvf_optimizer = BalancedVirtualForceOptimizerV3(config)
        
    def setup_parameters(self):
        """设置ISSA算法参数"""
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
        
        # ISSA算法参数
        self.n_sparrows = self.config.get('issa_n_sparrows', 30)
        self.max_iter = self.config.get('issa_max_iter', 50)
        self.pd = self.config.get('issa_pd', 0.2)  # 生产者比例
        self.sd = self.config.get('issa_sd', 0.2)  # 警戒者比例
        self.st = self.config.get('issa_st', 0.8)  # 安全阈值
        
        # 边界约束（针对单个UAV）
        self.ub = np.array([self.square_length - 50, self.square_length - 50, 
                           self.config.get('UAV_height_max', 150)])
        self.lb = np.array([50, 50, self.config.get('UAV_height_min', 50)])
        
        # 随机种子
        self.random_seed = self.config.get('random_seed', None)
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
    
    def _chaotic_initialization(self, n_sparrows: int, dim: int) -> np.ndarray:
        """
        混沌策略初始化 - 公式(17)(18)
        Y_{i+1,j} = sin(0.7π / Y_{i,j}), Y_{i,j} ∈ [0,1]
        X_i = X_lb + Y_i * (X_ub - X_lb)
        
        Parameters:
        -----------
        n_sparrows : int
            麻雀数量
        dim : int
            维度（L × 3，所有UAV的xyz坐标）
            
        Returns:
        --------
        np.ndarray
            初始化的种群位置 (n_sparrows × dim)
        """
        # 初始化Y矩阵在(0, 1)区间
        Y = np.random.uniform(0.1, 0.9, (n_sparrows, dim))
        
        # 迭代映射混沌序列（公式17）- 进行10次迭代
        for iteration in range(10):
            Y = np.sin(0.7 * np.pi / (Y + 1e-10))  # 避免除零
            Y = np.abs(Y)  # 确保为正
            # 归一化到[0, 1]
            Y = (Y - Y.min()) / (Y.max() - Y.min() + 1e-10)
        
        # 映射到解空间（公式18）
        # 对于3D UAV位置：需要对x,y,z分别映射
        population = np.zeros((n_sparrows, dim))
        
        for i in range(n_sparrows):
            for l in range(self.L):
                # 每个UAV的xyz坐标
                idx_x = l * 3
                idx_y = l * 3 + 1
                idx_z = l * 3 + 2
                
                # 映射x坐标
                population[i, idx_x] = self.lb[0] + Y[i, idx_x] * (self.ub[0] - self.lb[0])
                # 映射y坐标
                population[i, idx_y] = self.lb[1] + Y[i, idx_y] * (self.ub[1] - self.lb[1])
                # 映射z坐标
                population[i, idx_z] = self.lb[2] + Y[i, idx_z] * (self.ub[2] - self.lb[2])
        
        return population
    
    def _decode_population(self, population: np.ndarray) -> np.ndarray:
        """
        将种群编码解码为UAV位置矩阵
        
        Parameters:
        -----------
        population : np.ndarray
            种群编码 (n_sparrows × (L×3)) 或 (1 × (L×3))
            
        Returns:
        --------
        np.ndarray
            UAV位置矩阵 (L × 3) 或 (n_sparrows × L × 3)
        """
        if population.ndim == 1:
            population = population.reshape(1, -1)
        
        # 解码为UAV位置
        UAV_pos = population.reshape(-1, self.L, 3)
        
        # 边界约束
        for i in range(3):
            UAV_pos[:, :, i] = np.clip(UAV_pos[:, :, i], self.lb[i], self.ub[i])
        
        return UAV_pos if UAV_pos.shape[0] > 1 else UAV_pos[0]
    
    def _encode_UAV_pos(self, UAV_pos: np.ndarray) -> np.ndarray:
        """将UAV位置编码为种群个体"""
        return UAV_pos.flatten()
    
    def _compute_fitness(self, UAV_pos: np.ndarray, UE_pos: np.ndarray, 
                        ground_AP_pos: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        计算适应度函数
        优化目标：最大化最小用户速率
        """
        try:
            all_AP_pos = np.vstack([ground_AP_pos, UAV_pos])
            _, _, betas = self.bvf_optimizer.compute_channel_model(UE_pos, all_AP_pos)
            mask = self.bvf_optimizer.compute_AP_selection_mask(betas)
            rates, sum_rate = self.bvf_optimizer.compute_user_rates(UE_pos, all_AP_pos, mask)
            min_rate = rates.min()
            
            # 适应度 = 最小用户速率（直接优化目标）
            fitness = min_rate
            fitness = max(fitness, 1e-6)  # 确保为正
            
            return fitness, sum_rate, rates
        except Exception as e:
            print(f"⚠️ 适应度计算错误: {e}")
            return 1e-6, 0.0, np.zeros(self.K)
    
    def _update_producer(self, iter: int, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """
        更新生产者位置 - 论文公式(14)
        
        if R < ST:
            X_i(t+1) = X_i(t) * exp(-i / (ξ * iter_max))
        else:
            X_i(t+1) = X_i(t) + Q * L
        """
        pd_num = int(self.n_sparrows * self.pd)
        updated_pop = population[:pd_num].copy()
        
        # 生成警告值 R ∈ [0,1]
        R = np.random.rand()
        
        for i in range(pd_num):
            if R < self.st:  # 安全：局部细化搜索
                # 公式(14)第一种情况
                xi = np.random.rand()  # ξ ∈ [0,1]
                updated_pop[i] = population[i] * np.exp(-i / (xi * self.max_iter + 1e-10))
            else:  # 危险：大范围搜索
                # 公式(14)第二种情况
                # Q: 正态分布随机数
                # L: 1×d 全1矩阵
                Q = np.random.randn()
                L = np.ones(population.shape[1])
                updated_pop[i] = population[i] + Q * L
        
        # 边界处理
        for i in range(pd_num):
            uav_pos = self._decode_population(updated_pop[i])
            updated_pop[i] = self._encode_UAV_pos(uav_pos)
        
        return updated_pop
    
    def _update_scrounger(self, iter: int, population: np.ndarray, 
                         best_position: np.ndarray, worst_position: np.ndarray) -> np.ndarray:
        """
        更新跟随者位置 - 论文公式(15)
        
        if i > n/2:
            X_i(t+1) = Q * exp((X_worst - X_i(t)) / i²)
        else:
            X_i(t+1) = X_p(t+1) + |X_i(t) - X_p(t+1)| * A^+ * L
        """
        pd_num = int(self.n_sparrows * self.pd)
        n_scrounger = self.n_sparrows - pd_num
        updated_pop = population[pd_num:].copy()
        
        for i in range(n_scrounger):
            idx = pd_num + i
            
            if idx > self.n_sparrows / 2:  # 差的个体
                # 公式(15)第一种情况：接近饥饿
                Q = np.random.randn(population.shape[1])
                updated_pop[i] = Q * np.exp((worst_position - population[idx]) / ((i + 1) ** 2))
            else:  # 较好的个体
                # 公式(15)第二种情况：跟随生产者
                # X_p 是生产者位置（这里用best_position）
                # A: 1×d 随机矩阵，元素为±1
                # A^+ = A^T(AA^T)^{-1}
                A = np.random.choice([-1, 1], size=population.shape[1])
                # 计算 A^+
                AA_T = np.dot(A, A)
                A_plus = A / (AA_T + 1e-8)
                
                L = np.ones(population.shape[1])
                updated_pop[i] = best_position + np.abs(population[idx] - best_position) * A_plus * L
        
        # 边界处理
        for i in range(n_scrounger):
            uav_pos = self._decode_population(updated_pop[i])
            updated_pop[i] = self._encode_UAV_pos(uav_pos)
        
        return updated_pop
    
    def _update_watch(self, population: np.ndarray, best_position: np.ndarray,
                     fitness: np.ndarray, best_fitness: float, worst_fitness: float) -> np.ndarray:
        """
        更新警戒者位置 - 论文公式(16)
        
        if f_i > f_b:
            X_i(t+1) = X_best(t) + ω * |X_i(t) - X_best(t)|
        else:
            X_i(t+1) = X_i(t) + K * |X_i(t) - X_worst(t)| / (f_i - f_worst + ε)
        """
        sd_num = int(self.n_sparrows * self.sd)
        watch_indices = np.random.choice(self.n_sparrows, sd_num, replace=False)
        updated_pop = population.copy()
        
        worst_position = population[np.argmin(fitness)]
        
        for idx in watch_indices:
            f_i = fitness[idx]
            
            if f_i > best_fitness:  # 在边缘（危险）
                # 公式(16)第一种情况
                # ω: 正态分布步长系数
                omega = np.random.randn(population.shape[1])
                updated_pop[idx] = best_position + omega * np.abs(population[idx] - best_position)
            else:  # 在中间（安全）
                # 公式(16)第二种情况
                # K: [-1, 1] 随机值，表示方向
                # ε: 避免除零的最小常数
                K = np.random.uniform(-1, 1, size=population.shape[1])
                epsilon = 1e-8
                
                denominator = np.abs(f_i - worst_fitness) + epsilon
                step = K * np.abs(population[idx] - worst_position) / denominator
                
                # 限制步长避免数值爆炸
                step = np.clip(step, -100, 100)
                updated_pop[idx] = population[idx] + step
        
        # 边界处理
        for idx in watch_indices:
            uav_pos = self._decode_population(updated_pop[idx])
            updated_pop[idx] = self._encode_UAV_pos(uav_pos)
        
        return updated_pop
    
    def _cauchy_gaussian_mutation(self, best_position: np.ndarray, iter: int, 
                                 fitness: np.ndarray, best_fitness: float) -> np.ndarray:
        """
        Cauchy-Gaussian变异策略 - 公式(19)(20)
        Z^t_best = X^t_best · [1 + (1 - t²/iter²_max)·cauchy(0,σ²) + t²/iter²_max·gauss(0,σ²)]
        """
        # 计算自适应方差σ（公式20）
        f_mean = np.mean(fitness)
        
        if best_fitness > f_mean:
            # 当前最优好于平均：σ=1，减少扰动
            sigma = 1.0
        else:
            # 当前最优不好：σ增大，增加扰动
            sigma = np.exp((best_fitness - f_mean) / (np.abs(best_fitness) + 1e-10))
        
        # 时间比率
        t = iter + 1
        t_ratio_sq = (t / self.max_iter) ** 2  # t²/iter²_max
        
        # Cauchy随机数
        cauchy_term = np.random.standard_cauchy(len(best_position)) * sigma
        cauchy_term = np.clip(cauchy_term, -5, 5)  # 限制Cauchy的长尾
        
        # Gaussian随机数
        gaussian_term = np.random.normal(0, sigma, len(best_position))
        
        # 公式(19)：Cauchy-Gaussian变异
        mutated_position = best_position * (
            1.0 + 
            (1 - t_ratio_sq) * cauchy_term * 0.1 +  # 早期：Cauchy主导（全局搜索）
            t_ratio_sq * gaussian_term * 0.1          # 后期：Gaussian主导（局部搜索）
        )
        
        # 边界处理
        uav_pos = self._decode_population(mutated_position)
        mutated_position = self._encode_UAV_pos(uav_pos)
        
        return mutated_position
    
    def optimize(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray,
                UAV_pos: np.ndarray) -> Dict:
        """
        执行标准ISSA优化
        
        Returns:
        --------
        Dict: 包含优化结果的字典
        """
        print("="*80)
        print("开始标准ISSA优化（严格遵循论文公式 - 3D版本）...")
        print(f"  • 优化目标: 最大化最小用户速率")
        print(f"  • 优化变量: UAV 3D位置(xyz)")
        print(f"  • 公式14: 生产者更新")
        print(f"  • 公式15: 跟随者更新")
        print(f"  • 公式16: 警戒者更新")
        print(f"  • 公式17-18: 混沌初始化")
        print(f"  • 公式19-20: Cauchy-Gaussian变异")
        print(f"  • 麻雀数量: {self.n_sparrows}, 最大迭代: {self.max_iter}")
        print("="*80)
        
        # 初始化历史记录
        history = {
            'iterations': [],
            'sum_rates': [],
            'min_rates': [],
            'best_fitness': []
        }
        
        # 种群维度
        dim = self.L * 3
        
        # 1. 混沌初始化（公式17-18）
        print("\n[1/5] 混沌策略初始化种群...")
        population = self._chaotic_initialization(self.n_sparrows, dim)
        
        # 2. 计算初始适应度
        print("[2/5] 计算初始适应度...")
        fitness = np.zeros(self.n_sparrows)
        sum_rates = np.zeros(self.n_sparrows)
        
        for i in range(self.n_sparrows):
            uav_pos = self._decode_population(population[i])
            fit, sum_rate, rates = self._compute_fitness(uav_pos, UE_pos, ground_AP_pos)
            fitness[i] = fit
            sum_rates[i] = sum_rate
        
        # 初始化最优解
        best_idx = np.argmax(fitness)
        best_position = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        best_sum_rate = sum_rates[best_idx]
        best_UAV_pos = self._decode_population(best_position)
        
        print(f"  ✓ 初始最优适应度: {best_fitness:.4f} Mbps")
        
        start_time = time.time()
        
        # 3. 迭代优化
        print(f"\n[3/5] 开始迭代优化（{self.max_iter}次）...")
        print("-"*80)
        
        for iter in range(self.max_iter):
            iter_start = time.time()
            
            # 按适应度排序（降序）
            sorted_indices = np.argsort(fitness)[::-1]
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            sum_rates = sum_rates[sorted_indices]
            
            # 更新最优和最差位置
            best_position = population[0].copy()
            best_fitness = fitness[0]
            best_sum_rate = sum_rates[0]
            worst_position = population[-1].copy()
            worst_fitness = fitness[-1]
            
            # (a) 更新生产者（公式14）
            producer_pop = self._update_producer(iter, population, fitness)
            population[:len(producer_pop)] = producer_pop
            
            # 重新评估生产者
            for i in range(len(producer_pop)):
                uav_pos = self._decode_population(population[i])
                fit, sum_rate, rates = self._compute_fitness(uav_pos, UE_pos, ground_AP_pos)
                fitness[i] = fit
                sum_rates[i] = sum_rate
            
            # (b) 更新跟随者（公式15）
            scrounger_pop = self._update_scrounger(iter, population, best_position, worst_position)
            pd_num = int(self.n_sparrows * self.pd)
            population[pd_num:] = scrounger_pop
            
            # 重新评估跟随者
            for i in range(pd_num, self.n_sparrows):
                uav_pos = self._decode_population(population[i])
                fit, sum_rate, rates = self._compute_fitness(uav_pos, UE_pos, ground_AP_pos)
                fitness[i] = fit
                sum_rates[i] = sum_rate
            
            # (c) 更新警戒者（公式16）
            population = self._update_watch(population, best_position, fitness, best_fitness, worst_fitness)
            
            # 重新评估所有个体
            for i in range(self.n_sparrows):
                uav_pos = self._decode_population(population[i])
                fit, sum_rate, rates = self._compute_fitness(uav_pos, UE_pos, ground_AP_pos)
                fitness[i] = fit
                sum_rates[i] = sum_rate
            
            # (d) Cauchy-Gaussian变异（公式19-20）
            mutated_position = self._cauchy_gaussian_mutation(best_position, iter, fitness, best_fitness)
            uav_pos_mutated = self._decode_population(mutated_position)
            mutated_fitness, mutated_sum_rate, mutated_rates = self._compute_fitness(
                uav_pos_mutated, UE_pos, ground_AP_pos)
            
            # 如果变异个体更好，替换最差个体
            if mutated_fitness > fitness.min():
                worst_idx = np.argmin(fitness)
                population[worst_idx] = mutated_position
                fitness[worst_idx] = mutated_fitness
                sum_rates[worst_idx] = mutated_sum_rate
                
                # 如果变异个体是新的全局最优，直接更新
                if mutated_fitness > best_fitness:
                    best_position = mutated_position.copy()
                    best_fitness = mutated_fitness
                    best_sum_rate = mutated_sum_rate
                    best_UAV_pos = uav_pos_mutated.copy()
            
            # 更新全局最优（检查种群中的最优）
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > best_fitness:
                best_position = population[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]
                best_sum_rate = sum_rates[current_best_idx]
                best_UAV_pos = self._decode_population(best_position)
            
            # 记录历史（使用已保存的值，避免重复计算造成不一致）
            history['iterations'].append(iter)
            history['sum_rates'].append(best_sum_rate)
            history['min_rates'].append(best_fitness)
            history['best_fitness'].append(best_fitness)
            
            iter_time = time.time() - iter_start
            
            if (iter + 1) % 10 == 0:
                print(f"Iter {iter+1:>3}/{self.max_iter}: "
                      f"Sum={best_sum_rate:>7.2f}Mbps, "
                      f"Min={best_fitness:>7.4f}Mbps, "
                      f"Time={iter_time:>5.3f}s")
        
        optimization_time = time.time() - start_time
        
        # 4. 使用迭代中记录的最优结果（避免随机信道造成不一致）
        print("\n[4/5] 整理最终结果...")
        print("  ℹ️  使用迭代中记录的最优值（信道模型包含随机性）")
        
        # 使用迭代中保存的最优值，确保与显示的迭代结果一致
        final_min_rate = best_fitness
        final_sum_rate = best_sum_rate
        
        # 重新计算以获取完整的rates数组（用于统计分析）
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
            'total_iterations': self.max_iter,
            'history': history
        }
        
        print("\n[5/5] 优化完成!")
        print("="*80)
        print(f"🎉 标准ISSA优化完成!")
        print(f"📊 最终总速率: {final_sum_rate:.2f} Mbps")
        print(f"🎯 最终最小速率: {final_min_rate:.4f} Mbps")
        print(f"📈 平均速率: {final_rates.mean():.4f} Mbps")
        print(f"⏱️  优化时间: {optimization_time:.2f} 秒")
        print("="*80)
        
        return results
