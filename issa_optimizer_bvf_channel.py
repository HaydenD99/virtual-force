"""
基于BVF信道模型的ISSA优化器
用于多UAV部署优化，与BVF方法进行对比
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 导入BVF的信道模型相关函数
from balanced_virtual_force_optimizer_v3 import BalancedVirtualForceOptimizerV3
import functionRlocalscattering
import SpectralEfficiencyDownlink


class ISSAOptimizerBVFChannel:
    """
    基于BVF信道模型的改进麻雀搜索算法(ISSA)优化器
    """
    
    def __init__(self, config: Dict):
        """
        初始化ISSA优化器
        
        Parameters:
        -----------
        config : Dict
            配置字典，与BVF优化器使用相同的配置
        """
        self.config = config
        self.setup_parameters()
        
        # 创建BVF优化器实例（用于复用信道模型计算）
        self.bvf_optimizer = BalancedVirtualForceOptimizerV3(config)
        
    def setup_parameters(self):
        """设置ISSA算法参数"""
        # 基本参数（复用BVF的配置）
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
        self.n_sparrows = self.config.get('issa_n_sparrows', 30)  # 麻雀数量（种群大小）
        self.max_iter = self.config.get('issa_max_iter', 100)  # 最大迭代次数
        self.pd = self.config.get('issa_pd', 0.2)  # 生产者比例
        self.sd = self.config.get('issa_sd', 0.2)  # 警戒者比例
        self.st = self.config.get('issa_st', 0.8)  # 安全阈值
        
        # 边界约束
        self.ub = np.array([self.square_length - 50, self.square_length - 50, 
                           self.config.get('UAV_height_max', 150)])
        self.lb = np.array([50, 50, self.config.get('UAV_height_min', 50)])
        
        # 随机种子
        self.random_seed = self.config.get('random_seed', None)
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
    
    def _chaotic_initialization(self, n_sparrows: int, dim: int) -> np.ndarray:
        """
        使用混沌策略初始化种群
        
        Parameters:
        -----------
        n_sparrows : int
            麻雀数量
        dim : int
            维度（L × 3，所有UAV的位置）
            
        Returns:
        --------
        np.ndarray
            初始化的种群位置 (n_sparrows × (L × 3))
        """
        Y = np.random.rand(n_sparrows, dim)

        for _ in range(10):
            Y = 4 * Y * (1 - Y)  # Logistic映射
            Y = np.clip(Y, 0, 1)  # 确保在(0, 1)内
        
        # 映射到解空间: X_i = X_lb + Y_i × (X_ub - X_lb)
        # 每个麻雀代表所有L个UAV的位置
        population = np.zeros((n_sparrows, dim))
        for i in range(n_sparrows):
            uav_positions = []
            for l in range(self.L):
                # 为每个UAV映射到位置
                j = l * 3
                x = self.lb[0] + Y[i, j] * (self.ub[0] - self.lb[0])
                y = self.lb[1] + Y[i, min(j+1, dim-1)] * (self.ub[1] - self.lb[1])
                z = self.heights['UAV']  # 固定高度
                uav_positions.extend([x, y, z])
            population[i] = np.array(uav_positions)
        
        return population
    
    def _decode_population(self, population: np.ndarray) -> np.ndarray:
        """
        将种群编码解码为UAV位置矩阵
        
        Parameters:
        -----------
        population : np.ndarray
            种群编码 (n_sparrows × (L×3))
            
        Returns:
        --------
        np.ndarray
            UAV位置矩阵 (L × 3)
        """
        # 每个个体编码为所有UAV的位置
        UAV_pos = population.reshape(-1, self.L, 3)
        # 边界约束
        UAV_pos = np.clip(UAV_pos, self.lb, self.ub)
        return UAV_pos
    
    def _compute_fitness(self, UAV_pos: np.ndarray, UE_pos: np.ndarray, 
                        ground_AP_pos: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        计算适应度函数
        
        Parameters:
        -----------
        UAV_pos : np.ndarray
            UAV位置 (L × 3)
        UE_pos : np.ndarray
            UE位置 (K × 3)
        ground_AP_pos : np.ndarray
            地面AP位置 (G × 3)
            
        Returns:
        --------
        Tuple[float, float, np.ndarray]
            (适应度值, 总速率, 各用户速率)
        """
        try:
            # 使用BVF优化器的信道模型计算速率
            all_AP_pos = np.vstack([ground_AP_pos, UAV_pos])
            _, _, betas = self.bvf_optimizer.compute_channel_model(UE_pos, all_AP_pos)
            mask = self.bvf_optimizer.compute_AP_selection_mask(betas)
            rates, sum_rate = self.bvf_optimizer.compute_user_rates(UE_pos, all_AP_pos, mask)
            min_rate = rates.min()
            
            # 返回适应度值（ISSA最大化适应度）
            epsilon = 1e-3
            fitness = min_rate
            
            # 确保适应度值为正
            fitness = max(fitness, 0.001)
            
            return fitness, sum_rate, rates
        except Exception as e:
            # 如果计算失败，返回很差的适应度
            print(f"适应度计算错误: {e}")
            return 0.001, 0.0, np.zeros(self.K)
    
    def _update_producer(self, iter: int, population: np.ndarray) -> np.ndarray:
        """
        更新生产者位置
        """
        pd_num = int(self.n_sparrows * self.pd)
        R = np.random.rand(pd_num)
        updated_pop = population[:pd_num].copy()
        
        for i in range(pd_num):
            if R[i] < self.st:
                xi = np.random.rand()
                updated_pop[i] = population[i] * np.exp(-i / (xi * self.max_iter + 1e-10))
            else:
                Q = np.random.normal(0, 1, population.shape[1])
                updated_pop[i] = population[i] + Q * 30  # 增大步长以提高搜索效率
        
        # 边界处理
        for i in range(pd_num):
            uav_pos = self._decode_population(updated_pop[i:i+1])[0]
            uav_pos = np.clip(uav_pos, self.lb, self.ub)
            updated_pop[i] = uav_pos.flatten()
        
        return updated_pop
    
    def _update_scrounger(self, population: np.ndarray, best_position: np.ndarray,
                         worst_position: np.ndarray) -> np.ndarray:
        """
        更新跟随者位置
        """
        pd_num = int(self.n_sparrows * self.pd)
        n_scrounger = self.n_sparrows - pd_num
        updated_pop = population[pd_num:].copy()
        
        for i in range(n_scrounger):
            idx = pd_num + i
            if idx > self.n_sparrows / 2:
                Q = np.random.normal(0, 1, population.shape[1])
                # 修复Bug: 使用i索引updated_pop，使用(i+1)避免除零
                updated_pop[i] = Q * np.exp((worst_position - population[idx]) / ((i+1) ** 2))
            else:
                A = np.array([1 if np.random.random() > 0.5 else -1 
                             for _ in range(population.shape[1])])
                A_star = A / (np.dot(A, A) + 1e-12)
                updated_pop[i] = population[idx] + np.abs(population[idx] - best_position) * A_star
        
        # 边界处理
        for i in range(n_scrounger):
            uav_pos = self._decode_population(updated_pop[i:i+1])[0]
            uav_pos = np.clip(uav_pos, self.lb, self.ub)
            updated_pop[i] = uav_pos.flatten()
        
        return updated_pop
    
    def _update_watch(self, population: np.ndarray, best_position: np.ndarray,
                     fitness: np.ndarray, best_fitness: float, worst_fitness: float) -> np.ndarray:
        """
        更新警戒者位置
        """
        sd_num = int(self.n_sparrows * self.sd)
        watch_indices = np.random.choice(self.n_sparrows, sd_num, replace=False)
        updated_pop = population.copy()
        
        for idx in watch_indices:
            f_i = fitness[idx]
            if f_i > best_fitness:
                omega = np.random.normal(0, 1, population.shape[1])
                updated_pop[idx] = best_position + omega * np.abs(population[idx] - best_position)
            else:
                K = np.random.uniform(-1, 1, population.shape[1])
                epsilon = 1e-10
                worst_pos = population[np.argmin(fitness)]
                updated_pop[idx] = population[idx] + K * (
                    np.abs(population[idx] - worst_pos) / (f_i - worst_fitness + epsilon)
                )
        
        # 边界处理
        for idx in watch_indices:
            uav_pos = self._decode_population(updated_pop[idx:idx+1])[0]
            uav_pos = np.clip(uav_pos, self.lb, self.ub)
            updated_pop[idx] = uav_pos.flatten()
        
        return updated_pop
    
    def _cauchy_gaussian_mutation(self, best_position: np.ndarray, iter: int, 
                                 fitness: np.ndarray, best_fitness: float) -> np.ndarray:
        """
        Cauchy-Gaussian变异策略
        """
        # 计算自适应方差
        if best_fitness > np.mean(fitness):
            sigma = 1.0
        else:
            f_mean = np.mean(fitness)
            sigma = np.exp(np.abs(best_fitness - f_mean) / (np.abs(best_fitness) + 1e-10))
        
        # Cauchy-Gaussian变异
        t = iter
        t_max = self.max_iter
        t_ratio = (t / (t_max + 1)) ** 2
        
        # Cauchy分布随机数
        cauchy_term = np.random.standard_cauchy(len(best_position)) * sigma * 5
        
        # 高斯分布随机数
        gaussian_term = np.random.normal(0, sigma, len(best_position)) * 10
        
        # 变异位置
        mutated_position = (best_position * (1 + (1 - t_ratio) * cauchy_term * 0.1) + 
                          t_ratio * gaussian_term * 0.1)
        
        # 边界处理
        uav_pos = self._decode_population(mutated_position.reshape(1, -1))[0]
        uav_pos = np.clip(uav_pos, self.lb, self.ub)
        mutated_position = uav_pos.flatten()
        
        return best_position
    
    def optimize(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray,
                UAV_pos: np.ndarray) -> Dict:
        """
        执行ISSA优化
        
        Parameters:
        -----------
        UE_pos : np.ndarray
            UE位置
        ground_AP_pos : np.ndarray
            地面AP位置
        UAV_pos : np.ndarray
            初始UAV位置
            
        Returns:
        --------
        Dict
            优化结果
        """
        print("开始ISSA优化（基于BVF信道模型）...")
        
        # 初始化历史记录
        history = {
            'iterations': [],
            'sum_rates': [],
            'min_rates': [],
            'best_fitness': []
        }
        
        # 种群维度：每个个体编码所有L个UAV的位置
        dim = self.L * 3
        
        # 混沌初始化
        population = self._chaotic_initialization(self.n_sparrows, dim)
        
        # 计算初始适应度
        fitness = np.zeros(self.n_sparrows)
        sum_rates = np.zeros(self.n_sparrows)
        min_rates = np.zeros(self.n_sparrows)
        
        for i in range(self.n_sparrows):
            uav_pos = self._decode_population(population[i:i+1])[0]
            fit, sum_rate, rates = self._compute_fitness(uav_pos, UE_pos, ground_AP_pos)
            fitness[i] = fit
            sum_rates[i] = sum_rate
            min_rates[i] = rates.min()
        
        # 初始化最优和最差
        best_idx = np.argmax(fitness)
        worst_idx = np.argmin(fitness)
        best_position = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        worst_position = population[worst_idx].copy()
        worst_fitness = fitness[worst_idx]
        
        best_min_rate = min_rates[best_idx]
        best_sum_rate = sum_rates[best_idx]
        best_UAV_pos = self._decode_population(best_position.reshape(1, -1))[0]
        
        start_time = time.time()
        
        # 迭代优化
        for iter in range(self.max_iter):
            iter_start_time = time.time()
            
            # 按适应度排序
            sorted_indices = np.argsort(fitness)[::-1]
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            
            # 更新最优和最差
            best_position = population[0].copy()
            best_fitness = fitness[0]
            worst_position = population[-1].copy()
            worst_fitness = fitness[-1]
            
            # 1. 更新生产者
            pd_num = int(self.n_sparrows * self.pd)
            producer_pop = self._update_producer(iter, population)
            population[:pd_num] = producer_pop
            
            # 重新计算生产者适应度
            for i in range(pd_num):
                uav_pos = self._decode_population(population[i:i+1])[0]
                fit, sum_rate, rates = self._compute_fitness(uav_pos, UE_pos, ground_AP_pos)
                fitness[i] = fit
                sum_rates[i] = sum_rate
                min_rates[i] = rates.min()
            
            # 2. 更新跟随者
            scrounger_pop = self._update_scrounger(population, best_position, worst_position)
            population[pd_num:] = scrounger_pop
            
            # 重新计算跟随者适应度
            for i in range(pd_num, self.n_sparrows):
                uav_pos = self._decode_population(population[i:i+1])[0]
                fit, sum_rate, rates = self._compute_fitness(uav_pos, UE_pos, ground_AP_pos)
                fitness[i] = fit
                sum_rates[i] = sum_rate
                min_rates[i] = rates.min()
            
            # 3. 更新警戒者
            watch_pop = self._update_watch(population, best_position, fitness, 
                                          best_fitness, worst_fitness)
            population = watch_pop
            
            # 重新计算警戒者适应度
            for i in range(self.n_sparrows):
                uav_pos = self._decode_population(population[i:i+1])[0]
                fit, sum_rate, rates = self._compute_fitness(uav_pos, UE_pos, ground_AP_pos)
                fitness[i] = fit
                sum_rates[i] = sum_rate
                min_rates[i] = rates.min()
            
            # 4. Cauchy-Gaussian变异
            mutated_position = self._cauchy_gaussian_mutation(best_position, iter, 
                                                              fitness, best_fitness)
            uav_pos_mutated = self._decode_population(mutated_position.reshape(1, -1))[0]
            mutated_fitness, mutated_sum_rate, mutated_rates = self._compute_fitness(
                uav_pos_mutated, UE_pos, ground_AP_pos)
            
            # 5. 更新最优解
            if mutated_fitness > best_fitness:
                best_position = mutated_position.copy()
                best_fitness = mutated_fitness
                best_min_rate = mutated_rates.min()
                best_sum_rate = mutated_sum_rate
                best_UAV_pos = uav_pos_mutated.copy()
                
                # 替换最差个体
                worst_idx = np.argmin(fitness)
                population[worst_idx] = mutated_position.copy()
                fitness[worst_idx] = mutated_fitness
            
            # 更新全局最优
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > best_fitness:
                best_position = population[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]
                best_min_rate = min_rates[current_best_idx]
                best_sum_rate = sum_rates[current_best_idx]
                best_UAV_pos = self._decode_population(best_position.reshape(1, -1))[0]
            
            # 记录历史
            history['iterations'].append(iter)
            history['sum_rates'].append(best_sum_rate)
            history['min_rates'].append(best_min_rate)
            history['best_fitness'].append(best_fitness)
            
            iter_time = time.time() - iter_start_time
            
            if (iter + 1) % 10 == 0:
                print(f"Iter {iter+1}/{self.max_iter}: Sum={best_sum_rate:.2f}Mbps, "
                      f"Min={best_min_rate:.4f}Mbps, Time={iter_time:.3f}s")
        
        optimization_time = time.time() - start_time
        
        # 计算最终结果
        final_rates, final_sum_rate = self.bvf_optimizer.compute_user_rates(
            UE_pos, np.vstack([ground_AP_pos, best_UAV_pos]), 
            self.bvf_optimizer.compute_AP_selection_mask(
                self.bvf_optimizer.compute_channel_model(
                    UE_pos, np.vstack([ground_AP_pos, best_UAV_pos]))[2]))
        final_min_rate = final_rates.min()
        
        results = {
            'optimized_UAV_pos': best_UAV_pos,
            'final_sum_rate': final_sum_rate,
            'final_min_rate': final_min_rate,
            'final_rates': final_rates,
            'optimization_time': optimization_time,
            'total_iterations': self.max_iter,
            'history': history
        }
        
        print(f"\n🎉 ISSA优化完成!")
        print(f"📊 最终总速率: {final_sum_rate:.2f} Mbps")
        print(f"🎯 最终最小速率: {final_min_rate:.4f} Mbps")
        print(f"⏱️  优化时间: {optimization_time:.2f} 秒")
        
        return results

