"""
基于Cell-Free MIMO的遗传算法UAV位置优化器
用于与虚拟力算法进行性能对比
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

class GeneticAlgorithmOptimizer:
    """基于Cell-Free MIMO的遗传算法优化器"""
    
    def __init__(self, config: Dict):
        """
        初始化遗传算法优化器
        
        Args:
            config: 配置字典，包含所有系统参数
        """
        self.config = config
        self.setup_parameters()
        
    def setup_parameters(self):
        """设置系统参数"""
        # 基本参数
        self.square_length = self.config.get('square_length', 1000)
        self.K = self.config.get('num_UE', 60)  # UE数量
        self.L = self.config.get('num_UAV', 9)  # UAV数量  
        self.G = self.config.get('num_ground_AP', 4)  # 地面AP数量
        self.M = self.config.get('M', 4)  # 每个AP的天线数
        
        # 高度设置
        self.heights = {
            'UE': self.config.get('UE_height', 1.65),
            'ground_AP': self.config.get('ground_AP_height', 15.0),
            'UAV': self.config.get('UAV_height', 50.0)
        }
        
        # 信道参数
        self.alpha = self.config.get('alpha', 3.67)  # 路径损耗指数
        self.constant_term = self.config.get('constant_term', -30.5)  # 参考损耗
        self.sigma_sf = self.config.get('sigma_sf', 1)  # 阴影衰落标准差
        self.antenna_spacing = self.config.get('antenna_spacing', 0.5)  # 天线间距
        self.ASD_deg = self.config.get('ASD_deg', 10)  # 角度扩散
        
        # 通信参数
        self.B = self.config.get('B', 20e6)  # 带宽
        self.p = self.config.get('p', 100)  # 上行功率(mW)
        self.Pmax = self.config.get('Pmax', 1000)  # 下行功率(mW)
        self.noise_figure = self.config.get('noise_figure', 7)  # 噪声系数
        self.distance_vertical = self.config.get('distance_vertical', 150)  # 垂直距离
        
        # 导频参数
        self.tau_p = self.config.get('tau_p', self.K)  # 导频长度
        self.tau_c = self.config.get('tau_c', 200)  # 相干块长度
        self.prelogFactor = (self.tau_c - self.tau_p) / self.tau_c
        
        # 遗传算法参数
        self.population_size = self.config.get('population_size', 20)
        self.max_generations = self.config.get('max_generations', 100)
        self.crossover_rate = self.config.get('crossover_rate', 0.8)
        self.mutation_rate = self.config.get('mutation_rate', 0.15)
        self.elite_size = self.config.get('elite_size', 10)
        self.tournament_size = self.config.get('tournament_size', 5)
        self.num_serving_APs = self.config.get('num_serving_APs', 3)
        self.nbrOfRealizations = self.config.get('nbrOfRealizations', 50)
        
        # 计算噪声功率
        self.noise_variance_dBm = -174 + 10*np.log10(self.B) + self.noise_figure
        
        # 预分配数组
        self.eyeM = np.eye(self.M)

    def initialize_positions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """初始化所有节点位置"""
        # 初始化UE位置（随机分布）
        UE_xy = np.random.uniform(50, self.square_length-50, (self.K, 2))
        UE_pos = np.hstack([
            UE_xy,
            np.full((self.K, 1), self.heights['UE'])
        ])
        
        # 初始化地面AP位置（网格分布）
        ground_AP_pos = self._generate_ground_AP_positions()
        
        # 初始化UAV位置（随机分布）
        UAV_pos = self._generate_initial_UAV_positions()
        
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
    
    def _generate_initial_UAV_positions(self) -> np.ndarray:
        """随机生成UAV初始位置"""
        UAV_xy = np.random.uniform(100, self.square_length-100, (self.L, 2))
        UAV_pos = np.hstack([
            UAV_xy,
            np.full((self.L, 1), self.heights['UAV'])
        ])
        return UAV_pos

    def compute_channel_model(self, UE_pos: np.ndarray, AP_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算完整的多天线信道模型"""
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
        
        # ========= 生成信道矩阵 =========
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
                    
                    cond_num = np.linalg.cond(corr_matrix)
                    if cond_num > 1e12:
                        corr_matrix += 1e-6 * np.eye(self.M)
                    
                    Rsqrt = sl.sqrtm(corr_matrix)
                    
                    if not np.isfinite(Rsqrt).all():
                        try:
                            Rsqrt = np.linalg.cholesky(corr_matrix + 1e-6 * np.eye(self.M))
                        except np.linalg.LinAlgError:
                            Rsqrt = np.sqrt(np.abs(betas[k, l])) * np.eye(self.M)
                    
                    H[:, :, k, l] = Rsqrt @ CH[:, :, k, l]
                    
                except (np.linalg.LinAlgError, ValueError):
                    scaling = np.sqrt(np.abs(betas[k, l]) + 1e-12)
                    H[:, :, k, l] = scaling * CH[:, :, k, l]
        
        # ========= 导频训练和信道估计 =========
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
        
        # 计算完整信道模型
        H, Hhat, betas = self.compute_channel_model(UE_pos, AP_pos)
        
        # 应用AP选择掩码
        Hhat_uc = Hhat * mask[np.newaxis, np.newaxis, :, :]
        
        # 功率分配
        num_served_per_AP = mask.sum(axis=0)
        rho = np.zeros((self.K, L_total))
        for l in range(L_total):
            if num_served_per_AP[l] > 0:
                rho[mask[:, l], l] = self.Pmax / num_served_per_AP[l]
        gamma = np.sqrt(rho)
        
        # 计算下行速率
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

    def fitness_function(self, UAV_pos: np.ndarray, UE_pos: np.ndarray, 
                        ground_AP_pos: np.ndarray) -> float:
        """
        遗传算法适应度函数
        专注于最小速率优化（Max-Min Fairness）
        """
        # 合并所有AP位置
        all_AP_pos = np.vstack([ground_AP_pos, UAV_pos])
        
        # 计算性能
        try:
            _, _, betas = self.compute_channel_model(UE_pos, all_AP_pos)
            mask = self.compute_AP_selection_mask(betas)
            rates, sum_rate = self.compute_user_rates(UE_pos, all_AP_pos, mask)
            min_rate = rates.min()
            
            # 检查UAV间距离约束
            penalty = 0
            for i in range(self.L):
                for j in range(i+1, self.L):
                    dist = np.linalg.norm(UAV_pos[i, :2] - UAV_pos[j, :2])
                    if dist < 50:  # 最小间距50m
                        penalty += (50 - dist) * 0.1
            
            # 检查边界约束
            for i in range(self.L):
                x, y = UAV_pos[i, 0], UAV_pos[i, 1]
                if x < 50 or x > self.square_length - 50 or y < 50 or y > self.square_length - 50:
                    penalty += 10
            
            # 只针对最小速率优化
            fitness = min_rate * 10 - penalty
            return max(fitness, -100)  # 避免过度负值
            
        except Exception as e:
            return -100  # 计算失败时返回很低的适应度

    def create_initial_population(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray) -> List[np.ndarray]:
        """创建初始种群"""
        population = []
        
        # 生成多样化的初始解
        for i in range(self.population_size):
            if i < self.population_size // 4:
                # 25%随机分布
                UAV_pos = self._generate_initial_UAV_positions()
            elif i < self.population_size // 2:
                # 25%网格分布
                UAV_pos = self._generate_grid_UAV_positions()
            elif i < 3 * self.population_size // 4:
                # 25%基于用户密度分布
                UAV_pos = self._generate_density_based_UAV_positions(UE_pos)
            else:
                # 25%边界分布
                UAV_pos = self._generate_boundary_UAV_positions()
            
            population.append(UAV_pos)
        
        return population
    
    def _generate_grid_UAV_positions(self) -> np.ndarray:
        """网格分布UAV位置"""
        grid_size = int(np.ceil(np.sqrt(self.L)))
        spacing = (self.square_length - 200) / (grid_size + 1)
        
        positions = []
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= self.L:
                    break
                x = 100 + (i + 1) * spacing + np.random.normal(0, 30)
                y = 100 + (j + 1) * spacing + np.random.normal(0, 30)
                x = np.clip(x, 100, self.square_length - 100)
                y = np.clip(y, 100, self.square_length - 100)
                positions.append([x, y, self.heights['UAV']])
                count += 1
        
        # 如果数量不足，随机补充
        while len(positions) < self.L:
            x = np.random.uniform(100, self.square_length - 100)
            y = np.random.uniform(100, self.square_length - 100)
            positions.append([x, y, self.heights['UAV']])
        
        return np.array(positions[:self.L])
    
    def _generate_density_based_UAV_positions(self, UE_pos: np.ndarray) -> np.ndarray:
        """基于用户密度分布UAV位置（使用简单K-means实现）"""
        
        # 简单的K-means实现
        centers = self._simple_kmeans(UE_pos[:, :2], self.L)
        
        positions = []
        for i in range(self.L):
            center = centers[i]
            # 在聚类中心附近添加随机偏移
            x = center[0] + np.random.normal(0, 50)
            y = center[1] + np.random.normal(0, 50)
            x = np.clip(x, 100, self.square_length - 100)
            y = np.clip(y, 100, self.square_length - 100)
            positions.append([x, y, self.heights['UAV']])
        
        return np.array(positions)
    
    def _simple_kmeans(self, points: np.ndarray, k: int, max_iters: int = 20) -> np.ndarray:
        """简单的K-means聚类实现"""
        n_points = len(points)
        
        # 随机初始化中心点
        centers = points[np.random.choice(n_points, k, replace=False)]
        
        for _ in range(max_iters):
            # 分配点到最近的中心
            distances = np.sqrt(((points[:, np.newaxis] - centers) ** 2).sum(axis=2))
            assignments = np.argmin(distances, axis=1)
            
            # 更新中心点
            new_centers = []
            for i in range(k):
                cluster_points = points[assignments == i]
                if len(cluster_points) > 0:
                    new_centers.append(cluster_points.mean(axis=0))
                else:
                    new_centers.append(centers[i])  # 保持原中心
            
            new_centers = np.array(new_centers)
            
            # 检查收敛
            if np.allclose(centers, new_centers):
                break
                
            centers = new_centers
        
        return centers
    
    def _generate_boundary_UAV_positions(self) -> np.ndarray:
        """边界分布UAV位置"""
        positions = []
        perimeter = 4 * (self.square_length - 200)
        
        for i in range(self.L):
            # 沿边界均匀分布
            pos = (i / self.L) * perimeter
            
            if pos < self.square_length - 200:  # 下边界
                x = 100 + pos
                y = 100 + np.random.uniform(0, 100)
            elif pos < 2 * (self.square_length - 200):  # 右边界
                x = self.square_length - 100 - np.random.uniform(0, 100)
                y = 100 + (pos - (self.square_length - 200))
            elif pos < 3 * (self.square_length - 200):  # 上边界
                x = self.square_length - 100 - (pos - 2 * (self.square_length - 200))
                y = self.square_length - 100 - np.random.uniform(0, 100)
            else:  # 左边界
                x = 100 + np.random.uniform(0, 100)
                y = self.square_length - 100 - (pos - 3 * (self.square_length - 200))
            
            positions.append([x, y, self.heights['UAV']])
        
        return np.array(positions)

    def tournament_selection(self, population: List[np.ndarray], fitness_scores: List[float]) -> np.ndarray:
        """锦标赛选择"""
        tournament_indices = np.random.choice(len(population), self.tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """交叉操作"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # 随机选择交叉点
        crossover_points = np.random.randint(0, self.L, self.L // 2)
        
        for point in crossover_points:
            # 交换UAV位置
            child1[point], child2[point] = parent2[point].copy(), parent1[point].copy()
        
        return child1, child2

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """变异操作"""
        mutated = individual.copy()
        
        for i in range(self.L):
            if np.random.random() < self.mutation_rate:
                # 高斯变异
                sigma = 50  # 变异强度
                mutated[i, 0] += np.random.normal(0, sigma)
                mutated[i, 1] += np.random.normal(0, sigma)
                
                # 边界约束
                mutated[i, 0] = np.clip(mutated[i, 0], 100, self.square_length - 100)
                mutated[i, 1] = np.clip(mutated[i, 1], 100, self.square_length - 100)
        
        return mutated

    def optimize(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray) -> Dict:
        """执行遗传算法优化"""
        print("开始遗传算法优化...")
        
        # 创建初始种群
        population = self.create_initial_population(UE_pos, ground_AP_pos)
        
        # 记录历史
        history = {
            'generations': [], 'best_fitness': [], 'avg_fitness': [],
            'best_sum_rates': [], 'best_min_rates': [], 'UAV_positions': []
        }
        
        best_fitness = -np.inf
        best_individual = None
        best_generation = 0
        
        start_time = time.time()
        
        for generation in range(self.max_generations):
            # 计算适应度
            fitness_scores = []
            for individual in population:
                fitness = self.fitness_function(individual, UE_pos, ground_AP_pos)
                fitness_scores.append(fitness)
            
            # 更新最佳个体
            current_best_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()
                best_generation = generation
            
            # 计算性能指标
            all_AP_pos = np.vstack([ground_AP_pos, best_individual])
            _, _, betas = self.compute_channel_model(UE_pos, all_AP_pos)
            mask = self.compute_AP_selection_mask(betas)
            rates, sum_rate = self.compute_user_rates(UE_pos, all_AP_pos, mask)
            min_rate = rates.min()
            
            # 记录历史
            history['generations'].append(generation)
            history['best_fitness'].append(best_fitness)
            history['avg_fitness'].append(np.mean(fitness_scores))
            history['best_sum_rates'].append(sum_rate)
            history['best_min_rates'].append(min_rate)
            history['UAV_positions'].append(best_individual.copy())
            
            # 输出进度
            if generation % 10 == 0:
                print(f"Gen {generation}: Best Fitness={best_fitness:.4f}, "
                      f"Sum Rate={sum_rate:.2f}Mbps, Min Rate={min_rate:.4f}Mbps")
            
            # 创建下一代
            new_population = []
            
            # 精英保留
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # 生成新个体
            while len(new_population) < self.population_size:
                # 选择父母
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                # 交叉
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # 变异
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # 保持种群大小
            population = new_population[:self.population_size]
        
        optimization_time = time.time() - start_time
        
        # 最终评估
        final_all_AP_pos = np.vstack([ground_AP_pos, best_individual])
        _, _, final_betas = self.compute_channel_model(UE_pos, final_all_AP_pos)
        final_mask = self.compute_AP_selection_mask(final_betas)
        final_rates, final_sum_rate = self.compute_user_rates(UE_pos, final_all_AP_pos, final_mask)
        final_min_rate = final_rates.min()
        
        results = {
            'optimized_UAV_pos': best_individual,
            'final_sum_rate': final_sum_rate,
            'final_min_rate': final_min_rate,
            'final_rates': final_rates,
            'optimization_time': optimization_time,
            'total_generations': self.max_generations,
            'best_generation': best_generation,
            'history': history,
            'final_mask': final_mask,
            'best_fitness': best_fitness,
            'improvement_achieved': final_min_rate > 0.1
        }
        
        print(f"\n🎉 遗传算法优化完成!")
        print(f"✨ 最佳性能出现在第 {best_generation} 代")
        print(f"📊 最终总速率: {final_sum_rate:.2f} Mbps")
        print(f"🎯 最终最小速率: {final_min_rate:.4f} Mbps")
        print(f"📈 速率标准差: {final_rates.std():.4f} Mbps")
        print(f"⏱️  优化时间: {optimization_time:.2f} 秒")
        
        return results


def create_ga_config() -> Dict:
    """创建遗传算法配置"""
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
        
        # 遗传算法参数
        'population_size': 20,      # 种群大小
        'max_generations': 50,     # 最大代数
        'crossover_rate': 0.8,      # 交叉率
        'mutation_rate': 0.15,      # 变异率
        'elite_size': 8,            # 精英个体数量
        'tournament_size': 5,       # 锦标赛大小
        
        # 优化参数
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
    }


if __name__ == "__main__":
    # 测试遗传算法优化器
    config = create_ga_config()
    optimizer = GeneticAlgorithmOptimizer(config)
    
    # 初始化位置
    UE_pos, ground_AP_pos, _ = optimizer.initialize_positions()
    
    print(f"遗传算法优化器初始化完成:")
    print(f"UE数量: {len(UE_pos)}")
    print(f"地面AP数量: {len(ground_AP_pos)}")
    print(f"种群大小: {optimizer.population_size}")
    print(f"最大代数: {optimizer.max_generations}")
    
    # 执行优化
    results = optimizer.optimize(UE_pos, ground_AP_pos)
    
    print(f"\n遗传算法效果评估:")
    print(f"优化成功: {results['improvement_achieved']}")
    print(f"最终性能:")
    print(f"  - 总速率: {results['final_sum_rate']:.2f} Mbps")
    print(f"  - 最小速率: {results['final_min_rate']:.4f} Mbps")
    print(f"  - 优化时间: {results['optimization_time']:.2f} 秒")
