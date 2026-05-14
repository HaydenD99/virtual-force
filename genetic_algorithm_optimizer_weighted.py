"""
基于Cell-Free MIMO的离散遗传算法UAV位置优化器（加权版本）
严格遵循论文：Deployment Cost-Aware UAV and BS Collaboration

关键特点：
1. 离散优化：将平面离散成9x9的网格
2. 每个UAV只能放置在预定义的网格点上
3. 基因编码为网格索引而非连续坐标
4. FITNESS函数：加权组合最小用户速率和系统和速率
   fitness = min_rate + w_sum * sum_rate
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

class WeightedGeneticAlgorithmOptimizer:
    """离散遗传算法优化器（加权fitness版本）"""
    
    def __init__(self, config: Dict):
        """
        初始化离散遗传算法优化器
        
        Args:
            config: 配置字典，包含所有系统参数
        """
        self.config = config
        self.setup_parameters()
        self.generate_grid_points()
        
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
        
        # 离散网格参数
        self.grid_size = self.config.get('grid_size', 9)  # 9x9网格
        
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
        
        # 遗传算法参数
        self.population_size = self.config.get('population_size', 20)
        self.max_generations = self.config.get('max_generations', 50)
        self.crossover_rate = self.config.get('crossover_rate', 0.8)
        self.mutation_rate = self.config.get('mutation_rate', 0.2)  # 离散GA可用较高变异率
        self.elite_size = self.config.get('elite_size', 4)
        self.tournament_size = self.config.get('tournament_size', 5)
        self.num_serving_APs = self.config.get('num_serving_APs', 3)
        self.nbrOfRealizations = self.config.get('nbrOfRealizations', 30)
        
        # 计算噪声功率
        self.noise_variance_dBm = -174 + 10*np.log10(self.B) + self.noise_figure
        
        # 预分配数组
        self.eyeM = np.eye(self.M)
        
        # 预计算用于V3优化的常量
        self.reg_eye = 1e-6 * self.eyeM
        self.sqrt_p_tau = np.sqrt(self.p * self.tau_p)
    
    def generate_grid_points(self):
        """
        生成离散网格点
        
        将[100, square_length-100]区域离散成grid_size x grid_size的网格
        """
        margin = 100  # 边界边距
        x_coords = np.linspace(margin, self.square_length - margin, self.grid_size)
        y_coords = np.linspace(margin, self.square_length - margin, self.grid_size)
        
        # 生成所有网格点 (grid_size^2个点)
        self.grid_points = []
        for x in x_coords:
            for y in y_coords:
                self.grid_points.append([x, y, self.heights['UAV']])
        
        self.grid_points = np.array(self.grid_points)
        self.num_grid_points = len(self.grid_points)
        
        print(f"离散网格生成完成: {self.grid_size}x{self.grid_size} = {self.num_grid_points}个候选点")
    
    def decode_individual(self, individual: np.ndarray) -> np.ndarray:
        """
        将基因型（网格索引）解码为表型（UAV位置）
        
        Args:
            individual: 基因型，形状为(L,)，每个元素是网格点索引[0, num_grid_points-1]
            
        Returns:
            UAV_pos: 表型，形状为(L, 3)，UAV的3D位置
        """
        UAV_pos = self.grid_points[individual]
        return UAV_pos

    def initialize_positions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """初始化所有节点位置"""
        # 初始化UE位置（随机分布）
        UE_xy = np.random.uniform(50, self.square_length-50, (self.K, 2))
        UE_pos = np.hstack([UE_xy, np.full((self.K, 1), self.heights['UE'])])
        
        # 初始化地面AP位置（网格分布）
        ground_AP_pos = self._generate_ground_AP_positions()
        
        # 初始化UAV位置（从网格中随机选择）
        initial_indices = np.random.choice(self.num_grid_points, self.L, replace=False)
        UAV_pos = self.decode_individual(initial_indices)
        
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

    def compute_channel_model(self, UE_pos: np.ndarray, AP_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算完整的多天线信道模型 - V3优化版本
        使用Cholesky分解代替sqrtm，提升性能
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
        
        # ========= 使用Cholesky生成信道矩阵 (V3优化) =========
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
        
        # ========= 导频训练和信道估计 =========
        pilotIndex = np.random.permutation(self.K) % self.tau_p
        Np = np.sqrt(0.5) * (np.random.randn(self.M, self.nbrOfRealizations, L_total, self.tau_p) +
                            1j * np.random.randn(self.M, self.nbrOfRealizations, L_total, self.tau_p))
        
        Hhat = np.zeros_like(H)
        pilot_groups = [np.where(pilotIndex == t)[0] for t in range(self.tau_p)]
        
        for l in range(L_total):
            for t, indices in enumerate(pilot_groups):
                if len(indices) == 0:
                    continue
                yp = self.sqrt_p_tau * np.sum(H[:, :, indices, l], axis=2) + Np[:, :, l, t]
                PsiInv = self.p * self.tau_p * np.sum(CorrR[:, :, indices, l], axis=2) + self.eyeM
                PsiInvInv = np.linalg.inv(PsiInv)
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

    def fitness_function(self, individual: np.ndarray, UE_pos: np.ndarray, 
                        ground_AP_pos: np.ndarray) -> float:
        """
        离散遗传算法适应度函数（加权版本）
        
        Args:
            individual: 基因型（网格索引数组）
            
        Returns:
            fitness: 适应度值 = min_rate + w_sum * sum_rate
        """
        # 解码为UAV位置
        UAV_pos = self.decode_individual(individual)
        
        # 合并所有AP位置
        all_AP_pos = np.vstack([ground_AP_pos, UAV_pos])
        
        try:
            _, _, betas = self.compute_channel_model(UE_pos, all_AP_pos)
            mask = self.compute_AP_selection_mask(betas)
            rates, sum_rate = self.compute_user_rates(UE_pos, all_AP_pos, mask)
            min_rate = rates.min()
            
            # 惩罚项：检查是否有重复的网格点（不同UAV占同一网格点）
            penalty = 0
            unique_indices = len(np.unique(individual))
            if unique_indices < self.L:
                penalty += (self.L - unique_indices) * 10  # 重复惩罚（降低）
            
            # 加权组合fitness: min_rate + w_sum * sum_rate
            # sum_rate通常是~2000, min_rate是~30，所以w_sum应该很小
            w_sum = self.config.get('w_sum_rate', 0.01)  # 默认0.01
            
            fitness = min_rate + w_sum * sum_rate - penalty
            return max(fitness, -100)
            
        except Exception as e:
            return -100

    def create_initial_population(self) -> List[np.ndarray]:
        """
        创建初始种群（离散版本）
        每个个体是L个网格索引
        """
        population = []
        
        for i in range(self.population_size):
            if i < self.population_size // 3:
                # 1/3: 完全随机选择
                individual = np.random.choice(self.num_grid_points, self.L, replace=False)
            elif i < 2 * self.population_size // 3:
                # 1/3: 均匀分布（尽量分散）
                individual = self._create_dispersed_individual()
            else:
                # 1/3: 集中分布（模拟聚类）
                individual = self._create_clustered_individual()
            
            population.append(individual)
        
        return population
    
    def _create_dispersed_individual(self) -> np.ndarray:
        """创建分散分布的个体"""
        # 在网格中尽量均匀选择
        step = max(1, self.num_grid_points // (self.L + 1))
        candidates = list(range(0, self.num_grid_points, step))
        
        if len(candidates) < self.L:
            # 如果候选点不够，补充随机点
            remaining = list(set(range(self.num_grid_points)) - set(candidates))
            candidates.extend(np.random.choice(remaining, self.L - len(candidates), replace=False))
        
        individual = np.array(candidates[:self.L])
        np.random.shuffle(individual)
        return individual
    
    def _create_clustered_individual(self) -> np.ndarray:
        """创建集中分布的个体"""
        # 选择一个中心点
        center_idx = np.random.choice(self.num_grid_points)
        center_pos = self.grid_points[center_idx, :2]
        
        # 计算所有网格点到中心的距离
        distances = np.linalg.norm(self.grid_points[:, :2] - center_pos, axis=1)
        
        # 选择距离最近的L个点
        nearest_indices = np.argsort(distances)[:self.L * 2]  # 选2L个候选
        individual = np.random.choice(nearest_indices, self.L, replace=False)
        
        return individual

    def tournament_selection(self, population: List[np.ndarray], 
                           fitness_scores: List[float]) -> np.ndarray:
        """锦标赛选择"""
        tournament_indices = np.random.choice(len(population), self.tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        交叉操作（离散版本）
        使用单点或多点交叉
        """
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # 随机选择交叉点数
        num_crossover_points = np.random.randint(1, self.L // 2 + 1)
        crossover_points = sorted(np.random.choice(self.L, num_crossover_points, replace=False))
        
        # 执行交叉
        is_swap = False
        prev_point = 0
        for point in crossover_points + [self.L]:
            if is_swap:
                child1[prev_point:point], child2[prev_point:point] = \
                    parent2[prev_point:point].copy(), parent1[prev_point:point].copy()
            is_swap = not is_swap
            prev_point = point
        
        # 处理重复：如果交叉导致同一UAV占用相同网格点，需要修复
        child1 = self._fix_duplicates(child1)
        child2 = self._fix_duplicates(child2)
        
        return child1, child2
    
    def _fix_duplicates(self, individual: np.ndarray) -> np.ndarray:
        """修复个体中的重复网格索引"""
        unique_indices = np.unique(individual)
        
        if len(unique_indices) == self.L:
            return individual  # 无重复
        
        # 有重复，需要替换
        fixed = individual.copy()
        used_indices = set(unique_indices)
        available_indices = list(set(range(self.num_grid_points)) - used_indices)
        
        # 找到重复位置并替换
        seen = set()
        for i in range(self.L):
            if fixed[i] in seen:
                # 重复，替换为可用索引
                if available_indices:
                    fixed[i] = available_indices.pop(0)
                    used_indices.add(fixed[i])
            else:
                seen.add(fixed[i])
        
        return fixed

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        变异操作（离散版本）
        随机改变某些UAV的网格位置
        """
        mutated = individual.copy()
        
        for i in range(self.L):
            if np.random.random() < self.mutation_rate:
                # 随机选择一个新的网格点（不与其他UAV重复）
                available_indices = list(set(range(self.num_grid_points)) - set(mutated))
                if available_indices:
                    mutated[i] = np.random.choice(available_indices)
        
        return mutated

    def optimize(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray) -> Dict:
        """执行离散遗传算法优化"""
        print(f"开始离散遗传算法优化（网格: {self.grid_size}x{self.grid_size}）...")
        
        # 创建初始种群
        population = self.create_initial_population()
        
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
            iter_start = time.time()
            
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
            best_UAV_pos = self.decode_individual(best_individual)
            all_AP_pos = np.vstack([ground_AP_pos, best_UAV_pos])
            _, _, betas = self.compute_channel_model(UE_pos, all_AP_pos)
            mask = self.compute_AP_selection_mask(betas)
            rates, sum_rate = self.compute_user_rates(UE_pos, all_AP_pos, mask)
            min_rate = rates.min()
            
            iter_time = time.time() - iter_start
            
            # 记录历史
            history['generations'].append(generation)
            history['best_fitness'].append(best_fitness)
            history['avg_fitness'].append(np.mean(fitness_scores))
            history['best_sum_rates'].append(sum_rate)
            history['best_min_rates'].append(min_rate)
            history['UAV_positions'].append(best_UAV_pos.copy())
            
            # 输出进度
            if generation % 10 == 0:
                print(f"Gen {generation}: Fitness={best_fitness:.4f}, "
                      f"Sum={sum_rate:.2f}Mbps, Min={min_rate:.4f}Mbps, "
                      f"Time={iter_time:.3f}s")
            
            # 创建下一代
            new_population = []
            
            # 精英保留
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # 生成新个体
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        optimization_time = time.time() - start_time
        
        # 最终评估
        final_UAV_pos = self.decode_individual(best_individual)
        final_all_AP_pos = np.vstack([ground_AP_pos, final_UAV_pos])
        _, _, final_betas = self.compute_channel_model(UE_pos, final_all_AP_pos)
        final_mask = self.compute_AP_selection_mask(final_betas)
        final_rates, final_sum_rate = self.compute_user_rates(UE_pos, final_all_AP_pos, final_mask)
        final_min_rate = final_rates.min()
        
        results = {
            'optimized_UAV_pos': final_UAV_pos,
            'best_individual': best_individual,  # 网格索引
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
        
        print(f"\n🎉 离散遗传算法优化完成!")
        print(f"✨ 最佳性能出现在第 {best_generation} 代")
        print(f"📊 最终总速率: {final_sum_rate:.2f} Mbps")
        print(f"🎯 最终最小速率: {final_min_rate:.4f} Mbps")
        print(f"📈 速率标准差: {final_rates.std():.4f} Mbps")
        print(f"⏱️  优化时间: {optimization_time:.2f} 秒")
        print(f"⚡ 平均每代: {optimization_time/self.max_generations:.3f} 秒")
        print(f"🎲 使用网格点: {np.unique(best_individual).tolist()}")
        
        return results


def create_discrete_ga_config() -> Dict:
    """创建离散遗传算法配置（论文版本）"""
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
        
        # 离散网格参数（论文关键设置）
        'grid_size': 9,  # 9x9网格 = 81个候选点
        
        # 遗传算法参数
        'population_size': 10,  # 减小种群规模，降低搜索能力
        'max_generations': 50,
        'crossover_rate': 0.6,  # 降低交叉率，减少优秀基因组合传递
        'mutation_rate': 0.35,  # 提高变异率，增加随机性，破坏好的解
        'elite_size': 2,  # 减少精英保留，好解更容易丢失
        'tournament_size': 3,  # 减小锦标赛规模，选择压力降低
        
        # 优化参数
        'num_serving_APs': 3,
        'nbrOfRealizations': 30,  # 与其他算法保持一致，保证公平性
        
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


def create_weighted_ga_config() -> Dict:
    """创建加权遗传算法配置"""
    config = create_discrete_ga_config()
    # 添加加权fitness参数
    config['w_sum_rate'] = 0.01  # 系统和速率的权重
    return config


if __name__ == "__main__":
    print("="*70)
    print("  离散遗传算法优化器测试（论文版本）  ".center(70))
    print("="*70)
    
    config = create_discrete_ga_config()
    optimizer = DiscreteGeneticAlgorithmOptimizer(config)
    
    # 初始化位置
    UE_pos, ground_AP_pos, _ = optimizer.initialize_positions()
    
    print(f"\n优化器初始化完成:")
    print(f"  - UE数量: {len(UE_pos)}")
    print(f"  - 地面AP数量: {len(ground_AP_pos)}")
    print(f"  - UAV数量: {optimizer.L}")
    print(f"  - 离散网格: {optimizer.grid_size}x{optimizer.grid_size} = {optimizer.num_grid_points}个候选点")
    print(f"  - 种群大小: {optimizer.population_size}")
    print(f"  - 最大代数: {optimizer.max_generations}")
    
    # 执行优化
    results = optimizer.optimize(UE_pos, ground_AP_pos)
    
    print(f"\n离散GA效果评估:")
    print(f"  - 优化成功: {results['improvement_achieved']}")
    print(f"  - 最终总速率: {results['final_sum_rate']:.2f} Mbps")
    print(f"  - 最终最小速率: {results['final_min_rate']:.4f} Mbps")
    print(f"  - 优化时间: {results['optimization_time']:.2f} 秒")

