"""
Particle Swarm Optimization (PSO) for UAV Positioning
基于粒子群优化的UAV位置优化算法
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Optional

# 导入必要的函数
import functionRlocalscattering
import SpectralEfficiencyDownlink
from scipy import linalg as sl

class PSOOptimizer:
    """粒子群优化算法用于UAV位置优化"""
    
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
        
        # PSO参数
        self.swarm_size = self.config.get('swarm_size', 30)
        self.max_iterations = self.config.get('max_iterations', 100)
        self.w = self.config.get('w', 0.7)  # 惯性权重
        self.c1 = self.config.get('c1', 2.0)  # 个体学习因子
        self.c2 = self.config.get('c2', 2.0)  # 社会学习因子
        self.v_max = self.config.get('v_max', 50)  # 最大速度
        
        # 其他参数
        self.num_serving_APs = self.config.get('num_serving_APs', 3)
        self.nbrOfRealizations = self.config.get('nbrOfRealizations', 50)
        
        # 计算噪声功率
        self.noise_variance_dBm = -174 + 10*np.log10(self.B) + self.noise_figure
        self.eyeM = np.eye(self.M)

    def initialize_positions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """初始化所有节点位置"""
        # 初始化UE位置（随机分布）
        UE_xy = np.random.uniform(50, self.square_length-50, (self.K, 2))
        UE_pos = np.hstack([UE_xy, np.full((self.K, 1), self.heights['UE'])])
        
        # 初始化地面AP位置（网格分布）
        ground_AP_pos = self._generate_ground_AP_positions()
        
        # 使用K-means初始化UAV位置
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
        # 使用K-means聚类用户位置
        user_centers = self._adaptive_clustering(UE_pos[:, :2], self.L)
        
        positions = []
        for i in range(self.L):
            # 在用户聚类中心附近添加随机偏移
            base_pos = user_centers[i]
            angle = np.random.uniform(0, 2*np.pi)
            radius = np.random.uniform(50, 150)
            x = base_pos[0] + radius * np.cos(angle)
            y = base_pos[1] + radius * np.sin(angle)
            
            # 确保在边界内
            x = np.clip(x, 100, self.square_length - 100)
            y = np.clip(y, 100, self.square_length - 100)
            
            positions.append([x, y, self.heights['UAV']])
        
        return np.array(positions)
    
    def _adaptive_clustering(self, points: np.ndarray, k: int) -> np.ndarray:
        """自适应K-means聚类"""
        n_points = len(points)
        
        # 初始化中心点
        centers = np.zeros((k, 2))
        
        # K-means++初始化
        centers[0] = points[np.random.choice(n_points)]
        
        for i in range(1, k):
            distances = np.min([np.sum((points - c)**2, axis=1) for c in centers[:i]], axis=0)
            probabilities = distances / (distances.sum() + 1e-12)
            centers[i] = points[np.random.choice(n_points, p=probabilities)]
        
        # 迭代优化
        for _ in range(15):
            distances = np.sqrt(((points[:, np.newaxis] - centers) ** 2).sum(axis=2))
            assignments = np.argmin(distances, axis=1)
            
            for i in range(k):
                cluster_points = points[assignments == i]
                if len(cluster_points) > 0:
                    centers[i] = cluster_points.mean(axis=0)
        
        return centers

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

    def fitness_function(self, UAV_positions: np.ndarray, UE_pos: np.ndarray, 
                        ground_AP_pos: np.ndarray) -> float:
        """PSO适应度函数 - 基础版本"""
        try:
            # 基础的UAV位置处理
            UAV_pos = UAV_positions.reshape(self.L, 3)
            UAV_pos[:, 0] = np.clip(UAV_pos[:, 0], 50, self.square_length - 50)
            UAV_pos[:, 1] = np.clip(UAV_pos[:, 1], 50, self.square_length - 50)
            UAV_pos[:, 2] = self.heights['UAV']
            
            # 合并所有AP位置
            all_AP_pos = np.vstack([ground_AP_pos, UAV_pos])
            
            # 计算性能
            _, _, betas = self.compute_channel_model(UE_pos, all_AP_pos)
            mask = self.compute_AP_selection_mask(betas)
            rates, _ = self.compute_user_rates(UE_pos, all_AP_pos, mask)
            
            # 简单的适应度函数
            min_rate = rates.min()
            fitness = min_rate * 80  # 减少权重
            
            return fitness
        
        except Exception as e:
            return -1e6

    def optimize(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray,
                UAV_pos: np.ndarray) -> Dict:
        """执行PSO优化"""
        print("开始PSO（粒子群优化）算法...")
        
        start_time = time.time()
        
        # 初始化粒子群
        # 每个粒子代表所有UAV的x,y坐标
        dim = self.L * 2  # 每个UAV的x,y坐标
        
        # 初始化粒子位置和速度
        particles = np.zeros((self.swarm_size, dim))
        velocities = np.zeros((self.swarm_size, dim))
        
        # 用提供的初始UAV位置初始化第一个粒子
        particles[0] = UAV_pos[:, :2].flatten()
        
        # 其余粒子随机初始化
        for i in range(1, self.swarm_size):
            particles[i] = np.random.uniform(100, self.square_length-100, dim)
        
        # 初始化速度
        velocities = np.random.uniform(-self.v_max/2, self.v_max/2, (self.swarm_size, dim))
        
        # 个体最佳位置和适应度
        pbest = particles.copy()
        pbest_fitness = np.full(self.swarm_size, -np.inf)
        
        # 全局最佳位置和适应度
        gbest = None
        gbest_fitness = -np.inf
        
        # 历史记录
        history = {
            'iterations': [], 'sum_rates': [], 'min_rates': [], 
            'best_fitness': [], 'UAV_positions': []
        }
        
        for iteration in range(self.max_iterations):
            
            # 评估每个粒子
            for i in range(self.swarm_size):
                fitness = self.fitness_function(particles[i], UE_pos, ground_AP_pos)
                
                # 更新个体最佳
                if fitness > pbest_fitness[i]:
                    pbest_fitness[i] = fitness
                    pbest[i] = particles[i].copy()
                
                # 更新全局最佳
                if fitness > gbest_fitness:
                    gbest_fitness = fitness
                    gbest = particles[i].copy()
            
            # 更新粒子速度和位置
            for i in range(self.swarm_size):
                r1 = np.random.random(dim)
                r2 = np.random.random(dim)
                
                # 速度更新公式
                velocities[i] = (self.w * velocities[i] + 
                               self.c1 * r1 * (pbest[i] - particles[i]) +
                               self.c2 * r2 * (gbest - particles[i]))
                
                # 限制速度
                velocities[i] = np.clip(velocities[i], -self.v_max, self.v_max)
                
                # 位置更新
                particles[i] += velocities[i]
                
                # 边界约束
                particles[i] = np.clip(particles[i], 50, self.square_length - 50)
            
            # 记录最佳粒子的性能
            if gbest is not None:
                best_UAV_pos = gbest.reshape(self.L, 2)
                best_UAV_pos = np.column_stack([best_UAV_pos, np.full(self.L, self.heights['UAV'])])
                
                all_AP_pos = np.vstack([ground_AP_pos, best_UAV_pos])
                _, _, betas = self.compute_channel_model(UE_pos, all_AP_pos)
                mask = self.compute_AP_selection_mask(betas)
                rates, sum_rate = self.compute_user_rates(UE_pos, all_AP_pos, mask)
                min_rate = rates.min()
                
                history['iterations'].append(iteration)
                history['sum_rates'].append(sum_rate)
                history['min_rates'].append(min_rate)
                history['best_fitness'].append(gbest_fitness)
                history['UAV_positions'].append(best_UAV_pos.copy())
            
            # 输出进度
            if iteration % 10 == 0:
                print(f"Iter {iteration}: Best Fitness={gbest_fitness:.4f}, "
                      f"Min Rate={min_rate:.4f}Mbps, Sum Rate={sum_rate:.2f}Mbps")
        
        optimization_time = time.time() - start_time
        
        # 最终结果
        final_UAV_pos = gbest.reshape(self.L, 2)
        final_UAV_pos = np.column_stack([final_UAV_pos, np.full(self.L, self.heights['UAV'])])
        
        final_all_AP_pos = np.vstack([ground_AP_pos, final_UAV_pos])
        _, _, final_betas = self.compute_channel_model(UE_pos, final_all_AP_pos)
        final_mask = self.compute_AP_selection_mask(final_betas)
        final_rates, final_sum_rate = self.compute_user_rates(UE_pos, final_all_AP_pos, final_mask)
        final_min_rate = final_rates.min()
        
        results = {
            'optimized_UAV_pos': final_UAV_pos,
            'final_sum_rate': final_sum_rate,
            'final_min_rate': final_min_rate,
            'final_rates': final_rates,
            'optimization_time': optimization_time,
            'total_iterations': self.max_iterations,
            'history': history,
            'final_mask': final_mask,
            'improvement_achieved': final_min_rate > 0.1,
            'best_fitness': gbest_fitness
        }
        
        print(f"\n🎉 PSO优化完成!")
        print(f"📊 最终总速率: {final_sum_rate:.2f} Mbps")
        print(f"🎯 最终最小速率: {final_min_rate:.4f} Mbps")
        print(f"📈 速率标准差: {final_rates.std():.4f} Mbps")
        print(f"⏱️  优化时间: {optimization_time:.2f} 秒")
        print(f"🏆 最佳适应度: {gbest_fitness:.4f}")
        
        return results


def create_pso_config() -> Dict:
    """创建PSO配置"""
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
        
        # PSO参数
        'swarm_size': 20,
        'max_iterations': 100,
        'w': 0.7,          # 惯性权重
        'c1': 2.0,         # 个体学习因子
        'c2': 2.0,         # 社会学习因子
        'v_max': 50,       # 最大速度
        
        # 其他参数
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
    # 测试PSO优化器
    config = create_pso_config()
    optimizer = PSOOptimizer(config)
    
    # 初始化位置
    UE_pos, ground_AP_pos, UAV_pos = optimizer.initialize_positions()
    
    print(f"PSO优化器初始化完成:")
    print(f"UE数量: {len(UE_pos)}")
    print(f"地面AP数量: {len(ground_AP_pos)}")
    print(f"UAV数量: {len(UAV_pos)}")
    
    # 执行优化
    results = optimizer.optimize(UE_pos, ground_AP_pos, UAV_pos)
    
    print(f"\nPSO效果评估:")
    print(f"优化成功: {results['improvement_achieved']}")
    print(f"最终性能:")
    print(f"  - 总速率: {results['final_sum_rate']:.2f} Mbps")
    print(f"  - 最小速率: {results['final_min_rate']:.4f} Mbps")
    print(f"  - 优化时间: {results['optimization_time']:.2f} 秒")
