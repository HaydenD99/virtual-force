"""
Distributed UAV Deployment Algorithm Based on PSO Algorithm
基于分布式粒子群优化的无人机部署算法

论文: A Distributed Approach for User Association and UAV Deployment 
      in QoE-Aware Multi-UAV Networks

实现论文中的Algorithm 2
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Optional
from scipy import linalg as sl

# 导入必要的函数
import functionRlocalscattering
import SpectralEfficiencyDownlink


class DistributedPSOOptimizer:
    """
    分布式粒子群优化算法用于UAV部署
    
    Algorithm 2: Distributed UAV Deployment Algorithm Based on PSO Algorithm
    - Step 1: Initialization
    - Step 2: Velocity and position Update  
    - Step 3: Fitness calculation
    - Step 4: Best position update
    """
    
    def __init__(self, config: Dict):
        """
        初始化分布式PSO优化器
        
        参数:
            config: 配置字典，包含所有系统参数
        """
        self.config = config
        self.setup_parameters()
        
    def setup_parameters(self):
        """设置系统参数"""
        # ========== 基本参数 ==========
        self.square_length = self.config.get('square_length', 1000)
        self.K = self.config.get('num_UE', 60)  # 用户数
        self.L = self.config.get('num_UAV', 9)  # UAV数量
        self.G = self.config.get('num_ground_AP', 4)  # 地面AP数量
        self.M = self.config.get('M', 4)  # 天线数
        
        # ========== 高度设置 ==========
        self.heights = {
            'UE': self.config.get('UE_height', 1.65),
            'ground_AP': self.config.get('ground_AP_height', 15.0),
            'UAV': self.config.get('UAV_height', 50.0)
        }
        
        # ========== PSO算法参数 ==========
        # 对应论文中的参数
        self.N_particle = self.config.get('N_particle', 30)  # 粒子数量
        self.max_iterations = self.config.get('max_iterations', 100)  # 最大迭代次数
        
        # PSO标准参数 (对应论文中的公式15)
        self.w = self.config.get('w', 0.729)  # 惯性权重 (inertia weight)
        self.c1 = self.config.get('c1', 1.49445)  # 认知学习因子 (cognitive learning factor)
        self.c2 = self.config.get('c2', 1.49445)  # 社会学习因子 (social learning factor)
        
        # 速度约束
        self.v_max = self.config.get('v_max', 50.0)  # 最大速度
        self.v_min = self.config.get('v_min', -50.0)  # 最小速度
        
        # 位置边界约束
        self.pos_min = self.config.get('pos_min', 50.0)
        self.pos_max = self.square_length - self.pos_min
        
        # ========== 信道参数 ==========
        self.alpha = self.config.get('alpha', 3.67)  # 路径损耗指数
        self.constant_term = self.config.get('constant_term', -30.5)  # 路径损耗常数项
        self.sigma_sf = self.config.get('sigma_sf', 1)  # 阴影衰落标准差
        self.antenna_spacing = self.config.get('antenna_spacing', 0.5)  # 天线间距
        self.ASD_deg = self.config.get('ASD_deg', 10)  # 角度扩展
        
        # ========== 通信参数 ==========
        self.B = self.config.get('B', 20e6)  # 带宽 (Hz)
        self.p = self.config.get('p', 100)  # 导频功率 (mW)
        self.Pmax = self.config.get('Pmax', 1000)  # 最大发射功率 (mW)
        self.noise_figure = self.config.get('noise_figure', 7)  # 噪声系数 (dB)
        self.distance_vertical = self.config.get('distance_vertical', 150)  # 垂直距离
        
        # ========== 导频参数 ==========
        self.tau_p = self.config.get('tau_p', self.K)  # 导频长度
        self.tau_c = self.config.get('tau_c', 200)  # 相干时间
        self.prelogFactor = (self.tau_c - self.tau_p) / self.tau_c  # 预对数因子
        
        # ========== 其他参数 ==========
        self.num_serving_APs = self.config.get('num_serving_APs', 3)  # 服务AP数量
        self.nbrOfRealizations = self.config.get('nbrOfRealizations', 50)  # 信道实现数
        
        # 计算噪声功率
        self.noise_variance_dBm = -174 + 10*np.log10(self.B) + self.noise_figure
        self.eyeM = np.eye(self.M)
        
        print(f"✅ 分布式PSO优化器初始化完成")
        print(f"   - 粒子数量: {self.N_particle}")
        print(f"   - 最大迭代次数: {self.max_iterations}")
        print(f"   - PSO参数: w={self.w}, c1={self.c1}, c2={self.c2}")

    # ========================================================================
    # Step 1: Initialization (初始化)
    # ========================================================================
    
    def initialize_positions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        初始化所有节点位置
        
        返回:
            UE_pos: 用户位置 [K, 3]
            ground_AP_pos: 地面AP位置 [G, 3]
            UAV_pos: UAV位置 [L, 3]
        """
        # 初始化UE位置（随机分布）
        UE_xy = np.random.uniform(self.pos_min, self.pos_max, (self.K, 2))
        UE_pos = np.hstack([UE_xy, np.full((self.K, 1), self.heights['UE'])])
        
        # 初始化地面AP位置（网格分布）
        ground_AP_pos = self._generate_ground_AP_positions()
        
        # 使用K-means聚类初始化UAV位置
        UAV_pos = self._smart_initialize_UAVs(UE_pos, ground_AP_pos)
        
        return UE_pos, ground_AP_pos, UAV_pos
    
    def _generate_ground_AP_positions(self) -> np.ndarray:
        """生成地面AP位置（网格布局）"""
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
    
    def _smart_initialize_UAVs(self, UE_pos: np.ndarray, 
                               ground_AP_pos: np.ndarray) -> np.ndarray:
        """
        智能初始化UAV位置（基于K-means聚类）
        
        参数:
            UE_pos: 用户位置
            ground_AP_pos: 地面AP位置
            
        返回:
            UAV位置 [L, 3]
        """
        # 使用K-means聚类用户位置
        user_centers = self._kmeans_clustering(UE_pos[:, :2], self.L)
        
        positions = []
        for i in range(self.L):
            # 在用户聚类中心附近添加随机偏移
            base_pos = user_centers[i]
            angle = np.random.uniform(0, 2*np.pi)
            radius = np.random.uniform(30, 100)
            
            x = base_pos[0] + radius * np.cos(angle)
            y = base_pos[1] + radius * np.sin(angle)
            
            # 确保在边界内
            x = np.clip(x, self.pos_min, self.pos_max)
            y = np.clip(y, self.pos_min, self.pos_max)
            
            positions.append([x, y, self.heights['UAV']])
        
        return np.array(positions)
    
    def _kmeans_clustering(self, points: np.ndarray, k: int, 
                          max_iter: int = 20) -> np.ndarray:
        """
        K-means聚类算法
        
        参数:
            points: 数据点 [N, 2]
            k: 聚类数量
            max_iter: 最大迭代次数
            
        返回:
            聚类中心 [k, 2]
        """
        n_points = len(points)
        
        # K-means++初始化
        centers = np.zeros((k, 2))
        centers[0] = points[np.random.choice(n_points)]
        
        for i in range(1, k):
            # 计算每个点到最近中心的距离
            distances = np.min([np.sum((points - c)**2, axis=1) 
                              for c in centers[:i]], axis=0)
            # 概率选择下一个中心
            probabilities = distances / (distances.sum() + 1e-12)
            centers[i] = points[np.random.choice(n_points, p=probabilities)]
        
        # 迭代优化
        for _ in range(max_iter):
            # 分配点到最近的中心
            distances = np.sqrt(((points[:, np.newaxis] - centers) ** 2).sum(axis=2))
            assignments = np.argmin(distances, axis=1)
            
            # 更新中心
            for i in range(k):
                cluster_points = points[assignments == i]
                if len(cluster_points) > 0:
                    centers[i] = cluster_points.mean(axis=0)
        
        return centers
    
    def initialize_swarm(self, initial_UAV_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        初始化粒子群
        对应论文Algorithm 2中的Step 1: Initialization
        
        N_particle, P_i and V_i are initialized randomly within the range of their values.
        PB_i ← P_i, ∀i ∈ {1, 2, ..., N_particle}
        GB^t ← F_m(PB_i)
        
        参数:
            initial_UAV_pos: 初始UAV位置 [L, 3]
            
        返回:
            particles: 粒子位置 [N_particle, L*2] (每个粒子代表所有UAV的x,y坐标)
            velocities: 粒子速度 [N_particle, L*2]
        """
        dim = self.L * 2  # 每个UAV的x,y坐标
        
        # 初始化粒子位置
        particles = np.zeros((self.N_particle, dim))
        
        # 第一个粒子使用提供的初始位置
        particles[0] = initial_UAV_pos[:, :2].flatten()
        
        # 其余粒子在搜索空间内随机初始化
        for i in range(1, self.N_particle):
            particles[i] = np.random.uniform(self.pos_min, self.pos_max, dim)
        
        # 初始化速度（随机分布在[-v_max/2, v_max/2]）
        velocities = np.random.uniform(
            self.v_min/2, self.v_max/2, 
            (self.N_particle, dim)
        )
        
        return particles, velocities

    # ========================================================================
    # Step 2: Velocity and position Update (速度和位置更新)
    # ========================================================================
    
    def update_velocity_and_position(self, particles: np.ndarray, 
                                     velocities: np.ndarray,
                                     pbest: np.ndarray, 
                                     gbest: np.ndarray,
                                     iteration: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        更新粒子速度和位置
        对应论文Algorithm 2中的Step 2: Velocity and position Update
        
        V_i and P_i is updated according to (15) and (16).
        
        标准PSO速度更新公式 (15):
        V_i^{t+1} = w * V_i^t + c1 * r1 * (PB_i - P_i^t) + c2 * r2 * (GB - P_i^t)
        
        位置更新公式 (16):
        P_i^{t+1} = P_i^t + V_i^{t+1}
        
        参数:
            particles: 当前粒子位置 [N_particle, dim]
            velocities: 当前粒子速度 [N_particle, dim]
            pbest: 个体最佳位置 [N_particle, dim]
            gbest: 全局最佳位置 [dim]
            iteration: 当前迭代次数
            
        返回:
            updated_particles: 更新后的粒子位置
            updated_velocities: 更新后的粒子速度
        """
        dim = particles.shape[1]
        
        # 生成随机数 r1, r2 ∈ [0, 1]
        r1 = np.random.random((self.N_particle, dim))
        r2 = np.random.random((self.N_particle, dim))
        
        # 可选：使用线性递减的惯性权重
        # w_t = w_max - (w_max - w_min) * iteration / max_iterations
        w_current = self.w
        
        # 速度更新 (公式15)
        # V_i^{t+1} = w * V_i^t + c1 * r1 * (PB_i - P_i^t) + c2 * r2 * (GB - P_i^t)
        cognitive_component = self.c1 * r1 * (pbest - particles)  # 认知部分
        social_component = self.c2 * r2 * (gbest - particles)      # 社会部分
        
        updated_velocities = (w_current * velocities + 
                            cognitive_component + 
                            social_component)
        
        # 速度限制（防止粒子移动过快）
        updated_velocities = np.clip(updated_velocities, self.v_min, self.v_max)
        
        # 位置更新 (公式16)
        # P_i^{t+1} = P_i^t + V_i^{t+1}
        updated_particles = particles + updated_velocities
        
        # 边界约束（确保UAV位置在有效范围内）
        updated_particles = np.clip(updated_particles, self.pos_min, self.pos_max)
        
        # 如果粒子碰到边界，反弹速度
        boundary_violations = (updated_particles <= self.pos_min) | (updated_particles >= self.pos_max)
        updated_velocities[boundary_violations] *= -0.5
        
        return updated_particles, updated_velocities

    # ========================================================================
    # Step 3: Fitness calculation (适应度计算)
    # ========================================================================
    
    def compute_channel_model(self, UE_pos: np.ndarray, 
                              AP_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算完整的多天线信道模型
        
        参数:
            UE_pos: 用户位置 [K, 3]
            AP_pos: AP位置 [L_total, 3]
            
        返回:
            H: 真实信道 [M, nbrOfRealizations, K, L_total]
            Hhat: 估计信道 [M, nbrOfRealizations, K, L_total]
            betas: 大尺度衰落 [K, L_total]
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
                R[:, :, k, l] = functionRlocalscattering.R(
                    self.M, angles[k, l], self.ASD_deg
                )
        
        CorrR = betas[None, None, :, :] * R
        
        # ========= 生成信道矩阵 =========
        CH = np.sqrt(0.5) * (
            np.random.randn(self.M, self.nbrOfRealizations, self.K, L_total) +
            1j * np.random.randn(self.M, self.nbrOfRealizations, self.K, L_total)
        )
        H = np.empty_like(CH, dtype=complex)
        
        # 应用空间相关
        for k in range(self.K):
            for l in range(L_total):
                try:
                    corr_matrix = CorrR[:, :, k, l]
                    if not np.isfinite(corr_matrix).all():
                        H[:, :, k, l] = CH[:, :, k, l]
                        continue
                    
                    # 数值稳定性处理
                    cond_num = np.linalg.cond(corr_matrix)
                    if cond_num > 1e12:
                        corr_matrix += 1e-6 * np.eye(self.M)
                    
                    # 计算矩阵平方根
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
        Np = np.sqrt(0.5) * (
            np.random.randn(self.M, self.nbrOfRealizations, L_total, self.tau_p) +
            1j * np.random.randn(self.M, self.nbrOfRealizations, L_total, self.tau_p)
        )
        
        Hhat = np.zeros_like(H)
        
        for l in range(L_total):
            for t in range(self.tau_p):
                indices = np.where(pilotIndex == t)[0]
                if len(indices) == 0:
                    continue
                    
                # 接收导频信号
                yp = (np.sqrt(self.p * self.tau_p) * 
                     np.sum(H[:, :, indices, l], axis=2) + Np[:, :, l, t])
                
                # MMSE信道估计
                PsiInv = (self.p * self.tau_p * 
                         np.sum(CorrR[:, :, indices, l], axis=2) + self.eyeM)
                PsiInvInv = np.linalg.inv(PsiInv)
                
                for k in indices:
                    RPsi = CorrR[:, :, k, l] @ PsiInvInv
                    Hhat[:, :, k, l] = np.sqrt(self.p * self.tau_p) * RPsi @ yp
        
        return H, Hhat, betas
    
    def compute_AP_selection_mask(self, betas: np.ndarray) -> np.ndarray:
        """
        计算AP选择掩码（每个用户选择信号最强的几个AP）
        
        参数:
            betas: 大尺度衰落 [K, L_total]
            
        返回:
            mask: AP选择掩码 [K, L_total]
        """
        top_AP_indices = np.argpartition(
            betas, -self.num_serving_APs, axis=1
        )[:, -self.num_serving_APs:]
        
        mask = np.zeros((self.K, betas.shape[1]), dtype=bool)
        for k in range(self.K):
            mask[k, top_AP_indices[k]] = True
            
        return mask
    
    def compute_user_rates(self, UE_pos: np.ndarray, 
                          AP_pos: np.ndarray, 
                          mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        计算用户速率
        
        参数:
            UE_pos: 用户位置 [K, 3]
            AP_pos: AP位置 [L_total, 3]
            mask: AP选择掩码 [K, L_total]
            
        返回:
            rates: 每个用户的速率 (Mbps) [K]
            sum_rate: 总速率 (Mbps)
        """
        L_total = len(AP_pos)
        
        # 计算完整信道模型
        H, Hhat, betas = self.compute_channel_model(UE_pos, AP_pos)
        
        # 应用AP选择掩码
        Hhat_uc = Hhat * mask[np.newaxis, np.newaxis, :, :]
        
        # 功率分配（均匀分配给服务的用户）
        num_served_per_AP = mask.sum(axis=0)
        rho = np.zeros((self.K, L_total))
        for l in range(L_total):
            if num_served_per_AP[l] > 0:
                rho[mask[:, l], l] = self.Pmax / num_served_per_AP[l]
        gamma = np.sqrt(rho)
        
        # 计算下行速率（使用MR预编码）
        a_MR, B_MR = self._compute_a_B_MR(H, Hhat_uc, gamma)
        SE_MR = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(
            a_MR, B_MR, self.B, gamma, self.Pmax
        )
        
        # 转换为Mbps
        rates = SE_MR * self.prelogFactor / 1e6
        
        return rates, np.sum(rates)
    
    def _compute_a_B_MR(self, H: np.ndarray, Hhat_uc: np.ndarray, 
                       gamma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算MR预编码的信号和干扰项
        
        参数:
            H: 真实信道
            Hhat_uc: 估计信道（应用AP选择后）
            gamma: 功率分配系数
            
        返回:
            a_MR: 信号项
            B_MR: 干扰项
        """
        M, N, K, L = H.shape
        
        # MR预编码器（归一化的信道估计）
        w_MR = Hhat_uc / (np.linalg.norm(Hhat_uc, axis=0, keepdims=True) + 1e-12)
        
        # 有效信道增益
        a_MR = np.einsum('mnkl,mnkl->lk', np.conj(H), w_MR) / N
        
        # 干扰项
        interf_MR = np.einsum('mnkl,mnil->kiln', np.conj(H), w_MR).mean(axis=-1)
        
        B_MR = np.zeros((L, L, K, K), dtype=np.float64)
        
        for k in range(K):
            for i in range(K):
                B_MR[:, :, k, i] = np.outer(
                    interf_MR[k, i, :], interf_MR[k, i, :].conj()
                ).real
        
        interf2_MR = np.abs(interf_MR) ** 2
        for l in range(L):
            B_MR[l, l, :, :] = interf2_MR[:, :, l]
        
        a_MR = np.abs(a_MR)
        return a_MR, B_MR
    
    def fitness_function(self, particle: np.ndarray, 
                        UE_pos: np.ndarray, 
                        ground_AP_pos: np.ndarray) -> float:
        """
        适应度函数 F_m
        对应论文Algorithm 2中的Step 3: Fitness calculation
        
        F_m is calculated according to (17).
        
        论文中的适应度函数通常是最小用户速率的加权组合:
        F_m = w1 * min_rate + w2 * sum_rate
        
        参数:
            particle: 粒子（UAV位置） [L*2]
            UE_pos: 用户位置 [K, 3]
            ground_AP_pos: 地面AP位置 [G, 3]
            
        返回:
            fitness: 适应度值（越大越好）
        """
        try:
            # 重构UAV位置
            UAV_pos = particle.reshape(self.L, 2)
            UAV_pos = np.column_stack([
                UAV_pos, 
                np.full(self.L, self.heights['UAV'])
            ])
            
            # 边界约束
            UAV_pos[:, 0] = np.clip(UAV_pos[:, 0], self.pos_min, self.pos_max)
            UAV_pos[:, 1] = np.clip(UAV_pos[:, 1], self.pos_min, self.pos_max)
            
            # 合并所有AP位置
            all_AP_pos = np.vstack([ground_AP_pos, UAV_pos])
            
            # 计算信道和AP选择
            _, _, betas = self.compute_channel_model(UE_pos, all_AP_pos)
            mask = self.compute_AP_selection_mask(betas)
            
            # 计算用户速率
            rates, sum_rate = self.compute_user_rates(UE_pos, all_AP_pos, mask)
            
            # 适应度函数 (公式17)
            # 优化目标：最大化最小用户速率（QoE公平性）
            min_rate = rates.min()
            
            # 组合适应度（可以调整权重）
            w1 = self.config.get('w_min_rate', 1.0)  # 最小速率权重
            w2 = self.config.get('w_sum_rate', 0.01)  # 总速率权重
            
            fitness = w1 * min_rate + w2 * sum_rate
            
            return fitness
            
        except Exception as e:
            # 如果计算失败，返回很差的适应度
            return -1e10

    # ========================================================================
    # Step 4: Best position update (最佳位置更新)
    # ========================================================================
    
    def update_best_positions(self, particles: np.ndarray, 
                             fitness_values: np.ndarray,
                             pbest: np.ndarray, 
                             pbest_fitness: np.ndarray,
                             gbest: np.ndarray, 
                             gbest_fitness: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        更新个体最佳和全局最佳位置
        对应论文Algorithm 2中的Step 4: Best position update
        
        For each particle:
            PB_i is updated according to (18).
        End
        GB is updated according to (19).
        
        公式 (18) - 个体最佳更新:
        if F(P_i) > F(PB_i):
            PB_i = P_i
            
        公式 (19) - 全局最佳更新:
        GB = argmax_{i} F(PB_i)
        
        参数:
            particles: 当前粒子位置 [N_particle, dim]
            fitness_values: 当前适应度 [N_particle]
            pbest: 个体最佳位置 [N_particle, dim]
            pbest_fitness: 个体最佳适应度 [N_particle]
            gbest: 全局最佳位置 [dim]
            gbest_fitness: 全局最佳适应度
            
        返回:
            updated_pbest: 更新后的个体最佳位置
            updated_pbest_fitness: 更新后的个体最佳适应度
            updated_gbest: 更新后的全局最佳位置
            updated_gbest_fitness: 更新后的全局最佳适应度
        """
        # 个体最佳更新 (公式18)
        # 对每个粒子，如果当前适应度优于历史最佳，则更新
        improvement_mask = fitness_values > pbest_fitness
        pbest[improvement_mask] = particles[improvement_mask].copy()
        pbest_fitness[improvement_mask] = fitness_values[improvement_mask]
        
        # 全局最佳更新 (公式19)
        # 在所有个体最佳中找到最优的
        best_idx = np.argmax(pbest_fitness)
        if pbest_fitness[best_idx] > gbest_fitness:
            gbest = pbest[best_idx].copy()
            gbest_fitness = pbest_fitness[best_idx]
        
        return pbest, pbest_fitness, gbest, gbest_fitness

    # ========================================================================
    # Main Optimization Loop (主优化循环)
    # ========================================================================
    
    def optimize(self, UE_pos: np.ndarray, 
                ground_AP_pos: np.ndarray,
                UAV_pos: np.ndarray) -> Dict:
        """
        执行分布式PSO优化
        实现完整的Algorithm 2
        
        参数:
            UE_pos: 用户位置 [K, 3]
            ground_AP_pos: 地面AP位置 [G, 3]
            UAV_pos: 初始UAV位置 [L, 3]
            
        返回:
            results: 优化结果字典
        """
        print("\n" + "="*80)
        print("  分布式PSO优化算法 (Algorithm 2)  ".center(80))
        print("="*80)
        
        start_time = time.time()
        
        # ========== Step 1: Initialization ==========
        print("\n[Step 1] 初始化粒子群...")
        particles, velocities = self.initialize_swarm(UAV_pos)
        
        # 初始化个体最佳 PB_i ← P_i
        pbest = particles.copy()
        pbest_fitness = np.full(self.N_particle, -np.inf)
        
        # 计算初始适应度并设置全局最佳 GB^t ← F_m(PB_i)
        for i in range(self.N_particle):
            pbest_fitness[i] = self.fitness_function(particles[i], UE_pos, ground_AP_pos)
        
        best_idx = np.argmax(pbest_fitness)
        gbest = pbest[best_idx].copy()
        gbest_fitness = pbest_fitness[best_idx]
        
        print(f"   ✓ 粒子群初始化完成")
        print(f"   ✓ 初始全局最佳适应度: {gbest_fitness:.4f}")
        
        # 历史记录
        history = {
            'iterations': [],
            'sum_rates': [],
            'min_rates': [],
            'mean_rates': [],
            'best_fitness': [],
            'UAV_positions': []
        }
        
        # ========== Repeat: 迭代优化 ==========
        print(f"\n[开始迭代] 最大迭代次数: {self.max_iterations}")
        print("-" * 80)
        
        for iteration in range(self.max_iterations):
            # ========== For each particle ==========
            
            # ===== Step 2: Velocity and position Update =====
            particles, velocities = self.update_velocity_and_position(
                particles, velocities, pbest, gbest, iteration
            )
            
            # ===== Step 3: Fitness calculation =====
            fitness_values = np.array([
                self.fitness_function(particles[i], UE_pos, ground_AP_pos)
                for i in range(self.N_particle)
            ])
            
            # ========== End (for each particle) ==========
            
            # ===== Step 4: Best position update =====
            pbest, pbest_fitness, gbest, gbest_fitness = self.update_best_positions(
                particles, fitness_values, pbest, pbest_fitness, gbest, gbest_fitness
            )
            
            # ========== 记录当前最佳解的性能指标 ==========
            best_UAV_pos = gbest.reshape(self.L, 2)
            best_UAV_pos = np.column_stack([
                best_UAV_pos, 
                np.full(self.L, self.heights['UAV'])
            ])
            
            # 计算详细性能
            all_AP_pos = np.vstack([ground_AP_pos, best_UAV_pos])
            _, _, betas = self.compute_channel_model(UE_pos, all_AP_pos)
            mask = self.compute_AP_selection_mask(betas)
            rates, sum_rate = self.compute_user_rates(UE_pos, all_AP_pos, mask)
            
            min_rate = rates.min()
            mean_rate = rates.mean()
            
            # 保存历史
            history['iterations'].append(iteration)
            history['sum_rates'].append(sum_rate)
            history['min_rates'].append(min_rate)
            history['mean_rates'].append(mean_rate)
            history['best_fitness'].append(gbest_fitness)
            history['UAV_positions'].append(best_UAV_pos.copy())
            
            # 输出进度
            if iteration % 10 == 0 or iteration == self.max_iterations - 1:
                print(f"Iter {iteration:3d} | "
                      f"Fitness: {gbest_fitness:8.4f} | "
                      f"Min Rate: {min_rate:6.4f} Mbps | "
                      f"Sum Rate: {sum_rate:8.2f} Mbps | "
                      f"Mean Rate: {mean_rate:6.4f} Mbps")
        
        # ========== Until: 达到最大迭代次数 ==========
        
        optimization_time = time.time() - start_time
        
        # ========== 最终结果 ==========
        print("\n" + "="*80)
        print("  优化完成  ".center(80))
        print("="*80)
        
        final_UAV_pos = gbest.reshape(self.L, 2)
        final_UAV_pos = np.column_stack([
            final_UAV_pos, 
            np.full(self.L, self.heights['UAV'])
        ])
        
        final_all_AP_pos = np.vstack([ground_AP_pos, final_UAV_pos])
        _, _, final_betas = self.compute_channel_model(UE_pos, final_all_AP_pos)
        final_mask = self.compute_AP_selection_mask(final_betas)
        final_rates, final_sum_rate = self.compute_user_rates(
            UE_pos, final_all_AP_pos, final_mask
        )
        final_min_rate = final_rates.min()
        final_mean_rate = final_rates.mean()
        final_std_rate = final_rates.std()
        
        # 性能提升统计
        initial_sum_rate = history['sum_rates'][0]
        initial_min_rate = history['min_rates'][0]
        
        improvement_sum = ((final_sum_rate - initial_sum_rate) / 
                          (initial_sum_rate + 1e-6)) * 100
        improvement_min = ((final_min_rate - initial_min_rate) / 
                          (initial_min_rate + 1e-6)) * 100
        
        # 打印最终结果
        print(f"\n📊 最终性能指标:")
        print(f"   - 总速率 (Sum Rate):     {final_sum_rate:8.2f} Mbps")
        print(f"   - 最小速率 (Min Rate):   {final_min_rate:8.4f} Mbps")
        print(f"   - 平均速率 (Mean Rate):  {final_mean_rate:8.4f} Mbps")
        print(f"   - 速率标准差 (Std):      {final_std_rate:8.4f} Mbps")
        print(f"   - 最佳适应度 (Fitness):  {gbest_fitness:8.4f}")
        
        print(f"\n📈 性能提升:")
        print(f"   - 总速率提升:   {improvement_sum:+.2f}%")
        print(f"   - 最小速率提升: {improvement_min:+.2f}%")
        
        print(f"\n⏱️  计算时间:")
        print(f"   - 总优化时间: {optimization_time:.2f} 秒")
        print(f"   - 平均每代:   {optimization_time/self.max_iterations:.3f} 秒")
        
        # 构造返回结果
        results = {
            'optimized_UAV_pos': final_UAV_pos,
            'final_sum_rate': final_sum_rate,
            'final_min_rate': final_min_rate,
            'final_mean_rate': final_mean_rate,
            'final_std_rate': final_std_rate,
            'final_rates': final_rates,
            'optimization_time': optimization_time,
            'total_iterations': self.max_iterations,
            'history': history,
            'final_mask': final_mask,
            'best_fitness': gbest_fitness,
            'improvement_sum_rate': improvement_sum,
            'improvement_min_rate': improvement_min,
            'initial_sum_rate': initial_sum_rate,
            'initial_min_rate': initial_min_rate,
        }
        
        return results


# ============================================================================
# Configuration and Main Function
# ============================================================================

def create_distributed_pso_config() -> Dict:
    """
    创建分布式PSO配置
    
    返回:
        config: 配置字典
    """
    return {
        # ========== 基本参数 ==========
        'square_length': 1000,      # 区域边长 (m)
        'num_UE': 60,               # 用户数量
        'num_UAV': 9,               # UAV数量
        'num_ground_AP': 4,         # 地面AP数量
        'M': 4,                     # 天线数
        
        # ========== 高度设置 ==========
        'UE_height': 1.65,          # 用户高度 (m)
        'ground_AP_height': 15.0,   # 地面AP高度 (m)
        'UAV_height': 50.0,         # UAV高度 (m)
        
        # ========== PSO参数 ==========
        'N_particle': 30,           # 粒子数量
        'max_iterations': 50,       # 最大迭代次数
        'w': 0.729,                 # 惯性权重 (inertia weight)
        'c1': 1.49445,              # 认知学习因子 (cognitive)
        'c2': 1.49445,              # 社会学习因子 (social)
        'v_max': 50.0,              # 最大速度 (m/s)
        'v_min': -50.0,             # 最小速度 (m/s)
        'pos_min': 50.0,            # 位置下界
        
        # ========== 适应度函数权重 ==========
        'w_min_rate': 1.0,          # 最小速率权重
        'w_sum_rate': 0.01,         # 总速率权重
        
        # ========== 通信参数 ==========
        'num_serving_APs': 3,       # 每个用户服务的AP数
        'nbrOfRealizations': 30,    # 信道实现数
        
        # ========== 信道参数 ==========
        'alpha': 3.67,              # 路径损耗指数
        'constant_term': -30.5,     # 路径损耗常数项 (dB)
        'B': 20e6,                  # 带宽 (Hz)
        'p': 100,                   # 导频功率 (mW)
        'Pmax': 1000,               # 最大发射功率 (mW)
        'noise_figure': 7,          # 噪声系数 (dB)
        'distance_vertical': 150,   # 垂直距离 (m)
        'tau_p': 60,                # 导频长度
        'tau_c': 200,               # 相干时间
        'ASD_deg': 10,              # 角度扩展 (度)
    }


if __name__ == "__main__":
    """测试分布式PSO优化器"""
    print("\n" + "="*80)
    print("  测试分布式PSO优化器  ".center(80))
    print("  Based on Algorithm 2 from the Paper  ".center(80))
    print("="*80)
    
    # 创建配置
    config = create_distributed_pso_config()
    
    # 创建优化器
    optimizer = DistributedPSOOptimizer(config)
    
    # 初始化位置
    print("\n[初始化] 生成用户、地面AP和UAV位置...")
    UE_pos, ground_AP_pos, UAV_pos = optimizer.initialize_positions()
    
    print(f"   ✓ 用户数量 (K): {len(UE_pos)}")
    print(f"   ✓ 地面AP数量 (G): {len(ground_AP_pos)}")
    print(f"   ✓ UAV数量 (L): {len(UAV_pos)}")
    print(f"   ✓ 总AP数量: {len(ground_AP_pos) + len(UAV_pos)}")
    
    # 执行优化
    results = optimizer.optimize(UE_pos, ground_AP_pos, UAV_pos)
    
    # 最终总结
    print("\n" + "="*80)
    print("  测试完成  ".center(80))
    print("="*80)
    print(f"\n✅ 分布式PSO优化器测试成功！")
    print(f"   论文算法实现: Algorithm 2")
    print(f"   优化成功: {results['improvement_min_rate'] > 0}")

