import numpy as np
import random
import math
import pickle
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import progressbar
import multiprocessing as mp
import APLocation_Generation
from sklearn.cluster import KMeans
from scipy.linalg import sqrtm
from scipy.optimize import linear_sum_assignment
import functionRlocalscattering
import SpectralEfficiencyDownlink
np.random.seed(42)

POP_SIZE=250
# 全局常量（从 SimulationConfig 中读取）
class SimulationConfig:
    def __init__(self):
        self.square_length = 1000
        self.num_UE = 50
        self.num_UAV = 9
        self.num_AP_ground = 4
        self.recluster_period = 10
        self.step_size = 10
        self.alpha = 1e4
        self.beta = 300
        self.k_min = 1e5
        self.epsilon = 1e-6
        self.heights = {
            'UE': 1.65,
            'AP_ground': 15.0,
            'UAV': 50.0
        }
        self.distanceVertical = 10  # 垂直距离（米）
        self.constantTerm = -30.5  # 路径损耗常数项（dB）
        self.alpha_pathloss = 3.67  # 路径损耗指数
        self.p = 10  # UE 发射功率（mW）
        self.ASDdeg = 10  # 局部散射角度标准差（度）
        self.Pmax = 0.1  # AP 最大发射功率（W）
        self.B = 20e6  # 带宽（Hz）
        self.tau_p = self.num_UE  # 导频长度（正交）
        self.tau_c = 200  # 符号周期
        self.prelogFactor = (self.tau_c - self.tau_p) / self.tau_c
        self.noiseVariancedBm = -174 + 10 * np.log10(self.B) + 7  # 噪声功率（dBm）
        self.UAV_SEC_R = 20
        self.UAV_SEC_R2 = self.UAV_SEC_R ** 2


cfg = SimulationConfig()  # 实例化配置


# 信道相关函数
def compute_a_B_MR(H, Hhat_uc, gamma, Pmax):
    """
    计算 MR 权重下的 a_MR 和干扰矩阵 B_MR
    H: (M, N_realizations, K, L)
    Hhat_uc: (M, N_realizations, K, L)（已应用 mask）
    gamma: (K, L)（功率缩放因子）
    返回：
    a_MR: (L, K)（每个 AP 到 UE 的归一化信道）
    B_MR: (L, L, K, K)（干扰矩阵）
    """
    # 计算 a_MR（归一化后的信道估计期望）
    a_MR = np.mean(Hhat_uc * np.conj(H), axis=1).real  # 归一化 Hhat 的期望

    # 计算干扰项 B_MR（简化版本，假设所有 AP 的干扰贡献）
    L = H.shape[3]
    K = H.shape[2]
    B_MR = np.zeros((L, L, K, K), dtype='float')
    for l in range(L):
        for m in range(L):
            if l == m:
                B_MR[l, m, :, :] = np.abs(gamma[:, l]) ** 2 * np.abs(H[:, :, :, l]) ** 2
            else:
                B_MR[l, m, :, :] = np.abs(gamma[:, m] * H[:, :, :, m]) ** 2
    # 可能需要更精确的计算，根据您的具体信道模型调整
    return a_MR, B_MR


def generateSumRate(APXpositions, APYpositions, UEXpositions, UEYpositions):
    """
    计算总和速率，返回每个 UE 的速率、总速率和服务关系 mask
    APXpositions/APYpositions: 所有 AP（地面+UAV）的二维坐标（一维数组）
    UEXpositions/UEYpositions: UE 的二维坐标（一维数组）
    返回:
    rates: (K,) 每个 UE 的速率（Mbps）
    total_rate: 系统总速率（Mbps）
    mask: (K, L) 服务关系矩阵（L是AP总数）
    """
    L = len(APXpositions)
    K = len(UEXpositions)

    AP_positions = np.stack([APXpositions, APYpositions], axis=1)  # (L,2)
    UE_positions = np.stack([UEXpositions, UEYpositions], axis=1)  # (K,2)

    # 计算距离和角度（向量化）
    diff = UE_positions[:, None, :] - AP_positions[None, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1) + cfg.distanceVertical ** 2)
    angles = np.arctan2(diff[..., 1], diff[..., 0])

    # 信道增益（包含 UE 发射功率 p 的贡献）
    p_dB = 10 * np.log10(cfg.p)  # 转换为 dBm
    channelGaindB = cfg.constantTerm + p_dB - cfg.alpha_pathloss * 10 * np.log10(distances)
    channelGainOverNoise = channelGaindB - cfg.noiseVariancedBm
    betas = 10 ** (channelGainOverNoise / 10)  # (K, L)

    # 空间相关矩阵 R（批量计算）
    R = np.zeros((cfg.M, cfg.M, K, L), dtype='complex')
    for k in range(K):
        for l in range(L):
            R[:, :, k, l] = functionRlocalscattering.R(
                M=cfg.M, angle=angles[k, l], ASDdeg=cfg.ASDdeg
            )
    CorrR = betas[None, None, :, :] * R  # (M,M,K,L)

    # 生成信道矩阵 H（假设 M=4 天线）
    CH = np.sqrt(0.5) * (
            np.random.randn(cfg.M, 100, K, L) +
            1j * np.random.randn(cfg.M, 100, K, L)
    )
    H = np.empty_like(CH, dtype='complex')
    for k in range(K):
        for l in range(L):
            Rsqrt = sqrtm(CorrR[:, :, k, l])
            H[:, :, k, l] = Rsqrt @ CH[:, :, k, l]

    # 导频分配与信道估计（假设 N_realizations=100）
    pilotIndex = np.random.permutation(K)
    noise_power_linear = 10 ** (cfg.noiseVariancedBm / 10) * 1e-3
    Np = np.sqrt(0.5 * noise_power_linear) * (
            np.random.randn(cfg.M, 100, L, cfg.tau_p) +
            1j * np.random.randn(cfg.M, 100, L, cfg.tau_p)
    )
    eyeM = np.eye(cfg.M)

    Hhat = np.zeros_like(H)
    Hhat_MMSE_MeanSquare = np.zeros((K, L))
    for l in range(L):
        for t in range(cfg.tau_p):
            indices = np.where(pilotIndex == t)[0]
            if len(indices) == 0:
                continue
            # 合并所有导频用户
            yp = np.sqrt(cfg.p * cfg.tau_p) * np.sum(H[:, :, indices, l], axis=2) + Np[:, :, l, t]
            # 计算 PsiInv
            PsiInv = (cfg.p * cfg.tau_p * np.sum(CorrR[:, :, indices, l], axis=2)) + eyeM
            PsiInvInv = np.linalg.inv(PsiInv)
            for k in indices:
                RPsi = CorrR[:, :, k, l] @ PsiInvInv
                Hhat[:, :, k, l] = np.sqrt(cfg.p * cfg.tau_p) * RPsi @ yp
                Hhat_MMSE_MeanSquare[k, l] = (cfg.p * cfg.tau_p / cfg.M) * np.real(
                    np.trace(RPsi @ CorrR[:, :, k, l])
                )

    # ========= AP选择和功率分配 =========
    # 基于信道估计质量选择前 3 个 AP（替换为 mask 的生成逻辑）
    main_AP = np.argsort(Hhat_MMSE_MeanSquare, axis=1)[:, -3:]  # 每个 UE 选信道质量最好的 3 个 AP
    mask = np.zeros((K, L), dtype=bool)
    np.put_along_axis(mask, main_AP, True, axis=1)  # mask 标记服务 AP（L是AP总数，包括地面AP和UAV）

    numServedPerAP = mask.sum(axis=0)
    rho = np.zeros((K, L))
    for l in range(L):
        if numServedPerAP[l] > 0:
            rho[mask[:, l], l] = cfg.Pmax / numServedPerAP[l]
    gamma = np.sqrt(rho)

    # 下行信道矩阵（过滤非服务 AP）
    Hhat_uc = Hhat * mask[None, None, :, :]

    # 计算 a_MR 和干扰矩阵 B_MR
    a_MR, B_MR = compute_a_B_MR(H, Hhat_uc, gamma, cfg.Pmax)

    # 计算 SE（假设 SpectralEfficiencyDownlink 已实现）
    SE_MR_equal = np.zeros((K,))
    SE_MR_equal[:] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(
        a_MR, B_MR, cfg.B, gamma, cfg.Pmax
    )

    # 应用预因子并转换为 Mbps
    total_rate = np.sum(SE_MR_equal) * cfg.prelogFactor / 1e6
    return SE_MR_equal * cfg.prelogFactor / 1e6, total_rate, mask


class UE:
    def __init__(self):
        self.loc = [-1, -1]
        self.list_loc = []


class UAV:
    def __init__(self, loc):
        self.loc = loc
        self.client_list = []


class Solution:
    def __init__(self):
        self.mat = np.empty((cfg.square_length, cfg.square_length))
        self.list = []
        self.fit = -np.inf


def Create_UE_list_loc(ue_space):
    for ue in ue_space:
        ue.list_loc = [[x, y] for x in range(cfg.square_length) for y in range(cfg.square_length)]


def Create_UAV_matrix_from_kmeans(ue_coords):
    """
    通过 K-means 聚类生成初始 UAV 位置（二维）
    """
    kmeans = KMeans(n_clusters=cfg.num_UAV, random_state=0).fit(ue_coords)
    centroids_xy = kmeans.cluster_centers_
    uav_list = [UAV(center.astype(int).tolist()) for center in centroids_xy]
    matrix = np.zeros((cfg.square_length, cfg.square_length), dtype=int)
    for uav in uav_list:
        matrix[uav.loc[0], uav.loc[1]] = 1
    return matrix, uav_list


def cal_distance2(loc1, loc2):
    return (loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2


def Fit_func(solution, ue_coords, AP_ground_coords):
    # 提取 UAV 的位置
    UAV_coords = np.array([uav.loc for uav in solution.list])
    # 合并所有 AP（地面 AP + UAV）
    AP_all = np.concatenate([AP_ground_coords[:, :2], UAV_coords], axis=0)
    L = len(AP_all)

    # 计算速率和 mask
    try:
        rates, _, mask = generateSumRate(
            APXpositions=AP_all[:, 0],
            APYpositions=AP_all[:, 1],
            UEXpositions=ue_coords[:, 0],
            UEYpositions=ue_coords[:, 1]
        )
    except Exception as e:
        print(f"Error in generateSumRate: {e}")
        return -np.inf

    # 提取 UAV 对应的 mask 部分（假设前 G 列是地面 AP）
    G = AP_ground_coords.shape[0]
    mask_uav = mask[:, G:]

    # 确保每个 UE 至少有一个 UAV 服务（可选）
    if np.any(np.all(mask_uav == False, axis=1)):
        return -np.inf  # 无效解

    return np.min(rates)  # 适应度为最小用户速率


def cal_fit(solution_list, ue_coords, AP_ground_coords):
    for solution in solution_list:
        solution.fit = Fit_func(solution, ue_coords, AP_ground_coords)


def Sort_fitness(solution_list):
    solution_list.sort(key=lambda x: x.fit, reverse=True)


def RouletteWheelSelection(solution_list):
    valid_sols = [sol for sol in solution_list if sol.fit != -np.inf]
    if not valid_sols:
        return solution_list[0], solution_list[1]

    total = sum(sol.fit for sol in valid_sols)
    if total <= 0:
        return valid_sols[0], valid_sols[1]

    r1 = random.uniform(0, total)
    r2 = random.uniform(0, total)
    selected1, selected2 = None, None

    current = 0
    for sol in valid_sols:
        current += sol.fit
        if r1 <= current:
            selected1 = sol
            break

    current = 0
    for sol in valid_sols:
        current += sol.fit
        if r2 <= current:
            selected2 = sol
            break

    return selected1, selected2


def Rand_Exchange(solution_list, children_space, AP_ground_coords):
    sol1, sol2 = RouletteWheelSelection(solution_list)

    # 随机切分交叉（矩阵切片）
    c_row = random.randint(0, cfg.square_length - 1)
    c_col = random.randint(0, cfg.square_length - 1)

    child1_mat = np.copy(sol2.mat)
    child1_mat[:c_row, :c_col] = sol1.mat[:c_row, :c_col]
    child1_mat[c_row:, c_col:] = sol1.mat[c_row:, c_col:]

    child2_mat = np.copy(sol1.mat)
    child2_mat[:c_row, :c_col] = sol2.mat[:c_row, :c_col]
    child2_mat[c_row:, c_col:] = sol2.mat[c_row:, c_col:]

    # 创建子代 UAV 列表并检查安全距离
    def _build_child(child_mat):
        uav_list = [UAV([x, y]) for x in range(cfg.square_length) for y in range(cfg.square_length) if
                    child_mat[x, y] == 1]
        new_uav_list = []
        seen = np.zeros((cfg.square_length, cfg.square_length), dtype=bool)
        for uav in uav_list:
            x, y = uav.loc
            if seen[x, y]:
                continue
            conflict = False
            for other in new_uav_list:
                if cal_distance2(uav.loc, other.loc) <= cfg.UAV_SEC_R2:
                    conflict = True
                    break
            if not conflict:
                new_uav_list.append(uav)
                seen[x, y] = True
        return new_uav_list, child_mat

    child1_uavs, child1_mat = _build_child(child1_mat)
    child2_uavs, child2_mat = _build_child(child2_mat)

    child1 = Solution()
    child1.list = child1_uavs
    child1.mat = child1_mat

    child2 = Solution()
    child2.list = child2_uavs
    child2.mat = child2_mat

    children_space.append(child1)
    children_space.append(child2)


def Create_children(solution_list, AP_ground_coords):
    children_space = []
    for _ in range(POP_SIZE // 2):
        Rand_Exchange(solution_list, children_space, AP_ground_coords)
    return children_space


def Check_security_distance(solution):
    # 安全距离检查已集成到交叉操作中
    pass


def Draw_result(ue_space, best_solution, AP_ground_coords):
    ue_coords = np.array([ue.loc for ue in ue_space])
    UAV_coords = np.array([uav.loc for uav in best_solution.list])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(ue_coords[:, 0], ue_coords[:, 1], c='blue', marker='x', s=50, label='UE')
    ax.scatter(AP_ground_coords[:, 0], AP_ground_coords[:, 1], c='green', marker='^', s=100, label='Ground AP')
    ax.scatter(UAV_coords[:, 0], UAV_coords[:, 1], c='red', marker='v', s=100, label='UAV')

    # 根据 mask 绘制连接关系
    _, _, mask = generateSumRate(
        APXpositions=np.concatenate([AP_ground_coords[:, 0], UAV_coords[:, 0]]),
        APYpositions=np.concatenate([AP_ground_coords[:, 1], UAV_coords[:, 1]]),
        UEXpositions=ue_coords[:, 0],
        UEYpositions=ue_coords[:, 1]
    )
    G = AP_ground_coords.shape[0]
    mask_uav = mask[:, G:]  # 提取 UAV 部分的 mask

    for ue_idx in range(mask_uav.shape[0]):
        served_UAVs = np.where(mask_uav[ue_idx])[0]
        for uav_idx in served_UAVs:
            ue_x, ue_y = ue_coords[ue_idx]
            uav_x, uav_y = UAV_coords[uav_idx]
            ax.plot([ue_x, uav_x], [ue_y, uav_y], 'b--', alpha=0.3, linewidth=1)

    ax.set_title("UAV Deployment with User-Centric Links")
    ax.legend()
    ax.grid(True)
    plt.show()


def main(filepre):
    # 加载用户位置（假设 UE 的 loc 是二维坐标）
    with open(f'{filepre}.pickle', 'rb') as f:
        ue_space = pickle.load(f)
    ue_coords = np.array([ue.loc for ue in ue_space])

    # 生成地面 AP 的位置（三维）
    G = cfg.num_AP_ground
    AP_ground_xy = APLocation_Generation.RandomAPLocations(G, cfg.square_length)
    AP_ground_coords = np.array([
        [np.real(xy), np.imag(xy), cfg.heights['AP_ground']]
        for xy in AP_ground_xy
    ]).astype(int)

    # 初始化解空间（使用 K-means）
    solution_list = []
    for _ in range(POP_SIZE):
        solution = Solution()
        # 使用 K-means 初始化 UAV 位置
        solution.mat, solution.list = Create_UAV_matrix_from_kmeans(ue_coords)
        # 检查安全距离
        # Check_security_distance(solution)  # 安全距离现在在交叉时处理
        solution_list.append(solution)

    # 进化过程
    num_generations = 60
    widgets = ["Optimizing: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=num_generations).start()

    for gen in range(num_generations):
        pbar.update(gen + 1)
        children = Create_children(solution_list, AP_ground_coords)
        all_solutions = solution_list + children
        cal_fit(all_solutions, ue_coords, AP_ground_coords)
        Sort_fitness(all_solutions)
        solution_list = all_solutions[:POP_SIZE]

    pbar.finish()

    best_solution = solution_list[0]
    Draw_result(ue_space, best_solution, AP_ground_coords)

    # 计算总吞吐量（基于最优解）
    UAV_final_coords = np.array([uav.loc for uav in best_solution.list])
    _, total_rate, _ = generateSumRate(
        APXpositions=np.concatenate([AP_ground_coords[:, 0], UAV_final_coords[:, 0]]),
        APYpositions=np.concatenate([AP_ground_coords[:, 1], UAV_final_coords[:, 1]]),
        UEXpositions=ue_coords[:, 0],
        UEYpositions=ue_coords[:, 1]
    )

    print(f"Optimal fitness (min user rate): {best_solution.fit:.2f} Mbps")
    print(f"Total system rate: {total_rate:.2f} Mbps")
    return best_solution.fit


if __name__ == "__main__":
    filepre = 'data/list_ue60_1'
    main(filepre)