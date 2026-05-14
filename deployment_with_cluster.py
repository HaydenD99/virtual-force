import numpy as np
from sklearn.cluster import KMeans
import APLocation_Generation
import generateUCCC
import generateUCC
import rateGenerationTest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(4)

##问题：没有把分簇的信道和力结合起来，力还是全部的，还有就是参数需要调整，模型需要变复杂

# ===================== 配置部分 =====================
class SimulationConfig:
    """
    仿真配置类：
    - square_length: 环境边长（x,y 方向）
    - num_UE: 用户数量
    - num_UAV: UAV 数量（簇数）
    - num_AP_ground: 地面固定 AP 数量
    - recluster_period: 重聚类周期（迭代轮数）
    - step_size: UAV 每步移动距离比例
    - alpha: 簇间 UAV 排斥权重
    - beta: AP 协同力权重
    - k_min: 最大化最小速率的吸引力系数
    - epsilon: 避免除零的小量
    - heights: 各实体固定高度
    """
    def __init__(self):
        self.square_length = 1000
        self.num_UE =60
        self.num_UAV = 9
        self.num_AP_ground = 4
        self.recluster_period = 10
        self.step_size = 10
        self.alpha = 1e6
        self.beta = 300
        self.k_min = 1e5
        self.epsilon = 1e-6
        # 固定高度：UE, AP_ground, UAV
        self.heights = {
            'UE': 1.65,
            'AP_ground': 15.0,
            'UAV': 50.0
        }

# ===================== 信道与速率计算占位 =====================
# def compute_sum_rate(APX, APY, UEX, UEY,UAV_centers):
#     """原有的总和速率函数，返回总速率（bit/s）"""
#     # TODO: 调用已有模块计算总和速率
#
#     return rateGenerationTest.generate_simplified_rates(APXpositions=APX, APYpositions=APY,
#         UEXpositions=UEX, UEYpositions=UEY,
#         cluster_labels=labels,
#         UAV_positions_xy=UAV_centers,
#         Pmax=0.1, B=20e6,
#         noiseVariancedBm=-174+10*np.log10(20e6))[1]
def compute_sum_rate(APX, APY, UEX, UEY,UAV_centers):
    """原有的总和速率函数，返回总速率（bit/s）"""
    # TODO: 调用已有模块计算总和速率
    return generateUCCC.generateSumRate(APXpositions=APX, APYpositions=APY,
        UEXpositions=UEX, UEYpositions=UEY)[1]

def compute_per_user_rate(APX, APY, UEX, UEY,UAV_centers):
    """
    新增函数：计算每个 UE 的下行速率，返回 shape=(K,) 的速率数组
    """
    # TODO: 调用已有模块或 SINR/SE 模型，返回每个用户速率
    return generateUCCC.generateSumRate(APXpositions=APX, APYpositions=APY,
                                       UEXpositions=UEX, UEYpositions=UEY)[0],generateUCCC.generateSumRate(APXpositions=APX, APYpositions=APY,
                                       UEXpositions=UEX, UEYpositions=UEY)[2]
# def compute_per_user_rate(APX, APY, UEX, UEY,UAV_centers):
#     """
#     新增函数：计算每个 UE 的下行速率，返回 shape=(K,) 的速率数组
#     """
#     # TODO: 调用已有模块或 SINR/SE 模型，返回每个用户速率
#     return rateGenerationTest.generate_simplified_rates(APXpositions=APX, APYpositions=APY,
#         UEXpositions=UEX, UEYpositions=UEY,
#         cluster_labels=labels,
#         UAV_positions_xy=UAV_centers,
#         Pmax=0.1, B=20e6,
#         noiseVariancedBm=-174+10*np.log10(20e6))[0]

# ===================== 力场函数 =====================
# def compute_attraction_to_UE(uav_pos, ue_pos, weights=None):
#     """
#     改造：支持权重加权吸引（针对速率），weights shape=(k,)
#     UAV 只在 x-y 平面移动，z 分量力为 0
#     返回与 uav_pos (1,3) 相同的力向量。
#     """
#     if ue_pos.size == 0:
#         return np.zeros_like(uav_pos)
#     # 先在 x-y 平面计算加权质心
#     if weights is None:
#         centroid_xy = ue_pos[:, :2].mean(axis=0)
#     else:
#         centroid_xy = np.average(ue_pos[:, :2], axis=0, weights=weights)
#     # 计算 x-y 方向上的单位向量
#     diff_xy = centroid_xy - uav_pos[:, :2]
#     norm_xy = np.linalg.norm(diff_xy, axis=1, keepdims=True) + 1e-6
#     unit_xy = diff_xy / norm_xy
#     # 返回三维力：z 分量为 0
#     return np.hstack([unit_xy, np.zeros((uav_pos.shape[0], 1))])

def compute_attraction_to_UE(uav_pos, ue_pos, weights=None, Ka=1.0):
    """
    改造版：
    - 沿用 virtual_fa_cellfree 的力设计
    - UAV 只在 x-y 平面移动，z 分量为 0
    - 支持权重加权（可按速率、信道增益等加权）
    - 支持批量 uav_pos: shape=(N_uav, 3)
    - 返回与 uav_pos 相同 shape 的力向量
    """

    N_uav = uav_pos.shape[0]
    N_ue = ue_pos.shape[0]

    if N_ue == 0:
        return np.zeros_like(uav_pos)

    if weights is None:
        weights = np.ones(N_ue)
    weights = weights / (np.sum(weights) + 1e-12)  # 归一化防止爆炸

    Fa_total = np.zeros((N_uav, 3))

    for i in range(N_uav):
        force_xy = np.zeros(2)
        for k in range(N_ue):
            dx = ue_pos[k, 0] - uav_pos[i, 0]
            dy = ue_pos[k, 1] - uav_pos[i, 1]
            dist_sq = dx ** 2 + dy ** 2 + 1e-6
            dist = np.sqrt(dist_sq)

            # 分量计算，Ka * weight * dx / dist^2
            force_xy[0] += Ka * weights[k] * dx / dist_sq
            force_xy[1] += Ka * weights[k] * dy / dist_sq

        Fa_total[i, :2] = force_xy
        # Fa_total[i, 2] 默认为 0

    return Fa_total


def compute_repulsion_between_UAV(uav_pos):
    """
    计算 UAV 之间的排斥力，三维版但只在 x-y 平面有作用
    """
    L = uav_pos.shape[0]
    forces = np.zeros((L, 3))
    for i in range(L):
        for j in range(i+1, L):
            diff_xy = uav_pos[i, :2] - uav_pos[j, :2]
            d = np.linalg.norm(diff_xy) + 1e-6
            if d < 100:
                f_xy = diff_xy / (d**2)
                forces[i, :2] += f_xy
                forces[j, :2] -= f_xy
    return forces

def compute_AP_UAV_sync_force(uav_pos, ap_pos, k_sync=1e3, d0=150):
    """
    计算地面 AP 对 UAV 的协同弹簧力，三维版但只在 x-y 面
    """
    L = uav_pos.shape[0]
    G = ap_pos.shape[0]
    forces = np.zeros((L, 3))
    for i in range(G):
        diff_xy = uav_pos[:, :2] - ap_pos[i, :2]
        dists = np.linalg.norm(diff_xy, axis=1, keepdims=True) + 1e-6
        f_xy = -k_sync * (dists - d0) * (diff_xy / dists)
        mask = (dists < d0).astype(float)
        forces[:, :2] += (f_xy * mask)
    return forces



def plot_optimization_comparison(UE_pos, AP_ground, UAV_initial, UAV_optimized):
    """输入参数明确标注初始和优化后位置"""
    plt.figure(figsize=(18, 6))

    # 子图1：初始部署
    plt.subplot(1, 3, 1)
    plt.scatter(UE_pos[:, 0], UE_pos[:, 1], c='blue', alpha=0.6, label='UE')
    # plt.scatter(AP_ground[:, 0], AP_ground[:, 1], c='red', marker='^', s=100, label='Ground AP')
    plt.scatter(UAV_initial[:, 0], UAV_initial[:, 1], c='orange', marker='x', s=100, linewidths=2,
                label='UAV (Initial)')
    plt.title("Initial Deployment (2D)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # 子图2：优化后部署
    plt.subplot(1, 3, 2)
    plt.scatter(UE_pos[:, 0], UE_pos[:, 1], c='blue', alpha=0.6)
    # plt.scatter(AP_ground[:, 0], AP_ground[:, 1], c='red', marker='^', s=100)
    plt.scatter(UAV_optimized[:, 0], UAV_optimized[:, 1], c='green', marker='P', s=100, label='UAV (Optimized)')
    # 绘制移动轨迹
    for j in range(UAV_initial.shape[0]):
        plt.plot([UAV_initial[j, 0], UAV_optimized[j, 0]],
                 [UAV_initial[j, 1], UAV_optimized[j, 1]], 'k--', alpha=0.3)
    plt.title("Optimized Deployment (2D)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # 子图3：3D对比
    ax = plt.subplot(1, 3, 3, projection='3d')
    ax.scatter(UE_pos[:, 0], UE_pos[:, 1], UE_pos[:, 2], c='blue', alpha=0.3, label='UE')
    # ax.scatter(AP_ground[:, 0], AP_ground[:, 1], AP_ground[:, 2], c='red', marker='^', s=100, label='AP')
    # 初始位置（半透明）
    ax.scatter(UAV_initial[:, 0], UAV_initial[:, 1], UAV_initial[:, 2],
               c='orange', marker='x', s=100, alpha=0.7, label='Initial UAV')
    # 优化后位置
    ax.scatter(UAV_optimized[:, 0], UAV_optimized[:, 1], UAV_optimized[:, 2],
               c='green', marker='P', s=100, label='Optimized UAV')
    # 轨迹线
    for j in range(UAV_initial.shape[0]):
        ax.plot([UAV_initial[j, 0], UAV_optimized[j, 0]],
                [UAV_initial[j, 1], UAV_optimized[j, 1]],
                [UAV_initial[j, 2], UAV_optimized[j, 2]], 'k--', alpha=0.3)
    ax.set_title("3D Deployment Comparison")
    ax.set_zlabel('Height (m)')
    ax.legend()

    plt.tight_layout()
    plt.savefig('optimization_comparison.png', dpi=300)
    plt.show()



# ===================== 主流程 =====================
if __name__ == "__main__":
    cfg = SimulationConfig()

    # 1) 随机初始化 UE（三维）
    UE_xy = np.random.uniform(0, cfg.square_length, (cfg.num_UE, 2))
    UE_pos = np.hstack([
        UE_xy,
        np.full((cfg.num_UE, 1), cfg.heights['UE'])
    ])  # shape=(K,3)

    # 2) 随机生成地面 AP 固定位置（三维）
    #AP_xy = np.random.uniform(0, cfg.square_length, (cfg.num_AP_ground, 2))
    AP_xy=APLocation_Generation.RandomAPLocations(cfg.num_AP_ground,cfg.square_length)

    AP_ground = np.hstack([
        np.real(AP_xy),
        np.imag(AP_xy),
        np.full((cfg.num_AP_ground, 1), cfg.heights['AP_ground'])
    ])  # shape=(G,3)
    #print(AP_ground)
    # 3) 初次对 UE 进行 K-means 聚类 (仅用 x-y)
    kmeans = KMeans(n_clusters=cfg.num_UAV, random_state=0).fit(UE_pos[:, :2])
    labels = kmeans.labels_
    centroids_xy = kmeans.cluster_centers_
    # UAV 初始三维位置：z=50m
    # UAV_pos = np.hstack([
    #     centroids_xy,
    #     np.full((cfg.num_UAV, 1), cfg.heights['UAV'])
    # ])  # shape=(L,3)
    # UAV_xy = APLocation_Generation.RandomAPLocations(cfg.num_UAV, cfg.square_length)
    UAV_xy = np.random.uniform(0, cfg.square_length, (cfg.num_UAV, 2))

    UAV_pos = np.hstack([
        UAV_xy,
        np.full((cfg.num_UAV, 1), cfg.heights['UAV'])
    ])


    best_min_rate = -np.inf
    best_positions = None
    # 保存初始聚类中心位置（优化前）
    UAV_initial_pos = UAV_pos.copy()  # 深拷贝初始位置

    # 4) 迭代优化 Loop
    for t in range(100):
        if t % cfg.recluster_period == 0 and t > 0:
            kmeans = KMeans(n_clusters=cfg.num_UAV, random_state=t).fit(UE_pos[:, :2])
            labels = kmeans.labels_

        # 计算每个 UE 的速率
        # AP_all = np.vstack([AP_ground, UAV_pos])
        AP_all = UAV_pos.copy()
        rates,mask = compute_per_user_rate(
            AP_all[:,0:1], AP_all[:,1:2],
            UE_pos[:,0:1], UE_pos[:,1:2],UAV_pos[:,0:2]
        )
        min_rate = rates.min()
        # if min_rate > best_min_rate:
        #     best_min_rate = min_rate
        #     best_positions = UAV_pos.copy()

        # 计算总受力
        forces = np.zeros_like(UAV_pos)
        for j in range(cfg.num_UAV):
            # ue_idx = np.where(labels == j)[0]
            # ue_j = UE_pos[ue_idx]
            ue_j_indices = np.where(mask[:, j])[0]  # mask_uav 的列 j 对应 UAV j
            if len(ue_j_indices) == 0:
                continue
            ue_j = UE_pos[ue_j_indices]
            rates_j = rates[ue_j_indices]
            # 权重推动对速率低的用户更敏感
            # w_j = 1.0 / (rates[ue_idx] + cfg.epsilon)
            # w_j /= (w_j.sum() + cfg.epsilon)
            # forces[j] += cfg.k_min * compute_attraction_to_UE(
            #     UAV_pos[j:j+1], ue_j, weights=w_j
            # )[0]
            # w_j = 1.0 / (rates_j + cfg.epsilon)
            # w_j /= (w_j.sum() + cfg.epsilon)
            w_j=(rates_j.max() - rates_j) / (rates_j.sum() + cfg.epsilon)
            forces[j] += cfg.k_min * compute_attraction_to_UE(
                UAV_pos[j:j+1], ue_j, weights=w_j
            )[0]
        forces += cfg.alpha * compute_repulsion_between_UAV(UAV_pos)
        # forces += cfg.beta * compute_AP_UAV_sync_force(UAV_pos, AP_ground)
        # UAV 只在 x-y 面移动：归一化 & 更新
        norms = np.linalg.norm(forces[:, :2], axis=1, keepdims=True) + cfg.epsilon
        diff = cfg.step_size * (forces[:, :2] / norms)
        UAV_pos[:, :2] += diff
        UAV_pos[:, :2] = np.clip(UAV_pos[:, :2], 0, cfg.square_length)

        print(f"Iter {t}: min_rate={min_rate:.4f}")

    print("Optimization complete.")
    print("Best min_rate=", best_min_rate)
    print("Best UAV positions (x, y, z):")
    print(best_positions)
    # 在优化完成后调用（添加到主流程末尾）
    # 在优化完成后调用（使用保存的初始位置）
    print("Visualizing optimization results...")
    plot_optimization_comparison(UE_pos, AP_ground, UAV_initial_pos, UAV_pos)
