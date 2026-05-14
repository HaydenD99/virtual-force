import numpy as np
import random
import math
import operator
import pickle
import time
import matplotlib.pyplot as plt
import progressbar
import copy
import generateUCCC
np.random.seed(42)  # 固定随机种子，确保可复现
# 全局常量
ROW = 1000  # 区域行数（网格）
COL = 1000  # 区域列数（网格）
glo_hight = 5  # 无人机固定高度

# 路径损耗参数
f = 1.4e9  # 载频 1.4 GHz
d0 = 1  # 参考距离 1米
c = 3e8  # 光速（m/s）
alpha = 3.5  # 路径损耗指数

# 用户和无人机参数
p_s_dBm = 20  # 无人机发射功率（dBm）
p_n_dBm = -95  # 噪声功率（dBm）

P_s = 20
P_n = 10**(-95/10)/1000
# 无人机安全距离约束（水平距离 ≥5 米）
UAV_SEC_R = 20  # 米
UAV_SEC_R2 = UAV_SEC_R ** 2

# 目标无人机数量
UAV_NUM = 9  # 全局变量：每个解的无人机数量严格限制为10

# 种群大小
POP_SIZE = 50  # 遗传算法种群数量

CLUSTER_SIZE=8
class UE:
    def __init__(self):
        self.loc = [-1, -1]  # 用户位置坐标 [x, y]


class UAV:
    def __init__(self, loc):
        self.loc = loc  # 无人机位置坐标 [x, y]


class Solution:
    def __init__(self):
        self.mat = np.zeros((ROW, COL), dtype=int)  # 无人机部署矩阵
        self.list = []  # 存储无人机对象的列表
        self.fit = -np.inf  # 适应度（最小用户速率）


def cal_distance2(loc1, loc2):
    return (loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2


def Create_UE_list_loc(ue_list):
    """生成用户可部署位置列表（全区域）"""
    for ue in ue_list:
        ue.list_loc = [[x, y] for x in range(ROW) for y in range(COL)]


def Create_UAV_matrix(ue_space):
    """生成满足安全距离的无人机部署（严格限制为UAV_NUM架）"""
    uav_list = []
    matrix = np.zeros((ROW, COL), dtype=int)

    target_num = UAV_NUM
    attempt_limit = 100000  # 最大尝试次数
    attempts = 0

    while len(uav_list) < target_num and attempts < attempt_limit:
        x = random.randint(0, ROW - 1)
        y = random.randint(0, COL - 1)
        new_uav = UAV([x, y])
        valid = True

        for existing in uav_list:
            if cal_distance2(existing.loc, new_uav.loc) <= UAV_SEC_R2:
                valid = False
                break

        if valid and matrix[x][y] == 0:
            uav_list.append(new_uav)
            matrix[x][y] = 1
            attempts = 0  # 重置尝试计数器
        else:
            attempts += 1

    # 强制限制无人机数量为target_num（即使尝试次数耗尽）
    if len(uav_list) < target_num:
        print(f"警告：生成无人机数量过少（{len(uav_list)}架），可能影响优化效果")

    return matrix, uav_list[:target_num]  # 截断到目标数量（防止超过）


def Create_solution_space(ue_space, solution_list):
    """初始化种群（确保每个解至少有8架无人机）"""
    for _ in range(POP_SIZE):
        solution = Solution()
        solution.mat, solution.list = Create_UAV_matrix(ue_space)

        # 强制保留至少8架无人机
        while len(solution.list) < 8:
            solution.mat, solution.list = Create_UAV_matrix(ue_space)

        # 截断到目标数量（UAV_NUM=10）
        solution.list = solution.list[:UAV_NUM]
        solution.mat = np.zeros((ROW, COL), dtype=int)
        for uav in solution.list:
            solution.mat[uav.loc[0], uav.loc[1]] = 1

        solution_list.append(solution)


# def Check_security_distance(solution):
#     """检查安全距离并强制限制数量到UAV_NUM"""
#     uav_list = solution.list
#     new_uav_list = []
#     matrix = np.zeros((ROW, COL), dtype=int)
#
#     for uav in uav_list:
#         x, y = uav.loc
#         if matrix[x][y] == 1:
#             continue
#         conflict = False
#         for other in new_uav_list:
#             if cal_distance2(uav.loc, other.loc) <= UAV_SEC_R2:
#                 conflict = True
#                 break
#         if not conflict:
#             new_uav_list.append(uav)
#             matrix[x][y] = 1
#
#     # 强制数量限制
#     if len(new_uav_list) > UAV_NUM:
#         random.shuffle(new_uav_list)
#         new_uav_list = new_uav_list[:UAV_NUM]
#
#     solution.list = new_uav_list
#     solution.mat = np.zeros((ROW, COL), dtype=int)
#     for uav in solution.list:
#         solution.mat[uav.loc[0], uav.loc[1]] = 1

def Check_security_distance(solution):
    """改进的安全距离检查"""
    uav_list = solution.list
    if not uav_list:
        return

    # 按某种优先级排序（如中心优先或随机）
    uav_list.sort(key=lambda u: (u.loc[0] - ROW / 2) ** 2 + (u.loc[1] - COL / 2) ** 2)

    valid_uavs = [uav_list[0]]
    occupied_mat = np.zeros((ROW, COL), dtype=bool)

    # 标记安全距离范围内的所有位置
    x, y = uav_list[0].loc
    min_x = max(0, x - int(UAV_SEC_R))
    max_x = min(ROW - 1, x + int(UAV_SEC_R))
    min_y = max(0, y - int(UAV_SEC_R))
    max_y = min(COL - 1, y + int(UAV_SEC_R))
    occupied_mat[min_x:max_x + 1, min_y:max_y + 1] = True

    for uav in uav_list[1:]:
        x, y = uav.loc
        if not occupied_mat[x, y]:
            valid_uavs.append(uav)
            # 更新占用区域
            min_x = max(0, x - int(UAV_SEC_R))
            max_x = min(ROW - 1, x + int(UAV_SEC_R))
            min_y = max(0, y - int(UAV_SEC_R))
            max_y = min(COL - 1, y + int(UAV_SEC_R))
            occupied_mat[min_x:max_x + 1, min_y:max_y + 1] = True

            if len(valid_uavs) >= UAV_NUM:
                break

    # 确保数量一致
    if len(valid_uavs) < UAV_NUM:
        # 尝试补充新的无人机
        attempts = 0
        while len(valid_uavs) < UAV_NUM and attempts < 1000:
            x, y = random.randint(0, ROW - 1), random.randint(0, COL - 1)
            if not occupied_mat[x, y]:
                valid_uavs.append(UAV([x, y]))
                min_x = max(0, x - int(UAV_SEC_R))
                max_x = min(ROW - 1, x + int(UAV_SEC_R))
                min_y = max(0, y - int(UAV_SEC_R))
                max_y = min(COL - 1, y + int(UAV_SEC_R))
                occupied_mat[min_x:max_x + 1, min_y:max_y + 1] = True
            attempts += 1

    solution.list = valid_uavs[:UAV_NUM]
    solution.mat = np.zeros((ROW, COL), dtype=int)
    for uav in solution.list:
        solution.mat[uav.loc[0], uav.loc[1]] = 1

def fspl(dist):
    """自由空间路径损耗（dB）"""
    if dist == 0:
        return 20 * math.log10(4 * math.pi * f * d0 / c)
    return 20 * math.log10(4 * math.pi * f * d0 / c) + 10 * alpha * math.log10(dist / d0)


def dBm2w(dBm):
    """dBm转瓦特"""
    return 0.001 * 10 ** (dBm / 10)


def pathlossA2G(dist2, h=10):
    """Consistent path loss calculation"""
    dist3d = math.sqrt(dist2 + h ** 2)
    return 20 * math.log10(4 * math.pi * f * d0 / c) + 10 * alpha * math.log10(dist3d / d0)

def get_user_clusters(channel, N=CLUSTER_SIZE):
    num_ue, num_bs = channel.shape
    user_clusters = []
    for k in range(num_ue):
        gains = np.abs(channel[k, :])
        sorted_idx = np.argsort(gains)[::-1]
        cluster = sorted_idx[:N].tolist()
        user_clusters.append(cluster)
    return user_clusters

# def calculate_min_rate(UE_list, BS_list):
#     """Corrected channel model implementation"""
#     num_ue = len(UE_list)
#     num_bs = len(BS_list)
#     channel = np.zeros((num_ue, num_bs), dtype=np.complex128)
#
#     # Calculate channel matrix
#     for k in range(num_ue):
#         for m in range(num_bs):
#             dist2 = (UE_list[k].loc[0] - BS_list[m].loc[0]) ** 2 + \
#                     (UE_list[k].loc[1] - BS_list[m].loc[1]) ** 2
#             pl_dB = pathlossA2G(dist2)
#             beta = 10 ** (-pl_dB / 10)
#             g = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
#             channel[k, m] = np.sqrt(beta)  # Include fading
#
#     rates = np.zeros(num_ue)
#     for ue in range(num_ue):
#         # Sum complex signals first
#         signal = 0
#         for bs in range(num_bs):
#             signal += channel[ue, bs] * np.sqrt(P_s)  # Include transmit power
#
#         signal_power = np.abs(signal) ** 2
#         snr = signal_power / P_n
#         rates[ue] = np.log2(1 + snr)
#     #print(rates)
#
#     return np.min(rates)

def cal_user_rates(UE_list, BS_list, channel):
    user_clusters = get_user_clusters(channel)
    num_ue = len(UE_list)
    num_bs = len(BS_list)
    # Noise_watt = 10 ** (Noise / 10) / 1000  # 正确转换dBm到瓦特
    rates = np.zeros(num_ue)
    uav_served_users = [[] for _ in range(num_bs)]
    for k in range(num_ue):
        for m in user_clusters[k]:
            uav_served_users[m].append(k)
    # 功率分配矩阵
    P_mk = np.zeros((num_bs, num_ue))
    for m in range(num_bs):
        num_served = len(uav_served_users[m])
        if num_served > 0:
            for k in uav_served_users[m]:
                P_mk[m, k] = P_s / num_served
        # for k in range(num_ue):
        #     # 有用信号功率（来自服务用户k的所有无人机）
        #     signal_power = sum(
        #         (abs(channel[k, m]) ** 2) * P_mk[m, k]
        #         for m in user_clusters[k]
        #     )
        #     # 干扰功率（来自服务同一用户的其他用户）
        #     interference_power = 0
        #     for m in user_clusters[k]:
        #         for u in uav_served_users[m]:
        #             if u != k:  # 同一无人机服务的其他用户
        #                 interference_power += (abs(channel[k, m]) ** 2) * P_mk[m, u]
        #     sinr = signal_power / (interference_power + P_n)
        #     rates[k] = np.log2(1 + sinr)

        for k in range(num_ue):
            signal_power = 0
            interference_power = 0
            for m in range(num_bs):
                if m in user_clusters[k]:
                    signal_power += (np.abs(channel[k, m])**2) * P_mk[m, k]
                else:
                    # 计算基站m的干扰功率（假设基站m的功率均分给其服务用户）
                    num_served_m = len(uav_served_users[m])
                    if num_served_m >0:
                        interference_power += (np.abs(channel[k, m])**2) * (P_s / num_served_m)
            snr = signal_power / (interference_power + P_n)
            rates[k] = math.log2(1 + snr)
    return rates

# def cal_user_rates(UE_list, BS_list, channel):
#     """
#     基于3GPP A2G信道模型计算每个用户的速率（含大尺度和小尺度衰落）
#
#     参数:
#         UE_list: 地面用户坐标列表，形状为 (N, 3)
#         BS_list: 无人机坐标列表，形状为 (M, 3)
#         user_clusters: 用户-无人机关联关系
#
#     返回:
#         每个用户的速率数组 (bps/Hz)
#     """
#
#     # Noise_watt = 10 ** (Noise / 10) / 1000  # 噪声功率(W)
#     user_clusters = get_user_clusters(channel)
#     num_ue = len(UE_list)
#     num_bs = len(BS_list)
#     # assert channel.shape == (num_bs, num_ue), \
#     #     f"Channel matrix shape {channel.shape} != ({num_bs}, {num_ue})"
#
#     # 预计算每个无人机服务的用户列表
#     uav_served_users = [[] for _ in range(num_bs)]
#     for k in range(num_ue):
#         for m in user_clusters[k]:
#             uav_served_users[m].append(k)
#
#     # 功率分配矩阵 (M x N)
#     P_mk = np.zeros((num_bs, num_ue))
#     for m in range(num_bs):
#         num_served = len(uav_served_users[m])
#         if num_served > 0:
#             for k in uav_served_users[m]:
#                 P_mk[m, k] = P_s / num_served
#
#     # 计算每个用户的速率
#     rates = np.zeros(num_ue)
#     for k in range(num_ue):
#         # 有用信号功率（来自服务用户k的所有无人机）
#         signal_power = sum(
#             (abs(channel[k, m]) ** 2) * P_mk[m, k]
#             for m in user_clusters[k]
#         )
#         # 干扰功率（来自服务同一用户的其他用户）
#         interference_power = 0
#         for m in user_clusters[k]:
#             for u in uav_served_users[m]:
#                 if u != k:  # 同一无人机服务的其他用户
#                     interference_power += (abs(channel[k, m]) ** 2) * P_mk[m, u]
#
#         # SINR和速率计算
#         sinr = signal_power / (interference_power + P_n)
#         rates[k] = np.log2(1 + sinr)
#     return rates
# def calculate_min_rate(UE_list, BS_list, P_max=20):
#     """
#     Cell-free下行MRT+SINR速率，严格限制每个UAV总发射功率不超过P_max（W）
#     """
#     num_ue = len(UE_list)
#     num_bs = len(BS_list)
#     H = np.zeros((num_ue, num_bs), dtype=np.complex128)
#
#     # 生成信道矩阵 H[k, m]
#     for k in range(num_ue):
#         for m in range(num_bs):
#             dist2 = (UE_list[k].loc[0] - BS_list[m].loc[0]) ** 2 + \
#                     (UE_list[k].loc[1] - BS_list[m].loc[1]) ** 2
#             pl_dB = pathlossA2G(dist2)
#             beta = 10 ** (-pl_dB / 10)
#             g = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
#             H[k, m] = np.sqrt(beta) * g
#
#     # MRT预编码
#     W = np.conj(H.T)  # shape (num_bs, num_ue)
#     for k in range(num_ue):
#         norm = np.linalg.norm(W[:, k])
#         if norm > 0:
#             W[:, k] /= norm
#
#     # 功率归一化：每个UAV的总发射功率不超过P_max
#     P_mk = np.zeros((num_bs, num_ue))
#     for m in range(num_bs):
#         norm2 = np.sum(np.abs(W[m, :]) ** 2)
#         if norm2 > 0:
#             for k in range(num_ue):
#                 P_mk[m, k] = P_max * (np.abs(W[m, k]) ** 2) / norm2
#         else:
#             P_mk[m, :] = 0
#
#     rates = np.zeros(num_ue)
#     for k in range(num_ue):
#         # 有用信号
#         signal = 0
#         for m in range(num_bs):
#             signal += H[k, m] * np.sqrt(P_mk[m, k]) * W[m, k]
#         signal_power = np.abs(signal) ** 2
#
#         # 干扰
#         interference = 0
#         for j in range(num_ue):
#             if j == k:
#                 continue
#             interf = 0
#             for m in range(num_bs):
#                 interf += H[k, m] * np.sqrt(P_mk[m, j]) * W[m, j]
#             interference += np.abs(interf) ** 2
#
#         sinr = signal_power / (P_n + interference)
#         rates[k] = np.log2(1 + sinr)
#
#     return np.min(rates)


def cellfree_channel(UE_list, BS_list):
    num_ue = len(UE_list)
    num_bs = len(BS_list)
    channel = np.zeros((num_ue, num_bs), dtype=np.complex128)
    for k in range(num_ue):
        for m in range(num_bs):
            dx = UE_list[k].loc[0] - BS_list[m].loc[0]
            dy = UE_list[k].loc[1] - BS_list[m].loc[1]
            dist2 = math.sqrt(dx**2 + dy**2)
            h = 10
            pl_dB = pathlossA2G(dist2, h)
            beta = 10 ** (-pl_dB / 10)
            g = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
            channel[k, m] = g * math.sqrt(beta)  # 使用math.sqrt而非np.sqrt
    return channel

# def Fit_func(solution, ue_coords):
#     """Wrapper that converts coordinates to UE objects"""
#     if not solution.list:
#         return -np.inf
#
#     # Create temporary UE objects
#     ue_objects = [UE() for _ in ue_coords]
#     for ue, coord in zip(ue_objects, ue_coords):
#         ue.loc = coord
#     channel = cellfree_channel(ue_objects, solution.list)
#     rates = cal_user_rates(ue_objects, solution.list, channel)
#     # 提取所有 UAV 的 x 和 y 坐标
#     x_positions = [uav.loc[0] for uav in solution.list]  # 所有 UAV 的 x 坐标
#     y_positions = [uav.loc[1] for uav in solution.list]  # 所有 UAV 的 y 坐标
#
#     # 转换为 NumPy 数组
#     APXpositions = np.array(x_positions)
#     APYpositions = np.array(y_positions)
#
#     rates = generateUCCC.generateSumRate(APXpositions=APXpositions, APYpositions=APYpositions,
#                                        UEXpositions=ue_objects[:,0], UEYpositions=ue_objects[:,1])[0]
#
#     return np.min(rates)

def Fit_func(solution, ue_coords):
    """Wrapper that converts coordinates to UE objects"""
    if not solution.list:
        return -np.inf

    # Create temporary UE objects
    ue_objects = [UE() for _ in ue_coords]
    for ue, coord in zip(ue_objects, ue_coords):
        ue.loc = coord

    channel = cellfree_channel(ue_objects, solution.list)
    rates = cal_user_rates(ue_objects, solution.list, channel)

    # 提取所有 UAV 的 x 和 y 坐标
    x_positions = [uav.loc[0] for uav in solution.list]  # 所有 UAV 的 x 坐标
    y_positions = [uav.loc[1] for uav in solution.list]  # 所有 UAV 的 y 坐标

    # 提取所有 UE 的 x 和 y 坐标
    ue_x_positions = [ue.loc[0] for ue in ue_objects]  # 所有 UE 的 x 坐标
    ue_y_positions = [ue.loc[1] for ue in ue_objects]  # 所有 UE 的 y 坐标

    # 转换为 NumPy 数组
    APXpositions = np.array(x_positions)
    APYpositions = np.array(y_positions)
    UEXpositions = np.array(ue_x_positions)
    UEYpositions = np.array(ue_y_positions)

    rates = generateUCCC.generateSumRate(
        APXpositions=APXpositions,
        APYpositions=APYpositions,
        UEXpositions=UEXpositions,
        UEYpositions=UEYpositions
    )[0]

    return np.min(rates)

def cal_fit(solution_list, ue_coords):
    """计算所有解的适应度"""
    for solution in solution_list:
        solution.fit = Fit_func(solution, ue_coords)


def Sort_fitness(solution_list):
    """按适应度排序解空间"""
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


def Rand_Exchange(solution_list, children_space):
    sol1, sol2 = RouletteWheelSelection(solution_list)

    c_row = random.randint(0, ROW - 1)
    c_col = random.randint(0, COL - 1)

    child1_mat = np.copy(sol2.mat)
    child1_mat[:c_row, :c_col] = sol1.mat[:c_row, :c_col]
    child1_mat[c_row:, c_col:] = sol1.mat[c_row:, c_col:]

    child2_mat = np.copy(sol1.mat)
    child2_mat[:c_row, :c_col] = sol2.mat[:c_row, :c_col]
    child2_mat[c_row:, c_col:] = sol2.mat[c_row:, c_col:]

    # 转换为无人机列表并检查安全距离
    child1 = Solution()
    child1.mat = child1_mat
    child1.list = [UAV([x, y]) for x in range(ROW) for y in range(COL) if child1_mat[x][y] == 1]
    Check_security_distance(child1)

    child2 = Solution()
    child2.mat = child2_mat
    child2.list = [UAV([x, y]) for x in range(ROW) for y in range(COL) if child2_mat[x][y] == 1]
    Check_security_distance(child2)

    # 添加变异操作
    if random.random() < 0.2:
        Mutate(child1)
    if random.random() < 0.2:
        Mutate(child2)

    children_space.append(child1)
    children_space.append(child2)


# def Mutate(solution):
#     if not solution.list:
#         return
#
#     idx = random.randint(0, len(solution.list) - 1)
#     uav = solution.list[idx]
#     x, y = uav.loc
#
#     dx = random.randint(-5, 5)
#     dy = random.randint(-5, 5)
#     new_x = max(0, min(x + dx, ROW - 1))
#     new_y = max(0, min(y + dy, COL - 1))
#
#     valid = True
#     for other in solution.list:
#         if [new_x, new_y] == other.loc:
#             valid = False
#             break
#         if cal_distance2([new_x, new_y], other.loc) <= UAV_SEC_R2:
#             valid = False
#             break
#
#     if valid:
#         uav.loc = [new_x, new_y]
#         solution.mat = np.zeros((ROW, COL), dtype=int)
#         for uav in solution.list:
#             solution.mat[uav.loc[0], uav.loc[1]] = 1
def Mutate(solution):
    if not solution.list:
        return

    idx = random.randint(0, len(solution.list) - 1)
    uav = solution.list[idx]
    x, y = uav.loc

    dx = random.randint(-5, 5)
    dy = random.randint(-5, 5)
    new_x = max(0, min(x + dx, ROW - 1))
    new_y = max(0, min(y + dy, COL - 1))

    valid = True
    for other in solution.list:
        if [new_x, new_y] == other.loc:
            valid = False
            break
        if cal_distance2([new_x, new_y], other.loc) <= UAV_SEC_R2:
            valid = False
            break

    if valid:
        uav.loc = [new_x, new_y]
        solution.mat = np.zeros((ROW, COL), dtype=int)
        for uav in solution.list:
            solution.mat[uav.loc[0], uav.loc[1]] = 1

def Create_children(solution_list):
    children = []
    for _ in range(POP_SIZE // 2):
        Rand_Exchange(solution_list, children)
    return children


# def Draw_result(ue_list, solution, title="Deployment Result"):
#     """绘制部署结果"""
#     plt.figure(figsize=(10, 10))
#
#     # 绘制用户位置
#     ue_coords = [ue.loc for ue in ue_list]
#     plt.scatter(*zip(*ue_coords), color='blue', marker='x', label='UE')
#
#     # 绘制无人机位置
#     uav_coords = [uav.loc for uav in solution.list]
#     plt.scatter(*zip(*uav_coords), color='red', marker='v', label='UAV')
#
#     # 绘制安全距离范围
#     for uav in solution.list:
#         circle = plt.Circle(uav.loc, UAV_SEC_R, color='r', fill=False, alpha=0.2)
#         plt.gca().add_patch(circle)
#
#     plt.xlim(-5, ROW + 5)
#     plt.ylim(-5, COL + 5)
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

def Draw_result(ue_space, best_solution):
    """绘制最终部署结果（仅显示UE和UAV位置）"""
    ue_coords = [ue.loc for ue in ue_space]
    uav_coords = [uav.loc for uav in best_solution.list]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(*zip(*ue_coords), color='blue', marker='x', label='UE')
    ax.scatter(*zip(*uav_coords), color='red', marker='v', label='UAV')

    ax.set_xlim(-5, ROW + 5)
    ax.set_ylim(-5, COL + 5)
    ax.set_aspect('equal')
    ax.legend()
    plt.title("Cell-Free MIMO deployment optimization with max-min user rate")
    plt.grid(True)
    plt.show()

def main():
    num_ue = 60  # 假设你要生成 60 个 UE
    area_size = 1000  # 假设 UE 分布在 1000x1000 的区域内

    # 随机生成 UE 坐标 (x, y)
    ue_coords = np.random.uniform(0, area_size, size=(num_ue, 2))  # shape=(60, 2)
    # 创建 UE 对象列表（保持和原代码一致的结构）
    ue_space = []
    for coord in ue_coords:
        ue = UE()
        ue.loc = coord  # 假设 UE 类有 loc 属性存储坐标
        ue_space.append(ue)
    Create_UE_list_loc(ue_space)
    # 初始化种群
    solution_list = []
    Create_solution_space(ue_space, solution_list)
    initial_solution = solution_list[0]
    ue_coords = [ue.loc for ue in ue_space]
    # Draw_result(initial_solution, ue_coords)
    start_total = time.time()

    num_generations = 100
    widgets = ["全局进度: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=num_generations).start()

    for gen in range(num_generations):
        pbar.update(gen + 1)

        # 创建子代
        children = Create_children(solution_list)

        # 合并种群
        all_solutions = solution_list + children

        # -----------关键：每一代都做一次安全距离检查-----------
        for sol in all_solutions:
            Check_security_distance(sol)
        # ---------------------------------------------------

        # 计算适应度
        ue_coords = [ue.loc for ue in ue_space]
        cal_fit(all_solutions, ue_coords)

        # 保留前POP_SIZE个解
        Sort_fitness(all_solutions)
        solution_list = all_solutions[:POP_SIZE]

        # 打印每代信息
        current_best = solution_list[0].fit
        elapsed = time.time() - start_total
        print(f"迭代第{gen + 1}/{num_generations}轮 | 最优适应度：{current_best:.4f} bps/Hz | 耗时：{elapsed:.3f}秒")

    pbar.finish()

    # 统计总耗时
    total_time = time.time() - start_total
    print(f"\n总耗时：{total_time:.2f}秒")

    best_solution = solution_list[0]
    Draw_result(ue_space, best_solution)

    print(f"最终最小用户速率：{best_solution.fit:.4f} Mbps/Hz")
    return best_solution.fit


if __name__ == "__main__":

    main()
