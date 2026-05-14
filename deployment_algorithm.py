import numpy as np
import matplotlib.pyplot as plt
import APLocation_Generation
import generateUCC
np.random.seed(4873256)
# 参数设置
L = 20  # 无人机数量 (AP数量)
K = 60  # UE数量
squareLength = 100  # 区域边长 (m)
height = 200  # 无人机固定高度 (m)
dt = 1.0  # 时间步长 (s)
max_iter = 100  # 最大迭代次数
epsilon = 1e-2  # 收敛阈值
# Get AP locations and keep them fixed for all the setups
APpositions = APLocation_Generation.RandomAPLocations(L, squareLength)
APXpositions = APpositions.real
APYpositions = APpositions.imag
# print(APXpositions)
##在这里需要改成每个setup都是不同的位置 从setup开始插入遗传算法的模块
# np.save('new_storage/APpositions.npy', APpositions)
# APpositions = np.load(filename + 'APpositions.npy')
# posXY = np.random.uniform(
#         low=0,
#         high=squareLength,
#         size=(K, 2))
# UEXpositions = posXY[:, 0:1]
# UEYpositions = posXY[:, 1:2]
# UEpositions = UEXpositions + 1j * UEYpositions

# 随机初始化无人机位置 (x, y, z)
drones_pos = np.random.uniform(0, squareLength, (L, 3))  # (L, 3), z=height
drones_pos[:,0]=APXpositions[:,0]
drones_pos[:,1]=APYpositions[:,0]
drones_pos[:, 2] = height  # 固定高度

# 随机初始化UE位置 (x, y, z=0)
UE_pos = np.random.uniform(0, squareLength, (K, 3))  # (K, 3), z=0
UE_pos[:, 2] = 0  # UE在地面上
generateUCC.plot_positions_both(drones_pos[:,0],drones_pos[:,1],UE_pos[:, 0],UE_pos[:,1])
# 力场参数
k_a = 1e3 # UE吸引力系数
k_r = 1e9  # 无人机间排斥力系数
R_opt = 100.0  # 期望的无人机间距


# 主循环
prev_sum_rate = 0
for iter in range(max_iter):
    # 计算当前sum rate
    #current_sum_rate = calculate_sum_rate(drones_pos, UE_pos)
    #print(drones_pos[:,0])
    current_sum_rate = generateUCC.gerateSumRate(drones_pos[:,0].reshape(-1,1),drones_pos[:,1].reshape(-1,1),UE_pos[:,0].reshape(-1,1),UE_pos[:,1].reshape(-1,1))
    print(f"current iter is {iter},sum rate is {current_sum_rate}")

    # 检查收敛
    if abs(current_sum_rate - prev_sum_rate) < epsilon:
        print(f"Converged at iteration {iter}, sum_rate={current_sum_rate:.2f}")
        break
    prev_sum_rate = current_sum_rate

    # 计算每个无人机的合力
    forces = np.zeros((L, 2))  # 仅考虑x,y平面

    # 1. UE吸引力
    for i in range(L):
        for k in range(K):
            dist_vec = UE_pos[k, :2] - drones_pos[i, :2]
            dist = np.linalg.norm(dist_vec)
            #print(dist)
            if dist > 0:
                forces[i] += k_a * dist_vec / dist ** 2  # 吸引力方向指向UE
                #print(forces[i])

    # 2. 无人机间排斥力
    for i in range(L):
        for j in range(i + 1, L):
            dist_vec = drones_pos[i, :2] - drones_pos[j, :2]
            dist = np.linalg.norm(dist_vec)
            if dist > 0:
                # 排斥力方向远离其他无人机
                rep_force = k_r * (1 / dist - 1 / R_opt) * (dist_vec / dist ** 3) if dist < R_opt else 0
                #print(rep_force)
                forces[i] += rep_force
                forces[j] -= rep_force  # 反作用力

    # 更新无人机位置（仅更新x,y，z固定）
    drones_pos[:, :2] += dt * forces  # 简单欧拉积分

    # 边界检查（防止无人机飞出区域）
    drones_pos[:, 0] = np.clip(drones_pos[:, 0], 0, squareLength)
    drones_pos[:, 1] = np.clip(drones_pos[:, 1], 0, squareLength)

# 可视化最终部署
def plot_final_position(UE_pos, drones_pos):
    plt.figure(figsize=(6, 6))
    plt.scatter(UE_pos[:, 0], UE_pos[:, 1], c='blue', label='UEs', marker='^', s=60)
    plt.scatter(drones_pos[:, 0], drones_pos[:, 1], c='red', label='Drones', marker='o', s=100)
    plt.xlim(0, squareLength)
    plt.ylim(0, squareLength)
    plt.legend()
    plt.title("Drone Deployment (UE Attraction + Repulsion)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True)
    plt.savefig('dep.png')
plot_final_position(UE_pos, drones_pos)
print(f"Final sum rate: {current_sum_rate:.2f}")