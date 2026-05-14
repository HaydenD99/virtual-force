# 参数
import math
# 网格范围

X_BOUND = [0,99]
Y_BOUND = [0,99]
# 每个格子的长度 200m
grid = 200/1000
# 网格划分
X_SIZE = X_BOUND[1] - X_BOUND[0] + 1
Y_SIZE = Y_BOUND[1] - Y_BOUND[0] + 1
# 格点总数
area_n = X_SIZE *Y_SIZE  #可部署位置处

pi = math.pi
# 规模
DNA_SIZE = area_n * 2
POP_SIZE = 1000 #小一点
# 遗传时 概率
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.01
# 迭代次数
N_GENERATIONS = 5
# 用户视距比例
NLOS_RATE = 0
# 视距与非视距通信距离
R_LOS = 25#/grid
R_NLOS = 3#0.2/grid
R_forest = 18
#R_mbs = 15
#us_M = 140  # 用户数
# 地面基站  受破坏比例
basedestroyrate = 0.9
# 地面基站的通信距离
R_base = 2/grid
# 无人机之间的距离
distance_comm = 5.6/grid
# 安全距离
safe_dist = 1.7*R_LOS

num1 = 1.01
num2 = 0
num3 = 3
num4 = 0

eita_LOS = 0.1
eita_NLOS = 21