import numpy as np
import scipy.linalg as sl
import APLocation_Generation
import functionRlocalscattering
#Pathloss exponent
alpha = 3.67

#Average channels gain in dB at a reference distance of 1 meter.
constantTerm = -30.5

#Standard deviation of shadow fading
sigma_sf = 1

#Define the antenna spacing (in number of wavelengths)
antennaSpacing = 1/2 #Half wavelength distance

squareLength=1000
L=16
K=20
M=4
tau_p=K ##pilot index

#Communication bandwidth
B = 20e6

#Total uplink transmit power per UE (mW)
p = 100

#Maximum downlink transmit power per AP (mW)
Pmax = 1000

#Define noise figure at AP (in dB)
noiseFigure = 7

#Store identity matrix of size M x M
eyeM = np.identity(M)

#Compute noise power
noiseVariancedBm = -174 + 10*np.log10(B) + noiseFigure

#Angular standard deviation in the local scattering model (in degrees)
ASD_deg = 10

def generate_cell_free_channels(L, K, M, APpositions, distanceVertical=10,apply_shadowing=False):
    """
    生成Cell-Free MIMO系统的信道模型

    参数:
    L: AP的数量
    K: UE的数量
    M: 每个AP的天线数量
    squareLength: 二维区域的边长（米）
    alpha: 路径损耗指数
    constantTerm: 参考距离（1米）的路径损耗常数（dB）
    noiseVariancedBm: 噪声功率（dBm）
    ASD_deg: 散射角的标准差（度）
    antennaSpacing: 天线间距（以波长为单位）
    distanceVertical: AP和UE的垂直距离（米）
    sigma_sf: 阴影衰落的标准差（dB）
    apply_shadowing: 是否应用阴影衰落（布尔值）
    nbrOfRealizations: 每个配置的信道实现次数

    返回:
    H: 信道矩阵 (M, 实现次数, K, L)
    APpositions: AP的位置 (复数数组)
    UEpositions: UE的位置 (复数数组)
    betas: 信道增益（线性标度） (K, L)
    angles: UE与AP之间的角度 (K, L)
    R_corr: 空间相关矩阵 (M, M, K, L)
    """
    # 生成AP的位置（网格分布）

    APX = APpositions.real
    APY = APpositions.imag

    # 生成UE的位置（均匀随机分布）
    UEpositions=generate_UE_positions(K, squareLength)
    # UEXY = np.random.uniform(0, squareLength, (K, 2))
    # UEpositions = UEXY[:, 0] + 1j * UEXY[:, 1]

    # 计算AP和UE之间的距离和角度
    UE_X = UEpositions.real.reshape(-1, 1)
    UE_Y = UEpositions.imag.reshape(-1, 1)
    distances = np.zeros((K, L))
    channelGaindB = np.zeros((K, L))
    angles= np.zeros((K, L))
    R_corr = np.zeros((M, M, K, L), dtype='complex')
    for k in range(0, K):
        Xdist = np.matlib.repmat(UE_X[k, 0], L, 1) - APX
        Xdistabs = np.abs(Xdist)
        temp = np.asarray(Xdistabs > squareLength / 2).nonzero()[0]
        Xdist[temp, 0] = (squareLength - Xdistabs[temp, 0]) * np.sign(-Xdist[temp, 0])
        Ydist = np.matlib.repmat(UE_Y[k, 0], L, 1) - APY
        Ydistabs = np.abs(Ydist)
        temp = np.asarray(Ydistabs > squareLength / 2).nonzero()[0]
        Ydist[temp, 0] = (squareLength - Ydistabs[temp, 0]) * np.sign(-Ydist[temp, 0])
        distances[k, :] = np.sqrt(distanceVertical ** 2 + Xdist[:, 0] ** 2 + Ydist[:, 0] ** 2)
        channelGaindB[k, :] = constantTerm - alpha * 10 * np.log10(distances[k, :])

        # Go through all APs
        for j in range(0, L):
            # Compute nominal angle between the new UE k and AP l
            angles[k, j] = np.angle(Xdist[j] + 1j * Ydist[j])

            R_corr[:, :, k, j] = functionRlocalscattering.R(M, angles[k, j], ASD_deg)
    # # 计算dx和dy，考虑周期性边界条件
    # dx = UE_X - APX
    # dy = UE_Y - APY
    #
    # # 调整dx和dy为环形距离（超过half的用另一边的最短距离）
    # dx = dx - squareLength * np.round(dx / squareLength)
    # dx = np.where(dx > squareLength / 2, dx - squareLength, dx)
    # dx = np.where(dx < -squareLength / 2, dx + squareLength, dx)
    #
    # dy = dy - squareLength * np.round(dy / squareLength)
    # dy = np.where(dy > squareLength / 2, dy - squareLength, dy)
    # dy = np.where(dy < -squareLength / 2, dy + squareLength, dy)
    #
    # # 计算三维距离和角度
    # distances = np.sqrt(dx ** 2 + dy ** 2 + distanceVertical ** 2)
    # angles = np.arctan2(dy, dx)  # 角度以弧度表示
    #
    # # 计算路径损耗后的信道增益（dB）
    # channelGaindB = constantTerm - alpha * 10 * np.log10(distances)
    #
    # # 应用阴影衰落（可选）
    # if apply_shadowing:
    #     channelGaindB = apply_shadowing_effect(channelGaindB, sigma_sf)
    #
    # 计算信道增益（线性标度）
    channelGainOverNoise = channelGaindB - noiseVariancedBm
    betas = 10 ** (channelGainOverNoise / 10)
    # R_corr = np.zeros((M, M, K, L), dtype='complex')
    # for k in range(K):
    #     for l in range(L):
    #         # 生成空间相关矩阵R（需依赖外部模块functionRlocalscattering）
    #         R_corr = functionRlocalscattering.R(M, angles[k,l], ASD_deg)

    # 生成复高斯随机信道向量CH
    CH_real = np.random.randn(M, K, L)
    CH_imag = np.random.randn(M, K, L)
    CH = np.sqrt(0.5) * (CH_real + 1j * CH_imag)

    # 构建信道矩阵H
    H = np.zeros((M, K, L), dtype=np.complex128)
    corrR = np.zeros((M, M, K, L), dtype='complex')
    for l in range(L):
        for k in range(K):
            # 计算当前的协方差矩阵
            corrR[:,:,k,l] = betas[k, l] * R_corr[:, :, k, l]
            Rsqrt = sl.sqrtm(corrR[:,:,k,l])  # 矩阵平方根
            H[:, k, l] = Rsqrt @ CH[:,  k, l]

    return H,corrR

def generate_system_model(H,CorrR):
    pilotIndex = np.random.permutation(K)
    Np = np.sqrt(0.5) * (
                np.random.randn(M, L, tau_p) + 1j * np.random.randn(M, L, tau_p))
    # Prepare to store results
    Hhat = np.zeros((M, K, L), dtype='complex')
    Hhat_MMSE_MeanSquare = np.zeros((K, L), dtype='float')

    # Go through all APs
    for l in range(0, L):

        for t in range(0, tau_p):
            # Compute processed pilot signal for all UEs that use pilot t
            yp = np.sqrt(p * tau_p) * np.sum(H[:, t == pilotIndex, l], 1) + Np[:, l, t]

            # Compute the matrix that is inverted in the MMSE estimator
            PsiInv = (p * tau_p * np.sum(CorrR[:, :, t == pilotIndex, l], 2) + eyeM)

            # Go through all UEs that use pilot t
            for k in np.argwhere(t == pilotIndex):
                RPsi = np.matmul(CorrR[:, :, int(k), l], np.linalg.inv(PsiInv))
                Hhat[ :, int(k), l] = np.sqrt(p * tau_p) * np.matmul(RPsi, yp)
                Hhat_MMSE_MeanSquare[int(k), l] = (p * tau_p / M) * np.real(
                    np.trace(np.matmul(RPsi, CorrR[:, :, int(k), l])))
    a_MR = np.zeros((L, K), dtype='complex')
    w_MR = np.zeros((M, K, L), dtype='complex')
    B_MR = np.zeros((L, L, K, K), dtype='complex')
    interf_MR = np.zeros((K, K, L), dtype='complex')
    interf2_MR = np.zeros((K, K, L), dtype='float')

    for l in range(L):
        V_MR=Hhat[:,:,l]
        w_MR=V_MR/np.linalg.norm(V_MR,axis=0)



    return Hhat

def apply_shadowing_effect(channelGaindB, sigma_sf):
    """
    添加截断阴影衰落（截断范围±3dB）
    """
    K, L = channelGaindB.shape
    for k in range(K):
        # 生成截断的高斯噪声
        perturbation = sigma_sf * np.random.randn(L)
        while np.any(perturbation > 3) or np.any(perturbation < -3):
            # 找出超过范围的索引
            indices = np.where((perturbation > 3) | (perturbation < -3))
            perturbation[indices] = sigma_sf * np.random.randn(len(indices[0]))

        channelGaindB[k] += perturbation
    return channelGaindB


def generate_spatial_correlation_matrices(M, K, L, angles, ASD_deg, antennaSpacing):
    """
    根据角度和散射角标准差生成空间相关矩阵（调用外部模块）
    """
    ASD_rad = np.deg2rad(ASD_deg)
    R_corr = np.zeros((M, M, K, L), dtype=np.complex128)

    for k in range(K):
        for l in range(L):
            theta = angles[k, l]
            # 调用外部模块的R函数（需用户提供具体实现）
            R_corr[:, :, k, l] = functionRlocalscattering.R(M, theta, ASD_deg)

    return R_corr

def generate_UE_positions(K, squareLength):
    """
    生成UE的随机位置（均匀分布）
    """
    posXY = np.random.uniform(0, squareLength, (K, 2))
    return posXY[:, 0] + 1j * posXY[:, 1]



APpositions = APLocation_Generation.RandomAPLocations(L, squareLength)
HH,CorrR=generate_cell_free_channels(L, K, M, APpositions, distanceVertical=10,apply_shadowing=False)
Hhat=generate_system_model(HH,CorrR)
print(Hhat)