import numpy as np
from scipy import linalg as sl
from scipy.spatial import cKDTree
import functionRlocalscattering
import SpectralEfficiencyDownlink
#Number of UAV APs
L = 9

#Number of UEs
K = 60

#Select length of pilot of UEs
tau_p = K #Orthogonal sequences if tau_p=K, else tau_p = 10

# Select length of coherence block
tau_c = 200

prelogFactor = (tau_c-tau_p)/(tau_c)

#Number of AP antennas
M = 4

#Select the number of setups with random UE locations
nbrOfSetups = 100

#Select the number of channels realizations per setup
nbrOfRealizations = 100

## Model parameters

#Set the length in meters of the total square area
squareLength = 1000

#Number of APs per dimension
nbrAPsPerDim = int(np.sqrt(L))##modification of np.int to int, same with the following modification

#Pathloss exponent
alpha = 3.67

#Average channels gain in dB at a reference distance of 1 meter.
constantTerm = -30.5

#Standard deviation of shadow fading
sigma_sf = 1

#Define the antenna spacing (in number of wavelengths)
antennaSpacing = 1/2 #Half wavelength distance

#Distance between APs in vertical/horizontal direction
interAPDistance = int(squareLength/nbrAPsPerDim)

## Propagation parameters

#Communication bandwidth
B = 20e6

#Total uplink transmit power per UE (mW)
p = 100

#Maximum downlink transmit power per AP (mW)
Pmax = 1000

#Compute downlink power per UE in case of equal power allocation
rhoEqual = (Pmax/K)*np.ones((K,L))

#Square roots of power coefficients for equal power allocation
gammaEqual = np.sqrt(rhoEqual)

#Prepare power coefficients for the benchmark in [12]
rho_Giovanni19 = np.zeros((K,L,nbrOfSetups))

#Vertical distance between APs and UEs
distanceVertical = 150

#Define noise figure at AP (in dB)
noiseFigure = 7

#Compute noise power
noiseVariancedBm = -174 + 10*np.log10(B) + noiseFigure

#Angular standard deviation in the local scattering model (in degrees)
ASDdeg = 10

#Store identity matrix of size M x M
eyeM = np.identity(M)
SE_MR_equal = np.zeros((K,))
def generateSumRate(APXpositions, APYpositions, UEXpositions, UEYpositions):
    L = len(APXpositions)
    K = len(UEXpositions)

    AP_positions = np.stack([APXpositions[:, 0], APYpositions[:, 0]], axis=1)
    UE_positions = np.stack([UEXpositions[:, 0], UEYpositions[:, 0]], axis=1)
    # AP_positions = np.stack([APXpositions, APYpositions], axis=1)
    # UE_positions = np.stack([UEXpositions, UEYpositions], axis=1)

    # ========= 计算距离和角度（向量化） =========
    diff = UE_positions[:, None, :] - AP_positions[None, :, :]  # (K, L, 2)
    distances = np.sqrt(np.sum(diff**2, axis=-1) + distanceVertical ** 2)  # (K, L)
    angles = np.arctan2(diff[..., 1], diff[..., 0])  # (K, L)

    channelGaindB = constantTerm - alpha * 10 * np.log10(distances)
    channelGainOverNoise = channelGaindB - noiseVariancedBm
    betas = 10 ** (channelGainOverNoise / 10)  # (K, L)

    # ========= 空间相关矩阵 R（批量计算） =========
    R = np.zeros((M, M, K, L), dtype='complex')
    for k in range(K):
        for l in range(L):
            R[:, :, k, l] = functionRlocalscattering.R(M, angles[k, l], ASDdeg)

    CorrR = betas[None, None, :, :] * R

    # ========= 生成信道矩阵 =========
    CH = np.sqrt(0.5) * (np.random.randn(M, nbrOfRealizations, K, L) +
                          1j * np.random.randn(M, nbrOfRealizations, K, L))
    H = np.empty_like(CH, dtype='complex')
    for k in range(K):
        for l in range(L):
            Rsqrt = sl.sqrtm(CorrR[:, :, k, l])
            H[:, :, k, l] = Rsqrt @ CH[:, :, k, l]

    # ========= 导频分配 =========
    pilotIndex = np.random.permutation(K)
    Np = np.sqrt(0.5) * (np.random.randn(M, nbrOfRealizations, L, tau_p) +
                         1j * np.random.randn(M, nbrOfRealizations, L, tau_p))
    eyeM = np.eye(M)

    Hhat = np.zeros_like(H)
    Hhat_MMSE_MeanSquare = np.zeros((K, L))

    for l in range(L):
        for t in range(tau_p):
            indices = np.where(pilotIndex == t)[0]
            if len(indices) == 0:
                continue
            yp = np.sqrt(p * tau_p) * np.sum(H[:, :, indices, l], axis=2) + Np[:, :, l, t]
            PsiInv = (p * tau_p * np.sum(CorrR[:, :, indices, l], axis=2) + eyeM)
            PsiInvInv = np.linalg.inv(PsiInv)
            for k in indices:
                RPsi = CorrR[:, :, k, l] @ PsiInvInv
                Hhat[:, :, k, l] = np.sqrt(p * tau_p) * RPsi @ yp
                Hhat_MMSE_MeanSquare[k, l] = (p * tau_p / M) * np.real(np.trace(RPsi @ CorrR[:, :, k, l]))

    # ========= AP选择和功率分配 =========
    nearest_indices = np.argpartition(distances, 6, axis=1)[:, :3]
    mask = np.zeros((K, L), dtype=bool)
    np.put_along_axis(mask, nearest_indices, True, axis=1)

    numServedPerAP = mask.sum(axis=0)
    rho = np.zeros((K, L))
    for l in range(L):
        if numServedPerAP[l] > 0:
            rho[mask[:, l], l] = Pmax / numServedPerAP[l]
    gamma = np.sqrt(rho)

    # ========= 下行速率计算 =========
    Hhat_uc = Hhat * mask[np.newaxis, np.newaxis, :, :]

    a_MR, B_MR = compute_a_B_MR(H, Hhat_uc, gamma, Pmax)  # 推荐将 MR 相关计算封装单独函数加速

    SE_MR_equal[:] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(a_MR, B_MR, B, gamma, Pmax)
    return SE_MR_equal / 1e6, np.sum(SE_MR_equal) / 1e6, mask

def compute_a_B_MR(H, Hhat_uc, gamma, Pmax):
    M, N, K, L = H.shape

    # Beamforming vectors
    w_MR = Hhat_uc / (np.linalg.norm(Hhat_uc, axis=0, keepdims=True) + 1e-12)  # (M, N, K, L)

    # Compute a_MR efficiently: a_MR[j, k] = E[ conj(h_{j,k})^T w_{j,k} ]
    # a_MR = np.einsum('mnkl,mnkl->lk', np.conj(H), w_MR).T / N  # shape (L, K)
    a_MR = np.einsum('mnkl,mnkl->lk', np.conj(H), w_MR) / N  # shape (L, K)

    # Compute interference power
    interf_MR = np.einsum('mnkl,mnil->kiln', np.conj(H), w_MR).mean(axis=-1)  # (K, K, L)

    B_MR = np.zeros((L, L, K, K), dtype=np.float64)

    for k in range(K):
        for i in range(K):
            # B_MR[:, :, k, i] = interf_MR[k, i, :].reshape(L, 1) @ interf_MR[k, i, :].conj().reshape(1, L)
            B_MR[:, :, k, i] = np.outer(interf_MR[k, i, :], interf_MR[k, i, :].conj()).real

    # Correct diagonal
    interf2_MR = np.abs(interf_MR) ** 2  # (K, K, L)
    for l in range(L):
        B_MR[l, l, :, :] = interf2_MR[:, :, l]  # (K, K)

    a_MR = np.abs(a_MR)
    return a_MR, B_MR
