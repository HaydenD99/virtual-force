import numpy as np
import scipy.linalg as sl
import APLocation_Generation
import functionRlocalscattering
import SpectralEfficiencyDownlink

np.random.seed(4873256)#int(time.time()))   # To get the same UE distribution every time you try something new
#Number of APs
L = 16

#Number of UEs
K = 20

#Select length of pilot of UEs
tau_p = K #Orthogonal sequences if tau_p=K, else tau_p = 10

#Select length of coherence block
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
distanceVertical = 10

#Define noise figure at AP (in dB)
noiseFigure = 7

#Compute noise power
noiseVariancedBm = -174 + 10*np.log10(B) + noiseFigure

#Angular standard deviation in the local scattering model (in degrees)
ASDdeg = 10

#Store identity matrix of size M x M
eyeM = np.identity(M)
SE_MR_equal = np.zeros((K,))

APpositions = APLocation_Generation.RandomAPLocations(L, squareLength)
# Get AP locations and keep them fixed for all the setups
##在这里需要改成每个setup都是不同的位置 从setup开始插入遗传算法的模块
# np.save('new_storage/APpositions.npy', APpositions)
# APpositions = np.load(filename + 'APpositions.npy')
APXpositions = APpositions.real
APYpositions = APpositions.imag

# Go through each random setup
def gerateSumRate():
    # Output simulation progress
    # print(n, 'setups out of', nbrOfSetups)

    UEpositions = np.zeros((K, 1), dtype='complex')
    distances = np.zeros((K, L))

    # Prepare to store normalized spatial correlation matrices
    R = np.zeros((M, M, K, L), dtype='complex')  ##归一化的空间相关矩阵

    # Prepare to store average channels gain numbers (in dB)
    channelGaindB = np.zeros((K, L))
    # Generate random UE locations together
    posXY = np.random.uniform(
        low=0,
        high=squareLength,
        size=(K, 2))
    UEXpositions = posXY[:, 0:1]
    UEYpositions = posXY[:, 1:2]
    UEpositions = UEXpositions + 1j * UEYpositions
    # start = time.perf_counter()
    angletoUE = np.zeros((K, L))

    for k in range(0, K):
        Xdist = np.matlib.repmat(UEXpositions[k, 0], L, 1) - APXpositions
        Xdistabs = np.abs(Xdist)
        temp = np.asarray(Xdistabs > squareLength / 2).nonzero()[0]
        Xdist[temp, 0] = (squareLength - Xdistabs[temp, 0]) * np.sign(-Xdist[temp, 0])
        Ydist = np.matlib.repmat(UEYpositions[k, 0], L, 1) - APYpositions
        Ydistabs = np.abs(Ydist)
        temp = np.asarray(Ydistabs > squareLength / 2).nonzero()[0]
        Ydist[temp, 0] = (squareLength - Ydistabs[temp, 0]) * np.sign(-Ydist[temp, 0])
        distances[k, :] = np.sqrt(distanceVertical ** 2 + Xdist[:, 0] ** 2 + Ydist[:, 0] ** 2)
        channelGaindB[k, :] = constantTerm - alpha * 10 * np.log10(distances[k, :])

        # Go through all APs
        for j in range(0, L):
            # Compute nominal angle between the new UE k and AP l
            angletoUE[k, j] = np.angle(Xdist[j] + 1j * Ydist[j])

            R[:, :, k, j] = functionRlocalscattering.R(M, angletoUE[k, j], ASDdeg)

    # end = time.perf_counter() - start
    # print('\n Time: ', end)
    # Generate random perturbations (shadowing) truncated at 3 dB
    for k1 in range(0,K):
        perturbation = sigma_sf*np.random.randn(1,L)
        bool1 = np.logical_or(perturbation > 3, perturbation < -3)
        while np.sum(bool1) != 0:
            perturbation[bool1] = sigma_sf*np.random.randn(1,np.sum(bool1)).reshape(np.sum(bool1))
            bool1 = np.logical_or(perturbation > 3, perturbation < -3)

        channelGainPerturbed = channelGaindB[k1,:] + perturbation
        channelGaindB[k1,:] = channelGainPerturbed

    channelGainOverNoise = channelGaindB - noiseVariancedBm
    H = np.zeros((M, nbrOfRealizations, K, L), dtype='complex')
    CH = np.sqrt(0.5) * (np.random.randn(M, nbrOfRealizations, K, L) + 1j * np.random.randn(M, nbrOfRealizations, K, L))
    betas = np.zeros((K, L))
    CorrR = np.zeros((M, M, K, L), dtype='complex')

    for j2 in range(0, L):
        for k2 in range(0, K):
            betas[k2, j2] = (10 ** (channelGainOverNoise[k2, j2] / 10))
            CorrR[:, :, k2, j2] = betas[k2, j2] * R[:, :, k2, j2]
            Rsqrt = sl.sqrtm(CorrR[:, :, k2, j2])
            H[:, :, k2, j2] = np.matmul(Rsqrt, CH[:, :, k2, j2])

    # Perform channels estimation
    # Pilot assignment
    # pilotIndex = pilot_assignment.assign_pilots(K, tau_p, betas)
    # For random pilot assignment
    # pilotIndex = np.mod(np.random.permutation(K), tau_p)
    pilotIndex = np.random.permutation(K)

    # Generate realizations of normalized noise
    Np = np.sqrt(0.5) * (
                np.random.randn(M, nbrOfRealizations, L, tau_p) + 1j * np.random.randn(M, nbrOfRealizations, L, tau_p))

    # Prepare to store results
    Hhat = np.zeros((M, nbrOfRealizations, K, L), dtype='complex')
    Hhat_MMSE_MeanSquare = np.zeros((K, L), dtype='float')

    # Go through all APs
    for l in range(0, L):

        for t in range(0, tau_p):
            # Compute processed pilot signal for all UEs that use pilot t
            yp = np.sqrt(p * tau_p) * np.sum(H[:, :, t == pilotIndex, l], 2) + Np[:, :, l, t]

            # Compute the matrix that is inverted in the MMSE estimator
            PsiInv = (p * tau_p * np.sum(CorrR[:, :, t == pilotIndex, l], 2) + eyeM)

            # Go through all UEs that use pilot t
            for k in np.argwhere(t == pilotIndex):
                RPsi = np.matmul(CorrR[:, :, int(k), l], np.linalg.inv(PsiInv))
                Hhat[:, :, int(k), l] = np.sqrt(p * tau_p) * np.matmul(RPsi, yp)
                Hhat_MMSE_MeanSquare[int(k), l] = (p * tau_p / M) * np.real(
                    np.trace(np.matmul(RPsi, CorrR[:, :, int(k), l])))

    w_MR = np.zeros((M, K, L), dtype='complex')

    # a_MR[:,k] is a_k in the paper
    a_MR = np.zeros((L, K), dtype='complex')

    # B_MR[:,:,k,i] is B_{ki} in the paper
    B_MR = np.zeros((L, L, K, K), dtype='complex')

    interf_MR = np.zeros((K, K, L), dtype='complex')

    interf2_MR = np.zeros((K, K, L), dtype='float')

    for n1 in range(0, nbrOfRealizations):
        for j3 in range(0, L):
            V_MR = Hhat[:, n1, :, j3]
            w_MR[:, :, j3] = V_MR / np.linalg.norm(V_MR, axis=0)

        for j4 in range(0, L):
            for k4 in range(0, K):
                a_MR[j4, k4] = a_MR[j4, k4] + np.matmul(np.conj(H[:, n1, k4, j4]), w_MR[:, k4, j4]) / nbrOfRealizations

                for i4 in range(0, K):
                    interf_MR[k4, i4, j4] = interf_MR[k4, i4, j4] + np.matmul(np.conj(H[:, n1, k4, j4]),
                                                                              w_MR[:, i4, j4]) / nbrOfRealizations

                    interf2_MR[k4, i4, j4] = interf2_MR[k4, i4, j4] + np.power(
                        np.abs(np.matmul(np.conj(H[:, n1, k4, j4]), w_MR[:, i4, j4])), 2) / nbrOfRealizations


    for k5 in range(0, K):
        for i5 in range(0, K):
            B_MR[:, :, k5, i5] = np.matmul(interf_MR[k5, i5, :].reshape(L, 1),
                                           np.conj(interf_MR[k5, i5, :].reshape(1, L)))

    for j5 in range(0, L):
        B_MR[j5, j5, :, :] = interf2_MR[:, :, j5]  ##接入点自身的干扰信息

    a_MR = np.abs(a_MR)
    B_MR = np.real(B_MR)

    SE_MR_equal[:] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(a_MR, B_MR, B, gammaEqual, Pmax)
    return np.sum(SE_MR_equal)/1e6
print(gerateSumRate())