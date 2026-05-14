#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 23:34:47 2019
Modified May 2025 to integrate simplified pathloss & clustering
"""

import numpy as np
from sklearn.cluster import KMeans

def Calculate_SINR_and_SE_DL(signal, interference, B, gammaEqual, noiseVariance):
    """
    signal:    (L, K)  每个 AP→UE 的有效通道增益 γ·β
    interference: (L, L, K, K)
    B:         带宽 (Hz)
    gammaEqual: (K, L)
    noiseVariance: 噪声功率 (线性)
    返回:   SE_MR_equal, 形状 (K,), 单位 bit/s
    """
    L, K = signal.shape
    SE_MR_equal = np.zeros(K)
    for k in range(K):
        # 11期望信号功率：
        num = (signal[:, k:k+1].T @ gammaEqual[k:k+1, :].T)**2  # shape (1,1)
        # 22 把噪声当做baseline，再加干扰
        den = noiseVariance
        for i in range(K):
            # UE i 在所有 AP 的发射对 UE k 构成干扰
            den += (gammaEqual[i:i+1, :]
                    @ interference[:, :, k, i]
                    @ gammaEqual[i:i+1, :].T)
        SE_MR_equal[k] = B * np.log2(1 + num/den)
    return SE_MR_equal


def generate_simplified_rates(
    APXpositions, APYpositions,
    UEXpositions, UEYpositions,
    cluster_labels,           # (K,) 每 UE 的簇编号
    UAV_positions_xy,         # (Lu,2)
    Pmax,                     # 单个 AP/UAV 发射功率 (线性)
    B,                        # 带宽 Hz
    noiseVariancedBm,         # dBm
    constantTerm=-30.5,
    alpha_ground=3.67,
    alpha_uav=2.0,
    h_AP=15.0,
    h_UE=1.65,
    h_UAV=50.0
):
    K = len(UEXpositions)
    Lg = len(APXpositions)
    Lu = UAV_positions_xy.shape[0]
    # L = Lg + Lu
    L=Lu

    # 1) 服务掩码
    mask = np.zeros((K, L), bool)
    for k in range(K):
        mask[k, cluster_labels[k]] = True
        # mask[k, Lg + cluster_labels[k]] = True

    # 2) 坐标三维
    # AP_pos = np.column_stack([APXpositions, APYpositions, np.full(Lg, h_AP)])
    UAV_pos = np.column_stack([UAV_positions_xy[:,0], UAV_positions_xy[:,1], np.full(Lu, h_UAV)])
    # allAP_pos = np.vstack([AP_pos, UAV_pos])
    allAP_pos = UAV_pos
    UE_pos = np.column_stack([UEXpositions, UEYpositions, np.full(K, h_UE)])

    # 3) 距离
    dists = np.linalg.norm(UE_pos[:,None,:] - allAP_pos[None,:,:], axis=2)  # (K,L)

    # 4) 路损指数
    # exponents = np.hstack([np.full(Lg, alpha_ground), np.full(Lu, alpha_uav)])[None,:]
    exponents = np.full(Lu,alpha_uav)[None,:]
    # 5) 大尺度增益 β (linear)
    beta_dB = constantTerm - 10 * exponents * np.log10(dists + 1e-9)
    beta_lin = 10**(beta_dB/10)  # (K,L)

    # 6) 功率分配 γ^2 = ρ
    numServed = mask.sum(axis=0)  # (L,)
    rho = np.zeros((K,L))
    for j in range(L):
        if numServed[j] > 0:
            rho[mask[:,j], j] = Pmax/numServed[j]
    gamma = np.sqrt(rho)          # (K,L)

    # 7) 构造 signal 和 interference
    #    signal[j,k] = γ[k,j] * β[k,j]
    signal = (gamma * beta_lin).T  # (L,K)

    #    interference[j1,j2,k,i] = γ[i,j2] * β[k,j2] if j1==j2 else 0
    interference = np.zeros((L, L, K, K))
    for j in range(L):
        # 仅对角元有值
        # UE i 用 AP j 的信号干扰 UE k
        # 量级 ~ γ[i,j]·β[k,j]
        interference[j, j, :, :] = rho[:, j][None, :] * beta_lin[:, j][None, :]
        # shape (1,K) 广播到 (K,K)
        # axis0: k (受害 UE), axis1: i (干扰 UE)
    # 乘以 mask：没有服务的 AP 不计
    # 如果 AP j 不服务 UE i 或 k，则干扰/信号为零
    for k in range(K):
        for j in range(L):
            if not mask[k,j]:
                signal[j,k] = 0
                rho[k,j] = 0
                gamma[k,j] = 0
                interference[j,j,k,:] = 0
    # 8) 调用 SINR 和 SE 计算
    # 注意：noiseVariancedBm 是 dBm，需要转换到线性瓦特
    noise_lin_W = 10 ** (noiseVariancedBm / 10) * 1e-3
    # （dBm→dBW：减 30，再 10^(·/10)）
    noise_lin_W = 10 ** ((noiseVariancedBm - 30) / 10)

    SE_bps = Calculate_SINR_and_SE_DL(
        signal, interference,
        B,
        gamma,  # shape (K,L)
        noise_lin_W
    )
    SE_Mbps = SE_bps   # 转换到 Mbps
    return SE_Mbps, np.sum(SE_Mbps)


# if __name__ == "__main__":
#     # 示例数据
#     K, Lg, Lu = 100, 16, 4
#     np.random.seed(0)
#     UEX, UEY = np.random.rand(K)*1000, np.random.rand(K)*1000
#     APX, APY = np.random.rand(Lg)*1000, np.random.rand(Lg)*1000
#
#     # 外部聚类
#     ue_xy = np.column_stack([UEX, UEY])
#     kmeans = KMeans(n_clusters=Lu, random_state=0).fit(ue_xy)
#     labels = kmeans.labels_
#     UAV_centers = kmeans.cluster_centers_
#     #print(UAV_centers)
#
#     # 运行
#     SEs, total = generate_simplified_rates(
#         APXpositions=APX, APYpositions=APY,
#         UEXpositions=UEX, UEYpositions=UEY,
#         cluster_labels=labels,
#         UAV_positions_xy=UAV_centers,
#         Pmax=0.1, B=20e6,
#         noiseVariancedBm=-174+10*np.log10(20e6)
#     )
#     print("每 UE SE (Mbps):", SEs)
#     print("总 SE (Mbps):", total)
