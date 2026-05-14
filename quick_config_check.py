"""
Cell-Free 配置快速可行性检验 (无优化，只评估随机部署的速率分布)
几秒内完成，用于验证 (G+L)*M >> K 的条件是否满足
"""
import numpy as np
import warnings; warnings.filterwarnings('ignore')

from balanced_virtual_force_optimizer_v6 import BalancedVirtualForceOptimizerV6, create_v6_config

M       = 4
SEEDS   = [42, 73, 99]

CONFIGS = {
    'OLD (K=60,L=9,G=4)': dict(K=60, L=9, G=4),
    'A: K=20,L=9,G=4':    dict(K=20, L=9, G=4),
    'B: K=30,L=9,G=4':    dict(K=30, L=9, G=4),
    'C: K=30,L=9,G=6':    dict(K=30, L=9, G=6),
    'D: K=40,L=9,G=6':    dict(K=40, L=9, G=6),
}

def gap_layout(G, sq=1000):
    if G == 4:
        return np.array([[x, y, 15.0] for x in [250,750] for y in [250,750]])
    elif G == 6:
        return np.array([[x, y, 15.0]
                         for x in [200,500,800] for y in [333,667]])
    elif G == 9:
        sp = sq/4
        return np.array([[(i+1)*sp,(j+1)*sp,15.0]
                         for i in range(3) for j in range(3)])
    n = int(np.ceil(np.sqrt(G))); sp = sq/(n+1)
    return np.array([[(i+1)*sp,(j+1)*sp,15.0]
                     for i in range(n) for j in range(n)])[:G]

def uav_grid(L, sq=1000):
    n = int(np.ceil(np.sqrt(L))); sp = sq/(n+1)
    return np.array([[(i+1)*sp,(j+1)*sp,50.0]
                     for i in range(n) for j in range(n)])[:L]

print(f"\n{'='*72}")
print(f"  快速配置可行性检验  (M={M}, num_serving_APs=3, 随机部署)")
print(f"{'='*72}")
print(f"{'配置':<24} {'比值':>6} {'min_r':>7} {'mean_r':>8} {'JFI':>7} {'0-rate':>7}")
print(f"  {'-'*68}")

for cfg_name, cfg in CONFIGS.items():
    K, L, G = cfg['K'], cfg['L'], cfg['G']
    ratio = (G+L)*M / K

    v6_cfg = create_v6_config()
    v6_cfg.update({'num_UE':K,'num_UAV':L,'num_ground_AP':G,
                   'tau_p':K,'num_serving_APs':3,'nbrOfRealizations':30})
    ev = BalancedVirtualForceOptimizerV6(v6_cfg)

    mrs, means, jfis, zeros = [], [], [], []
    for s in SEEDS:
        np.random.seed(s)
        UE_pos = np.column_stack([np.random.uniform(50,950,(K,2)), np.ones(K)*1.65])
        gAP    = gap_layout(G)
        UAV_pos = uav_grid(L)
        all_AP = np.vstack([gAP, UAV_pos])

        _, _, betas = ev.compute_channel_model(UE_pos, all_AP)
        mask = ev.compute_AP_selection_mask(betas)
        rates, _ = ev.compute_user_rates(UE_pos, all_AP, mask)

        # JFI_eff
        mu = mask[:,G:]; mg = mask[:,:G]
        bu = betas[:,G:]; bg = betas[:,:G]
        gcov = np.array([bg[k,np.where(mg[k])[0]].sum() for k in range(K)])
        eff = np.zeros(L)
        for l in range(L):
            for k in np.where(mu[:,l])[0]:
                eff[l] += bu[k,l]/(gcov[k]+bu[k,l]+1e-12)
        s2 = eff.sum()
        jfi = float(s2**2/(L*(eff**2).sum()+1e-12)) if s2>1e-10 else 1.0

        mrs.append(float(rates.min()))
        means.append(float(rates.mean()))
        jfis.append(jfi)
        zeros.append(int((rates < 0.1).sum()))

    flag = '✓✓' if ratio>=2.0 else ('✓' if ratio>=1.5 else '✗')
    print(f"  {cfg_name:<22} {ratio:>5.2f}{flag}"
          f" {np.mean(mrs):>7.2f}"
          f" {np.mean(means):>8.2f}"
          f" {np.mean(jfis):>7.4f}"
          f" {np.mean(zeros):>6.1f}users")

print(f"\n  说明: '0-rate' 是平均速率<0.1 Mbps的用户数（应为0）")
print(f"  建议: 选择 ratio>=2.0、min_rate>5 Mbps 且 0-rate=0 的配置")
print(f"{'='*72}\n")
