"""
Load-Balanced BVF V6 — V6 + 地面感知蒙特卡罗负载均衡

核心设计:
  Phase 1: 纯 V6 优化 → 获得高 min_rate 的稳健基线 P1
  Phase 2: 地面感知蒙特卡罗局部搜索:
    1. 地面感知负载度量: UAV 有效负载按用户对该 UAV 的依赖度加权
       (用户同时被强地面 AP 覆盖 → 对 UAV 依赖低 → 有效负载贡献小)
    2. 地面感知扰动: 过载 UAV 在地面覆盖好的区域 → 大幅移动;
       在地面覆盖差的区域 → 小幅微调; 轻载 UAV → 移向覆盖空洞
    3. 归一化联合目标: joint_score = w_min*(min_rate/ref_rate) + w_jfi*JFI_eff
       — min_rate 和 JFI_eff 均归一化后共同参与评估, 允许 min_rate 适当下降
       — 相较于旧版不对称 penalty 形式, 此目标是连续的、双向的
    
  安全: P1 与 P2 均以 joint_score 比较, 取优者; 最差等同于 V6 (P1 兜底)
"""

import numpy as np
from typing import Dict
from balanced_virtual_force_optimizer_v6 import BalancedVirtualForceOptimizerV6


class LoadBalancedBVF_V6(BalancedVirtualForceOptimizerV6):

    def __init__(self, config: Dict):
        super().__init__(config)

        self.enable_load_balance = config.get('enable_load_balance', True)
        self.load_threshold = config.get('load_threshold', 1.3)
        self.backhaul_capacity = config.get('backhaul_capacity', 500.0)

        # 归一化联合目标参数: joint = w_min*(min_rate/ref_rate) + w_jfi*JFI_eff
        # w_min + w_jfi 无需等于 1, 两者共同决定相对权重
        # ref_rate: min_rate 的参考值 (将 min_rate 归一到 ~[0,1] 量级)
        # 含义: ref_rate=60 时, min_rate 每降 6 Mbps ≈ JFI_eff 降 0.05 的代价
        self.w_min = config.get('w_min', 0.5)
        self.w_jfi = config.get('w_jfi', 0.5)
        self.ref_rate = config.get('ref_rate', 60.0)

        self.mc_rounds = config.get('mc_rounds', 8)
        self.mc_candidates = config.get('mc_candidates', 50)
        self.mc_radius = config.get('mc_radius', 60.0)

        self.eval_seed = config.get('eval_seed', 99999)

    # ── 地面感知负载度量 ──

    def compute_load_metrics(self, rates, mask, betas):
        """
        全局 AP 负载视图 + UAV 依赖度加权有效负载 + 地面覆盖质量.

        Returns dict:
          num_served_all: (G+L,) 每个 AP 服务的用户数
          ground_load: (G,) 地面 AP 负载
          uav_user_count: (L,) UAV 原始用户数
          effective_load: (L,) UAV 依赖度加权有效负载
          ground_coverage: (K,) 每个用户的地面 AP 覆盖质量 (sum of ground betas)
          avg_eff_load, overloaded, underloaded: 基于 effective_load 计算
        """
        num_served_all = mask.sum(axis=0).astype(float)
        ground_load = num_served_all[:self.G]
        uav_user_count = num_served_all[self.G:]

        mask_uav = mask[:, self.G:]
        mask_ground = mask[:, :self.G]

        # 每个用户的地面覆盖质量: 被选中服务的地面 AP 的 beta 之和
        ground_coverage = np.zeros(self.K)
        for k in range(self.K):
            serving_ground = np.where(mask_ground[k])[0]
            if len(serving_ground) > 0:
                ground_coverage[k] = betas[k, serving_ground].sum()

        # UAV 依赖度加权有效负载
        effective_load = np.zeros(self.L)
        for l in range(self.L):
            served_users = np.where(mask_uav[:, l])[0]
            for k in served_users:
                uav_beta = betas[k, self.G + l]
                serving_ground = np.where(mask_ground[k])[0]
                ground_beta_sum = betas[k, serving_ground].sum() if len(serving_ground) > 0 else 0.0
                dependency = uav_beta / (ground_beta_sum + uav_beta + 1e-12)
                effective_load[l] += dependency

        avg_eff = effective_load.mean() if effective_load.mean() > 0 else 1e-6

        # 吞吐量负载 (兼顾回程容量)
        throughput = np.zeros(self.L)
        for l in range(self.L):
            served = mask_uav[:, l]
            if served.any():
                throughput[l] = rates[served].sum()
        bh_load = throughput / self.backhaul_capacity

        # 综合负载: max(有效负载归一化, 回程占用)
        eff_norm = effective_load / max(self.K, 1)
        load_index = np.maximum(eff_norm, bh_load)
        avg_load = load_index.mean() if load_index.mean() > 0 else 1e-6

        return {
            'num_served_all': num_served_all,
            'ground_load': ground_load,
            'uav_user_count': uav_user_count,
            'effective_load': effective_load,
            'ground_coverage': ground_coverage,
            'load_index': load_index,
            'avg_load': avg_load,
            'overloaded': load_index > avg_load * self.load_threshold,
            'underloaded': load_index < avg_load * 0.75,
        }

    @staticmethod
    def jfi(x):
        s = x.sum()
        return float(s ** 2 / (len(x) * (x ** 2).sum() + 1e-12)) if s > 1e-10 else 1.0

    def joint_score(self, min_rate, jfi_val):
        """
        归一化联合目标: min_rate 和 JFI_eff 对等参与。
        min_rate 除以 ref_rate 归一化到与 JFI_eff (0~1) 相近量级。
        w_min 越高则更偏重吞吐; w_jfi 越高则更偏重公平。
        """
        return self.w_min * (min_rate / self.ref_rate) + self.w_jfi * jfi_val

    # ── 确定性评估 ──

    def _det_eval(self, UE_pos, ground_AP_pos, UAV_pos):
        """确定性评估: 返回 min_rate, sum_rate, JFI_eff, joint_score, rates, load_metrics"""
        state = np.random.get_state()
        np.random.seed(self.eval_seed)
        all_AP = np.vstack([ground_AP_pos, UAV_pos])
        _, _, betas = self.compute_channel_model(UE_pos, all_AP)
        mask = self.compute_AP_selection_mask(betas)
        rates, sum_rate = self.compute_user_rates(UE_pos, all_AP, mask)
        np.random.set_state(state)

        load_metrics = self.compute_load_metrics(rates, mask, betas)
        jfi_val = self.jfi(load_metrics['effective_load'])
        js = self.joint_score(float(rates.min()), jfi_val)
        return float(rates.min()), float(sum_rate), jfi_val, js, rates, load_metrics

    # ── 地面感知智能扰动 ──

    def _smart_perturbation(self, pos, load_metrics, UE_pos, radius):
        """
        地面感知扰动策略:
        - 过载 UAV + 地面覆盖好 → 大幅移动 (用户有地面 AP 兜底)
        - 过载 UAV + 地面覆盖差 → 小幅微调 (用户依赖该 UAV, 不能走太远)
        - 轻载 UAV → 优先移向 "地面覆盖空洞 + 用户密集" 的区域
        """
        new_pos = pos.copy()
        overloaded = load_metrics['overloaded']
        underloaded = load_metrics['underloaded']
        load_index = load_metrics['load_index']
        avg = load_metrics['avg_load']
        ground_coverage = load_metrics['ground_coverage']

        mask_uav_count = load_metrics['uav_user_count']

        # 地面覆盖中位数 — 低于此值认为是覆盖空洞
        gc_median = np.median(ground_coverage) if ground_coverage.max() > 0 else 1e-6

        for l in range(self.L):
            if overloaded[l]:
                # 该 UAV 服务的用户的平均地面覆盖质量
                if mask_uav_count[l] > 0:
                    avg_gc = ground_coverage.mean()
                    # 用该 UAV 的 effective_load 中的 dependency 反推:
                    # effective_load 高 → 用户依赖度高 → 地面覆盖差
                    eff_l = load_metrics['effective_load'][l]
                    raw_count = mask_uav_count[l]
                    avg_dependency = eff_l / (raw_count + 1e-6)

                    if avg_dependency < 0.4:
                        # 低依赖: 用户有地面 AP 兜底, 大幅移动
                        scale = radius * (1.2 + (load_index[l] / avg - 1.0))
                    else:
                        # 高依赖: 用户靠这个 UAV, 只做小幅微调
                        scale = radius * 0.3
                else:
                    scale = radius * 0.5

                offset = np.random.randn(2) * scale
                new_pos[l, :2] += offset

            elif underloaded[l]:
                # 轻载 UAV: 寻找覆盖空洞 + 用户密集的区域
                gap_users = np.where(ground_coverage < gc_median)[0]

                if len(gap_users) > 0:
                    # 向覆盖空洞用户的加权质心移动
                    gap_gc = ground_coverage[gap_users]
                    weights = 1.0 / (gap_gc + 1e-6)
                    weights /= weights.sum()
                    target = (UE_pos[gap_users, :2] * weights[:, None]).sum(axis=0)

                    direction = target - pos[l, :2]
                    d = np.linalg.norm(direction) + 1e-6
                    base_offset = direction / d * radius * 0.6
                    noise = np.random.randn(2) * radius * 0.25
                    new_pos[l, :2] += base_offset + noise
                else:
                    # 无覆盖空洞: 向最近过载 UAV 方向移动
                    ol_indices = np.where(overloaded)[0]
                    if len(ol_indices) > 0:
                        dists = [np.linalg.norm(pos[l, :2] - pos[j, :2])
                                 for j in ol_indices]
                        nearest_ol = ol_indices[np.argmin(dists)]
                        direction = pos[nearest_ol, :2] - pos[l, :2]
                        d = np.linalg.norm(direction) + 1e-6
                        base_offset = direction / d * radius * 0.5
                        noise = np.random.randn(2) * radius * 0.3
                        new_pos[l, :2] += base_offset + noise
                    else:
                        new_pos[l, :2] += np.random.randn(2) * radius * 0.4
            else:
                new_pos[l, :2] += np.random.randn(2) * radius * 0.3

        new_pos[:, :2] = np.clip(new_pos[:, :2], 50, self.square_length - 50)
        return new_pos

    # ── 蒙特卡罗搜索 ──

    def _monte_carlo_search(self, UE_pos, ground_AP_pos, init_pos, init_load, init_js):
        best_pos = init_pos.copy()
        best_js = init_js
        best_min, _, best_jfi, _, _, _ = self._det_eval(UE_pos, ground_AP_pos, best_pos)

        # 收敛历史: [(round_label, jfi, min_rate, joint_score), ...]
        history = [('P1', best_jfi, best_min, best_js)]

        eff = init_load['effective_load']
        print(f"  MC start: JointScore={best_js:.4f} | Min={best_min:.2f} | "
              f"JFI_eff={best_jfi:.4f} | EffLoad=[{','.join(f'{x:.2f}' for x in eff)}]")

        cur_pos = init_pos.copy()
        cur_load = init_load
        radius = self.mc_radius

        for rd in range(self.mc_rounds):
            improved = False

            for _ in range(self.mc_candidates):
                cand = self._smart_perturbation(cur_pos, cur_load, UE_pos, radius)
                c_min, _, c_jfi, c_js, _, c_load = self._det_eval(
                    UE_pos, ground_AP_pos, cand)

                if c_js > best_js:
                    best_js = c_js
                    best_pos = cand.copy()
                    best_min = c_min
                    best_jfi = c_jfi
                    cur_pos = cand.copy()
                    cur_load = c_load
                    improved = True

            if not improved:
                radius *= 0.7

            history.append((f'R{rd}', best_jfi, best_min, best_js))
            print(f"  MC round {rd}: JointScore={best_js:.4f} | Min={best_min:.2f} | "
                  f"JFI_eff={best_jfi:.4f} | r={radius:.0f} | "
                  f"{'improved' if improved else 'no change'}")

            if radius < 5:
                break

        eff_final = cur_load['effective_load']
        print(f"  MC done: JointScore={best_js:.4f} | Min={best_min:.2f} | "
              f"JFI_eff={best_jfi:.4f} | EffLoad=[{','.join(f'{x:.2f}' for x in eff_final)}]")
        return best_pos, best_js, best_min, best_jfi, history

    # ── 优化入口 ──

    def optimize(self, UE_pos, ground_AP_pos, UAV_pos) -> Dict:
        print("=" * 80)
        print("  Load-Balanced BVF V6 (Ground-Aware MC Search)".center(80))
        print("=" * 80)
        print(f"  P1: Pure V6 | P2: {self.mc_rounds}x{self.mc_candidates} "
              f"ground-aware MC search")
        print(f"  JointScore = {self.w_min}*(min_rate/{self.ref_rate}) + "
              f"{self.w_jfi}*JFI_eff  [equal weights, normalized]")
        print(f"  JFI on dependency-weighted effective UAV load")
        print("=" * 80)

        # ═══════ Phase 1: 纯 V6 ═══════
        p1_result = super().optimize(UE_pos, ground_AP_pos, UAV_pos)
        p1_pos = p1_result['optimized_UAV_pos']

        p1_min, p1_sum, p1_jfi, p1_js, p1_rates, p1_load = \
            self._det_eval(UE_pos, ground_AP_pos, p1_pos)
        ol1 = int(p1_load['overloaded'].sum())

        print(f"\n  P1: Min={p1_min:.2f} | JFI_eff={p1_jfi:.4f} | "
              f"JointScore={p1_js:.4f} | OL={ol1}/{self.L}")
        print(f"      UAV users: {[int(x) for x in p1_load['uav_user_count']]}")
        eff1 = p1_load['effective_load']
        print(f"      Eff load:  [{', '.join(f'{x:.2f}' for x in eff1)}]")

        if not self.enable_load_balance:
            return self._build_result(p1_pos, p1_min, p1_jfi, p1_js, p1_rates, p1_result)

        # ═══════ Phase 2: 地面感知 MC 搜索 ═══════
        # 始终运行 Phase 2 — joint_score 允许 min_rate 适当下降换取 JFI 提升
        print(f"\n  -- Phase 2: Ground-Aware Monte Carlo Search --")
        p2_pos, p2_js, p2_min, p2_jfi, mc_history = \
            self._monte_carlo_search(UE_pos, ground_AP_pos, p1_pos, p1_load, p1_js)

        if p2_js > p1_js:
            _, p2_sum, _, _, p2_rates, p2_load = self._det_eval(
                UE_pos, ground_AP_pos, p2_pos)
            d_min = p2_min - p1_min
            d_jfi = p2_jfi - p1_jfi
            d_js = p2_js - p1_js
            print(f"\n  -> P2 wins | dMin={d_min:+.2f} | dJFI_eff={d_jfi:+.4f} | "
                  f"dJointScore={d_js:+.4f}")
            print(f"      UAV users: {[int(x) for x in p2_load['uav_user_count']]}")
            eff2 = p2_load['effective_load']
            print(f"      Eff load:  [{', '.join(f'{x:.2f}' for x in eff2)}]")
            return self._build_result(p2_pos, p2_min, p2_jfi, p2_js, p2_rates,
                                      p1_result, mc_history)
        else:
            print(f"\n  -> P1 wins (JointScore: P1={p1_js:.4f}, P2={p2_js:.4f})")
            return self._build_result(p1_pos, p1_min, p1_jfi, p1_js, p1_rates,
                                      p1_result, mc_history)

    def _build_result(self, pos, min_rate, jfi_val, joint_s, rates, p1_result,
                      mc_history=None):
        p1_hist = p1_result.get('history', {})
        return {
            'optimized_UAV_pos': pos,
            'final_min_rate': float(min_rate),
            'final_sum_rate': float(rates.sum()),
            'final_rates': rates,
            'final_jfi': float(jfi_val),
            'final_joint_score': float(joint_s),
            'mc_jfi_history': mc_history or [],
            'history': {
                'min_rates': p1_hist.get('min_rates', []),
                'sum_rates': p1_hist.get('sum_rates', []),
                'jfis': [jfi_val],
            },
            'best_iteration': p1_result.get('best_iteration', 0),
            'restart_count': p1_result.get('restart_count', 0),
        }


def create_lb_v6_config() -> Dict:
    from balanced_virtual_force_optimizer_v6 import create_v6_config
    cfg = create_v6_config()
    cfg.update({
        'enable_load_balance': True,
        'load_threshold': 1.3,
        'backhaul_capacity': 500.0,
        # 归一化联合目标参数
        'w_min': 0.5,
        'w_jfi': 0.5,
        'ref_rate': 60.0,
        'mc_rounds': 8,
        'mc_candidates': 50,
        'mc_radius': 60.0,
        'eval_seed': 99999,
    })
    return cfg
