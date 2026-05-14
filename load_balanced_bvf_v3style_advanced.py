"""
LB-BVF V3-Pro Final — 负载均衡虚拟力终极版

三大支柱:

[支柱1] 负载感知力场（继承并强化自 V3-Pro v7）
  ▸ 幅度缩放: eff_load > 1.3× avg → ×0.55 (过载减弱引力)
              eff_load < 0.75× avg → ×1.6  (轻载增强引力)
  ▸ 方向偏置 (dep_bias=0.45): 过载 UAV 合力方向向地面 AP 覆盖充足区域偏转
      dep_bias = 1 - 0.45 × β_uav/(β_uav+β_ground)
      高依赖用户(地面差) → 保留吸引  低依赖用户(地面好) → 减弱吸引
      → 过载 UAV 自然漂向可卸载区域，高依赖用户不被抛弃
  ▸ 负载度量: eff_load = Σ β_uav/(β_uav+β_ground)  (Cell-free感知，betas稳定)

[支柱2] JFI优先双目标框架（新设计）
  ▸ JointScore = 0.35*(min/ref) + 0.65*JFI_eff   (JFI权重更高)
  ▸ 软下限保护: min_rate < floor_rate(48 Mbps) → JS × (min/floor)²
      效果: JFI提升0.1 ≈ min_rate提升11 Mbps 等价
           低于48 Mbps 时 min_rate 被非线性惩罚，防崩溃
  ▸ V3-Pro v7 重启策略 (proven 9/10):
      过载 UAV → 远离服务簇质心 (分散)
      轻载 UAV → 向最低速率用户移动 (精准救援, 非质心陷阱)

[支柱3] 实时3D高度势能力（架构融合设计）
  ▸ 高度势能力直接作为虚拟力框架的第三维分量 F_z，与 (x,y) 力统一更新
  ▸ 过载 UAV: 向上斥力 F_z_rep = max(0, ratio-thr) × k_z_rep
      (飞高 → 覆盖半径扩大 → 流量自然稀释)
  ▸ 弹簧回归力: F_z_spring = -k_z_spring × (h-h_opt)/(h_max-h_opt)
      (防止无限攀升, 轻载/均衡时自动回落至最优高度)
  ▸ 平衡高度解析式:
      h_eq = h_opt + (h_max-h_opt) × excess × k_z_rep / k_z_spring
      例: 超载比=1.4 → h_eq≈63m | 超载比=1.6 → h_eq≈90m | 超载比=2.0 → h_eq≈143m
  ▸ 单向向上设计 (只有过载才推高，轻载/均衡弹簧回归 h_opt):
      过载 UAV (ratio>thr): 向上斥力 + 弹簧 → h_eq = h_opt + span × excess × k_z_rep/k_z_spring
        e.g. ratio=1.4→h_eq≈70m | ratio=1.6→h_eq≈90m | ratio=2.0→h_eq≈143m
      轻载/均衡 UAV: 弹簧回归 h_opt=50m (保持最优信号和覆盖稳定)
  ▸ 高度约束: [50, 150]m | alt_step_max=1.5m/iter (含4段衰减)
  ▸ z-力与 (x,y) 力独立归一化, 互不干扰
"""

import numpy as np
from typing import Dict, List
from balanced_virtual_force_optimizer_v6 import BalancedVirtualForceOptimizerV6


class LoadBalancedBVF_V3Style(BalancedVirtualForceOptimizerV6):

    def __init__(self, config: Dict):
        super().__init__(config)

        # ── 幅度缩放参数 ──
        self.overload_damping = config.get('overload_damping', 0.55)
        self.underload_boost  = config.get('underload_boost', 1.6)
        self.load_threshold   = config.get('load_threshold', 1.3)

        # ── 方向偏置强度 ──
        self.dep_bias_strength = config.get('dep_bias_strength', 0.45)

        # ── JFI 优先双目标框架 ──
        self.w_min      = config.get('w_min', 0.35)
        self.w_jfi      = config.get('w_jfi', 0.65)
        self.ref_rate   = config.get('ref_rate', 60.0)
        self.floor_rate = config.get('floor_rate', 48.0)   # 软下限 = 0.8 × ref
        self.eval_seed  = config.get('eval_seed', 99999)

        # ── 运动学阻尼 ──
        self.damping_coeff = config.get('damping_coeff', 0.15)
        self._velocity = None

        # ── 3D 实时高度势能力参数 ──
        self.k_z_rep      = config.get('k_z_rep', 0.8)       # 过载向上斥力强度
        self.k_z_spring   = config.get('k_z_spring', 0.6)   # 弹簧回归力强度
        self.alt_step_max = config.get('alt_step_max', 1.5)  # 最大高度步长 (m/iter)
        self.h_min_uav    = config.get('h_min_uav', 50.0)    # 最低飞行高度 (m)
        self.h_max_uav    = config.get('h_max_uav', 150.0)  # 最高飞行高度 (m)
        self.h_opt_uav    = config.get('h_opt_uav', 50.0)   # 均衡目标高度 (m)

        # ── 内部状态 ──
        self._load_factors  = np.ones(self.L)
        self._eff_load      = np.zeros(self.L)
        self._ground_cov    = np.zeros(self.K)
        self._jfi_current   = 1.0

    # ================================================================
    #  负载度量
    # ================================================================

    def _compute_ground_aware_load(self, mask, betas):
        """
        依赖度加权有效负载:
          eff_load_l = Σ_{k∈S_l} β_{k,l}^UAV / (β_{k,l}^UAV + Σ_g β_{k,g}^ground)
        基于 betas (大尺度衰落), 无随机噪声, 适合迭代内使用.
        """
        mask_uav    = mask[:, self.G:]
        betas_uav   = betas[:, self.G:]
        betas_gnd   = betas[:, :self.G]
        mask_gnd    = mask[:, :self.G]

        ground_cov = np.zeros(self.K)
        for k in range(self.K):
            sg = np.where(mask_gnd[k])[0]
            ground_cov[k] = betas_gnd[k, sg].sum() if len(sg) > 0 else 0.0

        eff_load = np.zeros(self.L)
        for l in range(self.L):
            served = np.where(mask_uav[:, l])[0]
            for k in served:
                dep = betas_uav[k, l] / (ground_cov[k] + betas_uav[k, l] + 1e-12)
                eff_load[l] += dep
        return eff_load, ground_cov

    @staticmethod
    def _jfi(x):
        s = x.sum()
        return float(s**2 / (len(x) * (x**2).sum() + 1e-12)) if s > 1e-10 else 1.0

    # ================================================================
    #  负载状态更新  (betas-based, 稳定)
    # ================================================================

    def _update_load_state(self, mask, betas):
        self._eff_load, self._ground_cov = self._compute_ground_aware_load(mask, betas)
        self._jfi_current = self._jfi(self._eff_load)

        avg = self._eff_load.mean() if self._eff_load.mean() > 1e-6 else 1e-6
        for l in range(self.L):
            ratio = self._eff_load[l] / avg
            if ratio > self.load_threshold:
                self._load_factors[l] = self.overload_damping
            elif ratio < 0.75:
                self._load_factors[l] = self.underload_boost
            else:
                self._load_factors[l] = 1.0

    # ================================================================
    #  JFI 优先 JointScore  (软下限保护)
    # ================================================================

    def joint_score(self, min_rate, jfi_val):
        """
        JS = 0.35*(min/ref) + 0.65*JFI_eff
        当 min_rate < floor_rate 时乘以二次衰减因子，防止 min_rate 崩溃.
        """
        raw = self.w_min * (min_rate / self.ref_rate) + self.w_jfi * jfi_val
        if min_rate < self.floor_rate:
            floor_factor = (min_rate / self.floor_rate) ** 2
            return raw * floor_factor
        return raw

    # ================================================================
    #  负载感知普适引力  (幅度缩放 + 方向偏置)
    # ================================================================

    def _compute_universal_attraction(self, UE_pos, UAV_pos, rates,
                                      mask_uav, betas_uav):
        """
        过载 UAV:
          幅度 × overload_damping
          方向偏置: dep_bias = 1 - dep_bias_strength × dep
            dep→1 (高依赖) → 保留引力  dep→0 (地面可接管) → 减弱引力
            → 合力向地面覆盖充足方向偏转, 诱导低依赖用户切换 AP
        轻载 UAV:
          幅度 × underload_boost  (主动填补覆盖空洞)
        """
        forces = np.zeros((self.L, 3))
        rate_weights = 1.0 / (rates + 0.01)
        rate_weights = rate_weights / rate_weights.mean()
        decay = self.force_params['distance_decay_factor']

        s = self.dep_bias_strength

        for l in range(self.L):
            force_total = np.zeros(3)
            is_overloaded = self._load_factors[l] < 1.0

            for k in range(self.K):
                direction = UE_pos[k] - UAV_pos[l]
                dist     = np.linalg.norm(direction[:2]) + 1e-6
                unit_dir = direction / (np.linalg.norm(direction) + 1e-6)

                dist_weight    = 1.0 / (1 + (dist / (decay * 1.4))**2)
                channel_weight = np.sqrt(betas_uav[k, l] + 1e-12)
                service_weight = 0.3 if mask_uav[k, l] else 1.0

                # 方向偏置: 仅对过载 UAV 的已服务用户生效
                if is_overloaded and mask_uav[k, l]:
                    dep = betas_uav[k, l] / (
                        self._ground_cov[k] + betas_uav[k, l] + 1e-12)
                    dep_bias = 1.0 - s * dep
                else:
                    dep_bias = 1.0

                total_w  = (rate_weights[k] * channel_weight
                            * dist_weight * service_weight * dep_bias)
                force_total += self.force_params['K_universal'] * total_w * unit_dir

            forces[l, :2] = force_total[:2] * self._load_factors[l]
        return forces

    # ================================================================
    #  3D 高度势能力  (过载斥力 + 弹簧回归)
    # ================================================================

    def _compute_altitude_force(self, UAV_pos):
        """
        单向高度势能力 (标量, 正=向上):

        过载 UAV (ratio > load_threshold):
          F_z = excess × k_z_rep  -  k_z_spring × (h - h_opt) / span
          h_eq = h_opt + span × excess × k_z_rep / k_z_spring
          e.g. ratio=1.4→70m | ratio=1.6→90m | ratio=2.0→143m

        轻载/均衡 UAV:
          弹簧 → h_opt=50m (不主动降低，保持覆盖稳定)
        """
        avg   = self._eff_load.mean() if self._eff_load.mean() > 1e-6 else 1e-6
        h_opt = self.h_opt_uav
        span  = self.h_max_uav - h_opt   # 100m

        F_z = np.zeros(self.L)
        for l in range(self.L):
            ratio  = self._eff_load[l] / avg
            h_cur  = UAV_pos[l, 2]
            excess = max(0.0, ratio - self.load_threshold)

            F_z_rep    = excess * self.k_z_rep
            F_z_spring = -self.k_z_spring * (h_cur - h_opt) / (span + 1e-9)
            F_z[l]     = F_z_rep + F_z_spring

        return F_z

    def compute_balanced_virtual_forces(self, UE_pos, ground_AP_pos, UAV_pos,
                                        rates, mask, betas):
        """
        在基类 6 分量力 (含负载感知引力 via MRO) 基础上附加高度势能力 F_z.
        z 分量独立于 (x,y), 不干扰水平收敛.
        """
        forces = super().compute_balanced_virtual_forces(
            UE_pos, ground_AP_pos, UAV_pos, rates, mask, betas)
        forces[:, 2] = self._compute_altitude_force(UAV_pos)
        return forces

    # ================================================================
    #  阻尼位置更新  (4段步长衰减, μ=0.15)
    # ================================================================

    def update_positions(self, UAV_pos, forces, iteration, current_min_rate):
        """
        4段步长衰减 (适配 80 轮):
          0–19:  ×1.2  快速展开
          20–49: ×1.0  主收敛
          50–69: ×0.6  精细化
          70+:   ×0.3  微调锁定 (每次重启后也有机会用到)
        阻尼: F_net = F_virtual - μ × v_prev
        """
        new_pos = UAV_pos.copy()

        if   iteration < 20: stage_factor = 1.2
        elif iteration < 50: stage_factor = 1.0
        elif iteration < 70: stage_factor = 0.6
        else:                stage_factor = 0.3

        if hasattr(self, 'last_min_rate'):
            if   current_min_rate > self.last_min_rate * 1.02: perf_factor = 1.05
            elif current_min_rate < self.last_min_rate * 0.98: perf_factor = 0.9
            else:                                               perf_factor = 1.0
        else:
            perf_factor = 1.0

        adaptive_step = np.clip(self.step_size * stage_factor * perf_factor, 2, 38)

        # 阻尼修正
        xy_force = forces[:, :2].copy()
        if self._velocity is not None:
            xy_force = xy_force - self.damping_coeff * self._velocity

        force_norms = np.linalg.norm(xy_force, axis=1)
        max_force   = np.max(force_norms) if np.max(force_norms) > 0 else 1.0
        displacement = adaptive_step * (xy_force / max_force)

        # 周期性探索扰动 (随迭代衰减)
        pert_str = max(1.5, 6 * (1 - iteration / self.max_iterations))
        if iteration % 10 == 0 and iteration > 0:
            pert_str *= 1.3
        displacement += np.random.normal(0, pert_str, (self.L, 2))

        new_pos[:, :2] += displacement
        new_pos[:, :2]  = np.clip(new_pos[:, :2], 50, self.square_length - 50)

        # ── z (高度) 独立更新 ──
        F_z     = forces[:, 2]
        max_fz  = np.max(np.abs(F_z)) if np.max(np.abs(F_z)) > 1e-6 else 1.0
        alt_cap = self.alt_step_max * stage_factor
        dz      = np.clip(F_z / max_fz * alt_cap, -alt_cap, alt_cap)
        new_pos[:, 2] = np.clip(new_pos[:, 2] + dz, self.h_min_uav, self.h_max_uav)

        self._velocity     = displacement.copy()
        self.last_min_rate = current_min_rate
        return new_pos, float(np.sum(np.linalg.norm(displacement, axis=1)))

    # ================================================================
    #  确定性评估
    # ================================================================

    def _det_eval(self, UE_pos, ground_AP_pos, UAV_pos):
        state = np.random.get_state()
        np.random.seed(self.eval_seed)
        all_AP = np.vstack([ground_AP_pos, UAV_pos])
        _, _, betas = self.compute_channel_model(UE_pos, all_AP)
        mask = self.compute_AP_selection_mask(betas)
        rates, sum_r = self.compute_user_rates(UE_pos, all_AP, mask)
        np.random.set_state(state)

        eff_load, _ = self._compute_ground_aware_load(mask, betas)
        jfi_val = self._jfi(eff_load)
        js = self.joint_score(float(rates.min()), jfi_val)
        return float(rates.min()), float(sum_r), jfi_val, js, rates

    # ================================================================
    #  主优化循环
    # ================================================================

    def optimize(self, UE_pos, ground_AP_pos, UAV_pos) -> Dict:
        print("=" * 80)
        print("  LB-BVF V3-Pro Final (JFI-Priority + 3D Real-time)".center(80))
        print("=" * 80)
        print(f"  damping={self.overload_damping} | boost={self.underload_boost}"
              f" | thr={self.load_threshold} | dep_bias={self.dep_bias_strength}")
        print(f"  w_min={self.w_min} | w_jfi={self.w_jfi} | floor={self.floor_rate} Mbps"
              f" | damp_coeff={self.damping_coeff}")
        print(f"  k_z_rep={self.k_z_rep} | k_z_spring={self.k_z_spring}"
              f" | alt_step={self.alt_step_max}m | h=[{self.h_min_uav},{self.h_max_uav}]m"
              f" | h_opt={self.h_opt_uav}m")
        print("=" * 80)

        current_pos   = UAV_pos.copy()
        self._velocity = None

        best_js   = -np.inf
        best_pos  = UAV_pos.copy()
        best_iter = 0
        no_improve    = 0
        restart_count = 0

        history = {'iterations': [], 'min_rates': [], 'sum_rates': [],
                   'movements': [], 'jfi_history': []}

        for iteration in range(self.max_iterations):
            all_AP = np.vstack([ground_AP_pos, current_pos])
            _, _, betas = self.compute_channel_model(UE_pos, all_AP)
            mask = self.compute_AP_selection_mask(betas)
            rates, sum_r = self.compute_user_rates(UE_pos, all_AP, mask)
            min_r = float(rates.min())

            self._update_load_state(mask, betas)
            jfi_eff = self._jfi_current
            js = self.joint_score(min_r, jfi_eff)

            if js > best_js:
                best_js = js;  best_pos = current_pos.copy()
                best_iter = iteration;  no_improve = 0
            else:
                no_improve += 1

            history['iterations'].append(iteration)
            history['min_rates'].append(min_r)
            history['sum_rates'].append(float(sum_r))
            history['jfi_history'].append(jfi_eff)

            # V3-Pro v7 重启 (proven 9/10)
            if (no_improve >= self.restart_threshold
                    and restart_count < self.max_restarts
                    and iteration < self.max_iterations - 10):
                print(f"  Restart #{restart_count + 1}")
                avg_eff = self._eff_load.mean() if self._eff_load.mean() > 1e-6 else 1e-6
                perturb = np.zeros((self.L, 2))
                for l in range(self.L):
                    ratio = self._eff_load[l] / avg_eff
                    if ratio > 1.1:
                        served = np.where(mask[:, self.G + l])[0]
                        if len(served) > 0:
                            centroid = UE_pos[served, :2].mean(axis=0)
                            away = best_pos[l, :2] - centroid
                            d = np.linalg.norm(away) + 1e-6
                            perturb[l] = (away / d) * self.perturbation_strength * 0.7
                        else:
                            perturb[l] = np.random.normal(0, self.perturbation_strength, 2)
                    elif ratio < 0.7:
                        # 向最低速率用户移动 (V3-Pro v7 proven策略)
                        low_k  = np.argmin(rates)
                        toward = UE_pos[low_k, :2] - best_pos[l, :2]
                        d = np.linalg.norm(toward) + 1e-6
                        perturb[l] = (toward / d) * self.perturbation_strength * 0.5
                    else:
                        perturb[l] = np.random.normal(
                            0, self.perturbation_strength * 0.5, 2)

                current_pos = best_pos.copy()
                current_pos[:, :2] += perturb
                current_pos[:, 2]   = self.h_opt_uav   # 重启时回归均衡高度
                current_pos[:, :2]  = np.clip(
                    current_pos[:, :2], 100, self.square_length - 100)
                no_improve    = 0
                restart_count += 1
                self._velocity = None
                continue

            forces = self.compute_balanced_virtual_forces(
                UE_pos, ground_AP_pos, current_pos, rates, mask, betas)
            current_pos, movement = self.update_positions(
                current_pos, forces, iteration, min_r)
            history['movements'].append(movement)

            if iteration % 10 == 0:
                lf_str  = ','.join(f'{x:.2f}' for x in self._load_factors)
                eff_str = ','.join(f'{x:.1f}' for x in self._eff_load)
                print(f"  Iter {iteration:3d} | Min={min_r:.2f}"
                      f" | JFI={jfi_eff:.4f} | JS={js:.4f}"
                      f" | LF=[{lf_str}] | EL=[{eff_str}]")

        p_min, p_sum, p_jfi, p_js, p_rates = self._det_eval(
            UE_pos, ground_AP_pos, best_pos)
        print(f"\n  Final (det): Min={p_min:.2f} | JFI_eff={p_jfi:.4f}"
              f" | JS={p_js:.4f} | best_iter={best_iter}"
              f" | h=[{','.join(f'{x:.0f}' for x in best_pos[:, 2])}]m")

        return {
            'optimized_UAV_pos': best_pos,
            'final_min_rate': p_min, 'final_sum_rate': p_sum,
            'final_rates': p_rates,  'final_jfi': p_jfi,
            'final_joint_score': p_js,
            'mc_jfi_history': history['jfi_history'],
            'history': history,
            'best_iteration': best_iter,
            'restart_count': restart_count,
        }


def create_lb_v3style_config() -> Dict:
    from balanced_virtual_force_optimizer_v6 import create_v6_config
    cfg = create_v6_config()
    cfg.update({
        # 负载均衡力场参数
        'overload_damping':  0.55,
        'underload_boost':   1.6,
        'load_threshold':    1.3,
        'dep_bias_strength': 0.45,   # 方向偏置 (自适应收缩)
        # JFI 优先目标函数
        'w_min':      0.35,
        'w_jfi':      0.65,
        'ref_rate':   60.0,
        'floor_rate': 48.0,          # 软下限 = 0.8 × ref
        'eval_seed':  99999,
        # 阻尼
        'damping_coeff': 0.15,
        # 3D 实时高度势能力 (单向向上)
        'k_z_rep':      0.8,    # 过载向上斥力强度
        'k_z_spring':   0.6,    # 弹簧回归强度
        'alt_step_max': 1.5,    # 最大高度步长 (m/iter)
        'h_min_uav':    50.0,   # 最低飞行高度 (m)
        'h_max_uav':    150.0,  # 最高飞行高度 (m)
        'h_opt_uav':    50.0,   # 均衡目标高度 (m)
        # 迭代次数
        'max_iterations': 80,
    })
    return cfg
