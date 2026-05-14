"""
Dynamic Load-Balanced BVF (Dynamic LB-BVF)
==========================================

设计理念 (Three + One Pillars):

[支柱1] 负载感知力场 (继承 LB-BVF V3-Pro)
  ▸ 依赖度加权有效负载 eff_load (Cell-free感知)
  ▸ 幅度缩放: 过载×0.55 / 轻载×1.6
  ▸ 方向偏置 dep_bias=0.45: 过载UAV合力偏向地面AP覆盖充足区域

[支柱2] JFI + MinRate 双公平性框架 (继承 LB-BVF V3-Pro)
  ▸ 三目标 JointScore = w_min*(min/ref) + w_jfi*JFI + w_ee*EE_norm
  ▸ 软下限保护: min_rate < floor_rate → JS × (min/floor)²
  ▸ EE_norm = (1 - E_step/E_ref) 归一化能效奖励

[支柱3] 实时3D高度势能力 (继承 LB-BVF V3-Pro)
  ▸ 过载向上斥力 + 弹簧回归 h_opt=50m
  ▸ 高度约束 [50, 150]m

[支柱4] 能量守恒力 (新增, 适配动态场景)
  ▸ 第8虚拟力: F_energy = -K_e * P(V)/P_hover * dist_factor * budget_factor * unit_dir
  ▸ 非线性阻力: 大位移(高速飞行)面临强得多的阻力, 小调整几乎不受惩罚
  ▸ 能量预算因子随累计能耗增长而增强, 防止能量耗尽
  ▸ `optimize_one_step` API: 动态单步跟踪, 返回 (pos, min_rate, sum_rate, energy_J, dist_m)

联合评估机制:
  JS_dyn = w_min*(min_rate/ref) + w_jfi*JFI_eff + w_ee*(1 - E_step/E_ref)
         × softfloor_penalty(min_rate)

  作用: 同时感知通信公平性 + 负载均衡 + 飞行能效
       小调整=能效高=奖励; 大跳跃=能效低=抑制; 低速率=强惩罚
"""

import numpy as np
from typing import Dict, Tuple

from load_balanced_bvf_v3style_advanced import LoadBalancedBVF_V3Style, create_lb_v3style_config
from run_dynamic_energy_comparison import UAVEnergyModel


class DynamicLoadBalancedBVF(LoadBalancedBVF_V3Style):
    """
    动态负载均衡 BVF: 融合 LB-BVF V3-Pro (负载+公平+3D高度) 与能量感知力场
    """

    def __init__(self, config: Dict, energy_model: UAVEnergyModel = None):
        super().__init__(config)

        # ── 能量模型 ──
        self.energy_model = energy_model or UAVEnergyModel()
        self.P_norm = self.energy_model.P_hover          # 悬停功率 (归一化基准)

        # ── 能量守恒力参数 (第8虚拟力) ──
        self.K_energy         = config.get('K_energy', 1.5e4)       # 基础能量力强度 (降低以允许更多重分配)
        self.w_energy_base    = config.get('w_energy_base', 0.12)   # 能量力权重 (降低以留空间给LB力)
        self.energy_budget    = config.get('energy_budget', 1000e3) # 总能量预算 (J)
        self.flight_speed     = config.get('uav_flight_speed', 10.0) # m/s
        self._step_iter       = 0   # 当前步内迭代计数 (用于预热抑制)

        # ── 三目标联合评分权重 ──
        self.w_min  = config.get('w_min', 0.30)   # min_rate 权重 (提升至0.30以改善min-rate)
        self.w_jfi  = config.get('w_jfi', 0.50)   # JFI 权重 (主导)
        self.w_ee   = config.get('w_ee', 0.20)    # 能效权重 (略降以让出空间给min_rate)
        self.E_ref  = config.get('E_ref', None)    # 能效参考 (None=自动: L×hover×dt)

        # ── 动态场景参数 ──
        self.time_step       = config.get('time_step', 5.0)    # 时间步长 (s)
        self.max_displacement = config.get('max_displacement', 50.0)  # 单步最大位移 (m)

        # ── 能量状态 ──
        self.cumulative_energy = 0.0
        self._prev_UAV_pos     = None   # 上一步 UAV 位置 (能量力基准)

        # 调整通信力权重以为能量力腾出比例
        scale = 1.0 - self.w_energy_base
        fp = self.force_params
        for k in ['w_critical', 'w_interference', 'w_universal',
                  'w_cooperation', 'w_separation', 'w_boundary']:
            if k in fp:
                fp[k] = fp.get(k, 1.0/6) * scale

        # ── 关键用户力增强 (改善 min_rate) ──
        # 动态场景中用户持续移动，需要更强的"追最差用户"能力
        k_boost = config.get('K_critical_boost', 2.0)
        fp['K_critical'] *= k_boost

    # ================================================================
    #  能量守恒力 (第8虚拟力)
    # ================================================================

    def _compute_energy_force(self, UAV_pos: np.ndarray,
                               prev_pos: np.ndarray) -> np.ndarray:
        """
        能量守恒力: 阻止无人机做大位移飞行.

        物理直觉:
          飞行速度 V = dist/dt → 推进功率 P(V) → 非线性增长
          小调整 (~5m, V=1m/s):  P≈P_hover → force ≈ K × 1.0 (微弱)
          中等移动 (~30m, V=6m/s): P≈1.5×P_hover → force ≈ K × 1.5 (适中)
          大幅跳跃 (~100m, V=20m/s): P≈3×P_hover → force ≈ K × 3.0 (强烈)

        方向: 从当前位置指向上一步位置 (回拉力).
        """
        forces = np.zeros((self.L, 3))

        if prev_pos is None:
            return forces

        for l in range(self.L):
            displacement = UAV_pos[l, :2] - prev_pos[l, :2]
            dist = np.linalg.norm(displacement)

            if dist < 0.5:
                continue

            unit_dir = displacement / (dist + 1e-6)

            # 功率因子: 速度越大阻力越强 (非线性)
            velocity = min(dist / max(self.time_step, 1.0), 25.0)
            P_flight = self.energy_model.propulsion_power(velocity)
            power_factor = P_flight / self.P_norm

            # 距离因子: 超出 max_displacement 后额外惩罚
            if dist <= self.max_displacement:
                dist_factor = dist / self.max_displacement
            else:
                dist_factor = 1.0 + 2.0 * (dist - self.max_displacement) / (
                    self.max_displacement + 1e-6)

            # 预算因子: 累计能耗越多, 阻力越强
            energy_ratio = self.cumulative_energy / (self.energy_budget + 1e-6)
            budget_factor = 1.0 + 3.0 * np.clip(energy_ratio, 0.0, 1.0)

            force_mag = self.K_energy * power_factor * dist_factor * budget_factor
            forces[l, :2] = -force_mag * unit_dir

        return forces

    # ================================================================
    #  综合力 (6通信力 + 高度势能力 + 能量守恒力)
    # ================================================================

    def compute_balanced_virtual_forces(self, UE_pos, ground_AP_pos, UAV_pos,
                                        rates, mask, betas) -> np.ndarray:
        """
        = LB-BVF V3-Pro 6力 + z高度势能力 + 能量守恒力 (第8力)
        能量力权重随累计能耗动态增加 (0.20 → 0.45).
        """
        # 父类: 通信力 + 高度势能力
        forces = super().compute_balanced_virtual_forces(
            UE_pos, ground_AP_pos, UAV_pos, rates, mask, betas)

        # 第8力: 能量守恒力
        # 激活条件: ① 有上一步位置 ② 非首次部署(有累计能耗) ③ 前3次迭代预热抑制
        if (self._prev_UAV_pos is not None
                and self.cumulative_energy > 0.0
                and self._step_iter >= 3):
            f_energy = self._compute_energy_force(UAV_pos, self._prev_UAV_pos)
            # 动态权重: 累计能耗越多, 能量力权重越大
            w_e = min(0.35, self.w_energy_base +
                      0.20 * (self.cumulative_energy / (self.energy_budget + 1e-6)))
            forces[:, :2] += w_e * f_energy[:, :2]
        self._step_iter += 1

        return forces

    # ================================================================
    #  三目标联合评分 (动态版)
    # ================================================================

    def joint_score_dynamic(self, min_rate: float, jfi_val: float,
                             energy_step_J: float) -> float:
        """
        JS_dyn = w_min*(min/ref) + w_jfi*JFI_eff + w_ee*(1 - E_step/E_ref)
        当 min_rate < floor_rate 时: JS × (min/floor)²

        energy_step_J: 本步实际能耗 (焦耳)
        E_ref 默认 = L × P_hover × dt × 2  (悬停2倍作为参考上界)
        """
        # 能效归一化参考
        if self.E_ref is None:
            E_ref = self.L * self.P_norm * self.time_step * 2.0
        else:
            E_ref = self.E_ref

        EE_norm = float(np.clip(1.0 - energy_step_J / (E_ref + 1e-6), 0.0, 1.0))

        raw = (self.w_min  * (min_rate / self.ref_rate) +
               self.w_jfi  * jfi_val +
               self.w_ee   * EE_norm)

        # 软下限保护
        if min_rate < self.floor_rate:
            raw *= (min_rate / self.floor_rate) ** 2

        return float(raw)

    # ================================================================
    #  确定性评估 (带能效)
    # ================================================================

    def _det_eval_dyn(self, UE_pos, ground_AP_pos, UAV_pos,
                      energy_step_J: float = 0.0):
        """返回 (min_rate, sum_rate, jfi_val, js_dyn, rates)"""
        state = np.random.get_state()
        np.random.seed(self.eval_seed)
        all_AP = np.vstack([ground_AP_pos, UAV_pos])
        _, _, betas = self.compute_channel_model(UE_pos, all_AP)
        mask = self.compute_AP_selection_mask(betas)
        rates, sum_r = self.compute_user_rates(UE_pos, all_AP, mask)
        np.random.set_state(state)

        eff_load, _ = self._compute_ground_aware_load(mask, betas)
        jfi_val = self._jfi(eff_load)
        js = self.joint_score_dynamic(float(rates.min()), jfi_val, energy_step_J)
        return float(rates.min()), float(sum_r), jfi_val, js, rates

    # ================================================================
    #  动态单步优化 (核心接口)
    # ================================================================

    def optimize_one_step(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray,
                          UAV_pos_init: np.ndarray, max_iter: int = 15,
                          dt: float = None) -> Tuple[np.ndarray, float, float, float, float]:
        """
        动态单步优化 (用于连续时间步跟踪).

        参数:
            UE_pos         : 当前时间步的用户位置 (K, 3)
            ground_AP_pos  : 地面AP位置 (G, 3)
            UAV_pos_init   : 上一时间步末尾的 UAV 位置 (L, 3)
            max_iter       : 本步迭代次数
            dt             : 时间步长 (s), None 使用 self.time_step

        返回:
            (optimized_UAV_pos, min_rate, sum_rate, energy_J, total_dist_m)
        """
        if dt is not None:
            self.time_step = dt

        self._prev_UAV_pos = UAV_pos_init.copy()
        self._velocity     = None
        self._step_iter    = 0   # 重置预热计数器

        current_pos = UAV_pos_init.copy()
        best_js     = -np.inf
        best_pos    = UAV_pos_init.copy()
        no_improve  = 0

        orig_max_iter      = self.max_iterations
        orig_restart_thr   = self.restart_threshold
        self.max_iterations   = max_iter
        self.restart_threshold = max(max_iter + 1, 999)  # 单步内禁用重启

        try:
            for iteration in range(max_iter):
                all_AP = np.vstack([ground_AP_pos, current_pos])
                _, _, betas = self.compute_channel_model(UE_pos, all_AP)
                mask = self.compute_AP_selection_mask(betas)
                rates, sum_r = self.compute_user_rates(UE_pos, all_AP, mask)
                min_r = float(rates.min())

                self._update_load_state(mask, betas)
                jfi_val = self._jfi_current

                # 实时能耗估计 (用于力计算内的评估)
                step_E = self._estimate_step_energy(UAV_pos_init, current_pos)
                js = self.joint_score_dynamic(min_r, jfi_val, step_E)

                if js > best_js:
                    best_js  = js
                    best_pos = current_pos.copy()
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= max(5, max_iter // 3):
                    break

                forces = self.compute_balanced_virtual_forces(
                    UE_pos, ground_AP_pos, current_pos, rates, mask, betas)
                current_pos, _ = self.update_positions(
                    current_pos, forces, iteration, min_r)

        finally:
            self.max_iterations   = orig_max_iter
            self.restart_threshold = orig_restart_thr

        # ── 实际能量消耗 (飞行能耗) ──
        energy_J, total_dist = self.energy_model.total_energy_for_repositioning(
            UAV_pos_init, best_pos, flight_speed=self.flight_speed)
        self.cumulative_energy += energy_J

        # 确定性最终评估
        p_min, p_sum, p_jfi, p_js, _ = self._det_eval_dyn(
            UE_pos, ground_AP_pos, best_pos, energy_J)

        return best_pos, p_min, p_sum, energy_J, total_dist

    def _estimate_step_energy(self, from_pos: np.ndarray,
                               to_pos: np.ndarray) -> float:
        """粗估当前位移对应的能耗 (用于力计算阶段评分, 不更新累计)"""
        E, _ = self.energy_model.total_energy_for_repositioning(
            from_pos, to_pos, flight_speed=self.flight_speed)
        return E

    # ================================================================
    #  静态部署优化 (向后兼容, 使用原 JFI-priority JS)
    # ================================================================

    def optimize(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray,
                 UAV_pos: np.ndarray) -> Dict:
        """
        静态部署模式 (与 LB-BVF V3-Pro 接口相同).
        能量力在静态模式下不激活 (prev_pos=None).
        """
        self._prev_UAV_pos = None
        return super().optimize(UE_pos, ground_AP_pos, UAV_pos)


# ================================================================
#  配置工厂
# ================================================================

def create_dynamic_lb_config(K: int = 60, L: int = 9, G: int = 4) -> Dict:
    """创建 Dynamic LB-BVF 配置"""
    cfg = create_lb_v3style_config()
    cfg.update({
        # 系统规模
        'num_UE':        K,
        'num_UAV':       L,
        'num_ground_AP': G,
        'tau_p':         K,
        'num_serving_APs': 3,
        'max_iterations':  80,

        # 三目标权重 (min_rate 提权以保证全面领先)
        'w_min': 0.30,
        'w_jfi': 0.50,
        'w_ee':  0.20,
        'E_ref': None,   # 自动: L × P_hover × dt × 2

        # 能量守恒力
        'K_energy':        1.5e4,   # 降低以留出更多JFI优化空间
        'w_energy_base':   0.12,
        'energy_budget':   1000e3,  # 1000 kJ 总预算
        'uav_flight_speed': 10.0,  # m/s

        # ── 关键用户力增强 (改善 min_rate) ──
        'K_critical_boost':  2.0,   # K_critical 乘以该倍数, 增强追最差用户能力

        # ── 负载均衡力增强参数 (维持JFI) ──
        'overload_damping':  0.30,  # 过载UAV幅度强抑制 (默认0.55)
        'underload_boost':   2.5,   # 轻载UAV幅度强增益 (默认1.6)
        'dep_bias_strength': 0.55,  # 方向偏置增强 (默认0.45)

        # 动态场景
        'time_step':        5.0,
        'max_displacement': 50.0,
    })
    return cfg
