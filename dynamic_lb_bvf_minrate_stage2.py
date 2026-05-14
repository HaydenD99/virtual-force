"""
DE2VF-MR-S2: 两阶段 + Top-k 最差用户硬约束 变体
=================================================
目标: 提升动态场景 min-rate 尾部表现。

核心机制:
1) 两阶段优化
   - 阶段A (前 phase_a_ratio): 以 min-rate 导向为主
   - 阶段B (后续): 恢复综合目标 (min + jfi + energy)
2) Top-k 最差用户硬约束
   - 记录候选解下最差 k 用户的平均速率
   - 若该值显著下降(超过阈值), 候选直接拒绝
"""

import numpy as np
from typing import Dict, Tuple

from dynamic_lb_bvf import DynamicLoadBalancedBVF, create_dynamic_lb_config


class DynamicLoadBalancedBVFMinRateStage2(DynamicLoadBalancedBVF):
    def __init__(self, config: Dict, energy_model=None):
        super().__init__(config, energy_model)

        self.phase_a_ratio = config.get('phase_a_ratio', 0.40)
        self.phase_a_min_boost = config.get('phase_a_min_boost', 0.28)
        self.worst_q = config.get('worst_q', 0.20)
        self.k_worst_force = config.get('k_worst_force', 1.2e4)

        self.topk_ratio = config.get('topk_ratio', 0.20)
        self.topk_drop_tol = config.get('topk_drop_tol', 0.25)  # Mbps

    @staticmethod
    def _mean_bottom_k(rates: np.ndarray, k: int) -> float:
        if len(rates) == 0:
            return 0.0
        k = max(1, min(k, len(rates)))
        idx = np.argpartition(rates, k - 1)[:k]
        return float(np.mean(rates[idx]))

    def _joint_score_two_phase(self, min_rate: float, jfi_val: float,
                               energy_step_J: float, phase_a: bool) -> float:
        if self.E_ref is None:
            E_ref = self.L * self.P_norm * self.time_step * 2.0
        else:
            E_ref = self.E_ref
        ee = float(np.clip(1.0 - energy_step_J / (E_ref + 1e-6), 0.0, 1.0))

        if phase_a:
            w_min_eff = min(0.72, self.w_min + self.phase_a_min_boost)
            w_ee_eff = min(0.15, self.w_ee)
            w_jfi_eff = max(1e-6, 1.0 - w_min_eff - w_ee_eff)
        else:
            w_min_eff, w_jfi_eff, w_ee_eff = self.w_min, self.w_jfi, self.w_ee

        raw = (w_min_eff * (min_rate / self.ref_rate)
               + w_jfi_eff * jfi_val
               + w_ee_eff * ee)
        if min_rate < self.floor_rate:
            raw *= (min_rate / self.floor_rate) ** 2
        return float(raw)

    def _compute_worst_pull(self, UE_pos: np.ndarray, UAV_pos: np.ndarray,
                            rates: np.ndarray, mask: np.ndarray) -> np.ndarray:
        forces = np.zeros((self.L, 3))
        if len(rates) == 0:
            return forces

        n_worst = max(1, int(np.ceil(len(rates) * self.worst_q)))
        worst_idx = np.argsort(rates)[:n_worst]
        for k in worst_idx:
            serving_uavs = np.where(mask[k, self.G:])[0]
            if len(serving_uavs) == 0:
                continue
            ue_xy = UE_pos[k, :2]
            share = 1.0 / len(serving_uavs)
            for l in serving_uavs:
                vec = ue_xy - UAV_pos[l, :2]
                d = np.linalg.norm(vec)
                if d < 1e-6:
                    continue
                forces[l, :2] += self.k_worst_force * share * (vec / (d + 1e-6))
        return forces

    def optimize_one_step(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray,
                          UAV_pos_init: np.ndarray, max_iter: int = 15,
                          dt: float = None) -> Tuple[np.ndarray, float, float, float, float]:
        if dt is not None:
            self.time_step = dt

        self._prev_UAV_pos = UAV_pos_init.copy()
        self._velocity = None
        self._step_iter = 0

        current_pos = UAV_pos_init.copy()
        best_pos = current_pos.copy()
        best_js = -np.inf
        no_improve = 0

        phase_a_iters = max(1, int(np.ceil(max_iter * self.phase_a_ratio)))

        orig_max_iter = self.max_iterations
        orig_restart_thr = self.restart_threshold
        self.max_iterations = max_iter
        self.restart_threshold = max(max_iter + 1, 999)

        try:
            for iteration in range(max_iter):
                phase_a = iteration < phase_a_iters

                all_AP = np.vstack([ground_AP_pos, current_pos])
                _, _, betas = self.compute_channel_model(UE_pos, all_AP)
                mask = self.compute_AP_selection_mask(betas)
                rates, _ = self.compute_user_rates(UE_pos, all_AP, mask)
                min_r = float(rates.min())

                self._update_load_state(mask, betas)
                jfi_val = self._jfi_current
                e_cur = self._estimate_step_energy(UAV_pos_init, current_pos)
                js = self._joint_score_two_phase(min_r, jfi_val, e_cur, phase_a)

                if js > best_js:
                    best_js = js
                    best_pos = current_pos.copy()
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= max(5, max_iter // 3):
                    break

                forces = self.compute_balanced_virtual_forces(
                    UE_pos, ground_AP_pos, current_pos, rates, mask, betas)

                if phase_a:
                    f_worst = self._compute_worst_pull(UE_pos, current_pos, rates, mask)
                    forces[:, :2] += 0.22 * f_worst[:, :2]

                cand_pos, _ = self.update_positions(current_pos, forces, iteration, min_r)

                # Top-k 硬约束: 阻止尾部显著恶化
                all_AP_c = np.vstack([ground_AP_pos, cand_pos])
                _, _, betas_c = self.compute_channel_model(UE_pos, all_AP_c)
                mask_c = self.compute_AP_selection_mask(betas_c)
                rates_c, _ = self.compute_user_rates(UE_pos, all_AP_c, mask_c)

                k_bot = max(1, int(np.ceil(self.K * self.topk_ratio)))
                bot_now = self._mean_bottom_k(rates, k_bot)
                bot_cand = self._mean_bottom_k(rates_c, k_bot)

                if bot_cand + self.topk_drop_tol < bot_now:
                    continue

                current_pos = cand_pos

        finally:
            self.max_iterations = orig_max_iter
            self.restart_threshold = orig_restart_thr

        energy_J, total_dist = self.energy_model.total_energy_for_repositioning(
            UAV_pos_init, best_pos, flight_speed=self.flight_speed)
        self.cumulative_energy += energy_J

        p_min, p_sum, _, _, _ = self._det_eval_dyn(UE_pos, ground_AP_pos, best_pos, energy_J)
        return best_pos, p_min, p_sum, energy_J, total_dist


def create_dynamic_lb_minrate_stage2_config(K: int = 30, L: int = 9, G: int = 6) -> Dict:
    cfg = create_dynamic_lb_config(K=K, L=L, G=G)
    cfg.update({
        'phase_a_ratio': 0.40,
        'phase_a_min_boost': 0.28,
        'worst_q': 0.20,
        'k_worst_force': 1.2e4,
        'topk_ratio': 0.20,
        'topk_drop_tol': 0.25,
    })
    return cfg
