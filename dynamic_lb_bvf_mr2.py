"""
DE2VF-MR2: Two-Phase + Top-k Hard Constraint
=============================================
新变体（不改动现有 DE2VF）：
1) 两阶段优化：前期强拉 min-rate，后期回收综合目标
2) Top-k 最差用户硬约束：候选解不能显著恶化尾部用户
"""

import numpy as np
from typing import Dict, Tuple

from dynamic_lb_bvf import DynamicLoadBalancedBVF, create_dynamic_lb_config


class DynamicLoadBalancedBVFMR2(DynamicLoadBalancedBVF):
    def __init__(self, config: Dict, energy_model=None):
        super().__init__(config, energy_model)
        self.phase_split = config.get('phase_split', 0.4)
        self.worst_q = config.get('worst_q', 0.20)
        self.tail_tol = config.get('tail_tol', 0.15)  # 允许尾部均值最多下降 0.15 Mbps
        self.k_tail_pull = config.get('k_tail_pull', 1.8e4)

    def _tail_set(self, rates: np.ndarray):
        n = len(rates)
        n_worst = max(1, int(np.ceil(n * self.worst_q)))
        return np.argsort(rates)[:n_worst]

    def _tail_mean(self, rates: np.ndarray):
        idx = self._tail_set(rates)
        return float(np.mean(rates[idx]))

    def _tail_pull_force(self, UE_pos, UAV_pos, rates, mask):
        forces = np.zeros((self.L, 3))
        worst_idx = self._tail_set(rates)
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
                forces[l, :2] += self.k_tail_pull * share * (vec / (d + 1e-6))
        return forces

    def _score_phase(self, min_rate: float, jfi_val: float, energy_step_J: float, phase_a: bool):
        if self.E_ref is None:
            e_ref = self.L * self.P_norm * self.time_step * 2.0
        else:
            e_ref = self.E_ref
        ee = float(np.clip(1.0 - energy_step_J / (e_ref + 1e-6), 0.0, 1.0))

        if phase_a:
            # 前期强拉 min-rate
            w_min, w_jfi, w_ee = 0.72, 0.24, 0.04
        else:
            # 后期恢复综合目标
            w_min, w_jfi, w_ee = 0.36, 0.48, 0.16

        raw = w_min * (min_rate / self.ref_rate) + w_jfi * jfi_val + w_ee * ee
        if min_rate < self.floor_rate:
            raw *= (min_rate / self.floor_rate) ** 2
        return float(raw)

    def optimize_one_step(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray,
                          UAV_pos_init: np.ndarray, max_iter: int = 15,
                          dt: float = None) -> Tuple[np.ndarray, float, float, float, float]:
        if dt is not None:
            self.time_step = dt

        self._prev_UAV_pos = UAV_pos_init.copy()
        self._velocity = None
        self._step_iter = 0

        current_pos = UAV_pos_init.copy()
        best_js = -np.inf
        best_pos = current_pos.copy()
        no_improve = 0

        orig_max_iter = self.max_iterations
        orig_restart_thr = self.restart_threshold
        self.max_iterations = max_iter
        self.restart_threshold = max(max_iter + 1, 999)

        try:
            for it in range(max_iter):
                all_AP = np.vstack([ground_AP_pos, current_pos])
                _, _, betas = self.compute_channel_model(UE_pos, all_AP)
                mask = self.compute_AP_selection_mask(betas)
                rates, _ = self.compute_user_rates(UE_pos, all_AP, mask)
                min_r = float(rates.min())
                tail_ref = self._tail_mean(rates)

                self._update_load_state(mask, betas)
                jfi_val = self._jfi_current
                step_E = self._estimate_step_energy(UAV_pos_init, current_pos)
                phase_a = it < int(max_iter * self.phase_split)
                js = self._score_phase(min_r, jfi_val, step_E, phase_a)

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
                    f_tail = self._tail_pull_force(UE_pos, current_pos, rates, mask)
                    forces[:, :2] += 0.20 * f_tail[:, :2]

                cand_pos, _ = self.update_positions(current_pos, forces, it, min_r)

                # 硬约束：候选不能明显恶化尾部
                all_AP_c = np.vstack([ground_AP_pos, cand_pos])
                _, _, betas_c = self.compute_channel_model(UE_pos, all_AP_c)
                mask_c = self.compute_AP_selection_mask(betas_c)
                rates_c, _ = self.compute_user_rates(UE_pos, all_AP_c, mask_c)
                tail_c = self._tail_mean(rates_c)
                if tail_c + self.tail_tol < tail_ref:
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


def create_dynamic_lb_mr2_config(K: int = 30, L: int = 9, G: int = 6) -> Dict:
    cfg = create_dynamic_lb_config(K=K, L=L, G=G)
    cfg.update({
        'phase_split': 0.4,
        'worst_q': 0.20,
        'tail_tol': 0.15,
        'k_tail_pull': 1.8e4,
    })
    return cfg
