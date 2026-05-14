"""
DE2VF-MR+: 新增算法变体 (不改动现有 DE2VF)
==========================================
目标: 在动态场景中进一步拉开 Min-Rate 优势。

设计思路:
1) 低速率自适应加权: min-rate 越低, 优化阶段临时提高 min-rate 权重。
2) 最差用户牵引力: 识别底部用户(按速率排序), 对服务这些用户的 UAV 施加定向牵引。
3) 微搜索精修: 在每步优化结束后做小范围局部搜索, 仅接受 min-rate 改善的候选。

说明:
- 这是独立新算法类, 不会修改现有 DynamicLoadBalancedBVF。
- 可直接替换运行脚本中的类进行 A/B 对比。
"""

import numpy as np
from typing import Dict, Tuple

from dynamic_lb_bvf import DynamicLoadBalancedBVF, create_dynamic_lb_config


class DynamicLoadBalancedBVFMinRatePlus(DynamicLoadBalancedBVF):
    """更强调 Min-Rate 的 DE2VF 变体"""

    def __init__(self, config: Dict, energy_model=None):
        super().__init__(config, energy_model)

        # 根据调参结果选择最优默认值
        self.mr_gain = config.get('mr_gain', 0.30)
        self.worst_q = config.get('worst_q', 0.20)  # 底部 20% 用户
        self.k_mr_force = config.get('k_mr_force', 2.2e4)
        self.refine_step_sizes = config.get('refine_step_sizes', [8.0, 5.0, 3.0])

    def _joint_score_dynamic_with_adaptive_min(self, min_rate: float, jfi_val: float,
                                               energy_step_J: float) -> float:
        if self.E_ref is None:
            E_ref = self.L * self.P_norm * self.time_step * 2.0
        else:
            E_ref = self.E_ref

        ee = float(np.clip(1.0 - energy_step_J / (E_ref + 1e-6), 0.0, 1.0))

        deficit = max(0.0, (self.floor_rate - min_rate) / (self.floor_rate + 1e-6))
        w_min_eff = min(0.70, self.w_min + self.mr_gain * deficit)
        remain = max(1e-6, 1.0 - w_min_eff - self.w_ee)
        w_jfi_eff = remain

        raw = w_min_eff * (min_rate / self.ref_rate) + w_jfi_eff * jfi_val + self.w_ee * ee
        if min_rate < self.floor_rate:
            raw *= (min_rate / self.floor_rate) ** 2
        return float(raw)

    def _compute_worst_user_pull(self, UE_pos: np.ndarray, UAV_pos: np.ndarray,
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
                forces[l, :2] += self.k_mr_force * share * (vec / (d + 1e-6))

        return forces

    def _local_refine_minrate(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray,
                              pos: np.ndarray, init_pos: np.ndarray,
                              base_min_rate: float) -> Tuple[np.ndarray, float]:
        cur_pos = pos.copy()
        cur_min = base_min_rate

        all_AP = np.vstack([ground_AP_pos, cur_pos])
        _, _, betas = self.compute_channel_model(UE_pos, all_AP)
        mask = self.compute_AP_selection_mask(betas)
        rates, _ = self.compute_user_rates(UE_pos, all_AP, mask)
        k_worst = int(np.argmin(rates))
        target_xy = UE_pos[k_worst, :2]
        serving_uavs = np.where(mask[k_worst, self.G:])[0]

        if len(serving_uavs) == 0:
            return cur_pos, cur_min

        for step_size in self.refine_step_sizes:
            improved = False
            for l in serving_uavs:
                cand = cur_pos.copy()
                vec = target_xy - cand[l, :2]
                d = np.linalg.norm(vec)
                if d < 1e-6:
                    continue
                cand[l, :2] += step_size * vec / (d + 1e-6)
                cand[l, 0] = np.clip(cand[l, 0], 50, self.square_length - 50)
                cand[l, 1] = np.clip(cand[l, 1], 50, self.square_length - 50)

                e_cand = self._estimate_step_energy(init_pos, cand)
                min_c, _, jfi_c, js_c, _ = self._det_eval_dyn(UE_pos, ground_AP_pos, cand, e_cand)
                js_cur = self._joint_score_dynamic_with_adaptive_min(
                    cur_min,
                    self._det_eval_dyn(UE_pos, ground_AP_pos, cur_pos, self._estimate_step_energy(init_pos, cur_pos))[2],
                    self._estimate_step_energy(init_pos, cur_pos)
                )

                if (min_c > cur_min + 0.01) or (min_c >= cur_min - 1e-6 and js_c > js_cur + 1e-4):
                    cur_pos = cand
                    cur_min = min_c
                    improved = True
            if not improved:
                continue

        return cur_pos, cur_min

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
        best_min = -np.inf
        no_improve = 0

        orig_max_iter = self.max_iterations
        orig_restart_thr = self.restart_threshold
        self.max_iterations = max_iter
        self.restart_threshold = max(max_iter + 1, 999)

        try:
            for iteration in range(max_iter):
                all_AP = np.vstack([ground_AP_pos, current_pos])
                _, _, betas = self.compute_channel_model(UE_pos, all_AP)
                mask = self.compute_AP_selection_mask(betas)
                rates, _ = self.compute_user_rates(UE_pos, all_AP, mask)
                min_r = float(rates.min())

                self._update_load_state(mask, betas)
                jfi_val = self._jfi_current

                step_E = self._estimate_step_energy(UAV_pos_init, current_pos)
                js = self._joint_score_dynamic_with_adaptive_min(min_r, jfi_val, step_E)

                if js > best_js:
                    best_js = js
                    best_pos = current_pos.copy()
                    best_min = min_r
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= max(5, max_iter // 3):
                    break

                forces = self.compute_balanced_virtual_forces(
                    UE_pos, ground_AP_pos, current_pos, rates, mask, betas)

                # 新增: 最差用户牵引力
                f_worst = self._compute_worst_user_pull(UE_pos, current_pos, rates, mask)
                forces[:, :2] += 0.22 * f_worst[:, :2]

                current_pos, _ = self.update_positions(current_pos, forces, iteration, min_r)

        finally:
            self.max_iterations = orig_max_iter
            self.restart_threshold = orig_restart_thr

        # 新增: 局部精修 min-rate
        best_pos, best_min = self._local_refine_minrate(
            UE_pos, ground_AP_pos, best_pos, UAV_pos_init, best_min)

        energy_J, total_dist = self.energy_model.total_energy_for_repositioning(
            UAV_pos_init, best_pos, flight_speed=self.flight_speed)
        self.cumulative_energy += energy_J

        p_min, p_sum, _, _, _ = self._det_eval_dyn(UE_pos, ground_AP_pos, best_pos, energy_J)
        return best_pos, p_min, p_sum, energy_J, total_dist


def create_dynamic_lb_minrate_plus_config(K: int = 30, L: int = 9, G: int = 6) -> Dict:
    cfg = create_dynamic_lb_config(K=K, L=L, G=G)
    cfg.update({
        'mr_gain': 0.30,
        'worst_q': 0.20,
        'k_mr_force': 2.2e4,
        'refine_step_sizes': [8.0, 5.0, 3.0],
    })
    return cfg
