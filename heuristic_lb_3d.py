"""
三维负载均衡启发式算法对比基准
=================================
三大竞争算法 (公平对比版):
  GA3D_LB   — 3D 离散遗传算法  + JFI 负载均衡目标
  PSO3D_LB  — 3D 粒子群优化    + JFI 负载均衡目标
  SSA3D_LB  — 3D 麻雀搜索算法  + JFI 负载均衡目标

对比创新点:
  1. 3D 搜索空间: z (高度) ∈ [h_min, h_max]
     - GA:  9×9×n_z 离散网格
     - PSO: 连续粒子, v_z ∈ [-v_z_max, v_z_max]
     - SSA: 连续个体, 含 z 维度
  2. 负载均衡适应度 (与 LB-BVF 完全一致):
     fitness = w_min*(min_rate/ref) + w_jfi*JFI_eff
     软下限: min_rate < floor → fitness × (min/floor)²
     JFI_eff = 依赖度加权 Jain's Fairness Index (Cell-free感知)

用法:
    from heuristic_lb_3d import GA3D_LB, PSO3D_LB, SSA3D_LB
    cfg = {...}
    alg = GA3D_LB(cfg)
    result = alg.optimize(UE_pos, gAP, init_UAV_pos)
    # result['optimized_UAV_pos'], result['final_min_rate'], result['final_jfi'], result['final_joint_score']
"""

import numpy as np
import time
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from balanced_virtual_force_optimizer_v6 import BalancedVirtualForceOptimizerV6, create_v6_config


# ======================================================================
#  共用 Mixin: 信道计算 + Cell-free JFI + 联合评分
# ======================================================================

class _ChannelMixin:
    """共享信道计算, 委托给 BVF V6 实例"""

    def _init_channel(self, config):
        self._ch = BalancedVirtualForceOptimizerV6(config)
        self.K = config.get('num_UE', 40)
        self.L = config.get('num_UAV', 6)
        self.G = config.get('num_ground_AP', 9)
        self.w_min   = config.get('w_min', 0.35)
        self.w_jfi   = config.get('w_jfi', 0.65)
        self.ref_rate  = config.get('ref_rate', 60.0)
        self.floor_rate = config.get('floor_rate', 48.0)
        self.eval_seed  = config.get('eval_seed', 99999)
        self.nbrReals_inner = config.get('nbrOfRealizations_inner', 20)
        self.nbrReals_final = config.get('nbrOfRealizations_final', 50)

    def _channel_and_rates(self, UE_pos, ground_AP_pos, UAV_pos, nReals=None):
        if nReals is not None:
            orig = self._ch.nbrOfRealizations
            self._ch.nbrOfRealizations = nReals
        all_AP = np.vstack([ground_AP_pos, UAV_pos])
        _, _, betas = self._ch.compute_channel_model(UE_pos, all_AP)
        mask  = self._ch.compute_AP_selection_mask(betas)
        rates, sr = self._ch.compute_user_rates(UE_pos, all_AP, mask)
        if nReals is not None:
            self._ch.nbrOfRealizations = orig
        return rates, float(sr), mask, betas

    def _jfi_eff(self, mask, betas):
        G, L, K = self.G, self.L, self.K
        mask_u = mask[:, G:]; mask_g = mask[:, :G]
        bu = betas[:, G:];    bg = betas[:, :G]
        gcov = np.array([bg[k, np.where(mask_g[k])[0]].sum() for k in range(K)])
        eff  = np.zeros(L)
        for l in range(L):
            for k in np.where(mask_u[:, l])[0]:
                eff[l] += bu[k, l] / (gcov[k] + bu[k, l] + 1e-12)
        s = eff.sum()
        return float(s**2 / (L * (eff**2).sum() + 1e-12)) if s > 1e-10 else 1.0

    def _joint_score(self, rates, mask, betas):
        mr  = float(rates.min())
        jfi = self._jfi_eff(mask, betas)
        raw = self.w_min * (mr / self.ref_rate) + self.w_jfi * jfi
        if mr < self.floor_rate:
            raw *= (mr / self.floor_rate) ** 2
        return raw, mr, jfi

    def _det_eval(self, UE_pos, gAP, UAV_pos):
        """确定性最终评估 (固定种子, 多实现)"""
        st = np.random.get_state()
        np.random.seed(self.eval_seed)
        rates, sr, mask, betas = self._channel_and_rates(
            UE_pos, gAP, UAV_pos, nReals=self.nbrReals_final)
        np.random.set_state(st)
        js, mr, jfi = self._joint_score(rates, mask, betas)
        return mr, float(sr), jfi, js, rates


# ======================================================================
#  GA3D_LB — 3D 离散遗传算法 + 负载均衡
# ======================================================================

class GA3D_LB(_ChannelMixin):
    """
    3D Discrete Genetic Algorithm with Load-Balanced fitness.
    网格: grid_xy × grid_xy × n_z  (例: 9×9×4=324 候选点)
    """

    def __init__(self, config: Dict):
        self._init_channel(config)
        sq = config.get('square_length', 1000)
        self.sq = sq
        self.margin  = config.get('grid_margin', 100)
        self.grid_xy = config.get('grid_size', 9)     # x,y 维度
        self.n_z     = config.get('n_z_levels', 4)    # z 高度档位
        self.h_min   = config.get('h_min_uav', 50.0)
        self.h_max   = config.get('h_max_uav', 150.0)

        # GA 超参数
        self.pop_size   = config.get('population_size', 15)
        self.n_gen      = config.get('max_generations', 30)
        self.cx_rate    = config.get('crossover_rate', 0.8)
        self.mut_rate   = config.get('mutation_rate', 0.25)
        self.elite_n    = config.get('elite_size', 3)
        self.tourn_k    = config.get('tournament_size', 4)
        self.max_no_imp = config.get('max_no_improve', 15)

        self._build_grid()

    def _build_grid(self):
        xs = np.linspace(self.margin, self.sq - self.margin, self.grid_xy)
        ys = np.linspace(self.margin, self.sq - self.margin, self.grid_xy)
        zs = np.linspace(self.h_min, self.h_max, self.n_z)
        pts = []
        for x in xs:
            for y in ys:
                for z in zs:
                    pts.append([x, y, z])
        self.grid = np.array(pts)
        self.n_pts = len(self.grid)

    def _decode(self, individual):
        return self.grid[individual]   # (L, 3)

    def _fitness(self, ind, UE_pos, gAP):
        UAV = self._decode(ind)
        try:
            rates, _, mask, betas = self._channel_and_rates(
                UE_pos, gAP, UAV, nReals=self.nbrReals_inner)
            js, _, _ = self._joint_score(rates, mask, betas)
            dup_pen = max(0, self.L - len(np.unique(ind))) * 0.05
            return js - dup_pen
        except:
            return -1.0

    def _tournament(self, fitvals):
        idx = np.random.choice(len(fitvals), self.tourn_k, replace=False)
        return idx[np.argmax(fitvals[idx])]

    def _crossover(self, p1, p2):
        if np.random.rand() > self.cx_rate:
            return p1.copy(), p2.copy()
        pt = np.random.randint(1, self.L)
        c1 = np.concatenate([p1[:pt], p2[pt:]])
        c2 = np.concatenate([p2[:pt], p1[pt:]])
        return c1, c2

    def _mutate(self, ind):
        child = ind.copy()
        for i in range(self.L):
            if np.random.rand() < self.mut_rate:
                child[i] = np.random.randint(self.n_pts)
        return child

    def _init_pop(self, init_UAV_pos):
        pop = []
        # seed: map init_UAV_pos to nearest grid points
        dists = np.linalg.norm(
            self.grid[:, np.newaxis] - init_UAV_pos[np.newaxis], axis=2)
        seed_ind = np.argmin(dists, axis=0)
        pop.append(seed_ind)
        # diversity
        for _ in range(self.pop_size - 1):
            if np.random.rand() < 0.5:
                pop.append(np.random.choice(self.n_pts, self.L, replace=False))
            else:
                step = max(1, self.n_pts // (self.L + 1))
                cands = list(range(0, self.n_pts, step))
                if len(cands) < self.L:
                    cands += list(range(len(cands), self.L))
                ind = np.array(cands[:self.L])
                np.random.shuffle(ind)
                pop.append(ind)
        return [np.array(p) for p in pop]

    def optimize(self, UE_pos, ground_AP_pos, UAV_pos_init) -> Dict:
        pop = self._init_pop(UAV_pos_init)
        fitvals = np.array([self._fitness(p, UE_pos, ground_AP_pos) for p in pop])

        best_fit = fitvals.max()
        best_ind = pop[int(np.argmax(fitvals))].copy()
        no_imp = 0

        for gen in range(self.n_gen):
            new_pop = []
            elite_idx = np.argsort(fitvals)[-self.elite_n:]
            for ei in elite_idx:
                new_pop.append(pop[ei].copy())

            while len(new_pop) < self.pop_size:
                p1 = pop[self._tournament(fitvals)]
                p2 = pop[self._tournament(fitvals)]
                c1, c2 = self._crossover(p1, p2)
                new_pop.extend([self._mutate(c1), self._mutate(c2)])

            pop = new_pop[:self.pop_size]
            fitvals = np.array([self._fitness(p, UE_pos, ground_AP_pos) for p in pop])

            gen_best_fit = fitvals.max()
            if gen_best_fit > best_fit + 1e-5:
                best_fit = gen_best_fit
                best_ind = pop[int(np.argmax(fitvals))].copy()
                no_imp = 0
            else:
                no_imp += 1

            if no_imp >= self.max_no_imp:
                break

        best_UAV = self._decode(best_ind)
        mr, sr, jfi, js, rates = self._det_eval(UE_pos, ground_AP_pos, best_UAV)
        return {
            'optimized_UAV_pos': best_UAV, 'final_min_rate': mr,
            'final_sum_rate': sr, 'final_jfi': jfi, 'final_joint_score': js,
            'final_rates': rates,
        }


# ======================================================================
#  PSO3D_LB — 3D 粒子群优化 + 负载均衡
# ======================================================================

class PSO3D_LB(_ChannelMixin):
    """
    3D Continuous PSO with Load-Balanced fitness.
    粒子维度: L×3 = [x0,y0,z0, x1,y1,z1, ...]
    """

    def __init__(self, config: Dict):
        self._init_channel(config)
        sq = config.get('square_length', 1000)
        self.sq = sq
        self.pos_min = config.get('pos_min', 50.0)
        self.pos_max = sq - self.pos_min
        self.h_min   = config.get('h_min_uav', 50.0)
        self.h_max   = config.get('h_max_uav', 150.0)

        # PSO 超参数
        self.N = config.get('N_particle', 20)
        self.n_iter = config.get('max_iterations', 35)
        self.w  = config.get('w', 0.729)
        self.c1 = config.get('c1', 1.49445)
        self.c2 = config.get('c2', 1.49445)
        self.v_max_xy = config.get('v_max', 50.0)
        self.v_max_z  = config.get('v_max_z', 20.0)
        self.max_no_imp = config.get('max_no_improve', 18)

    def _clip_particle(self, p: np.ndarray) -> np.ndarray:
        p = p.copy().reshape(self.L, 3)
        p[:, 0] = np.clip(p[:, 0], self.pos_min, self.pos_max)
        p[:, 1] = np.clip(p[:, 1], self.pos_min, self.pos_max)
        p[:, 2] = np.clip(p[:, 2], self.h_min, self.h_max)
        return p.flatten()

    def _clip_vel(self, v: np.ndarray) -> np.ndarray:
        v = v.copy().reshape(self.L, 3)
        v[:, :2] = np.clip(v[:, :2], -self.v_max_xy, self.v_max_xy)
        v[:, 2]  = np.clip(v[:, 2], -self.v_max_z,  self.v_max_z)
        return v.flatten()

    def _fitness(self, p, UE_pos, gAP):
        UAV = p.reshape(self.L, 3)
        try:
            rates, _, mask, betas = self._channel_and_rates(
                UE_pos, gAP, UAV, nReals=self.nbrReals_inner)
            js, _, _ = self._joint_score(rates, mask, betas)
            return js
        except:
            return -1.0

    def _init_swarm(self, init_UAV_pos):
        dim = self.L * 3
        particles = np.zeros((self.N, dim))
        velocities = np.zeros((self.N, dim))

        # seed particle = init positions
        particles[0] = init_UAV_pos.flatten()
        for i in range(1, self.N):
            xy = np.random.uniform(self.pos_min, self.pos_max, (self.L, 2))
            z  = np.random.uniform(self.h_min,  self.h_max, (self.L, 1))
            particles[i] = np.hstack([xy, z]).flatten()

        for i in range(self.N):
            vxy = np.random.uniform(-self.v_max_xy/2, self.v_max_xy/2, (self.L, 2))
            vz  = np.random.uniform(-self.v_max_z/2,  self.v_max_z/2, (self.L, 1))
            velocities[i] = np.hstack([vxy, vz]).flatten()

        return particles, velocities

    def optimize(self, UE_pos, ground_AP_pos, UAV_pos_init) -> Dict:
        particles, velocities = self._init_swarm(UAV_pos_init)
        fitvals = np.array([self._fitness(p, UE_pos, ground_AP_pos) for p in particles])

        pbest = particles.copy()
        pbest_fit = fitvals.copy()
        gbest_idx = int(np.argmax(fitvals))
        gbest = particles[gbest_idx].copy()
        gbest_fit = fitvals[gbest_idx]
        no_imp = 0

        for it in range(self.n_iter):
            r1 = np.random.rand(self.N, self.L * 3)
            r2 = np.random.rand(self.N, self.L * 3)
            velocities = (self.w * velocities
                          + self.c1 * r1 * (pbest - particles)
                          + self.c2 * r2 * (gbest - particles))
            velocities = np.array([self._clip_vel(v) for v in velocities])
            particles  = np.array([self._clip_particle(particles[i] + velocities[i])
                                    for i in range(self.N)])

            fitvals = np.array([self._fitness(p, UE_pos, ground_AP_pos) for p in particles])

            imp_mask = fitvals > pbest_fit
            pbest[imp_mask] = particles[imp_mask].copy()
            pbest_fit[imp_mask] = fitvals[imp_mask]

            cur_best_idx = int(np.argmax(pbest_fit))
            if pbest_fit[cur_best_idx] > gbest_fit + 1e-5:
                gbest = pbest[cur_best_idx].copy()
                gbest_fit = pbest_fit[cur_best_idx]
                no_imp = 0
            else:
                no_imp += 1

            if no_imp >= self.max_no_imp:
                break

        best_UAV = gbest.reshape(self.L, 3)
        mr, sr, jfi, js, rates = self._det_eval(UE_pos, ground_AP_pos, best_UAV)
        return {
            'optimized_UAV_pos': best_UAV, 'final_min_rate': mr,
            'final_sum_rate': sr, 'final_jfi': jfi, 'final_joint_score': js,
            'final_rates': rates,
        }


# ======================================================================
#  SSA3D_LB — 3D 麻雀搜索算法 + 负载均衡
# ======================================================================

class SSA3D_LB(_ChannelMixin):
    """
    3D Sparrow Search Algorithm with Load-Balanced fitness.
    个体维度: L×3 = [x0,y0,z0, ...]
    包含 OBL 初始化、生产者/加入者/警觉者角色。
    """

    def __init__(self, config: Dict):
        self._init_channel(config)
        sq = config.get('square_length', 1000)
        self.sq = sq
        self.pos_min = 50.0
        self.pos_max = sq - 50.0
        self.h_min   = config.get('h_min_uav', 50.0)
        self.h_max   = config.get('h_max_uav', 150.0)

        # SSA 超参数
        self.N = config.get('newssa_n_sparrows', 20)
        self.n_iter = config.get('newssa_max_iter', 35)
        self.PR = config.get('newssa_pr', 0.2)   # 生产者比例
        self.FR = config.get('newssa_fr', 0.15)  # 警觉者比例
        self.ST = config.get('newssa_st', 0.8)   # 安全阈值
        self.max_no_imp = config.get('max_no_improve', 18)

        self._dim = self.L * 3
        self._lb = np.tile([self.pos_min, self.pos_min, self.h_min], self.L)
        self._ub = np.tile([self.pos_max, self.pos_max, self.h_max], self.L)

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self._lb, self._ub)

    def _decode(self, x: np.ndarray) -> np.ndarray:
        return self._clip(x).reshape(self.L, 3)

    def _fitness(self, x, UE_pos, gAP):
        UAV = self._decode(x)
        try:
            rates, _, mask, betas = self._channel_and_rates(
                UE_pos, gAP, UAV, nReals=self.nbrReals_inner)
            js, _, _ = self._joint_score(rates, mask, betas)
            return js
        except:
            return -1.0

    def _init_pop(self, init_UAV_pos):
        pop = np.zeros((self.N, self._dim))
        # seed
        pop[0] = init_UAV_pos.flatten()
        # random
        for i in range(1, self.N):
            xy = np.random.uniform(self.pos_min, self.pos_max, (self.L, 2))
            z  = np.random.uniform(self.h_min,  self.h_max, (self.L, 1))
            pop[i] = np.hstack([xy, z]).flatten()
        # OBL: fold remaining half
        mid = (self._lb + self._ub) / 2
        for i in range(self.N // 2, self.N):
            pop[i] = 2 * mid - pop[i - self.N // 2]
        pop = np.array([self._clip(pop[i]) for i in range(self.N)])
        return pop

    def optimize(self, UE_pos, ground_AP_pos, UAV_pos_init) -> Dict:
        pop = self._init_pop(UAV_pos_init)
        fitvals = np.array([self._fitness(pop[i], UE_pos, ground_AP_pos)
                            for i in range(self.N)])

        best_idx = int(np.argmax(fitvals))
        gbest_x  = pop[best_idx].copy()
        gbest_fit = fitvals[best_idx]
        worst_idx = int(np.argmin(fitvals))
        no_imp = 0

        PR_n = max(1, int(self.N * self.PR))
        FR_n = max(1, int(self.N * self.FR))

        for s in range(self.n_iter):
            idx_sorted = np.argsort(fitvals)[::-1]
            alpha = np.random.rand()
            r = np.random.rand()

            # ── 生产者更新 ──
            for rank, i in enumerate(idx_sorted[:PR_n]):
                if r < self.ST:
                    pop[i] = pop[i] * np.exp(-s / (alpha * self.n_iter + 1e-9))
                else:
                    Q = np.random.randn()
                    L_vec = np.ones(self._dim)
                    pop[i] = pop[i] + Q * L_vec
                pop[i] = self._clip(pop[i])

            # ── 加入者更新 ──
            for i in idx_sorted[PR_n:]:
                if rank == 0:
                    worst_x = pop[idx_sorted[-1]]
                    pop[i] = pop[i] + np.random.randn(self._dim) * np.abs(pop[i] - worst_x)
                else:
                    xbest = pop[idx_sorted[0]]
                    A = np.random.choice([-1, 1], self._dim)
                    pop[i] = xbest + np.abs(pop[i] - xbest) * A
                pop[i] = self._clip(pop[i])

            # ── 警觉者更新 ──
            alert_indices = np.random.choice(self.N, FR_n, replace=False)
            xbest = pop[idx_sorted[0]]
            for i in alert_indices:
                fi = fitvals[i]
                if fi > fitvals[idx_sorted[-1]]:
                    pop[i] = xbest + np.random.randn(self._dim) * np.abs(pop[i] - xbest)
                else:
                    beta_val = np.random.randn()
                    pop[i] = pop[i] + beta_val * (np.abs(pop[i] - xbest) / (fi - fitvals[idx_sorted[-1]] + 1e-9))
                pop[i] = self._clip(pop[i])

            fitvals = np.array([self._fitness(pop[i], UE_pos, ground_AP_pos)
                                for i in range(self.N)])
            cur_best = int(np.argmax(fitvals))
            if fitvals[cur_best] > gbest_fit + 1e-5:
                gbest_x   = pop[cur_best].copy()
                gbest_fit = fitvals[cur_best]
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= self.max_no_imp:
                break

        best_UAV = self._decode(gbest_x)
        mr, sr, jfi, js, rates = self._det_eval(UE_pos, ground_AP_pos, best_UAV)
        return {
            'optimized_UAV_pos': best_UAV, 'final_min_rate': mr,
            'final_sum_rate': sr, 'final_jfi': jfi, 'final_joint_score': js,
            'final_rates': rates,
        }


# ======================================================================
#  共用: 单步优化包装 (动态场景)
# ======================================================================

def one_step_optimize(alg, UE_pos, gAP, UAV_pos_prev, max_iter,
                      energy_model=None, flight_speed=10.0,
                      w_min=0.30, w_jfi=0.50, w_ee=0.20,
                      ref_rate=60.0, floor_rate=48.0, dt=5.0):
    """
    动态单步优化包装 (能效对齐版).

    说明:
    1) 先运行启发式算法得到候选 new_pos。
    2) 再在 prev→new 线段上做轻量重评分，使用与动态场景一致的三目标:
       JS = w_min*(min/ref) + w_jfi*JFI + w_ee*(1 - E_step/E_ref)
       + min-rate 软下限惩罚。
    3) 选取 JS 最大的位置作为单步输出。

    返回: (new_pos, min_rate, sum_rate, energy_J, dist_m)
    """
    # 临时压缩迭代次数
    iter_attrs = {
        GA3D_LB: 'n_gen',
        PSO3D_LB: 'n_iter',
        SSA3D_LB: 'n_iter',
    }
    attr = iter_attrs.get(type(alg))
    orig_iter = getattr(alg, attr, None) if attr else None
    if attr:
        setattr(alg, attr, max_iter)

    res = alg.optimize(UE_pos, gAP, UAV_pos_prev)
    raw_new_pos = res['optimized_UAV_pos']

    if attr and orig_iter is not None:
        setattr(alg, attr, orig_iter)

    # 若无能量模型，退化为原逻辑
    if energy_model is None:
        return raw_new_pos, res['final_min_rate'], res['final_sum_rate'], 0.0, 0.0

    def _score_dyn(pos):
        rates, sr, mask, betas = alg._channel_and_rates(UE_pos, gAP, pos, nReals=alg.nbrReals_inner)
        mr = float(rates.min())
        jfi = alg._jfi_eff(mask, betas)
        e_step, dist = energy_model.total_energy_for_repositioning(UAV_pos_prev, pos, flight_speed=flight_speed)
        E_ref = alg.L * energy_model.P_hover * dt * 2.0
        ee = float(np.clip(1.0 - e_step / (E_ref + 1e-6), 0.0, 1.0))
        js = w_min * (mr / ref_rate) + w_jfi * jfi + w_ee * ee
        if mr < floor_rate:
            js *= (mr / floor_rate) ** 2
        return js, mr, float(sr), e_step, dist

    # 线段轻量搜索，避免启发式在动态场景大跳导致能效劣化
    best = None
    alphas = [1.00, 0.85, 0.70, 0.55, 0.40, 0.25]
    for a in alphas:
        cand = UAV_pos_prev + a * (raw_new_pos - UAV_pos_prev)
        js, mr, sr, e_step, dist = _score_dyn(cand)
        if (best is None) or (js > best[0]):
            best = (js, cand, mr, sr, e_step, dist)

    _, new_pos, mr, sr, energy_J, dist_m = best
    return new_pos, mr, sr, energy_J, dist_m


# ======================================================================
#  配置工厂
# ======================================================================

def create_heuristic_config(K=40, L=6, G=9) -> Dict:
    cfg = create_v6_config()
    cfg.update({
        'num_UE': K, 'num_UAV': L, 'num_ground_AP': G,
        'tau_p': K, 'num_serving_APs': 3, 'M': 4,
        'max_iterations': 35,
        # LB fitness
        'w_min': 0.35, 'w_jfi': 0.65, 'ref_rate': 60.0, 'floor_rate': 48.0,
        # 3D bounds
        'h_min_uav': 50.0, 'h_max_uav': 150.0,
        # Inner/outer eval realizations
        'nbrOfRealizations_inner': 20,
        'nbrOfRealizations_final': 50,
        # GA
        'grid_size': 9, 'n_z_levels': 4,
        'population_size': 15, 'max_generations': 30,
        'crossover_rate': 0.8, 'mutation_rate': 0.25,
        'elite_size': 3, 'tournament_size': 4, 'max_no_improve': 15,
        # PSO
        'N_particle': 20, 'w': 0.729, 'c1': 1.49445, 'c2': 1.49445,
        'v_max': 50.0, 'v_max_z': 20.0,
        # SSA
        'newssa_n_sparrows': 20, 'newssa_max_iter': 35,
        'newssa_pr': 0.2, 'newssa_fr': 0.15, 'newssa_st': 0.8,
    })
    return cfg
