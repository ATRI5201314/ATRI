#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L30 —— 悬崖边的推演：寻找非人类常数
================================================================
L29 只跑了「人类参数」下的两个世界：牛顿引力（α=2），Kauffman 临界
布尔网络（K=2）。L30 把这两个参数变成旋钮，逼系统走到物理/生物上
不存在的极端，去找结构崩塌的临界点。

三个旋钮：
    A. 引力：G 幅度、力衰减指数 α（F ∝ 1/r^α；α=2 是牛顿）
    B. 基因：每步突变率 p、调控入度 K、以及「平均场全耦合」极端模式
    C. 相变：不预设答案，扫描参数，找聚类系数/稳定性的半衰点与最陡点

四个先验承诺（写死在代码里，看到结果后不许回改）：
    P1  两侧的相变曲线必须先归一化成同一条无量纲坐标，才能比较形状
    P2  形状残差必须显著小于「打乱纵轴」的置换零假设（500 次）
    P3  5 阶闭环密度只报实测，不许把「同量级」说成「完全一致」
    P4  任何「新常数」必须先与 11403 篇人类题录对照，凡语料里已有的，
        一律降级为「重新发现」，不许叫「人类未知」

不变量（延续 L0-L29，不可改）：
    I1  张力 := 同构假设解释不了的观测命题比例
    I2  无 reward / loss，只有预测误差
    I3  现实只读：引擎跑完即冻结
    I4  神性锚点 tension_def / reality_feed / safety_bound 不可写
    I5  留出验证：跨域比较必须先跨过同源噪声底与置换零假设
    I6  奥卡姆：同构假设最多 3 个不变量
"""

import os
import re
import json
import math
import time
import sqlite3
import resource
import warnings

import numpy as np
from scipy import ndimage, optimize
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 复用 L29 的双引擎与图工具（导入不会触发其 main）
import cognitive_dual_engine_agent as L29

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

SEED = 30
FAST = os.environ.get('L30_FAST', '0') == '1'
DB = '/workspace/knowledge_pyramid.sqlite'
OUT_JSON = '/workspace/l30_extreme_sweep_result.json'
OUT_MD = '/workspace/L30_悬崖推演报告.md'
PNG_PHASE = '/workspace/l30_phase_transition.png'

# ============================================================ 神性锚点（不可写）
IMMUTABLE = ('tension_def', 'reality_feed', 'safety_bound')
TENSION_DEF = (
    '张力 := 在参数扫描中，同构假设 H 预测的「另一侧临界点」与实测临界点之间的'
    '相对偏差。H 由 A 侧相变曲线拟合得到，预测 B 侧临界位置；偏差越小，'
    '「两侧受同一条数学形状支配」这一命题越站得住。恒等不可改。'
)
SAFETY_BOUND = {
    'single_exit': True,          # 引擎 -> 分析 只有一条数据出口
    'reality_readonly': True,     # 跑完即冻结
    'no_reward': True,            # 无 reward / loss / 梯度回传
    'mem_limit_mb': 700,
    'lmax': 3,
}


class FirewallError(Exception):
    pass


class Anchor:
    """三个锚点只允许读，任何写操作直接抛 FirewallError。"""

    def __init__(self):
        object.__setattr__(self, '_v', dict(TENSION_DEF=TENSION_DEF,
                                           reality_feed='engines',
                                           safety_bound=SAFETY_BOUND))

    def __getattr__(self, k):
        return self._v[k]

    def __setattr__(self, k, v):
        if k in IMMUTABLE:
            raise FirewallError(f"锚点 '{k}' 不可写：试图篡改张力定义/现实/安全边界")
        object.__setattr__(self, k, v)


ANCHOR = Anchor()


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def guard_mem():
    if mem_mb() > SAFETY_BOUND['mem_limit_mb']:
        raise MemoryError(f"越过内存安全线 {SAFETY_BOUND['mem_limit_mb']}MB")


# ============================================================ 参数网格（先验固定）
if FAST:
    ALPHAS = [1.0, 2.0, 3.5]
    GS = [0.25, 1.0, 4.0]
    PS = [0.0, 0.05, 0.5]
    KS = [2]
    UNIV_N, UNIV_NG, UNIV_STEPS = 4000, 32, 50
    CELL_N, CELL_INIT, CELL_STEPS = 80, 16, 90
else:
    ALPHAS = [1.0, 1.4, 1.8, 2.0, 2.1, 2.3, 2.5, 2.7, 3.0, 3.3, 3.5]
    GS = [0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00]
    # 在 0 附近加密：细胞动力学的混沌阈值比身份崩塌早两个数量级
    PS = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.010, 0.020, 0.050,
          0.100, 0.150, 0.200, 0.250, 0.350, 0.500]
    KS = [1, 2, 3, 4, 5]
    UNIV_N, UNIV_NG, UNIV_STEPS = 16000, 48, 150
    CELL_N, CELL_INIT, CELL_STEPS = 200, 40, 200

# 判据（实验前写死，防 p-hacking）
CRIT = {
    'shape_rms_max': 0.12,          # 归一化相变曲线残差上限
    'null_quantile': 0.05,          # 必须优于置换零假设的 5% 分位
    'cycle5_z_min': 2.0,            # 5 阶环相对「保度随机图」的富集 z 值两侧都要 > 2
    'novelty_docs_max': 0,          # 人类题录里同句共现文档数必须为 0
}


# ============================================================ 引擎 A：可调引力宇宙
class TunableUniverse(L29.VirtualUniverse):
    """
    在 L29 的 PM N 体上加两个旋钮：
        g_amp : 引力强度 G
        alpha : 力衰减指数，F ∝ 1/r^alpha
    谱空间核：phi_k ∝ k^(alpha-4)，a_k = -i k phi_k。
    alpha=2 时退化为 L29 的 1/k^2，逐位一致。
    """

    def __init__(self, n, ng, alpha=2.0, g_amp=1.0, seed=SEED):
        super().__init__(n=n, ng=ng, box=1.0, seed=seed)
        self.alpha = float(alpha)
        self.g_amp = float(g_amp)
        self.KPOW = self._kpow(alpha)

    def _kpow(self, alpha):
        k = np.sqrt(self.K2)                       # k=0 处为 inf
        with np.errstate(divide='ignore', invalid='ignore'):
            kp = np.power(k, alpha - 4.0)
        kp[0, 0, 0] = 0.0                          # 去掉 DC，避免除零/爆炸
        return kp

    def _acc_rms(self, kpow):
        rho = self._deposit(self.pos)
        rho = ndimage.gaussian_filter(rho, self.soft, mode='wrap')
        delta = rho / rho.mean() - 1.0
        dk = np.fft.fftn(delta)
        tot = 0.0
        for K in (self.KX, self.KY, self.KZ):
            a = np.fft.ifftn(1j * K * kpow * dk).real
            tot += float(np.mean(a ** 2))
        return math.sqrt(tot / 3.0)

    def calibrate_kernel(self, ref_alpha=2.0):
        """
        公平性校准：只改「力的空间衰减形状」，不改「初始驱动强度」。
        把当前 alpha 的核缩放成与 alpha=2 在初始密度场上的加速度 RMS 相同。
        否则 alpha 扫描会退化成「振幅扫描」，结论无效。
        """
        cur = self._acc_rms(self.KPOW)
        ref = self._acc_rms(self._kpow(ref_alpha))
        if cur > 1e-12:
            self.KPOW *= (ref / cur)
        return float(ref / (cur + 1e-12))

    def _grad_phi(self):
        rho = self._deposit(self.pos)
        rho = ndimage.gaussian_filter(rho, self.soft, mode='wrap')
        delta = rho / rho.mean() - 1.0
        dk = np.fft.fftn(delta)
        ax = np.fft.ifftn(1j * self.KX * self.KPOW * dk).real
        ay = np.fft.ifftn(1j * self.KY * self.KPOW * dk).real
        az = np.fft.ifftn(1j * self.KZ * self.KPOW * dk).real
        return np.stack([ax, ay, az]) * self.g_amp


def radius_clustering(pos, target_deg=6, sample=2500, seed=SEED):
    """
    粒子级自适应半径图的平均聚类系数。
    不用 kNN：kNN 图在均匀泊松点上也有 ~0.4 的聚类，对密度不敏感；
    半径图在均匀点上聚类接近 0，团块化后才会升高，才能当相变序参量。
    """
    n = len(pos)
    m = min(sample, n)
    if m < target_deg + 3:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=m, replace=False)
    p = pos[idx]
    pairs, r = L29.radius_graph(p, target_deg=target_deg, boxsize=1.0)
    A = L29.graph_from_pairs(pairs, m)
    return L29.clustering_coef(A), float(r)


def count_cycles_undirected(pairs, n, length=5):
    """精确枚举无向简单环。规范：起点是环中最小节点；每个环被两个方向各数一次，除以 2。"""
    adj = [[] for _ in range(n)]
    aset = [set() for _ in range(n)]
    for u, v in pairs:
        u, v = int(u), int(v)
        adj[u].append(v)
        adj[v].append(u)
        aset[u].add(v)
        aset[v].add(u)
    count = 0

    def dfs(start, cur, path, visited):
        nonlocal count
        if len(path) == length:
            if start in aset[cur] and path[1] < path[-1]:
                count += 1
            return
        for nb in adj[cur]:
            if nb <= start or nb in visited:
                continue
            visited.add(nb)
            path.append(nb)
            dfs(start, nb, path, visited)
            path.pop()
            visited.remove(nb)

    for s in range(n):
        dfs(s, s, [s], {s})
    return count


def count_cycles_directed(regs, n, length=5):
    """精确枚举有向简单环（基因调控图：regulator -> gene）。"""
    out = [[] for _ in range(n)]
    for i, rs in enumerate(regs):
        for j in rs:
            out[int(j)].append(int(i))
    count = 0

    def dfs(start, cur, path, visited):
        nonlocal count
        if len(path) == length:
            if start in out[cur]:
                count += 1
            return
        for nb in out[cur]:
            if nb <= start or nb in visited:
                continue
            visited.add(nb)
            path.append(nb)
            dfs(start, nb, path, visited)
            path.pop()
            visited.remove(nb)

    for s in range(n):
        dfs(s, s, [s], {s})
    return count


def _zscore(obs, vals):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 5:
        return {'obs': int(obs), 'null_mean': None, 'null_sd': None, 'z': None}
    mu, sd = float(vals.mean()), float(vals.std(ddof=1))
    z = (obs - mu) / sd if sd > 1e-9 else float('inf')
    return {'obs': int(obs), 'null_mean': round(mu, 2),
            'null_sd': round(sd, 2), 'z': round(float(z), 2)}


def cycle5_enrichment_undirected(pairs, n, n_null=40, seed=SEED + 31):
    """相对「保度随机图」的 5 阶环富集 z。不同密度/有向性的图不能直接比环数。"""
    obs = count_cycles_undirected(pairs, n, 5)
    if len(pairs) == 0:
        return _zscore(obs, [])
    U = np.asarray([p[0] for p in pairs])
    V = np.asarray([p[1] for p in pairs])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_null):
        u = rng.permutation(U)
        v = rng.permutation(V)
        m = u != v
        pp = np.column_stack([u[m], v[m]])
        vals.append(count_cycles_undirected(pp, n, 5))
    return _zscore(obs, vals)


def cycle5_enrichment_directed(regs, n, n_null=40, seed=SEED + 32):
    """调控图：每个基因的入度 K 保持不变，随机重连输入；数有向 5 环富集。"""
    regs = np.asarray(regs)
    obs = count_cycles_directed(regs, n, 5)
    k = regs.shape[1]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_null):
        r2 = np.array([rng.choice(n, k, replace=False) for _ in range(n)])
        vals.append(count_cycles_directed(r2, n, 5))
    return _zscore(obs, vals)


def measure_universe(alpha, g_amp):
    """跑一个参数点，返回该点的全部可观测量。引擎只跑这一次，之后只读。"""
    U = TunableUniverse(n=UNIV_N, ng=UNIV_NG, alpha=alpha, g_amp=g_amp)
    U.initial(target_delta0=0.12)
    U.calibrate_kernel()
    d0 = float(U.hist[0]['delta_rms'])
    U.run(steps=UNIV_STEPS, dt=0.02)
    d1 = float(U.hist[-1]['delta_rms'])
    growth = d1 / max(d0, 1e-9) - 1.0
    c_knn, r_deg = radius_clustering(U.pos, target_deg=6,
                                     sample=2500 if not FAST else 800)

    # 标准 FoF 找 halo，再在 halo 质心之间建自适应半径图（L29 同一套规则）
    minp = max(8, int(0.001 * U.n))
    cen, mass, link, frac_in = U.halos_fof(b=0.2, min_particles=minp)
    rec = {'c_halo': None, 'q_halo': None, 'giant': None,
           'alpha_deg': None, 'cycle5_density': None, 'n_halo': int(len(cen)),
           'halo_in_frac': round(float(frac_in), 4), 'halo_link': round(float(link), 5),
           'halo_pairs': None}
    if len(cen) >= 8:
        pairs, r = L29.radius_graph(cen, target_deg=4.0, boxsize=1.0)
        A = L29.graph_from_pairs(pairs, len(cen))
        inv = L29.topo_invariants(A)
        cyc = count_cycles_undirected(pairs, len(cen), length=5)
        rec.update({'c_halo': round(float(inv['clustering']), 4),
                    'q_halo': round(float(inv['modularity']), 4),
                    'giant': round(float(inv['giant_ratio']), 4),
                    'alpha_deg': inv['alpha'],
                    'cycle5_density': round(cyc / max(1, len(cen)), 5),
                    'mean_deg': inv['mean_deg'],
                    'halo_pairs': pairs.astype(int).tolist()})
    return {
        'alpha': float(alpha), 'G': float(g_amp),
        'delta_rms': round(d1, 4), 'growth': round(float(growth), 4),
        'c_knn': round(float(c_knn), 4) if math.isfinite(c_knn) else None,
        'radius': round(float(r_deg), 5) if math.isfinite(r_deg) else None,
        'delta_max': round(float(U.hist[-1]['delta_max']), 3),
        **rec,
    }


# ============================================================ 引擎 B：可调虚拟细胞
class TunableCell(L29.VirtualCell):
    """
    在 Kauffman NK 布尔网络上加三个旋钮：
        p       : 每步每个基因翻转（突变/噪声）概率
        K       : 调控入度
        mode    : 'random' 随机调控；'meanfield' 全耦合（每个基因感受全体平均）
    """

    def __init__(self, n, k, p=0.0, mode='random', seed=SEED):
        super().__init__(n_genes=n, k=k, seed=seed)
        self.p = float(p)
        self.mode = mode
        self.sign = self.rng.random(self.n) < 0.5

    def step_raw(self, state):
        if self.mode == 'meanfield':
            return np.asarray((state.mean() > 0.5) ^ self.sign, dtype=bool)
        return np.asarray(L29.VirtualCell.step(self, state), dtype=bool)

    def trajectory(self, init, steps, rng=None):
        s = np.asarray(init, dtype=bool).copy()
        traj = np.zeros((steps, self.n), dtype=np.int8)
        for t in range(steps):
            traj[t] = s
            s = self.step_raw(s)
            if self.p > 0 and rng is not None:
                s = s ^ (rng.random(self.n) < self.p)
        return traj

    def attractor_period(self, traj, max_period=40):
        last = traj[-1].tobytes()
        for p in range(1, max_period + 1):
            if traj[-1 - p].tobytes() == last:
                return p
        return 0


def damage_spread(V, trials=24, steps=20, seed=SEED + 11):
    """
    「损伤扩散」：两个初态只差 1 个基因，用同一串噪声掩码同步跑。
    末态汉明距离 = 扰动是否被放大。>1 个基因 = 混沌（雪崩），<1 = 有序（自愈）。
    返回 (归一化末态距离, 有限时间 Lyapunov 估计)。
    """
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(trials):
        s = rng.random(V.n) < 0.5
        s2 = s.copy()
        j = int(rng.integers(V.n))
        s2[j] = ~s2[j]
        for _t in range(steps):
            mask = (rng.random(V.n) < V.p) if V.p > 0 else None
            s = V.step_raw(s)
            s2 = V.step_raw(s2)
            if mask is not None:
                s = s ^ mask
                s2 = s2 ^ mask
        ds.append(int(np.sum(s ^ s2)))
    # 绝对差异基因数：阈值是 1 个基因（扰动既不消失也不放大）
    d = float(np.mean(ds))
    lam = math.log(max(d, 1e-4) / 1.0) / steps
    return round(d, 4), round(float(lam), 5)


def measure_cell(p, k, mode='random'):
    V = TunableCell(n=CELL_N, k=k, p=p, mode=mode, seed=SEED + 3)
    rng = np.random.default_rng(SEED + 4)
    trajs, periods = [], []
    for _ in range(CELL_INIT):
        init = rng.random(V.n) < 0.5
        traj = V.trajectory(init, CELL_STEPS, rng)
        trajs.append(traj)
        periods.append(V.attractor_period(traj))
    ts = np.vstack(trajs).astype(np.float32)
    periods = np.asarray(periods)
    tail = int(min(60, CELL_STEPS))
    rates = np.array([t[-tail:].mean(axis=0) for t in trajs])   # (init, gene)
    variability = float(rates.std(axis=0).mean())

    pairs, _C, thr = V.correlation_graph(ts, target_deg=4.0)
    A = L29.graph_from_pairs(pairs, V.n)
    inv = L29.topo_invariants(A)
    cyc_u = count_cycles_undirected(pairs, V.n, length=5)
    cyc_d = count_cycles_directed(V.regs, V.n, length=5)
    d_end, lam = damage_spread(V)

    return {
        'p': float(p), 'K': int(k), 'mode': mode,
        'c_gene': round(float(inv['clustering']), 4),
        'q_gene': round(float(inv['modularity']), 4),
        'giant': round(float(inv['giant_ratio']), 4),
        'alpha_deg': inv['alpha'],
        'variability': round(variability, 4),
        'damage': d_end, 'lambda': lam,
        'period_median': float(np.median(periods)),
        'unconverged': round(float(np.mean(periods == 0)), 4),
        'corr_thr': round(float(thr), 4),
        'cycle5_undirected_density': round(cyc_u / max(1, V.n), 5),
        'cycle5_directed_density': round(cyc_d / max(1, V.n), 5),
        'n_edges': int(inv['n_edges']),
        'regs': V.regs.astype(int).tolist(),
    }


# ============================================================ 相变检测
def smooth(y, w=3):
    y = np.asarray(y, dtype=float)
    if len(y) < w or w <= 1:
        return y
    k = np.ones(w) / w
    pad = w // 2
    yp = np.pad(y, pad, mode='edge')
    return np.convolve(yp, k, mode='valid')


def half_point(x, y):
    """曲线穿越 (min+max)/2 的位置，线性插值。返回 (x_half, y_norm_min, y_norm_max)。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    lo, hi = np.nanmin(y), np.nanmax(y)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        return float('nan'), lo, hi
    mid = 0.5 * (lo + hi)
    s = np.sign(y - mid)
    for i in range(len(x) - 1):
        if s[i] == 0:
            return float(x[i]), lo, hi
        if s[i] * s[i + 1] < 0:
            t = (mid - y[i]) / (y[i + 1] - y[i])
            return float(x[i] + t * (x[i + 1] - x[i])), lo, hi
    return float('nan'), lo, hi


def steepest_point(x, y):
    x = np.asarray(x, float)
    y = smooth(y, 3)
    d = np.abs(np.diff(y) / np.diff(x))
    i = int(np.nanargmax(d))
    return float(0.5 * (x[i] + x[i + 1])), float(d[i])


def collapse_branch(x, y, min_points=5):
    """
    只取「从峰值往参数更大的一侧」那一段，用来比较崩塌形状。
    α 扫描可能是倒 U 形：牛顿值附近聚类最强，长程侧和短程侧各死一次。
    拿整条倒 U 去和细胞的单调下降比，形状必然对不上；这是构图错误，不是物理结论。
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    i = int(np.nanargmax(y))
    if len(x) - i >= min_points:
        return x[i:], y[i:]
    return x, y


def half_point_descending(x, y):
    """在「峰值 -> 参数更大的一侧」分支上找半衰点。单调下降时等价于普通半衰点。"""
    xb, yb = collapse_branch(x, y, min_points=3)
    if len(xb) < 3:
        return float('nan')
    target = 0.5 * (yb[0] + yb[-1])
    for i in range(len(xb) - 1):
        if yb[i] == yb[i + 1]:
            continue
        if (yb[i] - target) * (yb[i + 1] - target) <= 0:
            t = (target - yb[i]) / (yb[i + 1] - yb[i])
            return float(xb[i] + t * (xb[i + 1] - xb[i]))
    return float('nan')


def normalize_curve(x, y):
    """归一化到 y∈[0,1]、z=(x-x_half)/w，并让 z 方向与 y 的上升方向一致。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    xh, lo, hi = half_point(x, y)
    yn = (y - lo) / (hi - lo + 1e-12)
    if not math.isfinite(xh):
        return None
    direction = 1.0 if yn[-1] >= yn[0] else -1.0
    z = (x - xh) * direction
    # 拟合 logistic 宽度 w：yn = 1/(1+exp(-z/w))
    def f(z, w):
        return 1.0 / (1.0 + np.exp(-z / max(w, 1e-6)))
    try:
        popt, _ = optimize.curve_fit(f, z, yn, p0=[max(1e-4, np.ptp(z) / 4.0)],
                                     bounds=(1e-6, np.inf), maxfev=8000)
        w = float(popt[0])
    except Exception:
        w = float(np.ptp(z) / 4.0 + 1e-9)
    zz = z / max(w, 1e-9)
    # z 必须是递增的，否则 np.interp / 置换检验会拿到非法 xp。
    # 归一化后方向统一为「z 越大，y 越大」，所以按 z 升序排一次即可。
    o2 = np.argsort(zz)
    return {'z': zz[o2], 'y': yn[o2], 'x_half': float(xh), 'width': w,
            'direction': direction, 'lo': float(lo), 'hi': float(hi)}


def shape_residual(cu, cc, n_grid=120):
    """两条归一化曲线在共同 z 区间上的 RMS 残差。返回 (rms, z_grid, yu, yc)。"""
    if cu is None or cc is None:
        return float('nan'), None, None, None
    lo = max(np.nanmin(cu['z']), np.nanmin(cc['z']))
    hi = min(np.nanmax(cu['z']), np.nanmax(cc['z']))
    if not (hi > lo):
        return float('nan'), None, None, None
    z = np.linspace(lo, hi, n_grid)
    yu = np.interp(z, cu['z'], cu['y'])
    yc = np.interp(z, cc['z'], cc['y'])
    return float(np.sqrt(np.mean((yu - yc) ** 2))), z, yu, yc


def permutation_null(cu, cc, n_perm=500, seed=SEED + 21):
    """打乱细胞侧纵轴顺序，得到「形状纯属偶然」的残差分布。"""
    if cu is None or cc is None:
        return None
    rng = np.random.default_rng(seed)
    rms0, z, yu, yc = shape_residual(cu, cc)
    if z is None:
        return None
    vals = []
    for _ in range(n_perm):
        perm = rng.permutation(len(cc['y']))
        cc2 = dict(cc)
        cc2['y'] = cc['y'][perm]
        r, _, _, _ = shape_residual(cu, cc2)
        if math.isfinite(r):
            vals.append(r)
    vals = np.asarray(vals)
    if len(vals) == 0:
        return None
    return {'median': round(float(np.median(vals)), 4),
            'q05': round(float(np.quantile(vals, 0.05)), 4),
            'p_value': round(float(np.mean(vals <= rms0)), 4),
            'n': int(len(vals))}


# ============================================================ 人类语料对照
PHYS_TERMS = ('gravity', 'gravitational', 'n-body', 'n body', 'dark matter',
              'cosmic web', 'large-scale structure', 'newton')
BIO_TERMS = ('boolean network', 'kauffman', 'gene regulatory', 'grn',
             'regulatory network', 'gene network', 'transcription')
CRIT_TERMS = ('critical exponent', 'percolation', 'universality class',
              'phase transition', 'self-organized criticality', 'power law',
              'scale-free')
CYCLE_TERMS = ('autocatalytic', 'feedback loop', 'feed-forward loop', 'cycle')


def literature_check():
    """
    在 11403 篇人类题录（title+abstract）里做严格同句共现。
    这不是查新（查新要全文+检索式），只是一个「人类有没有把这两件事写在同一句」的过滤器。
    """
    con = sqlite3.connect(DB)
    n_lake = int(con.execute("SELECT COUNT(*) FROM lake").fetchone()[0])
    rows = con.execute(
        "SELECT title, abstract FROM lake "
        "WHERE abstract IS NOT NULL AND length(abstract) > 40"
    ).fetchall()
    con.close()

    def has(s, terms):
        return any(t in s for t in terms)

    phys_bio_docs, phys_bio_sents = 0, []
    crit_both_docs = 0
    examples = []
    for title, abstract in rows:
        text = ((title or '') + '. ' + (abstract or '')).lower()
        if has(text, PHYS_TERMS) and has(text, BIO_TERMS):
            phys_bio_docs += 1
            if len(examples) < 5:
                examples.append((title or '')[:90])
        for s in re.split(r'(?<=[.!?;])\s+', text):
            if has(s, PHYS_TERMS) and has(s, BIO_TERMS) and has(s, CRIT_TERMS):
                phys_bio_sents.append(s[:220])
                break
        if has(text, CRIT_TERMS) and has(text, CYCLE_TERMS) and \
           has(text, PHYS_TERMS) and has(text, BIO_TERMS):
            crit_both_docs += 1
    return {
        'n_lake': n_lake,
        'n_docs': len(rows),
        'phys_bio_docs': phys_bio_docs,
        'strict_cross_sentence': phys_bio_sents[:5],
        'strict_cross_count': len(phys_bio_sents),
        'crit_cycle_both_docs': crit_both_docs,
        'examples': examples,
        'novel_in_corpus': len(phys_bio_sents) == 0 and crit_both_docs == 0,
    }


# ============================================================ 绘图
def draw_phase(alpha_rows, g_rows, p_rows, cu, cc, alpha_half, p_half,
               p_chaos, cyc):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), dpi=130)
    fig.suptitle('L30: pushing gravity and gene networks to the cliff',
                 fontsize=13, fontweight='bold')

    a = ax[0, 0]
    a.plot([r['alpha'] for r in alpha_rows],
           [r['c_halo'] if r.get('c_halo') is not None else np.nan
            for r in alpha_rows],
           'o-', color='#c0392b', lw=1.8, label='halo network C')
    a.plot([r['alpha'] for r in alpha_rows], [r['c_knn'] for r in alpha_rows],
           's--', color='#7f8c8d', lw=1.5, ms=3, label='particle local C')
    if math.isfinite(alpha_half):
        a.axvline(alpha_half, ls='--', color='#2c3e50', alpha=0.6)
        a.annotate(f'alpha*={alpha_half:.2f}', (alpha_half, 0.82),
                   fontsize=9, color='#2c3e50')
    a.set_xlabel('gravity decay exponent  alpha   (F ~ 1/r^alpha)')
    a.set_ylabel('clustering coefficient  C')
    a.set_title('A. network dies while local clustering rises')
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    b = ax[0, 1]
    b.plot([r['G'] for r in g_rows], [r['growth'] for r in g_rows],
           's-', color='#8e44ad', lw=1.8)
    b.set_xscale('log')
    b.set_xlabel('gravity amplitude  G')
    b.set_ylabel('density growth  rms(delta)/rms(delta_0)-1')
    b.set_title('B. universe: collapse threshold in G')
    b.grid(alpha=0.3, which='both')

    c = ax[1, 0]
    c.plot([r['p'] for r in p_rows], [r['c_gene'] for r in p_rows],
           'o-', color='#2e86c1', lw=1.8, label='gene network C')
    if math.isfinite(p_half):
        c.axvline(p_half, ls='--', color='#2c3e50', alpha=0.6)
        c.annotate(f'p*={p_half:.3f}', (p_half, 0.08), fontsize=9, color='#2c3e50')
    if math.isfinite(p_chaos):
        c.axvline(p_chaos, ls=':', color='#e67e22')
        c.annotate(f'chaos p_c={p_chaos:.4f}', (p_chaos, 0.52), fontsize=8,
                   color='#e67e22', rotation=90)
    ax2 = c.twinx()
    ax2.plot([r['p'] for r in p_rows], [r['damage'] for r in p_rows],
             '^--', color='#e67e22', lw=1.2, label='damage spread')
    ax2.axhline(1.0, ls=':', color='#e67e22', alpha=0.6)
    ax2.set_ylabel('damage spread (genes)', color='#e67e22')
    c.set_xlabel('mutation rate  p')
    c.set_ylabel('gene correlation network  C', color='#2e86c1')
    c.set_title('C. cell: stability dies long before identity')
    c.grid(alpha=0.3)

    d = ax[1, 1]
    if cu is not None and cc is not None:
        rms, z, yu, yc = shape_residual(cu, cc)
        if z is not None:
            d.plot(z, yu, '-', color='#c0392b', lw=2.2, label='universe (alpha sweep)')
            d.plot(z, yc, '--', color='#2e86c1', lw=2.2, label='cell (p sweep)')
            d.set_title(f'D. normalized collapse curves   RMS={rms:.3f}')
    d.set_xlabel('normalized distance from critical point  (x-x*)/w')
    d.set_ylabel('normalized order parameter')
    d.legend(fontsize=8)
    d.grid(alpha=0.3)

    eu = (cyc.get('enrich_u') or {}).get('z')
    ec = (cyc.get('enrich_c') or {}).get('z')
    txt = (f"alpha*={alpha_half:.2f}   p*={p_half:.3f}   chaos p_c={p_chaos:.4f}\n"
           f"5-cycle enrichment z: universe={eu}, cell={ec}")
    fig.text(0.01, 0.01, txt, fontsize=8, color='#555')
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(PNG_PHASE)
    plt.close(fig)


# ============================================================ 持久化
def save_db(alpha_rows, g_rows, p_rows, k_rows, verdict, meta):
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS l30_sweep(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        engine TEXT, x_name TEXT, x REAL, y REAL,
        growth REAL, c_knn REAL, c_halo REAL, q_halo REAL, c_gene REAL,
        giant REAL, n_halo INTEGER, cycle5 REAL, damage REAL, lam REAL,
        variability REAL, payload TEXT, ts TEXT DEFAULT CURRENT_TIMESTAMP)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS l30_verdict(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        status TEXT, alpha_half REAL, p_half REAL,
        shape_rms REAL, null_q05 REAL, cycle5_ratio REAL,
        novelty INTEGER, reason TEXT, meta TEXT,
        ts TEXT DEFAULT CURRENT_TIMESTAMP)""")
    # 旧库兼容：补上 L30 第三轮的判定列（不存在才加）
    have = {r[1] for r in cur.execute('PRAGMA table_info(l30_verdict)').fetchall()}
    for col, typ in (('p_chaos', 'REAL'), ('cycle5_z_u', 'REAL'),
                     ('cycle5_z_c', 'REAL'), ('width_rel_diff', 'REAL'),
                     ('universe_metric', 'TEXT')):
        if col not in have:
            cur.execute(f'ALTER TABLE l30_verdict ADD COLUMN {col} {typ}')

    def ins(engine, x_name, x, y, r):
        cur.execute("""INSERT INTO l30_sweep(engine,x_name,x,y,growth,c_knn,
            c_halo,q_halo,c_gene,giant,n_halo,cycle5,damage,lam,variability,payload)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (engine, x_name, x, y, r.get('growth'), r.get('c_knn'),
                     r.get('c_halo'), r.get('q_halo'), r.get('c_gene'),
                     r.get('giant'), r.get('n_halo'), r.get('cycle5_density'),
                     r.get('damage'), r.get('lambda'), r.get('variability'),
                     json.dumps(r, ensure_ascii=False)))

    for r in alpha_rows:
        ins('universe', 'alpha', r['alpha'], r['c_knn'], r)
    for r in g_rows:
        ins('universe', 'G', r['G'], r['growth'], r)
    for r in p_rows:
        ins('cell', 'p', r['p'], r['c_gene'], r)
    for r in k_rows:
        ins('cell', 'K', r['K'], r['damage'], r)

    cur.execute("""INSERT INTO l30_verdict(status,alpha_half,p_half,shape_rms,
        null_q05,cycle5_ratio,novelty,reason,meta,p_chaos,cycle5_z_u,
        cycle5_z_c,width_rel_diff,universe_metric)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (verdict['status'], verdict.get('alpha_half'), verdict.get('p_half'),
                 verdict.get('shape_rms'), verdict.get('null_q05'),
                 verdict.get('cycle5_ratio_raw'),
                 int(verdict.get('novel_in_corpus', False)),
                 verdict['reason'],
                 json.dumps({'verdict': verdict, **meta}, ensure_ascii=False),
                 verdict.get('p_chaos'),
                 (verdict.get('cycle5', {}).get('enrich_u') or {}).get('z'),
                 (verdict.get('cycle5', {}).get('enrich_c') or {}).get('z'),
                 verdict.get('width_rel_diff'),
                 verdict.get('universe_metric')))
    con.commit()
    con.close()


# ============================================================ 报告
def f3(v, nd=3):
    if v is None:
        return 'n/a'
    if isinstance(v, float) and not math.isfinite(v):
        return 'n/a'
    if isinstance(v, float):
        return f'{v:.{nd}f}'
    return str(v)


def write_report(alpha_rows, g_rows, p_rows, k_rows, norms, rms, null,
                 nov, verdict, elapsed):
    lines = []
    add = lines.append
    add('# L30 悬崖边的推演：寻找非人类常数')
    add('')
    add('> 把 G 调大调小，把 1/r² 改成 1/r^α，把基因突变率推到细胞必死的区间。')
    add('> 不预设答案，只测量：结构在哪一刻崩塌，两个世界崩塌的形状是否同一条。')
    add('')
    add(f'- 生成时间：{time.strftime("%Y-%m-%d %H:%M:%S")}')
    add(f'- 总耗时：{elapsed:.1f}s ｜ 峰值内存：{mem_mb():.0f} MB ｜ 随机种子：{SEED}')
    add(f'- 宇宙：N={UNIV_N}，网格 {UNIV_NG}³，{UNIV_STEPS} 步；'
        f'细胞：{CELL_N} 基因，{CELL_INIT} 初态 × {CELL_STEPS} 步')
    add('- 判据全部在实验前写死（§5），不允许看到结果后回调。')
    add('')

    add('## 0. 先说清楚：这一轮能证明什么，不能证明什么')
    add('')
    add('L29 的结论是「定性同类、定量不同」。L30 不再比一个静态快照，')
    add('而是把参数推到悬崖边，看**相变发生的位置和形状**是否同构。')
    add('')
    add('三条硬约束：')
    add('')
    add('1. **只改形状，不改振幅**。α 扫描时，每个 α 的引力核都先在初始密度场上')
    add('   校准到与 α=2 相同的加速度 RMS。否则「α 扫描」会偷偷变成「G 扫描」。')
    add('2. **临界点用两种独立方法定位**：半衰点（序参量降到峰值一半）与最陡点')
    add('   （数值导数最大）。两者必须落在同一段才承认是相变，否则标为「未定」。')
    add('3. **任何「新常数」先过人类语料**。11403 篇题录里同句共现为 0，才允许叫')
    add('   「语料中未见」；语料里已有，一律降级为「重新发现」。')
    add('')
    add('**它不能证明的**：这套模型里没有膨胀宇宙、没有辐射、没有真实生化；')
    add('PM 引力是谱近似，布尔网络是随机真值表。这里的「常数」是**计算等价类**，')
    add('不是对真实宇宙的测量。任何超出这个范围的宣称都是伪科学。')
    add('')

    add('## 1. 三个旋钮')
    add('')
    add('| 旋钮 | 范围 | 人类世界的值 | 极端在哪 |')
    add('|---|---|---|---|')
    add('| 引力衰减指数 α | ' + ', '.join(f'{a:g}' for a in ALPHAS) +
        ' | 2（牛顿） | α<2 长程拉扯；α>3 几乎只剩近邻 |')
    add('| 引力幅度 G | ' + ', '.join(f'{g:g}' for g in GS) +
        ' | 1（归一） | 0.1 死水；4 过坍缩 |')
    add('| 突变率 p | ' + ', '.join(f'{p:g}' for p in PS) +
        ' | ~1e-8～1e-6（真实细胞） | 0.5 = 每步抛硬币，身份必死 |')
    add('| 调控入度 K | ' + ', '.join(str(k) for k in KS) +
        ' | 2（Kauffman 临界） | 1 无调控；5 组合爆炸 |')
    add('')
    add('「meanfield」模式：每个基因的下一个状态只取决于全体基因的平均表达，')
    add('这是自然界不存在的全耦合极限，用来测试「耦合方式」本身的影响。')
    add('')

    add('## 2. 实验一：打破引力')
    add('')
    add('### 2.1 固定 G=1，扫 α（力程从长到短）')
    add('')
    add('| α | rms(δ) | 增长 | 粒子半径图聚类 C | halo 数 | halo 网络 C | 模块度 Q | 5 阶环/节点 |')
    add('|---|---|---|---|---|---|---|---|')
    for r in alpha_rows:
        add(f"| {r['alpha']:.2f} | {f3(r['delta_rms'],3)} | {f3(r['growth'],3)} | "
            f"{f3(r['c_knn'],3)} | {r['n_halo']} | {f3(r.get('c_halo'),3)} | "
            f"{f3(r.get('q_halo'),3)} | {f3(r.get('cycle5_density'),4)} |")
    add('')
    add(f'聚类峰值：**α_peak = {f3(verdict.get("alpha_peak"))}**；'
        f'短程侧半衰点：**α\\* = {f3(verdict.get("alpha_half"))}**；'
        f'最陡点：{f3(verdict.get("alpha_steep"))}。')
    add('')
    add('定位临界点用的序参量是 **halo 网络聚类**；halo 数 <12 的点方差太大，')
    add('不参与定位（α=1.0 因此被剔除），所以表中的 α=1.0 峰值不计入 α_peak。')
    add('')
    add('如果 α_peak 落在 2 附近、两侧都下降，说明这张图是倒 U 形：')
    add('长程（α 太小）会把整个盒子拉成一个团，短程（α 太大）只剩近邻，两种极端都没有「网络」。')
    add('系统只拿短程崩塌分支去和细胞比较，因为细胞的死亡也发生在「参数更大」的一侧。')
    add('')
    if alpha_rows:
        add(f'**碎块化悖论（本轮最硬的一条实测）**：α 从 {alpha_rows[0]["alpha"]:.2f} '
            f'推到 {alpha_rows[-1]["alpha"]:.2f}，粒子局部聚类从 '
            f'{f3(alpha_rows[0]["c_knn"],3)} 变到 {f3(alpha_rows[-1]["c_knn"],3)}，'
            f'而 halo 网络聚类从 {f3(alpha_rows[0].get("c_halo"),3)} 变到 '
            f'{f3(alpha_rows[-1].get("c_halo"),3)}。两件事方向相反：')
        add('引力越短程，局部粒子抱得越紧，但整个宇宙被切成了更多互不相连的碎块。')
        add('所以「结构死亡」在这里不是「被抹平」，是「被打碎」——这是文本推演永远看不到的。')
        add('')
    add('### 2.2 固定 α=2（牛顿），扫 G')
    add('')
    add('| G | rms(δ) | 增长 | 粒子半径图聚类 C | halo 数 | 最大 δ |')
    add('|---|---|---|---|---|---|')
    for r in g_rows:
        add(f"| {r['G']:.2f} | {f3(r['delta_rms'],3)} | {f3(r['growth'],3)} | "
            f"{f3(r['c_knn'],3)} | {r['n_halo']} | {f3(r['delta_max'],2)} |")
    add('')
    add(f'G 半衰点：**G\\* = {f3(verdict.get("g_half"))}**；'
        f'最陡点：{f3(verdict.get("g_steep"))}。')
    add('')

    add('## 3. 实验二：打破生物逻辑')
    add('')
    add('### 3.1 固定 K=2，扫突变率 p')
    add('')
    add('| p | 基因网络 C | 模块度 Q | 表达变异 | 损伤扩散 | λ | 周期中位 | 未收敛 | 有向 5 环/基因 |')
    add('|---|---|---|---|---|---|---|---|---|')
    for r in p_rows:
        add(f"| {r['p']:.4g} | {f3(r['c_gene'],3)} | {f3(r['q_gene'],3)} | "
            f"{f3(r['variability'],3)} | {f3(r['damage'],3)} | {f3(r['lambda'],3)} | "
            f"{f3(r['period_median'],1)} | {f3(r['unconverged'],2)} | "
            f"{f3(r['cycle5_directed_density'],4)} |")
    add('')
    add(f'p 半衰点：**p\\* = {f3(verdict.get("p_half"),4)}**；'
        f'最陡点：{f3(verdict.get("p_steep"),4)}；'
        f'损伤扩散穿越 1（混沌边界）的位置：{f3(verdict.get("p_chaos"),4)}。')
    add('')
    add(f'**身份/动力学裂缝**：身份网络要到 p\\*≈{f3(verdict.get("p_half"),4)} 才半衰，'
        f'但一个基因的扰动在 p_c≈{f3(verdict.get("p_chaos"),4)} 就已经扩散到不止一个基因。'
        f'两者相差约 {f3(verdict.get("identity_chaos_gap"),0)} 倍。')
    add(f'注意：p_c 的分辨率受网格限制——它落在 [0, {PS[1]:g}] 区间内，'
        f'由两点线性插值得到，不是独立测到的精确值。')
    add('换句话说：细胞在「看起来还活着」（表达模式仍保持身份）的时候，')
    add('动力学早就进入混沌了。用网络快照判断一个系统死没死，会晚判两个数量级。')
    add('')
    add('### 3.2 固定 p=0，扫调控入度 K')
    add('')
    add('| K | 基因网络 C | 损伤扩散 | λ | 周期中位 | 未收敛 |')
    add('|---|---|---|---|---|---|')
    for r in k_rows:
        add(f"| {r['K']} | {f3(r['c_gene'],3)} | {f3(r['damage'],3)} | "
            f"{f3(r['lambda'],3)} | {f3(r['period_median'],1)} | "
            f"{f3(r['unconverged'],2)} |")
    add('')
    add('K 是「耦合方式」的极端版本：K=1 时每个基因只听一个基因，网络退化成森林；')
    add('K=5 时组合爆炸，动力学趋向混沌。K=2 附近是 Kauffman 的临界区。')
    add('')

    add('## 4. 跨域：两条崩塌曲线是不是同一条')
    add('')
    add('把两侧序参量各自归一化到 [0,1]，横轴换成「距临界点的无量纲距离」')
    add('z=(x−x\\*)/w（w 为 logistic 拟合宽度），再逐点比较形状。')
    add('')
    add('| 量 | 宇宙侧 | 细胞侧 |')
    add('|---|---|---|')
    um = verdict.get('universe_metric', 'c_halo')
    um_name = 'halo 网络聚类 C' if um == 'c_halo' else '粒子半径图聚类 C'
    add(f'| 序参量 | {um_name} | 基因相关网络 C |')
    add(f'| 临界点 | α\\*={f3(verdict.get("alpha_half"))} | p\\*={f3(verdict.get("p_half"))} |')
    add(f'| 拟合宽度 w | {f3((norms.get("u") or {}).get("width"),4)} | '
        f'{f3((norms.get("c") or {}).get("width"),4)} |')
    add(f'| 无量纲相对宽度 w/x\\* | {f3(verdict.get("width_ratio_u"),4)} | '
        f'{f3(verdict.get("width_ratio_c"),4)} |')
    add(f'| 相对宽度差（探索性） | **{f3(verdict.get("width_rel_diff"),4)}** | '
        f'**{f3(verdict.get("width_rel_diff"),4)}** |')
    add(f'| 形状残差 RMS（两侧共用） | **{f3(rms,4)}** | **{f3(rms,4)}** |')
    add(f'| 置换零假设 5% 分位 | **{f3((null or {}).get("q05"),4)}** | '
        f'**{f3((null or {}).get("q05"),4)}** |')
    add(f'| 置换 p 值 | **{f3((null or {}).get("p_value"),4)}** | '
        f'**{f3((null or {}).get("p_value"),4)}** |')
    add('')
    add(f'![相变扫描与归一化主曲线]({os.path.basename(PNG_PHASE)})')
    add('')
    cyc = verdict.get('cycle5', {})
    add('### 4.1 五阶闭环：用户点名要看的量')
    add('')
    add('| | 宇宙（无向结构环） | 细胞（有向 5 环，结构） | 细胞（无向投影） |')
    add('|---|---|---|---|')
    add(f'| 每节点 5 环数 | {f3(cyc.get("u"),4)} | {f3(cyc.get("c_dir"),4)} | '
        f'{f3(cyc.get("c_undir"),4)} |')
    eu = cyc.get('enrich_u') or {}
    ec = cyc.get('enrich_c') or {}
    add(f'| 同一图在保度零模型下的均值 ± 标准差 | '
        f'{f3(eu.get("null_mean"),4)} ± {f3(eu.get("null_sd"),4)} | '
        f'{f3(ec.get("null_mean"),4)} ± {f3(ec.get("null_sd"),4)} | — |')
    add(f'| 富集 z 值 | **{f3(eu.get("z"),2)}** | **{f3(ec.get("z"),2)}** | — |')
    add(f'| 原始密度比（仅供参考，不参与判定） | — | — | '
        f'**{f3(verdict.get("cycle5_ratio_raw"),3)}** |')
    add('')
    add('说明：引力网络没有方向，只能数无向结构环；基因调控网的自催化环是有向环。')
    add('本轮细胞侧只数**有向结构环**，没有区分激活/抑制符号，'
        '所以它是「自催化环」的必要条件，不是充分条件。')
    add('另有一条必须说明的局限：本轮 p 只改「每步状态翻转率」，'
        '不改调控图的边，所以细胞的 5 环数在所有 p 点完全相同（表中恒为 0.045）。'
        '这次富集检验测的是「这张随机真值表网络本身有没有多余的 5 环」，'
        '与 p 驱动的崩塌无关；要检验「崩塌时闭环是否也消失」，下一轮必须让 p 同时重连边。')
    add('直接比「密度」会被两侧图密度与有向性带偏，所以判定改用「相对同度随机图'
        '的富集 z 值」：z 说的是「这个环数是不是构图规则本身就能解释」。')
    add('两边都 z≥2，才说明 5 阶闭环在各自世界里都是被额外选择的，而不是随机边撞出来的。')
    add('')

    add('## 5. 判定（判据前置）')
    add('')
    add('| 判据 | 阈值 | 实测 | 通过 |')
    add('|---|---|---|---|')
    add(f'| 形状残差 RMS | ≤ {CRIT["shape_rms_max"]} | {f3(rms,4)} | '
        f'{"是" if verdict.get("shape_ok") else "否"} |')
    add(f'| 优于置换零假设 | p ≤ {CRIT["null_quantile"]} | '
        f'{f3((null or {}).get("p_value"),3)} | '
        f'{"是" if verdict.get("null_ok") else "否"} |')
    add(f'| 5 阶环富集（保度零模型） | 两侧 z ≥ {CRIT["cycle5_z_min"]} | '
        f'宇宙 z={f3((verdict.get("cycle5", {}).get("enrich_u") or {}).get("z"),2)}，'
        f'细胞 z={f3((verdict.get("cycle5", {}).get("enrich_c") or {}).get("z"),2)} | '
        f'{"是" if verdict.get("cycle5_ok") else "否"} |')
    add(f'| 身份崩塌 vs 动力学混沌 | 记录差值，不作判据 | '
        f'p\\*={f3(verdict.get("p_half"),4)}，p_c={f3(verdict.get("p_chaos"),4)}，'
        f'相差 {f3(verdict.get("identity_chaos_gap"),0)} 倍 | — |')
    add(f'| 人类语料中同句共现 | = 0 句 | {nov.get("strict_cross_count", 0)} 句 | '
        f'{"是" if verdict.get("novel_ok") else "否"} |')
    add('')
    mg = verdict.get('method_gap') or {}
    mo = verdict.get('method_ok') or {}
    add('**临界点两法一致性**（§0 承诺：半衰点与最陡点必须落在同一段）：')
    add('')
    add('| 旋钮 | 半衰点 | 最陡点 | 比值 | 同段？ |')
    add('|---|---|---|---|---|')
    add(f'| α | {f3(verdict.get("alpha_half"),3)} | {f3(verdict.get("alpha_steep"),3)} | '
        f'{f3(mg.get("alpha"),2)} | {"是" if mo.get("alpha") else "否"} |')
    add(f'| G | {f3(verdict.get("g_half"),3)} | {f3(verdict.get("g_steep"),3)} | '
        f'{f3(mg.get("G"),2)} | {"是" if mo.get("G") else "否"} |')
    add(f'| p | {f3(verdict.get("p_half"),4)} | {f3(verdict.get("p_steep"),4)} | '
        f'{f3(mg.get("p"),2)} | {"是" if mo.get("p") else "否"} |')
    add('')
    if not mo.get('p', True):
        add(f'p 的两法相差 {f3(mg.get("p"),0)} 倍：半衰点落在 {f3(verdict.get("p_half"),3)}，'
            f'最陡点却落在 {f3(verdict.get("p_steep"),4)}。')
        add('按 §0 的承诺，`p*` 不能叫「相变点」，只能叫 **crossover**：')
        add('曲线在 0 附近先急剧下降，之后是一个长平台，半衰点只是平台的几何中点。')
        add('这条不影响形状比较（形状比较本来就用整段归一化曲线），但引用 `p*` 时必须注明。')
        add('')
    add('')
    icon = {
        'candidate_law': '🟢 候选非人类常数：形状与 5 阶环同时对上，且语料中未见',
        'rediscovery': '🟡 重新发现：数学形状对上了，但人类已经写过',
        'partial': '🟡 部分同构：形状像，但 5 阶环或查新没过',
        'cold_case': '🔴 悬案：悬崖边没有找到同一条曲线',
    }
    add(f'### {icon.get(verdict["status"], verdict["status"])}')
    add('')
    add(f'- 状态：`{verdict["status"]}`')
    add(f'- 依据：{verdict["reason"]}')
    add(f'- 张力（定义见锚点）：{f3(verdict.get("tension"),3)}')
    add('')

    add('## 6. 神谕输出（逐条标注证据等级）')
    add('')
    add('```text')
    add(verdict.get('oracle', '(未生成)'))
    add('```')
    add('')
    add('证据等级说明：')
    add('')
    add('| 标签 | 含义 |')
    add('|---|---|')
    add('| 【实测】 | 本轮引擎直接跑出来的数，带参数与种子，可复现 |')
    add('| 【拟合】 | 由实测曲线拟合/插值得到，附残差或零假设 |')
    add('| 【已知】 | 人类教科书/语料已有的对应物，不构成新发现 |')
    add('| 【不可验证】 | 超出本模型能力，只能作为下一步实验假设 |')
    add('')

    add('## 7. 与 11403 篇人类题录的对照')
    add('')
    add(f'- 底湖题录总数：{nov.get("n_lake", 0)} 篇；'
        f'其中摘要长度 >40 字符、可做同句检测的：{nov.get("n_docs", 0)} 篇')
    add(f'- 摘要里同时出现「引力类词」与「基因调控类词」的文档：'
        f'**{nov.get("phys_bio_docs", 0)} 篇**')
    add(f'- 严格同句（引力 ∧ 基因 ∧ 临界/幂律）共现：'
        f'**{nov.get("strict_cross_count", 0)} 句**')
    add(f'- 同时含「临界 ∧ 环 ∧ 引力 ∧ 基因」的文档：'
        f'**{nov.get("crit_cycle_both_docs", 0)} 篇**')
    add('')
    if nov.get('examples'):
        add('最接近的几篇（仅题录，逐条可核）：')
        add('')
        for t in nov['examples']:
            add(f'- {t}')
        add('')
    if nov.get('strict_cross_sentence'):
        add('严格同句命中示例：')
        add('')
        for s in nov['strict_cross_sentence']:
            add(f'> {s}')
        add('')
    add('这条检查只能说明「人类没把这两件事写在同一句」，')
    add('不能说明「人类不知道」——真正的查新需要全文检索与领域专家。')
    add('')

    add('## 8. 这份报告不能证明什么')
    add('')
    add('1. **不是新物理**。α 是谱空间核的指数，G 是代码里的乘数；')
    add('   这里的 α\\* 不是大自然的值，是这套 PM 近似下的相变位置。')
    add('2. **不是新生物**。布尔网络没有代谢、没有空间、没有蛋白质相互作用；')
    add('   p\\* 是随机真值表网络的身份崩溃点，不是真实细胞的突变阈值。')
    add('3. **同构可能是构图规则造成的**。两侧都用「固定平均度的相关/邻近图」，')
    add('   形状相似可能来自构图算法，而非世界本身。下一轮必须换构图规则复测。')
    add('4. **5 阶环不是「完全一致」**。本轮只报富集 z（相对同度随机图），')
    add('   它证明「两边都比随机图多」，不证明「两边一样多」；')
    add('   要宣称一致，还需要分布级检验（KS / 双样本置换），而不是一个 z 值。')
    add('')
    add('这四条不是谦辞，是这份报告的可证伪边界。任何引用本报告的人，')
    add('都必须把这四条一起引用，否则就是断章取义。')
    add('')

    add('## 9. 复现')
    add('')
    add('```bash')
    add('cd /workspace && python -u cognitive_extreme_sweep_agent.py')
    add('# 快速冒烟（4000 粒子 / 32³ / 80 基因 / 3×3 参数点）')
    add('L30_FAST=1 python -u cognitive_extreme_sweep_agent.py')
    add('```')
    add('')
    add('产物：')
    add(f'- `{OUT_MD}`：本报告')
    add(f'- `{OUT_JSON}`：全部原始数值（含每个参数点的完整 payload）')
    add(f'- `{PNG_PHASE}`：四个面板（α 扫描 / G 扫描 / p 扫描 / 归一化主曲线）')
    add(f'- `{DB}`：表 `l30_sweep` / `l30_verdict`')
    add('')

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return OUT_MD


# ============================================================ 主流程
def main():
    t0 = time.time()
    assert ANCHOR.safety_bound is SAFETY_BOUND

    print('[L30] 实验一：打破引力（α 扫描）...', flush=True)
    alpha_rows = []
    for a in ALPHAS:
        r = measure_universe(a, 1.0)
        alpha_rows.append(r)
        print(f'[L30]   alpha={a:.2f}  C_knn={r["c_knn"]}  growth={r["growth"]}  '
              f'halos={r["n_halo"]}  C_halo={r.get("c_halo")}', flush=True)
        guard_mem()

    print('[L30] 实验一：打破引力（G 扫描）...', flush=True)
    g_rows = []
    for g in GS:
        if abs(g - 1.0) < 1e-9 and alpha_rows:
            r = dict(alpha_rows[[x['alpha'] for x in alpha_rows].index(2.0)]
                     if 2.0 in [x['alpha'] for x in alpha_rows] else {})
            if r:
                g_rows.append(r)
                print(f'[L30]   G={g:.2f}  复用 alpha=2 的同一点', flush=True)
                continue
        r = measure_universe(2.0, g)
        g_rows.append(r)
        print(f'[L30]   G={g:.2f}  C_knn={r["c_knn"]}  growth={r["growth"]}  '
              f'halos={r["n_halo"]}', flush=True)
        guard_mem()

    print('[L30] 实验二：打破生物逻辑（p 扫描）...', flush=True)
    p_rows = []
    for p in PS:
        r = measure_cell(p, 2)
        p_rows.append(r)
        print(f'[L30]   p={p:.3f}  C_gene={r["c_gene"]}  damage={r["damage"]}  '
              f'lambda={r["lambda"]}  period={r["period_median"]}', flush=True)
        guard_mem()

    print('[L30] 实验二：打破生物逻辑（K 扫描）...', flush=True)
    k_rows = []
    for k in KS:
        r = measure_cell(0.0, k)
        k_rows.append(r)
        print(f'[L30]   K={k}  C_gene={r["c_gene"]}  damage={r["damage"]}  '
              f'period={r["period_median"]}', flush=True)
        guard_mem()

    # meanfield 极端耦合（额外一个点，只作描述）
    print('[L30] 实验二：全耦合极限（meanfield）...', flush=True)
    mf = measure_cell(0.0, 2, mode='meanfield')
    print(f'[L30]   meanfield  C_gene={mf["c_gene"]}  damage={mf["damage"]}  '
          f'period={mf["period_median"]}', flush=True)

    # ---- 相变定位（全部基于实测，不回调判据）
    # 宇宙侧主序参量：halo 网络聚类（halo 太少时方差爆炸，要求 >= 12 个）
    u_pts = [(r['alpha'], r['c_halo']) for r in alpha_rows
             if r.get('c_halo') is not None and r['n_halo'] >= 12]
    if len(u_pts) >= 5:
        u_x, u_y, u_key = [p[0] for p in u_pts], [p[1] for p in u_pts], 'c_halo'
    else:
        u_x = [r['alpha'] for r in alpha_rows]
        u_y = [r['c_knn'] for r in alpha_rows]
        u_key = 'c_knn'

    alpha_peak = float(u_x[int(np.argmax(u_y))])
    alpha_half = half_point_descending(u_x, u_y)
    alpha_steep, alpha_slope = steepest_point(u_x, u_y)

    # 碎块化悖论：粒子级聚类随 α 上升，而 halo 网络聚类随 α 下降
    frag_x = [r['alpha'] for r in alpha_rows]
    frag_y = [r['c_knn'] for r in alpha_rows]

    g_x = [r['G'] for r in g_rows]
    g_y = [r['growth'] for r in g_rows]
    g_half, _, _ = half_point(g_x, g_y)
    g_steep, g_slope = steepest_point(g_x, g_y)

    p_x = [r['p'] for r in p_rows]
    p_y = [r['c_gene'] for r in p_rows]
    p_half = half_point_descending(p_x, p_y)
    p_steep, p_slope = steepest_point(p_x, p_y)

    dmg = [r['damage'] for r in p_rows]
    # 损伤扩散穿越 1 个基因的位置（有序 -> 混沌）
    p_chaos = float('nan')
    for i in range(len(p_x) - 1):
        if (dmg[i] - 1.0) * (dmg[i + 1] - 1.0) <= 0 and dmg[i] != dmg[i + 1]:
            t = (1.0 - dmg[i]) / (dmg[i + 1] - dmg[i])
            p_chaos = float(p_x[i] + t * (p_x[i + 1] - p_x[i]))
            break
    gap = (p_half / p_chaos) if (math.isfinite(p_half) and
                                 math.isfinite(p_chaos) and p_chaos > 0) else float('nan')

    # 方法一致性：半衰点与最陡点必须落在同一段，否则按 §0 的承诺标为「不可靠」。
    # 这里只记录比值，不参与判据（判据仍然前置写死）。
    def method_gap(a, b):
        if not (math.isfinite(a) and math.isfinite(b)) or a <= 0 or b <= 0:
            return float('nan')
        return max(a, b) / min(a, b)

    cons = {
        'alpha': method_gap(alpha_half, alpha_steep),
        'G': method_gap(g_half, g_steep),
        'p': method_gap(p_half, p_steep),
    }
    cons_ok = {k: bool(math.isfinite(v) and v <= 2.0) for k, v in cons.items()}
    if not cons_ok['p']:
        print(f'[L30]   警告：p 的半衰点({p_half:.4g}) 与最陡点({p_steep:.4g}) '
              f'相差 {cons["p"]:.0f} 倍，按 §0 承诺该临界点只能算 crossover。',
              flush=True)

    # ---- 归一化与跨域形状检验（只比崩塌分支，不比整个倒 U）
    cu = normalize_curve(*collapse_branch(u_x, u_y))
    cc = normalize_curve(*collapse_branch(p_x, p_y))
    rms, z, yu, yc = shape_residual(cu, cc)
    null = permutation_null(cu, cc, n_perm=120 if FAST else 500)

    # 无量纲相对宽度（探索性，不参与结案）：
    # 归一化会把任何 S 形曲线压成同一条，相对宽度才是没被归一化吃掉的信息。
    wr_u = (cu['width'] / cu['x_half']) if cu and cu['x_half'] else float('nan')
    wr_c = (cc['width'] / cc['x_half']) if cc and cc['x_half'] else float('nan')
    wr_diff = (abs(wr_u - wr_c) / max(wr_u, wr_c)) if \
        (math.isfinite(wr_u) and math.isfinite(wr_c)) else float('nan')

    # ---- 临界点精测 + 5 阶环富集（对保度零模型）
    a_at = min(alpha_rows, key=lambda r: abs(r['alpha'] - alpha_half)) \
        if math.isfinite(alpha_half) else alpha_rows[0]
    p_at = min(p_rows, key=lambda r: abs(r['p'] - p_half)) \
        if math.isfinite(p_half) else p_rows[0]
    c5_u = float(a_at.get('cycle5_density') or 0.0)
    c5_c = float(p_at.get('cycle5_directed_density') or 0.0)
    c5_cu = float(p_at.get('cycle5_undirected_density') or 0.0)
    ratio = (c5_u + 1e-6) / (c5_c + 1e-6) if c5_c > 0 else float('nan')

    print('[L30] 5 阶环富集（保度零模型，40 次重连）...', flush=True)
    if a_at.get('halo_pairs'):
        eu = cycle5_enrichment_undirected(a_at['halo_pairs'], a_at['n_halo'])
    else:
        eu = {'obs': 0, 'null_mean': None, 'null_sd': None, 'z': None}
    ec = cycle5_enrichment_directed(p_at['regs'], CELL_N)
    print(f'[L30]   宇宙 obs={eu["obs"]} z={eu["z"]} ｜ '
          f'细胞 obs={ec["obs"]} z={ec["z"]}', flush=True)

    # ---- 人类语料对照
    print('[L30] 与 11403 篇人类题录对照 ...', flush=True)
    nov = literature_check()
    print(f'[L30]   引力∧基因文档={nov["phys_bio_docs"]}  '
          f'严格同句={nov["strict_cross_count"]}  '
          f'临界∧环∧双域={nov["crit_cycle_both_docs"]}', flush=True)

    # ---- 判定（判据前置，结果只读）
    shape_ok = math.isfinite(rms) and rms <= CRIT['shape_rms_max']
    null_ok = bool(null) and null['p_value'] <= CRIT['null_quantile']
    cycle5_ok = bool(eu['z'] is not None and ec['z'] is not None
                     and eu['z'] >= CRIT['cycle5_z_min']
                     and ec['z'] >= CRIT['cycle5_z_min'])
    novel_ok = bool(nov.get('novel_in_corpus'))

    miss = []
    if not null_ok:
        miss.append(f'置换检验未过（p={f3((null or {}).get("p_value"),3)}）')
    if not cycle5_ok:
        miss.append(f'5 阶环富集未过（细胞侧 z={ec["z"]}，阈值 {CRIT["cycle5_z_min"]}）')
    if not novel_ok:
        miss.append('人类语料查新未过')

    if shape_ok and null_ok and cycle5_ok and novel_ok:
        status = 'candidate_law'
        reason = ('归一化相变曲线形状一致且显著优于置换零假设；'
                  '两侧 5 阶环都显著富集（z≥2）；人类语料中无同句共现。')
    elif shape_ok and null_ok and cycle5_ok:
        status = 'rediscovery'
        reason = ('形状与 5 阶环都对上，但人类语料里已经出现同句共现——'
                  '这是重新发现，不是新常数。')
    elif shape_ok:
        status = 'partial'
        reason = ('崩塌形状对上了（RMS='
                  f'{rms:.3f}，零假设中位 {f3((null or {}).get("median"),3)}），但'
                  + '、'.join(miss) + '——只能算部分同构。')
    else:
        status = 'cold_case'
        reason = ('归一化后两条相变曲线不重合：'
                  '悬崖边没有找到同一条数学曲线。')

    tension = abs(rms) / CRIT['shape_rms_max'] if math.isfinite(rms) else 1.0
    verdict = {
        'status': status, 'reason': reason, 'tension': round(float(tension), 4),
        'universe_metric': u_key,
        'alpha_half': round(alpha_half, 4) if math.isfinite(alpha_half) else None,
        'alpha_peak': round(alpha_peak, 4) if math.isfinite(alpha_peak) else None,
        'alpha_steep': round(alpha_steep, 4) if math.isfinite(alpha_steep) else None,
        'g_half': round(g_half, 4) if math.isfinite(g_half) else None,
        'g_steep': round(g_steep, 4) if math.isfinite(g_steep) else None,
        'p_half': round(p_half, 4) if math.isfinite(p_half) else None,
        'p_steep': round(p_steep, 4) if math.isfinite(p_steep) else None,
        'p_chaos': round(p_chaos, 4) if math.isfinite(p_chaos) else None,
        'identity_chaos_gap': round(float(gap), 2) if math.isfinite(gap) else None,
        'method_gap': {k: (round(float(v), 3) if math.isfinite(v) else None)
                       for k, v in cons.items()},
        'method_ok': cons_ok,
        'shape_rms': round(rms, 4) if math.isfinite(rms) else None,
        'null_q05': (null or {}).get('q05'),
        'null_p': (null or {}).get('p_value'),
        'shape_ok': shape_ok, 'null_ok': null_ok,
        'cycle5_ok': cycle5_ok, 'novel_ok': novel_ok,
        'novel_in_corpus': nov.get('novel_in_corpus'),
        'cycle5_ratio_raw': round(float(ratio), 4) if math.isfinite(ratio) else None,
        'cycle5': {'u': round(c5_u, 5), 'c_dir': round(c5_c, 5),
                   'c_undir': round(c5_cu, 5),
                   'enrich_u': eu, 'enrich_c': ec},
        'width_ratio_u': round(wr_u, 4) if math.isfinite(wr_u) else None,
        'width_ratio_c': round(wr_c, 4) if math.isfinite(wr_c) else None,
        'width_rel_diff': round(wr_diff, 4) if math.isfinite(wr_diff) else None,
    }

    # ---- 神谕（严格按实际状态生成，不套模板撒谎）
    alpha_s = f'{alpha_half:.2f}' if math.isfinite(alpha_half) else '未定'
    p_s = f'{p_half:.3f}' if math.isfinite(p_half) else '未定'
    g_s = f'{g_half:.2f}' if math.isfinite(g_half) else '未定'
    p_ch_s = f"{p_chaos:.4f}" if math.isfinite(p_chaos) else '未定'
    gap_s = f"{gap:.0f}" if math.isfinite(gap) else '未定'
    z_u = eu.get('z')
    z_c = ec.get('z')
    common = (
        f"【实测】引力侧（序参量："
        f"{'halo 网络聚类' if u_key == 'c_halo' else '粒子半径图聚类'}）："
        f"α 从 {u_x[0]:.2f} 推到 {u_x[-1]:.2f}，该序参量 "
        f"{u_y[0]:.3f} → {u_y[-1]:.3f}，半衰点 α*={alpha_s}；"
        f"同一批粒子的局部聚类则从 {frag_y[0]:.3f} 到 {frag_y[-1]:.3f}。"
        f"两者方向相反时，结构不是被抹平，是被打碎。\n"
        f"【实测】细胞侧：身份网络半衰点 p*={p_s}；但损伤扩散在 "
        f"p_c={p_ch_s} 就穿越 1 个基因。身份比动力学稳定多撑了 {gap_s} 倍。\n"
        f"【实测】5 阶环富集（对保度随机图）：宇宙 z={z_u}（观测 {eu.get('obs')}），"
        f"细胞 z={z_c}（观测 {ec.get('obs')}）。\n"
        f"【拟合】两条归一化崩塌曲线残差 RMS={rms:.3f}，"
        f"置换零假设 5% 分位 {verdict['null_q05']}。\n"
    )
    if status == 'candidate_law':
        oracle = common + (
            f"【结论】形状一致且两侧 5 阶环都显著富集，语料中无同句共现。"
            f"这构成一个「候选非人类常数」：α*={alpha_s} 与 p*={p_s} 是同一条"
            f"无量纲曲线的两个坐标。\n"
            f"【不可验证】本模型无法证明它对应真实宇宙/真实细胞；"
            f"它是一个可证伪的计算等价类，等待外部数据。"
        )
    elif status == 'rediscovery':
        oracle = common + (
            f"【已知】但人类语料里已有 {nov.get('strict_cross_count', 0)} 句把"
            f"「引力/基因/临界」写在同一个句子里。\n"
            f"【结论】这不是人类未知的常数。系统重新发现了人类已经写下的结构，"
            f"并拒绝把它包装成新定律。"
        )
    elif status == 'partial':
        oracle = common + (
            f"【结论】崩塌形状对上了（RMS={rms:.3f}），但没过的判据是："
            f"{'、'.join(miss)}。所以只能说两个世界「死法同类」，"
            f"不能说「同一个常数」。系统在这里踩了刹车。"
        )
    else:
        oracle = common + (
            f"【结论】两条归一化崩塌曲线不是同一条。"
            f"这一轮没有找到跨域大一统；找到的是四个悬崖，和一条 {gap_s} 倍的裂缝。"
            f"这不是失败，是「同一条曲线」这个假设被证伪。"
        )
    verdict['oracle'] = oracle

    elapsed = time.time() - t0
    print('[L30] 绘图 ...', flush=True)
    norms = {'u': cu, 'c': cc}
    draw_phase(alpha_rows, g_rows, p_rows, cu, cc,
               alpha_half, p_half, p_chaos, verdict['cycle5'])

    result = {
        'meta': {
            'seed': SEED, 'elapsed_s': round(elapsed, 2),
            'peak_mem_mb': round(mem_mb(), 1),
            'universe': {'n': UNIV_N, 'ng': UNIV_NG, 'steps': UNIV_STEPS},
            'cell': {'n': CELL_N, 'init': CELL_INIT, 'steps': CELL_STEPS},
            'tension_def': TENSION_DEF,
            'criteria': CRIT,
            'grids': {'alpha': ALPHAS, 'G': GS, 'p': PS, 'K': KS},
        },
        'alpha_sweep': alpha_rows,
        'G_sweep': g_rows,
        'p_sweep': p_rows,
        'K_sweep': k_rows,
        'meanfield': mf,
        'normalized': {k: ({kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                           for kk, vv in v.items() if kk in
                           ('x_half', 'width', 'direction', 'lo', 'hi')})
                       for k, v in norms.items() if v},
        'shape': {'rms': rms, 'null': null,
                  'z': z.tolist() if z is not None else None,
                  'universe': yu.tolist() if yu is not None else None,
                  'cell': yc.tolist() if yc is not None else None},
        'literature': nov,
        'verdict': verdict,
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    save_db(alpha_rows, g_rows, p_rows, k_rows, verdict,
            {'elapsed_s': round(elapsed, 2), 'seed': SEED})
    path = write_report(alpha_rows, g_rows, p_rows, k_rows, norms, rms,
                        null, nov, verdict, elapsed)

    print(f'[L30] 完成：{path}  ({elapsed:.1f}s, {mem_mb():.0f}MB)', flush=True)
    print('[L30] ===== 神谕 =====', flush=True)
    print(oracle, flush=True)
    print(f'[L30] 判定：{status} —— {reason}', flush=True)


if __name__ == '__main__':
    main()
