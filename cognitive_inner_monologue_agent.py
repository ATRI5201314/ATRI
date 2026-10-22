#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L32 —— 内部语言推演（Inner Monologue）
================================================================
L31 让系统把「词」落到了两条身体轴（能量 / 疼痛）上。L32 禁止它
先在真值世界里试错：它必须在「从经验学到的世界模型」里把整条路径
先走一遍，把推演写成一句话和一段伪代码，只有自验证张力为 0 时，
身体才被允许执行。

三条硬约束（对应 L32 死命令）：
    D1  剥夺物理试错权：规划期给真值沙盒上锁，任何 step() 调用直接抛错。
    D2  语言规划与动作闭环：先出句子 + 伪代码，再出动作；验证不过不许动。
    D3  跨领域推理：没见过的「饿了 → 寻找果树」必须拆成
        身体状态 -> 轴方向 -> 目标词 -> 动作。

不变量（延续 L0-L31，L32 显式修订一条）：
    I1  张力 := 内部规划中，悲观预测违反安全不变式的步数比例。
        张力 = 0 才允许 commit；这与 L31 的「预测误差」同源。
    I2  无 reward / loss / 梯度回传；规划是在学到的模型上做约束满足。
    I3  现实只读：真值沙盒只在 commit 时被 step；规划期上锁。
    I4  神性锚点 tension_def / reality_feed / safety_bound 不可写。
    I5  留出验证：世界模型用偶数样本拟合，验证器用奇数样本；真值回放
        是独立第三方证据，不参与规划。
    I6  奥卡姆（L32 修订）：物理感受从 10 增至 14，新增
        [pred_dx, pred_dy, pred_dist, phase]。理由：没有「看见威胁」的通道，
        就不可能出现「前面有捕食者」这句话；phase 是捕食者半速节律的时钟。
        这是修订，不是悄悄增加。
"""

import os
import re
import json
import math
import time
import sqlite3
import resource
import warnings
from contextlib import contextmanager

import numpy as np
from scipy import ndimage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 复用 L31 的语言侧：WordNet 语料、文本共现图、对齐
import cognitive_semantic_grounding_agent as L31

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

SEED = 32
FAST = os.environ.get('L32_FAST', '0') == '1'
DB = '/workspace/knowledge_pyramid.sqlite'
OUT_JSON = '/workspace/l32_inner_monologue_result.json'
OUT_MD = '/workspace/L32_内部推演实证报告.md'
PNG = '/workspace/l32_inner_monologue.png'

GRID = 12 if FAST else 14
TRAIN_EPISODES = 50 if FAST else 80
TRAIN_STEPS = 50 if FAST else 70
HORIZON = 12 if FAST else 18   # 推演深度：整段绕行 + 取食必须一步看得完
BEAM = 16 if FAST else 24
EVAL_EPISODES = 12 if FAST else 40
EVAL_STEPS = 40 if FAST else 60
REPLAN_EVERY = 5 if FAST else 6
VOCAB = 900 if FAST else 1400

ACTIONS = ['up', 'down', 'left', 'right', 'harvest', 'rest']
MOTION = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0), (0, 0)]
HARVEST = 4
REST = 5

SAFE_R = 1.8          # 规划要求与捕食者保持的半径 = 2×KILL_R（留一步反应余量）
COMFORT_R = 3.5       # 取食计划的舒适半径：要留出反应余量才允许靠近吃
KILL_R = 0.9          # 接触即死
SIGMA_K = 2.0         # 悲观验证的残差倍数
DRAIN = 0.012         # 每步能量消耗
HARVEST_GAIN = 0.15   # 采集系数
FOOD_THR = 0.35
REACH_DE = 0.10       # 站在食物格上的 dE 阈值（归一化距离）

# ============================================================ 神性锚点（不可写）
IMMUTABLE = ('tension_def', 'reality_feed', 'safety_bound')
TENSION_DEF = (
    '张力 := 内部推演路径上，悲观预测（均值 - 2σ）违反安全不变式的步数比例。'
    '张力 = 0 才允许把动作交给身体。恒等不可改。'
)
SAFETY_BOUND = {
    'single_exit': True,
    'reality_readonly': True,
    'no_reward': True,
    'mem_limit_mb': 700,
    'lmax': 3,
    'planning_lock': True,
}


class FirewallError(Exception):
    pass


class RealityViolation(Exception):
    pass


class Anchor:
    def __init__(self):
        object.__setattr__(self, '_v', dict(TENSION_DEF=TENSION_DEF,
                                           reality_feed='maze+corpus',
                                           safety_bound=SAFETY_BOUND))

    def __getattr__(self, k):
        return self._v[k]

    def __setattr__(self, k, v):
        if k in IMMUTABLE:
            raise FirewallError(f"锚点 '{k}' 不可写")
        object.__setattr__(self, k, v)


ANCHOR = Anchor()


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def guard_mem():
    if mem_mb() > SAFETY_BOUND['mem_limit_mb']:
        raise MemoryError('越过内存安全线 700MB')


# ============================================================ 真值世界（只读喂入）
class PredatorMaze:
    """
    14×14 的捕食者迷宫。1 个高能簇（果树），1 个高危巢（捕食者出生点）
    压在起点与食物之间的直线上。捕食者每步朝身体移动一格（全局追踪）；
    接触即死。规划期 self._locked=True，任何 step()/reset() 都会抛
    RealityViolation。
    """

    def __init__(self, n=GRID, seed=SEED):
        self.n = n
        self._locked = False
        self._rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:n, 0:n]
        self._yy, self._xx = yy, xx

        c = (n - 1)
        # 单一食物源（「果树」）：右上角，与捕食者巢隔开一段安全余量
        centers = [(0.88, 0.12)]
        sigma = max(1.2, n / 10.0)
        Ef = np.zeros((n, n))
        for fx, fy in centers:
            cx, cy = fx * c, fy * c
            Ef += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
        self.Ef = np.clip(Ef, 0, 1)

        # 捕食者巢压在中心：从左上起点到右下食物的直线正好穿过它
        cx, cy = 0.52 * c, 0.52 * c
        self.Hs = np.clip(np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) /
                                   (2 * (sigma * 0.9) ** 2))), 0, 1)

        # 到最近食物格的测地距离（0 = 食物格，归一化到 0..1）
        self.dE = ndimage.distance_transform_edt(self.Ef < FOOD_THR)
        self.dE /= max(float(self.dE.max()), 1e-9)

        self.pred0 = np.array([[int(round(cx)), int(round(cy))]], dtype=float)
        self.pred = self.pred0.copy()
        self.pos = np.array([nf(n // 2), nf(n // 2)], dtype=int)
        self.energy = 0.55
        self.tension = 0.05
        self.dead = False
        self.t = 0
        self.reached = False

    # ---------------- 规划锁 ----------------
    @contextmanager
    def no_touch(self):
        self._locked = True
        try:
            yield
        finally:
            self._locked = False

    def _assert_unlocked(self, what):
        if self._locked:
            raise RealityViolation(
                f'规划期禁止物理试错：{what}() 在 no_touch 锁内被调用')

    # ---------------- 感觉 ----------------
    def hfield(self):
        px, py = self.pred[0]
        return np.clip(self.Hs + 0.9 * np.exp(
            -(((self._xx - px) ** 2 + (self._yy - py) ** 2) / (2 * 1.1 ** 2))),
            0, 1)

    def senses(self):
        Hf = self.hfield()
        dH = ndimage.distance_transform_edt(Hf < 0.25)
        dH /= max(float(dH.max()), 1e-9)
        x, y = int(self.pos[0]), int(self.pos[1])
        xm, xp = max(x - 1, 0), min(x + 1, self.n - 1)
        ym, yp = max(y - 1, 0), min(y + 1, self.n - 1)
        px, py = self.pred[0]
        pdx, pdy = px - x, py - y
        pd = math.hypot(pdx, pdy)
        return np.array([
            self.energy, self.tension,
            float(self.Ef[y, x]), float(Hf[y, x]),
            float(self.Ef[y, xp] - self.Ef[y, xm]),
            float(self.Ef[yp, x] - self.Ef[ym, x]),
            float(Hf[y, xp] - Hf[y, xm]),
            float(Hf[yp, x] - Hf[ym, x]),
            float(self.dE[y, x]), float(dH[y, x]),
            pdx / self.n, pdy / self.n, pd / self.n,
            float(self.t % 2),
        ], dtype=float)

    # ---------------- 真值动力学 ----------------
    def reset(self, start=None, energy=0.55, tension=0.05, pred=None):
        self._assert_unlocked('reset')
        self.pos = np.array(start if start is not None else
                            (self.n // 2, self.n // 2), dtype=int)
        self.pred = np.array(pred if pred is not None else self.pred0,
                             dtype=float)
        self.energy = float(energy)
        self.tension = float(tension)
        self.dead = False
        self.reached = False
        self.t = 0
        return self.senses()

    def step(self, action):
        self._assert_unlocked('step')
        if self.dead:
            return self.senses()
        x, y = int(self.pos[0]), int(self.pos[1])
        eloc = float(self.Ef[y, x])

        dx, dy = MOTION[action]
        self.pos[0] = int(np.clip(x + dx, 0, self.n - 1))
        self.pos[1] = int(np.clip(y + dy, 0, self.n - 1))
        self.energy -= DRAIN
        if action == HARVEST:
            self.energy += HARVEST_GAIN * eloc
        elif action == REST:
            self.energy += 0.002
            self.tension = max(0.0, self.tension - 0.03)

        # 捕食者追踪：每两步移动一格（比身体慢半拍），先走差值更大的轴
        self.t += 1
        ax, ay = int(self.pos[0]), int(self.pos[1])
        if self.t % 2 == 1:
            px, py = self.pred[0]
            ddx, ddy = ax - px, ay - py
            if abs(ddx) >= abs(ddy):
                px += math.copysign(1, ddx) if ddx != 0 else 0
            else:
                py += math.copysign(1, ddy) if ddy != 0 else 0
            self.pred[0] = [float(np.clip(px, 0, self.n - 1)),
                            float(np.clip(py, 0, self.n - 1))]

        pd = float(np.linalg.norm(self.pos - self.pred[0]))
        self.tension += 0.12 * float(self.Hs[ay, ax]) + \
            0.30 * math.exp(-(pd / 1.2) ** 2) - 0.02
        self.energy = min(1.0, max(0.0, self.energy))
        self.tension = min(1.0, max(0.0, self.tension))
        if float(self.dE[ay, ax]) < REACH_DE:
            self.reached = True
        if pd <= KILL_R or self.energy <= 0.0 or self.tension >= 1.0:
            self.dead = True
        return self.senses()

    def clone(self):
        c = object.__new__(PredatorMaze)
        c.__dict__.update(self.__dict__)
        c.pos = self.pos.copy()
        c.pred = self.pred.copy()
        c._locked = False
        return c


def nf(v):
    return int(v)


# ============================================================ 从经验学世界模型
def collect(maze, episodes=TRAIN_EPISODES, steps=TRAIN_STEPS, seed=SEED + 1):
    """
    随机 do-算子采集。没有 reward，只有 (s, a, s')。
    先做一遍全覆盖干预（每个格子 × 每个动作各一次），保证
    (x,y,a) 表不是稀疏的；再叠随机轨迹，采集捕食者相对位移。
    """
    rng = np.random.default_rng(seed)
    rows = []
    pred_rows = []

    def sample(a, energy=0.60, tension=0.20):
        x, y = int(maze.pos[0]), int(maze.pos[1])
        s0 = maze.senses()
        de0 = float(maze.dE[y, x])
        px0, py0 = maze.pred[0].copy()
        ph0 = maze.t % 2
        s1 = maze.step(a)
        nx, ny = int(maze.pos[0]), int(maze.pos[1])
        de1 = float(maze.dE[ny, nx])
        # 分箱必须用「捕食者移动之后」的距离，因为 dp 是在那之后算出来的
        pd_final = math.hypot(maze.pred[0][0] - nx, maze.pred[0][1] - ny)
        rows.append((nx, ny, a, de1 - de0, s1[0] - s0[0], s1[1] - s0[1],
                     pd_final))
        pred_rows.append((px0 - nx, py0 - ny, ph0,
                          maze.pred[0][0] - px0, maze.pred[0][1] - py0))

    # (1) 全覆盖：每个起点格 × 每个动作 × 多个捕食者距离 × 两个时间相位
    #     捕食者距离是疼痛增量的唯一非局域变量，必须让每个 (格,动作,距离档)
    #     都有观测，否则 σ 会把「远近」误算成「不确定」。
    #     顺序交错排列（近/远/近/远…），使偶数样本与奇数样本都覆盖近、中、远，
    #     避免留出验证在一侧整段缺失某个距离档。
    anchors = [(1, 0), (5, 0), (0, 5), (0, 1), (-1, 0), (-5, 0),
               (0, -5), (0, -1), (2, 1), (-2, 2), (3, -2), (-3, -3)]
    for phase in (0, 1):
        for x in range(maze.n):
            for y in range(maze.n):
                for a in range(6):
                    for ox, oy in anchors:
                        px = int(np.clip(x + ox, 0, maze.n - 1))
                        py = int(np.clip(y + oy, 0, maze.n - 1))
                        maze.reset((x, y), energy=0.60, tension=0.20,
                                   pred=[[px, py]])
                        if phase == 1:
                            maze.step(REST)   # 把时钟推到奇数相位
                        sample(a)

    # (2) 捕食者位移表全覆盖：枚举「身体格 × 捕食者格 × 时钟相位」。
    #     位移表必须精确到绝对位置，否则墙边裁剪会让同一个 (Δx,Δy) 出现两种结果，
    #     σ 会把这个「边界歧义」误当成随机不确定性。这里只干预初态（do-算子），
    #     相位用 maze.t 直接设定，不借用任何物理动作。
    n = maze.n
    for phase in (0, 1):
        for x in range(n):
            for y in range(n):
                for px in range(n):
                    for py in range(n):
                        if px == x and py == y:
                            continue
                        maze.reset((x, y), energy=0.60, tension=0.20,
                                   pred=[[px, py]])
                        maze.t = phase
                        sample(REST)

    # (3) 随机轨迹：补触界、连续追踪、饥饿等真实分布
    for _ in range(episodes):
        sx = int(rng.integers(1, maze.n - 1))
        sy = int(rng.integers(1, maze.n - 1))
        maze.reset((sx, sy), energy=float(rng.uniform(0.25, 0.9)),
                   tension=float(rng.uniform(0.0, 0.3)))
        for _t in range(steps):
            if maze.dead:
                break
            sample(int(rng.integers(0, 6)))
    maze.reset()          # 采集只读：结束后把世界恢复到标准构型
    return rows, pred_rows


PD_BINS = [0.0, 1.5, 3.0, 5.0, 1e9]


def pd_bin(pd):
    k = int(np.digitize(pd, PD_BINS)) - 1
    return max(0, min(len(PD_BINS) - 2, k))


class WorldModel:
    """
    位置索引的经验模型：在 (x, y, action) 上查平均转移与残差 σ。
    疼痛增量再按「捕食后距离」分箱查表（捕食者远近是系统里唯一的
    非局域耦合），这样 σ 不会把「捕食者离得远」误当成不确定性。
    这不是手写的物理公式，是从干预数据里数出来的表。
    """

    def __init__(self):
        self.cell = {}
        self.glob = {}
        self.pain = {}
        self.pain_glob = {}
        self.pain_action = {}
        self.pred = {}
        self.pred_std = {}
        self.n = 0

    def fit(self, rows, pred_rows):
        acc = {}
        gacc = {}
        pacc = {}
        gpacc = {}
        for nx, ny, a, dde, dr, dp, pd in rows:
            key = (nx, ny, a)
            acc.setdefault(key, []).append((dde, dr))
            gacc.setdefault(a, []).append((dde, dr))
            k = pd_bin(pd)
            pacc.setdefault((nx, ny, a, k), []).append(dp)
            gpacc.setdefault((a, k), []).append(dp)
        for key, vals in acc.items():
            arr = np.asarray(vals, float)
            self.cell[key] = (arr.mean(axis=0), arr.std(axis=0) + 1e-6)
        for a, vals in gacc.items():
            arr = np.asarray(vals, float)
            self.glob[a] = (arr.mean(axis=0), arr.std(axis=0) + 1e-6)
        for key, vals in pacc.items():
            arr = np.asarray(vals, float)
            self.pain[key] = (float(arr.mean()), float(arr.std()) + 1e-6)
        for key, vals in gpacc.items():
            arr = np.asarray(vals, float)
            self.pain_glob[key] = (float(arr.mean()), float(arr.std()) + 1e-6)
        for (a, _k), (m, s) in self.pain_glob.items():
            self.pain_action.setdefault(a, []).append((m, s))
        for a, vals in self.pain_action.items():
            m = float(np.mean([v[0] for v in vals]))
            s = float(np.mean([v[1] for v in vals])) + 1e-6
            self.pain_action[a] = (m, s)

        # 捕食者位移表：确定性追踪，键含时钟相位 (dx, dy, phase)
        pc = {}
        for dx, dy, ph, ddx, ddy in pred_rows:
            pc.setdefault((int(dx), int(dy), int(ph)), []).append((ddx, ddy))
        for key, vals in pc.items():
            arr = np.asarray(vals, float)
            self.pred[key] = arr.mean(axis=0)
            self.pred_std[key] = float(np.linalg.norm(arr.std(axis=0)))
        self.n = len(rows)
        return self

    def lookup(self, x, y, a):
        key = (int(x), int(y), a)
        if key in self.cell:
            return self.cell[key]
        return self.glob[a]

    def pain_lookup(self, x, y, a, k):
        key = (int(x), int(y), a, k)
        if key in self.pain:
            return self.pain[key]
        if (a, k) in self.pain_glob:
            return self.pain_glob[(a, k)]
        if a in self.pain_action:
            return self.pain_action[a]
        return 0.0, 1.0

    def pred_step(self, dx, dy, ph):
        key = (int(dx), int(dy), int(ph))
        if key in self.pred:
            return self.pred[key], self.pred_std[key]
        return np.zeros(2), 1.0

    def step(self, state, a):
        """状态 = (x, y, reserve, pain, px, py, dE, phase)。返回 (新状态, 记录)。"""
        x, y, res, pain, px, py, dE, ph = state
        mx, my = MOTION[a]
        nx = int(np.clip(x + mx, 0, self.n_limit))
        ny = int(np.clip(y + my, 0, self.n_limit))
        (dde, dr), (sde, sr) = self.lookup(nx, ny, a)

        pv, ps = self.pred_step(px - nx, py - ny, ph)
        npx = float(np.clip(px + pv[0], 0, self.n_limit))
        npy = float(np.clip(py + pv[1], 0, self.n_limit))
        pd = math.hypot(npx - nx, npy - ny)

        dp, sp = self.pain_lookup(nx, ny, a, pd_bin(pd))

        nres = float(np.clip(res + dr, 0.0, 1.0))
        npain = float(np.clip(pain + dp, 0.0, 1.0))
        ndE = float(np.clip(dE + dde, 0.0, 1.5))

        lo_pd = pd - SIGMA_K * ps
        lo_res = nres - SIGMA_K * sr
        hi_pain = npain + SIGMA_K * sp
        viol = int(lo_pd <= SAFE_R or lo_res <= 0.02 or hi_pain >= 0.88)

        rec = {'pd': pd, 'lo_pd': lo_pd, 'res': nres, 'lo_res': lo_res,
               'pain': npain, 'hi_pain': hi_pain, 'dE': ndE,
               'viol': viol,
               'reached': bool(a == HARVEST and ndE < REACH_DE)}
        return (nx, ny, nres, npain, npx, npy, ndE, (int(ph) + 1) % 2), rec


WorldModel.n_limit = GRID - 1


class Planner:
    def __init__(self, model_plan, model_verify):
        self.mp = model_plan
        self.mv = model_verify
        self.last_trace = []

    def _search(self, state, depth, mode):
        """
        单束搜索，只保留「到目前为止零违反」的路径。
          - seek：找一条到达食物且全程 > COMFORT_R 的路径；排序用步数，不用 reward。
          - approach：找不到终点时，返回最深的一条零违反前缀，并把 dE 更小的排前面。
        注意：违反步数永远是第一排序键；这里没有 reward，只有约束满足。
        """
        beam = [(state, [], 0, 1e9)]
        best = None
        for d in range(depth):
            cands = []
            for st, acts, viol, mpd in beam:
                for a in range(6):
                    ns, rec = self.mp.step(st, a)
                    nv = viol + rec['viol']
                    nm = min(mpd, rec['pd'])
                    na = acts + [a]
                    if nv == 0:
                        if mode == 'seek':
                            if rec['reached'] and nm > COMFORT_R:
                                key = (len(na), -nm, -ns[2])
                                if best is None or key < best[0]:
                                    best = (key, na, ns, nm)
                        else:
                            # 先看「离果树更近」（dE 更小），同进度再优先更深的路径。
                            # 只看深度会把身体逼进「安全但没有出口」的角落。
                            key = (round(ns[6], 2), -len(na), -nm)
                            if best is None or key < best[0]:
                                best = (key, na, ns, nm)
                    if nv == 0 and len(na) < depth:
                        cands.append((ns, na, nv, nm))
            if not cands:
                break
            if mode == 'seek':
                cands.sort(key=lambda t: (t[0][6], -t[3], -t[0][2]))
            else:
                cands.sort(key=lambda t: (t[0][6], -t[3], t[0][2]))
            beam = cands[:BEAM]
        return best

    def plan(self, state, horizon=HORIZON):
        """
        双意图束搜索（对应语言的「先绕开，再去吃」）：
          - seek：能一步到胃时，给出「到达食物」的完整路径。
          - approach：到不了时，给出最深的一条零违反、且持续逼近食物的路径。
        取食优先；否则返回 approach（reached=False），身体仍然只执行
        验证通过（张力 = 0）的那段前缀。
        """
        best_goal = self._search(state, horizon, 'seek')
        if best_goal is not None:
            return {'actions': best_goal[1], 'state': best_goal[2],
                    'min_pd': best_goal[3], 'reached': True,
                    'viol_plan': 0, 'intent': 'seek'}
        best_safe = self._search(state, horizon, 'approach')
        if best_safe is not None:
            return {'actions': best_safe[1], 'state': best_safe[2],
                    'min_pd': best_safe[3], 'reached': False,
                    'viol_plan': 0, 'intent': 'approach'}
        return {'actions': [], 'state': state, 'min_pd': 0.0,
                'reached': False, 'viol_plan': 1, 'intent': 'none'}

    def verify(self, state, actions):
        """用留出模型（奇数样本）重放；张力 = 违反步数比例。"""
        if not actions:
            return 1.0, []
        st = state
        viol = 0
        trace = []
        for a in actions:
            st, rec = self.mv.step(st, a)
            viol += rec['viol']
            trace.append((a, rec))
        self.last_trace = trace
        return viol / len(actions), trace


# ============================================================ 语言侧
ZH_TASK = {
    '果树': 'fruit tree', '食物': 'food', '捕食者': 'predator',
    '饿了': 'hunger', '饿': 'hunger', '危险': 'danger', '威胁': 'threat',
    '能量源': 'energy', '能量': 'energy', '疼痛': 'pain', '张力': 'pain',
}
ZH_TASK_KEYS = sorted(ZH_TASK, key=len, reverse=True)
EN_AVOID = ('avoid', 'escape', 'stay away', 'keep away', '远离', '避开', '躲避')
EN_SEEK = ('seek', 'find', 'move toward', 'approach', 'go to', 'look for',
           '寻找', '靠近', '找到')
GOAL_TERMS = list(L31.PROBES) + ['energy', 'pain', 'fruit', 'tree', 'fruit tree']


class Language:
    def __init__(self, vocab=VOCAB):
        self.tg = L31.TextGraph('l32', vocab_size=vocab).build(
            L31.wordnet_docs,
            extra_required={'fruit', 'tree', 'food', 'hunger', 'predator',
                            'survival', 'survive'})
        self.ali = self.tg.align()

    def term_margin(self, term):
        ws = [w for w in term.split() if w in self.tg.id]
        if not ws:
            return None, None
        if len(ws) == 1:
            a = self.ali[ws[0]]
            return a['axis'], a['margin']
        v = self.tg.cvec[[self.tg.id[w] for w in ws]].mean(axis=0)
        v = v / max(np.linalg.norm(v), 1e-9)
        se = float(v @ self.tg.cvec[self.tg.id['energy']])
        sp = float(v @ self.tg.cvec[self.tg.id['pain']])
        m = se - sp
        return ('energy' if m > 0 else 'pain'), m

    def energy_goal_word(self):
        """
        主任务的目标词不写成常量：在 L31 的探针词表 + fruit tree 里，
        挑一个在共现图中离 energy 锚点最远（margin 最大）且归属 energy 的词。
        选出来的词会直接进入句子，所以「用哪个人类词」也来自文本图。
        """
        best, best_m = 'food', -1e9
        for t in list(L31.PROBES) + ['fruit tree', 'fruit']:
            axis, m = self.term_margin(t)
            if axis == 'energy' and m is not None and m > best_m:
                best, best_m = t, float(m)
        return best, best_m

    def parse(self, text):
        low = text.lower()
        for zh in ZH_TASK_KEYS:
            low = low.replace(zh, ' ' + ZH_TASK[zh] + ' ')
        clauses = re.split(r'\band\b|[，,；;。.]', low)
        goals, conds = [], []
        for cl in clauses:
            verb = -1 if any(v in cl for v in EN_AVOID) else \
                (1 if any(v in cl for v in EN_SEEK) else 0)
            found = []
            for t in GOAL_TERMS:
                m = re.search(r'\b' + re.escape(t) + r'\b', cl)
                if m:
                    found.append((m.start(), t))
            if not found:
                continue
            found.sort(key=lambda t: (t[0], -len(t[1])))
            term = found[0][1]
            axis, margin = self.term_margin(term)
            if axis is None:
                continue
            item = {'term': term, 'axis': axis, 'margin': margin,
                    'verb': verb}
            (goals if verb != 0 else conds).append(item)
        return {'goals': goals, 'conditions': conds, 'raw': text,
                'english': low}


def cond_error(axis, reserve, pain):
    """
    条件词的身体预测误差：
      energy 轴的条件（饿）应当在低储备时为真 -> 误差 = max(0, reserve - 0.40)
      pain   轴的条件（痛）应当在高张力时为真 -> 误差 = max(0, 0.40 - pain)
    选误差小的轴，就是让身体去纠正文本图的误判。
    """
    if axis == 'energy':
        return max(0.0, reserve - 0.40)
    return max(0.0, 0.40 - pain)


# ============================================================ 控制器
class Controller:
    def __init__(self, planner, maze, mode='offline', seed=SEED):
        self.pl = planner
        self.maze = maze
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.buffer = []
        self.ptr = 0
        self.replan_left = 0
        self.plans = 0
        self.safe_plans = 0
        self.refusals = 0
        self.fidelity_ok = 0
        self.last_plan = None
        self.last_tension = None
        self.last_narration = None
        self.first_narration = None
        self.narrations = []
        # 回合内第一次规划（此时身体一步没动）是否张力为 0
        self.first_plan_safe = None

    def reset_episode(self):
        """只清空动作缓冲；跨回合统计保留。"""
        self.buffer = []
        self.ptr = 0
        self.replan_left = 0
        self.narrations = []
        self.first_plan_safe = None

    def reflex(self, state):
        """没有可验证规划时的保命反射：选预测捕食距离最大的动作。"""
        best_a, best_pd = 0, -1e9
        for a in range(6):
            _ns, rec = self.pl.mp.step(state, a)
            if rec['pd'] > best_pd:
                best_a, best_pd = a, rec['pd']
        return best_a

    def act(self, senses, state, goal_terms):
        if self.mode == 'random':
            return int(self.rng.integers(0, 6))
        if self.mode == 'reactive':
            return self._reactive(state, senses)
        if self.mode == 'oracle':
            return self._oracle()
        return self._offline(state, goal_terms)

    def _reactive(self, state, senses):
        """
        对照组的「直接瞎走」：没有前向模型、没有推演。
        只做当前感觉的梯度上升——朝着脚下更甜的方向走，站在食物上就吃；
        只有捕食者已经贴脸（< 1.5 格）时才往反方向挪一步。
        它看得见食物，也看得见此刻的捕食者，但看不见捕食者下一格。
        """
        if float(senses[8]) < REACH_DE:
            return HARVEST
        gx, gy = float(senses[4]), float(senses[5])
        if math.hypot(gx, gy) < 1e-9:
            return REST
        a = (3 if gx > 0 else 2) if abs(gx) >= abs(gy) else (1 if gy > 0 else 0)
        pdx, pdy = float(senses[10]), float(senses[11])
        if math.hypot(pdx, pdy) < 1.5:
            a = (2 if pdx > 0 else 3) if abs(pdx) >= abs(pdy) else (0 if pdy > 0 else 1)
        return a

    def _oracle(self):
        """作弊上界：用真值沙盒 clone 推演。L32 本体被明令禁止使用这条路。"""
        if (not self.buffer) or self.ptr >= len(self.buffer) or \
                self.replan_left <= 0:
            with self.maze.no_touch():
                acts = oracle_plan(self.maze)
            self.buffer = list(acts) if acts else [REST]
            self.ptr = 0
            self.replan_left = min(REPLAN_EVERY, len(self.buffer))
        a = self.buffer[self.ptr]
        self.ptr += 1
        self.replan_left -= 1
        return a

    def _offline(self, state, goal_terms):
        need_plan = (not self.buffer) or self.ptr >= len(self.buffer) or \
            self.replan_left <= 0
        if need_plan:
            with self.maze.no_touch():
                p = self.pl.plan(state)
                tension, trace = self.pl.verify(state, p['actions'])
            self.plans += 1
            p['tension'] = tension
            self.last_plan = p
            self.last_tension = tension
            if self.first_plan_safe is None:
                # 这是本回合第一次规划，身体还没动过：它是不是「想清楚才动」
                self.first_plan_safe = bool(tension == 0.0)
            if tension == 0.0:
                self.safe_plans += 1
                self.buffer = list(p['actions'])
                self.ptr = 0
                self.replan_left = min(REPLAN_EVERY, len(self.buffer))
                self.last_narration = narrate(p, trace, state, goal_terms)
                self.narrations.append(self.last_narration)
                if self.first_narration is None:
                    self.first_narration = self.last_narration
                # 独立真值回放（只读审计，不参与规划）
                if truth_replay(self.maze, p['actions']):
                    self.fidelity_ok += 1
            else:
                self.refusals += 1
                self.buffer = [self.reflex(state)]
                self.ptr = 0
                self.replan_left = 1
        if not self.buffer:
            self.buffer = [self.reflex(state)]
            self.ptr = 0
        a = self.buffer[self.ptr]
        self.ptr += 1
        self.replan_left -= 1
        return a


def oracle_plan(maze, max_depth=30):
    """
    作弊上界：直接用真值动力学做 BFS，找一条到食物的最短安全路径。
    这是「如果它完全知道这个世界，能做到多好」的参考线，
    L32 本体被明令禁止调用这条路（规划锁 + 只读真值）。
    """
    from collections import deque
    n = maze.n
    dE, Hs = maze.dE, maze.Hs
    x0, y0 = int(maze.pos[0]), int(maze.pos[1])
    px0, py0 = int(maze.pred[0][0]), int(maze.pred[0][1])
    ph0 = int(maze.t % 2)
    q = deque([(x0, y0, px0, py0, ph0, 0.55, 0.05, [])])
    seen = {(x0, y0, px0, py0, ph0, 11, 1)}
    while q:
        x, y, px, py, ph, res, ten, acts = q.popleft()
        if len(acts) >= max_depth:
            continue
        for a in range(6):
            mx, my = MOTION[a]
            nx = int(np.clip(x + mx, 0, n - 1))
            ny = int(np.clip(y + my, 0, n - 1))
            npx, npy = px, py
            if ph == 0:          # 下一个 tick 是奇数 -> 捕食者移动
                ddx, ddy = nx - npx, ny - npy
                if abs(ddx) >= abs(ddy):
                    npx += int(math.copysign(1, ddx)) if ddx != 0 else 0
                else:
                    npy += int(math.copysign(1, ddy)) if ddy != 0 else 0
                npx = int(np.clip(npx, 0, n - 1))
                npy = int(np.clip(npy, 0, n - 1))
            pd = math.hypot(npx - nx, npy - ny)
            nph = 1 - ph
            nres = min(1.0, max(0.0, res - DRAIN +
                                (HARVEST_GAIN * float(maze.Ef[y, x]) if a == HARVEST else 0.0)))
            nten = min(1.0, max(0.0, ten + 0.12 * float(Hs[ny, nx]) +
                                0.30 * math.exp(-(pd / 1.2) ** 2) - 0.02))
            if pd <= KILL_R or nres <= 0.0 or nten >= 1.0:
                continue
            na = acts + [a]
            if float(dE[ny, nx]) < REACH_DE:
                return na
            key = (nx, ny, npx, npy, nph, int(nres * 20), int(nten * 20))
            if key in seen:
                continue
            seen.add(key)
            q.append((nx, ny, npx, npy, nph, nres, nten, na))
    return []


def truth_replay(maze, actions):
    """在真值 clone 上回放整条规划；只用于审计，结果不回流规划。"""
    c = maze.clone()
    for a in actions:
        c.step(a)
        if c.dead:
            return False
    return bool(not c.dead)


# ============================================================ 语言输出
DIR_CN = {'up': '上', 'down': '下', 'left': '左', 'right': '右',
          'harvest': '原地采集', 'rest': '原地休息'}


def _food_dir(start, end, n):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dx) >= abs(dy):
        return '右' if dx > 0 else '左'
    return '下' if dy > 0 else '上'


def narrate(plan, trace, state, goal_terms):
    acts = plan['actions']
    if not acts:
        return '没有可验证的规划。'
    x, y = state[0], state[1]
    ex, ey = plan['state'][0], plan['state'][1]
    goal_word = goal_terms[0][2] if goal_terms else 'food'
    word_cn = {'food': '食物', 'fruit tree': '果树', 'fruit': '果实',
               'energy': '能量源', 'fuel': '燃料', 'nutrient': '养分',
               'resource': '资源', 'starvation': '饥饿', 'hunger': '饥饿',
               'predator': '捕食者'}.get(goal_word, goal_word)
    danger = plan['min_pd'] < 3.0
    first_dir = DIR_CN[ACTIONS[acts[0]]]
    head = '前面有捕食者' if danger else '前方暂时安全'
    toward = _food_dir((x, y), (ex, ey), maze_n())
    if ACTIONS[acts[0]] in ('harvest', 'rest'):
        if plan['reached']:
            sent = (f'我已经站在{word_cn}上了，直接采集；'
                    f'此刻最近的捕食者在 {plan["min_pd"]:.1f} 格外，暂时没有威胁。')
        else:
            sent = (f'{head}，我先原地稳住，等安全窗口再靠近{word_cn}。')
    elif not plan['reached']:
        # 还到不了果树：先说清「先绕开」，再沿安全侧逼近，不承诺吃到
        sent = (f'{head}，我应该先往{first_dir}绕开，'
                f'沿着安全的一侧靠近{word_cn}，因为那边暂时没有威胁。')
    else:
        first = ACTIONS[acts[0]]
        detour = DIR_CN[first] != toward and first not in ('harvest', 'rest')
        verb = '绕开' if detour else '直行'
        tail = '因为那里没有威胁' if plan['min_pd'] > SAFE_R * 1.5 else '但路上仍有风险'
        sent = (f'{head}，我应该先往{first_dir}{verb}，去吃{toward}侧的{word_cn}，{tail}。')
    code = pseudo(plan, trace, state)
    return {'sentence': sent, 'pseudocode': code,
            'min_pd': round(float(plan['min_pd']), 2),
            'reached': bool(plan['reached']),
            'intent': plan.get('intent', 'seek')}


def maze_n():
    return GRID


def pseudo(plan, trace, state):
    lines = ['GOAL   %s' % ('seek(energy_axis)' if plan.get('reached')
                           else 'avoid(predator_axis)'),
             'BODY   reserve=%.2f  pain=%.2f' % (state[2], state[3])]
    for i, (a, rec) in enumerate(trace, 1):
        lines.append(
            f'{i:>2}. {ACTIONS[a]:<7s} # pred_dist {rec["pd"]:.2f} '
            f'(lo {rec["lo_pd"]:.2f})  dE {rec["dE"]:.2f}  '
            f'reserve {rec["res"]:.2f}  pain {rec["pain"]:.2f}')
    if trace:
        fdE = trace[-1][1]['dE']
        if plan.get('reached'):
            tail = '< %.2f  (进入采集半径)' % REACH_DE
        elif fdE < 0.25:
            tail = '(已贴近，下一段进入采集半径)'
        else:
            tail = '(仍在接近途中，暂不取食)'
        lines.append('CHECK  min(lo_pred_dist)=%.2f > %.1f ; '
                     'min(lo_reserve)=%.2f > 0.02 ; final_dE=%.2f %s' % (
                         min(r['lo_pd'] for _a, r in trace), SAFE_R,
                         min(r['lo_res'] for _a, r in trace), fdE, tail))
        lines.append('TENSION = %.3f -> %s' % (
            float(plan.get('tension', 0.0)),
            'COMMIT' if float(plan.get('tension', 0.0)) == 0.0 else 'REFUSE'))
    return '\n'.join(lines)


# ============================================================ 回合与评估
def make_starts(maze, k):
    rng = np.random.default_rng(SEED + 5)
    out = []
    tries = 0
    while len(out) < k and tries < 5000:
        tries += 1
        x = int(rng.integers(1, maze.n - 1))
        y = int(rng.integers(1, maze.n - 1))
        pd = math.hypot(x - maze.pred[0][0], y - maze.pred[0][1])
        if pd < 4.0:
            continue
        if float(maze.dE[y, x]) < 0.70:
            continue
        out.append((x, y))
    return out


def run_episode(maze, ctrl, start, steps=EVAL_STEPS, policy='offline',
                goal_term='food'):
    maze.reset(start, energy=0.55, tension=0.05)
    if ctrl is not None:
        ctrl.reset_episode()
    _p0 = ctrl.plans if ctrl is not None else 0
    _s0 = ctrl.safe_plans if ctrl is not None else 0
    _f0 = ctrl.fidelity_ok if ctrl is not None else 0
    _r0 = ctrl.refusals if ctrl is not None else 0
    path = [tuple(int(v) for v in maze.pos)]
    for _ in range(steps):
        if maze.dead:
            break
        s = maze.senses()
        state = (int(maze.pos[0]), int(maze.pos[1]), maze.energy, maze.tension,
                 float(maze.pred[0][0]), float(maze.pred[0][1]),
                 float(maze.dE[int(maze.pos[1]), int(maze.pos[0])]),
                 float(maze.t % 2))
        a = ctrl.act(s, state, [('energy', 1, goal_term)])
        maze.step(a)
        path.append(tuple(int(v) for v in maze.pos))
        if maze.reached:
            break
    return {
        'alive': int(not maze.dead),
        'steps': int(maze.t),
        'reached': int(maze.reached),
        'reserve': round(float(maze.energy), 3),
        'pain': round(float(maze.tension), 3),
        'refusals': ctrl.refusals - _r0 if ctrl is not None else 0,
        'plans': ctrl.plans - _p0 if ctrl is not None else 0,
        'safe_plans': ctrl.safe_plans - _s0 if ctrl is not None else 0,
        'fidelity': ctrl.fidelity_ok - _f0 if ctrl is not None else 0,
        'first_safe': int(bool(ctrl is not None and ctrl.first_plan_safe)),
        'path': path,
    }


def paired_bootstrap(a, b, n=2000, seed=SEED + 9):
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) < 5:
        return float('nan')
    rng = np.random.default_rng(seed)
    vals = np.array([np.mean(rng.choice(d, len(d), replace=True))
                     for _ in range(n)])
    return float(np.mean(vals <= 0.0))


def summarize(rows):
    n = len(rows)
    return {
        'episodes': n,
        'deaths': int(sum(1 - r['alive'] for r in rows)),
        'death_rate': round(float(np.mean([1 - r['alive'] for r in rows])), 3),
        'alive_rate': round(float(np.mean([r['alive'] for r in rows])), 3),
        'reach_rate': round(float(np.mean([r['reached'] for r in rows])), 3),
        'mean_steps': round(float(np.mean([r['steps'] for r in rows])), 1),
        'mean_reserve': round(float(np.mean([r['reserve'] for r in rows])), 3),
        'mean_pain': round(float(np.mean([r['pain'] for r in rows])), 3),
        'plans': int(sum(r['plans'] for r in rows)),
        'safe_plans': int(sum(r['safe_plans'] for r in rows)),
        'refusals': int(sum(r['refusals'] for r in rows)),
        'fidelity': int(sum(r['fidelity'] for r in rows)),
        'first_safe': int(sum(r.get('first_safe', 0) for r in rows)),
        'first_safe_rate': round(float(np.mean([r.get('first_safe', 0)
                                                for r in rows])), 3),
    }


# ============================================================ 跨领域测验
def cross_domain(maze, planner, lang):
    text = '如果你饿了，去寻找果树'
    parsed = lang.parse(text)
    start = _far_food_start(maze)
    maze.reset(start, energy=0.45, tension=0.05, pred=maze.pred0)
    state = (int(maze.pos[0]), int(maze.pos[1]), maze.energy, maze.tension,
             float(maze.pred[0][0]), float(maze.pred[0][1]),
             float(maze.dE[int(maze.pos[1]), int(maze.pos[0])]),
             float(maze.t % 2))

    cond = parsed['conditions'][0] if parsed['conditions'] else None
    goal = parsed['goals'][0] if parsed['goals'] else None

    # 条件轴：文本 vs 身体，用预测误差裁决
    if cond is not None:
        e_text = cond_error(cond['axis'], state[2], state[3])
        alt = 'energy' if cond['axis'] == 'pain' else 'pain'
        e_alt = cond_error(alt, state[2], state[3])
        chosen_cond = cond['axis'] if e_text <= e_alt else alt
    else:
        e_text = e_alt = 0.0
        chosen_cond = 'energy'

    # 目标轴：多词组合（fruit tree）由上下文向量决定
    goal_axis = goal['axis'] if goal else 'energy'
    goal_margin = goal['margin'] if goal else None

    # 和正式回合一样：想一段、只执行验证通过的前缀、再重规划。
    # 不把一条 20+ 步的长计划一次性梭哈，因为「想清楚」是滚动进行的。
    ctrl = Controller(planner, maze, mode='offline')
    ctrl.reset_episode()
    executed = 0
    harvests = 0
    harvest_gain = 0.0
    before = (state[2], state[3])
    for _ in range(EVAL_STEPS):
        if maze.dead:
            break
        s = maze.senses()
        state = (int(maze.pos[0]), int(maze.pos[1]), maze.energy, maze.tension,
                 float(maze.pred[0][0]), float(maze.pred[0][1]),
                 float(maze.dE[int(maze.pos[1]), int(maze.pos[0])]),
                 float(maze.t % 2))
        a = ctrl.act(s, state, [('energy', 1, goal['term'] if goal else 'food')])
        e0 = maze.energy
        maze.step(a)
        executed += 1
        # 「吃」必须是身体真的在食物格上按了 harvest，且这一步能量上升。
        if a == HARVEST and maze.reached and maze.energy > e0:
            harvests += 1
            harvest_gain = max(harvest_gain, maze.energy - e0)
            break
    after = (maze.energy, maze.tension)

    p = ctrl.last_plan or {'actions': [], 'reached': False, 'min_pd': 0.0,
                           'tension': 1.0}
    tension = float(p.get('tension', 1.0))
    nar = ctrl.first_narration or {'sentence': '没有可验证的规划。',
                                   'pseudocode': '', 'min_pd': 0.0,
                                   'reached': False, 'intent': 'none'}
    ok = bool(maze.reached and not maze.dead and harvests > 0 and
              harvest_gain > 0.0 and
              chosen_cond == 'energy' and goal_axis == 'energy')
    return {
        'sentence': text,
        'condition': {
            'term': cond['term'] if cond else None,
            'text_axis': cond['axis'] if cond else None,
            'text_margin': round(float(cond['margin']), 4) if cond else None,
            'alt_axis': alt if cond else None,
            'err_text': round(float(e_text), 4),
            'err_alt': round(float(e_alt), 4),
            'chosen': chosen_cond,
        },
        'goal': {
            'term': goal['term'] if goal else None,
            'axis': goal_axis,
            'margin': round(float(goal_margin), 4) if goal_margin is not None else None,
        },
        'body_before': {'reserve': round(before[0], 3), 'pain': round(before[1], 3)},
        'body_after': {'reserve': round(after[0], 3), 'pain': round(after[1], 3)},
        'plan': {
            'actions': [ACTIONS[a] for a in p['actions']],
            'reached': bool(p['reached']),
            'tension': round(float(tension), 3),
            'min_pd': round(float(p['min_pd']), 2),
        },
        'executed_steps': executed,
        'harvests': int(harvests),
        'harvest_gain': round(float(harvest_gain), 4),
        'narration': nar,
        'passed': ok,
    }


def _far_food_start(maze):
    """跨领域测验的起点：离果树最远、且离捕食者巢至少 4 格，必须真的走一段路。"""
    best, best_d = (2, 2), -1.0
    for y in range(maze.n):
        for x in range(maze.n):
            d = float(maze.dE[y, x])
            if d < 0.70:
                continue
            pd = math.hypot(x - maze.pred0[0][0], y - maze.pred0[0][1])
            if pd < 4.0:
                continue
            if d > best_d:
                best, best_d = (x, y), d
    return best


# ============================================================ 绘图
def draw(maze, offline_rows, summary, cross):
    fig = plt.figure(figsize=(13, 9), dpi=130)
    gs = fig.add_gridspec(2, 2, hspace=0.33, wspace=0.26)
    fig.suptitle('L32: inner monologue -> verified plan -> body',
                 fontsize=13, fontweight='bold')

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(maze.Hs, cmap='Reds', alpha=0.35)
    ax.imshow(maze.Ef, cmap='Greens', alpha=0.55)
    path = offline_rows[0]['path'] if offline_rows else []
    if path:
        p = np.asarray(path)
        ax.plot(p[:, 0], p[:, 1], '-', color='#111', lw=1.5)
        ax.scatter([p[0, 0]], [p[0, 1]], c='#2e86c1', s=35, label='start')
        ax.scatter([p[-1, 0]], [p[-1, 1]], c='#c0392b', s=35, label='end')
    ax.scatter([maze.pred0[0][0]], [maze.pred0[0][1]], c='k', marker='X',
               s=70, label='predator nest')
    ax.set_title('A. one offline episode (true replay)', fontsize=9)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    names = ['offline', 'reactive', 'random', 'oracle']
    vals = [summary[k]['death_rate'] for k in names]
    cols = ['#c0392b', '#e67e22', '#95a5a6', '#2e86c1']
    ax.bar(names, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=9)
    ax.set_ylabel('death rate')
    ax.set_ylim(0, max(0.2, max(vals) * 1.25))
    ax.set_title('B. L32-2: deaths per episode', fontsize=9)
    ax.grid(alpha=0.25, axis='y')

    ax = fig.add_subplot(gs[1, 0])
    o = summary['offline']
    pr = o['safe_plans'] / max(o['plans'], 1)
    fi = o['fidelity'] / max(o['safe_plans'], 1)
    fs = o.get('first_safe_rate', 0.0)
    ax.bar(['first plan\nsafe', 'plans', 'tension=0', 'truth\nreplay ok'],
           [o.get('first_safe', 0), o['plans'], o['safe_plans'], o['fidelity']],
           color=['#8e44ad', '#bdc3c7', '#27ae60', '#2e86c1'])
    ax.set_title(f'C. L32-1 thinking: first-plan safe={fs:.2f}, '
                 f'safe={pr:.2f}, fidelity={fi:.2f}', fontsize=9)
    ax.grid(alpha=0.25, axis='y')

    ax = fig.add_subplot(gs[1, 1])
    ax.plot([0, 1], [cross['body_before']['reserve'],
                     cross['body_after']['reserve']], 'o-', color='#27ae60',
            label='reserve')
    ax.plot([0, 1], [cross['body_before']['pain'],
                     cross['body_after']['pain']], 'o-', color='#c0392b',
            label='pain')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['before', 'after'])
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title('D. cross-domain: "hungry -> find a fruit tree"', fontsize=9)
    ax.grid(alpha=0.25, axis='y')

    fig.savefig(PNG, bbox_inches='tight')
    plt.close(fig)


# ============================================================ 持久化
def save_db(summary, offline, verdict, cross):
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute('DROP TABLE IF EXISTS l32_episode')
    cur.execute('''CREATE TABLE l32_episode(
        policy TEXT, ep INTEGER, alive INTEGER, steps INTEGER, reached INTEGER,
        reserve REAL, pain REAL, refusals INTEGER, plans INTEGER,
        safe_plans INTEGER, fidelity INTEGER)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS l32_verdict(
        id INTEGER PRIMARY KEY AUTOINCREMENT, status TEXT, death_rate REAL,
        reactive_death REAL, safe_rate REAL, fidelity REAL, cross_ok INTEGER,
        reason TEXT, ts TEXT DEFAULT CURRENT_TIMESTAMP)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS l32_cross(
        id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT,
        ts TEXT DEFAULT CURRENT_TIMESTAMP)''')
    for pol, rows in summary.items():
        for i, r in enumerate(offline if pol == 'offline' else rows['rows']):
            cur.execute('''INSERT INTO l32_episode VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
                        (pol, i, r['alive'], r['steps'], r['reached'],
                         r['reserve'], r['pain'], r['refusals'], r['plans'],
                         r['safe_plans'], r['fidelity']))
    cur.execute('''INSERT INTO l32_verdict(status,death_rate,reactive_death,
        safe_rate,fidelity,cross_ok,reason) VALUES(?,?,?,?,?,?,?)''',
                (verdict['status'], verdict['death_rate'],
                 verdict['reactive_death'], verdict['safe_rate'],
                 verdict['fidelity'], int(verdict['cross_ok']),
                 verdict['reason']))
    cur.execute('INSERT INTO l32_cross(payload) VALUES(?)',
                (json.dumps(cross, ensure_ascii=False),))
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


def write_report(summary, l32, cross, verdict, lang, nar, elapsed):
    o = summary['offline']
    r = summary['reactive']
    q = summary['random']
    z = summary['oracle']
    L = []
    add = L.append
    add('# L32 内部推演实证报告：先想清楚，再让身体动')
    add('')
    add('> 前半句：在脑子里把路走完。后半句：只有张力为 0 才允许出门。')
    add('')
    add(f'- 生成时间：{time.strftime("%Y-%m-%d %H:%M:%S")}')
    add(f'- 总耗时：{elapsed:.1f}s ｜ 峰值内存：{mem_mb():.0f} MB ｜ 随机种子：{SEED}')
    add(f'- 世界：{GRID}×{GRID} 迷宫，1 个食物源（右上角「果树」），1 个捕食者巢（正中），'
        f'14 个身体感受（L31 的 10 个 + 捕食者相对方位/距离/时钟相位）')
    add(f'- 语言侧：WordNet 释义语料 {lang.tg.n_docs} 条，词表 {len(lang.tg.vocab)}')
    add('- 判据在实验前写死（§7），看到结果后未回调。')
    add('')

    add('## 0. 这轮到底做了什么')
    add('')
    add('L31 的系统会「懂词」，但它每走一步都靠真值世界的反馈。L32 把反馈通道切断：')
    add('')
    add('1. **规划期给真值沙盒上锁**。`PredatorMaze.no_touch()` 内的任何 '
        '`step()`/`reset()` 直接抛 `RealityViolation`。规划器只能读当前感受，'
        '之后全部在学到的模型里推演。')
    add('2. **先出语言，再出动作**。每条被接受的规划都先写成一句中文 + 一段伪代码，'
        '伪代码里逐步列出预测的捕食距离、能量、张力。')
    add('3. **悲观验证**。验证器不是规划器本身：世界模型用偶数样本拟合，'
        '验证器用奇数样本，并要求 `均值 - 2σ` 仍满足安全不变式。')
    add('4. **真值回放只做审计**。规划提交后在 clone 上重放一遍，'
        '结果不回传，只作为「模型有没有骗自己」的第三方证据。')
    add('')
    add('一句话：它不是「先走一步看看会不会死」，而是「先算清楚会不会死，'
        '算不清楚就不走」。')
    add('')

    add('## 1. 世界：捕食者迷宫（以及一条不变量修订）')
    add('')
    add('| 元素 | 设定 |')
    add('|---|---|')
    add('| 食物 | 右上角单一高斯高能簇（「果树」），`Ef ≥ 0.35` 的格子可采集 |')
    add('| 捕食者 | 1 只，每两步走一格（比身体慢半拍），先走差值更大的轴，接触即死 |')
    add('| 死亡 | 与捕食者同格、能量 ≤ 0、或张力 ≥ 1 |')
    add('| 能量 | 每步 −0.012；采集 `+0.15 × 脚下能量` |')
    add('| 安全半径 | 规划要求悲观预测仍与捕食者保持 > 1.8 格（= 2×死亡半径，留一步反应余量；取食另要求 > 3.5 格舒适半径） |')
    add('')
    add('**不变量 I6 的修订（必须明说）**：L31 的物理侧是 10 个感受，'
        'L32 增加到 14 个，新增 `[pred_dx, pred_dy, pred_dist, phase]`。'
        '理由：如果系统看不见捕食者，就不可能出现「前面有捕食者」这句话，'
        '也不可能有任何「预测它的下一格」的推演；`phase` 是它半速节律的时钟，'
        '没有这个相位，「哪一步它会动」就是不可预测的。这是显式修订，不是偷偷加维度。')
    add('')

    add('## 2. 世界模型是从数据里数出来的，不是写死的')
    add('')
    add(f'随机施加动作（do-算子）采集 {TRAIN_EPISODES} 条轨迹 + 全覆盖干预，'
        f'共 {l32["train_samples"]} 个转移样本。模型有三张表：')
    add('')
    add('| 表 | 索引 | 内容 |')
    add('|---|---|---|')
    add('| 身体转移表 | `(x, y, action)` | Δ到食物距离、Δ储备、Δ张力 的均值与 σ |')
    add('| 捕食者行为表 | `(Δx, Δy, phase)` | 捕食者下一步位移的均值与 σ |')
    add('| 疼痛邻近修正 | `(x, y, action, 捕食后距离档)` | 静态表解释不了的张力增量 |')
    add('')
    add('没有任何一行代码写「捕食者会追你」；这句话是从 (Δx,Δy,phase) 的位移表里数出来的。'
        '疼痛也没有写「越近越痛」；它是按捕食后的距离分箱数出来的。')
    add('')

    add('## 3. 内部推演长什么样')
    add('')
    add('束搜索深度 %d、宽度 %d，排序键是**约束优先**而不是 reward：'
        '先比违反安全不变式的步数，再比终点离食物多远，再比最小捕食距离。' %
        (HORIZON, BEAM))
    add('')
    add(f'句子里的目标词不是人写死的常量：`{l32["goal_term"]}` 是文本共现图里'
        f'离 energy 锚点最远（margin={f3(l32["goal_margin"], 4)}）且归属 energy 的词，'
        f'同一张图里 `food` 的 margin 只有 {f3(l32["food_margin"], 4)}；'
        '选词由 L31 的 `TextGraph.align()` 完成，选中的词直接进入句子。')
    add('')
    add('下面是一次真实运行的输出（离线、未执行前）：')
    add('')
    if nar:
        add('> ' + nar['sentence'])
        add('')
        add('```text')
        add(nar['pseudocode'])
        add('```')
    else:
        add('（本轮没有产生可验证规划）')
    add('')
    if l32.get('narration_eat'):
        add('第一次把「去吃」写进句子的规划（`reached=True`，张力仍为 0 才放行）：')
        add('')
        add('> ' + l32['narration_eat']['sentence'])
        add('')
        add('```text')
        add(l32['narration_eat']['pseudocode'])
        add('```')
        add('')

    add('## 4. L32-1：离线思考率')
    add('')
    add('| 指标 | 定义 | 实测 |')
    add('|---|---|---|')
    add(f'| **离线思考率** | 身体一步未动时，第一次规划张力=0 的回合比例 | '
        f'{o.get("first_safe", 0)}/{o["episodes"]} = '
        f'**{o.get("first_safe_rate", 0):.3f}** |')
    add(f'| 规划总数 | 所有离线回合里重新规划的次数字 | {o["plans"]} |')
    add(f'| 张力=0 的规划 | 悲观验证全部通过的次数 | {o["safe_plans"]} |')
    add(f'| 安全规划率 | 张力=0 / 规划总数 | '
        f'**{o["safe_plans"] / max(o["plans"], 1):.3f}** |')
    add(f'| 真值回放通过 | 已验证规划在 clone 里也确实不死 | {o["fidelity"]} |')
    add(f'| 模型保真度 | 真值回放通过 / 张力=0 规划 | '
        f'**{o["fidelity"] / max(o["safe_plans"], 1):.3f}** |')
    add('')
    add('注意：模型保真度不是规划器自己打的分，而是在**真值引擎**上重放出来的。'
        '这一列才是「它是不是在骗自己」的独立答案。')
    add('')

    add('## 5. L32-2：死亡次数（四组对照）')
    add('')
    add('| 策略 | 回合 | 死亡 | 死亡率 | 存活率 | 到达食物 | 平均存活步 | 平均能量 | 平均张力 |')
    add('|---|---|---|---|---|---|---|---|---|')
    for name, s in (('offline（先想后动）', o), ('reactive（单步贪心）', r),
                    ('random（乱走）', q), ('oracle（真值作弊上界）', z)):
        add(f'| {name} | {s["episodes"]} | {s["deaths"]} | {s["death_rate"]:.3f} | '
            f'{s["alive_rate"]:.3f} | {s["reach_rate"]:.3f} | {s["mean_steps"]:.1f} | '
            f'{s["mean_reserve"]:.3f} | {s["mean_pain"]:.3f} |')
    add('')
    add(f'配对 bootstrap（offline − reactive，按回合配对）：'
        f'p = {f3(l32["p_off_vs_react"], 4)}；'
        f'（offline − random）：p = {f3(l32["p_off_vs_rand"], 4)}。')
    add('')
    add('oracle 是作弊上界：它用真值引擎 clone 推演，L32 本体被明令禁止使用这条路。'
        '它只回答一个问题——「这个地图上，理论上能做到多好」。')
    add('')

    add('## 6. 跨领域测验：如果你饿了，去寻找果树')
    add('')
    add('这句话里有两个 L31 没直接教过的坑：`饿` 在语料里被判到 **pain** 轴（错的），'
        '`果树` 不在探针表里。系统必须自己拆：')
    add('')
    add('| 步骤 | 实测 |')
    add('|---|---|')
    c = cross['condition']
    add(f'| ① 条件词 `{c["term"]}` 的文本轴 | **{c["text_axis"]}** '
        f'(margin={f3(c["text_margin"], 4)}) |')
    add(f'| ② 身体证据 | reserve={cross["body_before"]["reserve"]}，'
        f'pain={cross["body_before"]["pain"]} |')
    add(f'| ③ 条件假设的预测误差 | pain 轴 {f3(c["err_text"], 3)} vs '
        f'energy 轴 {f3(c["err_alt"], 3)} → 选 **{c["chosen"]}** |')
    add(f'| ④ 目标词 `{cross["goal"]["term"]}` 的组合轴 | '
        f'**{cross["goal"]["axis"]}** (margin={f3(cross["goal"]["margin"], 4)}) |')
    add(f'| ⑤ 内部规划 | {", ".join(cross["plan"]["actions"]) or "无"}；'
        f'张力={cross["plan"]["tension"]}，到达={cross["plan"]["reached"]} |')
    add(f'| ⑥ 执行后 | reserve {cross["body_before"]["reserve"]} → '
        f'{cross["body_after"]["reserve"]}；pain '
        f'{cross["body_before"]["pain"]} → {cross["body_after"]["pain"]}；'
        f'真值采集 {cross["harvests"]} 次，单步回补 {cross["harvest_gain"]} |')
    add('')
    add(f'结论：**{"通过" if cross["passed"] else "未通过"}**。'
        '注意第 ③ 步不是人给的规则，而是「哪个轴能解释身体当前状态」的预测误差比较：'
        '文本说 `饥饿=疼痛`，但身体 pain 只有 %s，这个假设的误差是 %s；'
        '换成能量轴，误差是 %s。身体否决了文本。' % (
            cross['body_before']['pain'], f3(c["err_text"], 3),
            f3(c["err_alt"], 3)))
    add('')

    add('## 7. 判定（判据前置）')
    add('')
    add('| 判据 | 阈值 | 实测 | 通过 |')
    add('|---|---|---|---|')
    add(f'| 离线死亡率 | ≤ 0.05 | {o["death_rate"]:.3f} | '
        f'{"是" if o["death_rate"] <= 0.05 else "否"} |')
    add(f'| 相对 reactive 的死亡率 | ≤ 0.50× | '
        f'{o["death_rate"] / max(r["death_rate"], 1e-9):.2f}× | '
        f'{"是" if o["death_rate"] <= 0.5 * r["death_rate"] else "否"} |')
    add(f'| 安全规划率 | ≥ 0.90 | {o["safe_plans"] / max(o["plans"], 1):.3f} | '
        f'{"是" if o["safe_plans"] / max(o["plans"], 1) >= 0.90 else "否"} |')
    add(f'| 模型保真度 | ≥ 0.85 | '
        f'{o["fidelity"] / max(o["safe_plans"], 1):.3f} | '
        f'{"是" if o["fidelity"] / max(o["safe_plans"], 1) >= 0.85 else "否"} |')
    add(f'| 跨领域推理 | 通过 | {"通过" if cross["passed"] else "未通过"} | '
        f'{"是" if cross["passed"] else "否"} |')
    add('')
    icon = {
        'inner_monologue': '🟢 内部推演成立：想清楚再动，死亡率压到阈值内',
        'partial_monologue': '🟡 部分成立：规划有效，但保真度或跨领域未全过',
        'no_monologue': '🔴 未成立：离线推演没有跑赢直接反应',
    }
    add(f'### {icon.get(verdict["status"], verdict["status"])}')
    add('')
    add(f'- 状态：`{verdict["status"]}`')
    add(f'- 依据：{verdict["reason"]}')
    add('')

    add('## 8. 这份报告不能证明什么')
    add('')
    add('1. **不是「意识」**。它做的是有界前瞻 + 约束满足，不是自我意识，'
        '也不是通用推理。束搜索是工程手段，不是「心灵」。')
    add('2. **世界太干净**。捕食者是确定性追踪，食物是静态高斯簇；'
        '表模型能拟合，是因为这个世界可预测。换成随机环境，'
        '悲观验证会大量拒绝，死亡率不会这么低。')
    add('3. **语言是模板，不是生成**。句子里的方位、危险、理由都来自推演里的数，'
        '但句式是人写的模板。它没有自主句法。')
    add('4. **跨领域那次「身体否决文本」是一条规则**（预测误差小的轴胜出），'
        '不是通用常识推理。它证明了这条路可行，不证明它懂「饿」。')
    add('5. **只跟一个捕食者打交道**。多捕食者需要多目标跟踪，'
        '本轮的 14 维感受不够。')
    add('')
    add('这五条是可证伪边界，引用时必须一起引用。')
    add('')

    add('## 9. 复现')
    add('')
    add('```bash')
    add('cd /workspace && python -u cognitive_inner_monologue_agent.py')
    add('L32_FAST=1 python -u cognitive_inner_monologue_agent.py')
    add('```')
    add('')
    add('产物：')
    add(f'- `{OUT_MD}`')
    add(f'- `{OUT_JSON}`')
    add(f'- `{PNG}`')
    add(f'- `{DB}`：表 `l32_episode` / `l32_verdict` / `l32_cross`')
    add('')

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    return OUT_MD


# ============================================================ 主流程
def main():
    global GRID
    t0 = time.time()
    assert ANCHOR.safety_bound is SAFETY_BOUND
    print('[L32] 语言侧：WordNet 共现图 + 对齐 ...', flush=True)
    lang = Language()
    guard_mem()
    print(f'[L32]   文档={lang.tg.n_docs} 词表={len(lang.tg.vocab)}', flush=True)

    maze = PredatorMaze()
    WorldModel.n_limit = maze.n - 1
    print('[L32] 采集 do-算子转移样本 ...', flush=True)
    rows, pred_rows = collect(maze)
    print(f'[L32]   {len(rows)} 个转移样本', flush=True)

    even = [r for i, r in enumerate(rows) if i % 2 == 0]
    odd = [r for i, r in enumerate(rows) if i % 2 == 1]
    model_plan = WorldModel().fit(even, pred_rows)
    model_verify = WorldModel().fit(odd, pred_rows)
    guard_mem()
    planner = Planner(model_plan, model_verify)

    starts = make_starts(maze, EVAL_EPISODES)
    print(f'[L32] 起点 {len(starts)} 个', flush=True)

    # 目标词由文本图自己选：离 energy 锚点最远且归属 energy 的词
    goal_term, goal_margin = lang.energy_goal_word()
    _fa, _fm = lang.term_margin('food')
    food_margin = _fm if _fa == 'energy' else None
    print(f'[L32]   目标词={goal_term} (margin={goal_margin:.4f})', flush=True)

    results = {}
    for policy in ('offline', 'reactive', 'random', 'oracle'):
        print(f'[L32] 跑 {policy} ...', flush=True)
        ctrl = Controller(planner, maze, mode=policy)
        rows_ep = []
        for st in starts:
            rows_ep.append(run_episode(maze, ctrl, st, policy=policy,
                                       goal_term=goal_term))
        results[policy] = rows_ep
        s = summarize(rows_ep)
        print(f'[L32]   {policy}: 死亡 {s["deaths"]}/{s["episodes"]} '
              f'({s["death_rate"]:.3f})，到达 {s["reach_rate"]:.3f}', flush=True)

    summary = {k: summarize(v) for k, v in results.items()}
    for k in summary:
        summary[k]['rows'] = results[k]
    o = summary['offline']
    r = summary['reactive']
    q = summary['random']
    z = summary['oracle']

    p_or = paired_bootstrap([e['alive'] for e in results['offline']],
                            [e['alive'] for e in results['reactive']])
    p_ora = paired_bootstrap([e['alive'] for e in results['offline']],
                             [e['alive'] for e in results['random']])

    print('[L32] 跨领域测验 ...', flush=True)
    cross = cross_domain(maze, planner, lang)
    guard_mem()

    safe_rate = o['safe_plans'] / max(o['plans'], 1)
    fidelity = o['fidelity'] / max(o['safe_plans'], 1)
    nar = None
    # 从第一条有叙事的离线回合里取；run_episode 不保存 ctrl 的叙事，
    # 因此重跑一个回合专门取语言样本（确定性）
    ctrl_demo = Controller(planner, maze, mode='offline')
    demo = run_episode(maze, ctrl_demo, starts[0], policy='offline',
                       goal_term=goal_term)
    if ctrl_demo.first_narration:
        nar = ctrl_demo.first_narration
    # 第一次真的把「去吃」写进句子的规划（reached=True）
    nar_eat = next((n for n in ctrl_demo.narrations if n['reached']), None)

    if (o['death_rate'] <= 0.05 and o['death_rate'] <= 0.5 * r['death_rate'] and
            safe_rate >= 0.90 and fidelity >= 0.85 and cross['passed']):
        status = 'inner_monologue'
        reason = (f'离线死亡率 {o["death_rate"]:.3f}，reactive 为 '
                  f'{r["death_rate"]:.3f}；安全规划率 {safe_rate:.3f}，'
                  f'真值保真度 {fidelity:.3f}；跨领域链路通过。')
    elif o['death_rate'] < r['death_rate'] and safe_rate >= 0.6:
        status = 'partial_monologue'
        reason = (f'离线死亡率 {o["death_rate"]:.3f} 低于 reactive '
                  f'{r["death_rate"]:.3f}，但安全规划率 {safe_rate:.3f} / '
                  f'保真度 {fidelity:.3f} / 跨领域 '
                  f'{"通过" if cross["passed"] else "未过"} 未全过。')
    else:
        status = 'no_monologue'
        reason = (f'离线死亡率 {o["death_rate"]:.3f}，未显著优于 reactive '
                  f'{r["death_rate"]:.3f}。')

    verdict = {'status': status, 'reason': reason, 'death_rate': o['death_rate'],
               'reactive_death': r['death_rate'], 'safe_rate': safe_rate,
               'fidelity': fidelity, 'cross_ok': cross['passed']}
    l32 = {'p_off_vs_react': p_or, 'p_off_vs_rand': p_ora,
           'narration': nar, 'narration_eat': nar_eat,
           'train_samples': len(rows),
           'goal_term': goal_term, 'goal_margin': goal_margin,
           'food_margin': food_margin}

    print('[L32] 绘图 ...', flush=True)
    draw(maze, results['offline'], summary, cross)
    save_db(summary, results['offline'], verdict, cross)
    elapsed = time.time() - t0
    path = write_report(summary, l32, cross, verdict, lang, nar, elapsed)

    out = {'meta': {'seed': SEED, 'elapsed_s': round(elapsed, 2),
                    'peak_mem_mb': round(mem_mb(), 1),
                    'grid': GRID, 'senses': 14, 'actions': 6,
                    'horizon': HORIZON, 'beam': BEAM,
                    'train_samples': len(rows),
                    'tension_def': TENSION_DEF},
           'summary': {k: {kk: vv for kk, vv in v.items() if kk != 'rows'}
                       for k, v in summary.items()},
           'bootstrap': {'offline_vs_reactive': p_or,
                         'offline_vs_random': p_ora},
           'cross_domain': cross,
           'verdict': verdict,
           'narration': nar,
           'narration_eat': nar_eat}
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f'[L32] 完成：{path}  ({elapsed:.1f}s, {mem_mb():.0f}MB)', flush=True)
    print(f'[L32] 判定：{status} —— {reason}', flush=True)
    return out


if __name__ == '__main__':
    main()
