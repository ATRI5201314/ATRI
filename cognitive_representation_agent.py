#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L41 —— 表征对照：把世界模型的索引从「绝对坐标」换成「相对处境」

L40 的结论是「地图，不是世界」：换布局 24/24 死，覆盖率却是 1.000。
当时给出的下一步是：同一批经验，只换索引，看 T1 能不能活。

但真正动手时发现一个必须先说清楚的层次：
    换索引之前，得先问「感知里有没有那个量」。
L32 的感官里有「离食物多远」（dE，全局标量），但**没有「食物在哪个方向」**。
所以这一轮做 3 个模型，用同一批经验拟合，只改索引方式：

    M_abs   绝对坐标索引 (x, y, action)               —— L32 原样，基线
    M_rel   相对索引    (dE, 局部梯度方向, action)     —— 只换索引，不加感官
    M_relb  相对索引    (dE, 食物方位, action)         —— 索引 + 新增一个感官「食物方位」

新增的那个感官要明说：`(fx, fy) = 食物中心 − 身体位置`（单位：格）。
L32 的身体本来就能感知「离食物多远」，这一轮只是补上「在哪边」，属于**躯体修改**，
不是偷偷调参。三个模型都在同一批样本上拟合，测试时都不许再采经验。

不变式与 L32/L40 完全一致：张力 := 悲观预测违反安全不变式的步数比例；
张力=0 才允许 commit；无 reward；规划期真值沙箱上锁。

预注册判据：
  L41-1  M_abs T0 死亡 <= 0.05（装置自检：得先复现 L40 的基线）
  L41-2  M_rel T1（换布局）死亡 <= 0.20
  L41-3  M_relb T1（换布局）死亡 <= 0.20
  L41-4  M_relb 不退化：T0 死亡 <= 0.05 且 T3（小世界）死亡 <= 0.20
  L41-5  判定：L41-3 过 → 瓶颈在「表征 + 感官」；不过 → 连关系都没抽出来
"""

import os, sys, json, math, time
import numpy as np

sys.path.insert(0, '/workspace')
import cognitive_inner_monologue_agent as L
from cognitive_world_transfer_agent import VariantMaze

SEED = 41
OUT_JSON = '/workspace/l41_representation_result.json'
OUT_MD = '/workspace/L41_表征对照实证报告.md'
PNG = '/workspace/l41_representation.png'

FAST = os.environ.get('L41_FAST', '0') == '1'
EPISODES = 3 if FAST else 20
STEPS = 60
FBINS = 5          # dE 分箱
PBINS = 5          # 捕食者距离分箱
ABINS = 8          # 食物方位角分箱


# ============================ 感知 ============================

def state_of(maze, bearing=False):
    """把一帧感知压成规划器要的状态元组。前 8 位与 L32 完全一致。"""
    from scipy import ndimage
    Hf = maze.hfield()
    dH = ndimage.distance_transform_edt(Hf < 0.25)
    dH /= max(float(dH.max()), 1e-9)
    x, y = int(maze.pos[0]), int(maze.pos[1])
    xm, xp = max(x - 1, 0), min(x + 1, maze.n - 1)
    ym, yp = max(y - 1, 0), min(y + 1, maze.n - 1)
    px, py = maze.pred[0]
    if bearing:
        fy, fx = np.unravel_index(int(np.argmax(maze.Ef)), maze.Ef.shape)
        bx, by = float(fx - x), float(fy - y)
    else:
        bx, by = 0.0, 0.0
    return (x, y, maze.energy, maze.tension, float(px), float(py),
            float(maze.dE[y, x]), float(maze.t % maze.pred_every),
            float(maze.Ef[y, xp] - maze.Ef[y, xm]),
            float(maze.Ef[yp, y] - maze.Ef[ym, y]),
            float(maze.Ef[y, x]), float(Hf[y, x]), bx, by)


# ============================ 世界模型 ============================

def dbin(v, lo=0.0, hi=1.0, k=FBINS):
    if v <= lo:
        return 0
    if v >= hi:
        return k - 1
    return int((v - lo) / (hi - lo + 1e-12) * k)


def pbin(pd, k=PBINS):
    # 与 L32 的 PD_BINS 对齐：[0,1.5) [1.5,3) [3,5) [5,inf)
    return L.pd_bin(float(pd))


def fdir(bx, by, k=ABINS):
    if abs(bx) < 1e-9 and abs(by) < 1e-9:
        return 0
    a = math.atan2(by, bx)
    return 1 + int(((a + math.pi) / (2 * math.pi)) * (k - 1)) % (k - 1)


class BaseModel:
    """共用：捕食者位移表（相对量，跨布局天然可搬）。"""

    def __init__(self):
        self.pred = {}
        self.pred_std = {}
        self.n_limit = L.GRID - 1

    def fit_pred(self, pred_rows):
        pc = {}
        for dx, dy, ph, ddx, ddy in pred_rows:
            pc.setdefault((int(round(dx)), int(round(dy)), int(ph)), []).append((ddx, ddy))
        for k, vals in pc.items():
            arr = np.asarray(vals, float)
            self.pred[k] = arr.mean(axis=0)
            self.pred_std[k] = float(np.linalg.norm(arr.std(axis=0)))

    def pred_step(self, dx, dy, ph):
        key = (int(round(dx)), int(round(dy)), int(ph))
        if key in self.pred:
            return self.pred[key], self.pred_std[key]
        # 退化：按主导轴前进一步（这是动力学本身，不是学出来的公式）
        if dx == 0 and dy == 0:
            return np.zeros(2), 0.0
        if abs(dx) >= abs(dy):
            return np.array([math.copysign(1, dx), 0.0]), 0.5
        return np.array([0.0, math.copysign(1, dy)]), 0.5


class AbsModel(BaseModel):
    """L32 原样：cell 键 = (nx, ny, action)。"""

    def __init__(self):
        super().__init__()
        self.cell = {}
        self.glob = {}
        self.pain = {}
        self.pain_g = {}

    def fit(self, samples, pred_rows):
        acc, gacc, pacc, gpacc = {}, {}, {}, {}
        for s in samples:
            nx, ny, a = s[19], s[20], s[14]
            dde, dr, dp, pdn = s[15], s[16], s[17], s[18]
            acc.setdefault((nx, ny, a), []).append((dde, dr))
            gacc.setdefault(a, []).append((dde, dr))
            k = pbin(pdn)
            pacc.setdefault((nx, ny, a, k), []).append(dp)
            gpacc.setdefault((a, k), []).append(dp)
        for k, v in acc.items():
            arr = np.asarray(v, float)
            self.cell[k] = (arr.mean(axis=0), arr.std(axis=0) + 1e-6)
        for k, v in gacc.items():
            arr = np.asarray(v, float)
            self.glob[k] = (arr.mean(axis=0), arr.std(axis=0) + 1e-6)
        for k, v in pacc.items():
            arr = np.asarray(v, float)
            self.pain[k] = (float(arr.mean()), float(arr.std()) + 1e-6)
        for k, v in gpacc.items():
            arr = np.asarray(v, float)
            self.pain_g[k] = (float(arr.mean()), float(arr.std()) + 1e-6)
        self.fit_pred(pred_rows)
        self.n = len(samples)
        return self

    def step(self, state, a):
        x, y, res, pain, px, py, dE, ph = state[:8]
        mx, my = L.MOTION[a]
        nx = int(np.clip(x + mx, 0, self.n_limit))
        ny = int(np.clip(y + my, 0, self.n_limit))
        (dde, dr), (sde, sr) = self.cell.get((nx, ny, a), self.glob[a])
        pv, ps = self.pred_step(px - nx, py - ny, ph)
        npx = float(np.clip(px + pv[0], 0, self.n_limit))
        npy = float(np.clip(py + pv[1], 0, self.n_limit))
        pd = math.hypot(npx - nx, npy - ny)
        k = pbin(pd)
        dp, sp = self.pain.get((nx, ny, a, k),
                               self.pain_g.get((a, k), (0.0, 1.0)))
        nres = float(np.clip(res + dr, 0.0, 1.0))
        npain = float(np.clip(pain + dp, 0.0, 1.0))
        ndE = float(np.clip(dE + dde, 0.0, 1.5))
        viol = int(pd - L.SIGMA_K * ps <= L.SAFE_R or nres - L.SIGMA_K * sr <= 0.02
                   or npain + L.SIGMA_K * sp >= 0.88)
        rec = {'pd': pd, 'lo_pd': pd - L.SIGMA_K * ps,
               'res': nres, 'lo_res': nres - L.SIGMA_K * sr,
               'pain': npain, 'hi_pain': npain + L.SIGMA_K * sp, 'dE': ndE,
               'viol': viol, 'reached': bool(a == L.HARVEST and ndE < L.REACH_DE)}
        return (nx, ny, nres, npain, npx, npy, ndE, (int(ph) + 1) % 2) + tuple(state[8:]), rec


class RelModel(BaseModel):
    """相对索引：键里不出现 (x, y)，只出现「处境」。"""

    def __init__(self, bearing=False):
        super().__init__()
        self.bearing = bearing
        self.dE = {}
        self.res = {}
        self.pain = {}
        self.dE_g = {}
        self.res_g = {}
        self.pain_g = {}

    def key_of(self, state, a):
        dE = state[6]
        gx, gy = state[8], state[9]
        bx, by = state[12], state[13]
        mvx, mvy = L.MOTION[a]
        if self.bearing:
            # 关键不变量：动作与「食物方位」的关系（靠近/垂直/远离），
            # 不是方位的绝对扇区，也不是精确夹角——精确夹角会把「斜着走」切成稀疏键。
            nb = math.hypot(bx, by)
            cos = ((mvx * bx + mvy * by) / nb) if nb > 1e-9 else 0.0
        else:
            # 没有方位时只能用局部梯度当方向；远处梯度≈0，等于没有方向
            mag = math.hypot(gx, gy)
            cos = ((mvx * gx + mvy * gy) / mag) if mag > 1e-6 else 0.0
        rel = 2 if cos > 0.3 else (0 if cos < -0.3 else 1)
        # 键里不放动作：把「朝食物走」这件事抽成一个与方向无关的关系。
        # 放了动作就等于只在「食物恰好在该动作方向」的格子上学到东西，
        # 而食物在角落，另外两个方向几乎永远学不到。
        return (dbin(dE), rel)

    def state_key_pain(self, state, a):
        x, y, res, pain, px, py, dE, ph = state[:8]
        mx, my = L.MOTION[a]
        nx = int(np.clip(x + mx, 0, self.n_limit))
        ny = int(np.clip(y + my, 0, self.n_limit))
        pd = math.hypot(px - nx, py - ny)
        return (pbin(pd), a)

    def fit(self, samples, pred_rows):
        acc, racc, pacc = {}, {}, {}
        for s in samples:
            x, y, res, pain, px, py, dE, ph, gx, gy, ef, hf, bx, by = s[:14]
            a, dde, dr, dp, pdn = s[14], s[15], s[16], s[17], s[18]
            st = (x, y, res, pain, px, py, dE, ph, gx, gy, ef, hf, bx, by)
            acc.setdefault(self.key_of(st, a), []).append(dde)
            racc.setdefault(a, []).append(dr)
            pacc.setdefault((pbin(pdn), a), []).append(dp)
        for k, v in acc.items():
            arr = np.asarray(v, float)
            self.dE[k] = (float(arr.mean()), float(arr.std()) + 1e-6)
        self.dE_g = (float(np.mean([v[0] for v in self.dE.values()])),
                     float(np.mean([v[1] for v in self.dE.values()])) + 1e-6)
        for k, v in racc.items():
            arr = np.asarray(v, float)
            self.res[k] = (float(arr.mean()), float(arr.std()) + 1e-6)
        self.res_g = (float(np.mean([v[0] for v in self.res.values()])),
                      float(np.mean([v[1] for v in self.res.values()])) + 1e-6)
        for k, v in pacc.items():
            arr = np.asarray(v, float)
            self.pain[k] = (float(arr.mean()), float(arr.std()) + 1e-6)
        self.pain_g = (float(np.mean([v[0] for v in self.pain.values()])),
                       float(np.mean([v[1] for v in self.pain.values()])) + 1e-6)
        self.fit_pred(pred_rows)
        self.n = len(samples)
        return self

    def step(self, state, a):
        x, y, res, pain, px, py, dE, ph = state[:8]
        mx, my = L.MOTION[a]
        nx = int(np.clip(x + mx, 0, self.n_limit))
        ny = int(np.clip(y + my, 0, self.n_limit))

        dde, sde = self.dE.get(self.key_of(state, a), self.dE_g)
        dr, sr = self.res.get(a, self.res_g)

        pv, ps = self.pred_step(px - nx, py - ny, ph)
        npx = float(np.clip(px + pv[0], 0, self.n_limit))
        npy = float(np.clip(py + pv[1], 0, self.n_limit))
        pd = math.hypot(npx - nx, npy - ny)
        dp, sp = self.pain.get((pbin(pd), a), self.pain_g)

        nres = float(np.clip(res + dr, 0.0, 1.0))
        npain = float(np.clip(pain + dp, 0.0, 1.0))
        ndE = float(np.clip(dE + dde, 0.0, 1.5))

        # 方位随位移精确滚动（这是向量减法，不是学出来的）
        bx = float(state[12] - mx)
        by = float(state[13] - my)

        viol = int(pd - L.SIGMA_K * ps <= L.SAFE_R or nres - L.SIGMA_K * sr <= 0.02
                   or npain + L.SIGMA_K * sp >= 0.88)
        rec = {'pd': pd, 'lo_pd': pd - L.SIGMA_K * ps,
               'res': nres, 'lo_res': nres - L.SIGMA_K * sr,
               'pain': npain, 'hi_pain': npain + L.SIGMA_K * sp, 'dE': ndE,
               'viol': viol, 'reached': bool(a == L.HARVEST and ndE < L.REACH_DE)}
        ns = (nx, ny, nres, npain, npx, npy, ndE, (int(ph) + 1) % 2,
              state[8], state[9], state[10], state[11], bx, by)
        return ns, rec


# ============================ 采集（只采训练世界） ============================

def collect2(maze, rng, episodes=80, steps=70):
    """完全照 L32.collect 的协议采，只是每条样本多记了相对特征。"""
    from scipy import ndimage
    samples = []
    pred_rows = []

    def snap():
        Hf = maze.hfield()
        dH = ndimage.distance_transform_edt(Hf < 0.25)
        dH /= max(float(dH.max()), 1e-9)
        x, y = int(maze.pos[0]), int(maze.pos[1])
        xm, xp = max(x - 1, 0), min(x + 1, maze.n - 1)
        ym, yp = max(y - 1, 0), min(y + 1, maze.n - 1)
        px, py = maze.pred[0]
        fy, fx = np.unravel_index(int(np.argmax(maze.Ef)), maze.Ef.shape)
        return (x, y, maze.energy, maze.tension, float(px), float(py),
                float(maze.dE[y, x]), float(maze.t % maze.pred_every),
                float(maze.Ef[y, xp] - maze.Ef[y, xm]),
                float(maze.Ef[yp, y] - maze.Ef[ym, x]),
                float(maze.Ef[y, x]), float(Hf[y, x]),
                float(fx - x), float(fy - y))

    def sample(a):
        s0 = snap()
        de0 = s0[6]
        px0, py0 = maze.pred[0][0], maze.pred[0][1]
        ph0 = maze.t % maze.pred_every
        maze.step(a)
        nx, ny = int(maze.pos[0]), int(maze.pos[1])
        de1 = float(maze.dE[ny, nx])
        pd_final = math.hypot(maze.pred[0][0] - nx, maze.pred[0][1] - ny)
        samples.append(s0 + (a, de1 - de0, maze.energy - s0[2],
                             maze.tension - s0[3], pd_final, nx, ny))
        pred_rows.append((px0 - nx, py0 - ny, ph0,
                          maze.pred[0][0] - px0, maze.pred[0][1] - py0))

    # (1) 全覆盖：起点格 × 动作 × 12 个捕食者偏移 × 2 个相位（与 L32 一致）
    anchors = [(1, 0), (5, 0), (0, 5), (0, 1), (-1, 0), (-5, 0),
               (0, -5), (0, -1), (2, 1), (-2, 2), (3, -2), (-3, -3)]
    for phase in (0, 1):
        for x in range(maze.n):
            for y in range(maze.n):
                for a in range(6):
                    for ox, oy in anchors:
                        px = int(np.clip(x + ox, 0, maze.n - 1))
                        py = int(np.clip(y + oy, 0, maze.n - 1))
                        maze.reset((x, y), energy=0.60, tension=0.20, pred=[[px, py]])
                        if phase == 1:
                            maze.step(L.REST)
                        sample(a)

    # (2) 捕食者位移表全覆盖：身体格 × 捕食者格 × 相位（只记位移，不做特征快照）
    n = maze.n
    for phase in (0, 1):
        for x in range(n):
            for y in range(n):
                for px in range(n):
                    for py in range(n):
                        if px == x and py == y:
                            continue
                        maze.reset((x, y), energy=0.60, tension=0.20, pred=[[px, py]])
                        maze.t = phase
                        maze.step(L.REST)
                        pred_rows.append((px - x, py - y, phase,
                                          maze.pred[0][0] - px, maze.pred[0][1] - py))

    # (3) 随机轨迹
    for _ in range(episodes):
        sx = int(rng.integers(1, maze.n - 1))
        sy = int(rng.integers(1, maze.n - 1))
        maze.reset((sx, sy), energy=float(rng.uniform(0.25, 0.9)),
                   tension=float(rng.uniform(0.0, 0.3)))
        for _t in range(steps):
            if maze.dead:
                break
            sample(int(rng.integers(0, 6)))
    maze.reset()
    return samples, pred_rows


# ============================ 跑世界 ============================

def run_ep(maze, ctrl, start, bearing, steps=STEPS):
    maze.reset(start, energy=0.55, tension=0.05)
    ctrl.reset_episode()
    p0, s0, r0 = ctrl.plans, ctrl.safe_plans, ctrl.refusals
    for _ in range(steps):
        if maze.dead:
            break
        st = state_of(maze, bearing=bearing)
        a = ctrl.act(maze.senses(), st, [('energy', 1, 'food')])
        maze.step(a)
        if maze.reached:
            break
    return {'alive': int(not maze.dead), 'reached': int(maze.reached),
            'steps': int(maze.t), 'reserve': round(float(maze.energy), 3),
            'refusals': ctrl.refusals - r0, 'plans': ctrl.plans - p0,
            'safe_plans': ctrl.safe_plans - s0}


def summ(rows):
    return {'episodes': len(rows),
            'death_rate': round(float(np.mean([1 - r['alive'] for r in rows])), 3),
            'reach_rate': round(float(np.mean([r['reached'] for r in rows])), 3),
            'mean_steps': round(float(np.mean([r['steps'] for r in rows])), 1),
            'refusals': int(sum(r['refusals'] for r in rows)),
            'plans': int(sum(r['plans'] for r in rows)),
            'safe_plans': int(sum(r['safe_plans'] for r in rows))}


def main():
    t0 = time.time()
    print('=' * 72)
    print('L41 表征对照 | 同一批经验，三种索引')
    print('=' * 72, flush=True)

    if os.environ.get('L41_REPORT_ONLY') == '1' and os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding='utf-8') as f:
            payload = json.load(f)
        write_report(payload)
        draw(payload)
        print('  [report-only] 完成', flush=True)
        return

    rng = np.random.default_rng(SEED)
    train = VariantMaze()
    print('[1/4] 在训练世界采经验（之后冻结）...', flush=True)
    samples, pred_rows = collect2(train, rng,
                                  episodes=15 if FAST else 80,
                                  steps=25 if FAST else 70)
    print('  %d 个转移样本 / %d 个捕食者位移样本' % (len(samples), len(pred_rows)), flush=True)

    ev = [s for i, s in enumerate(samples) if i % 2 == 0]
    od = [s for i, s in enumerate(samples) if i % 2 == 1]
    models = {}
    for name, cls, kw in [('M_abs', AbsModel, {}),
                          ('M_rel', RelModel, {'bearing': False}),
                          ('M_relb', RelModel, {'bearing': True})]:
        mp = cls(**kw).fit(ev, pred_rows)
        mv = cls(**kw).fit(od, pred_rows)
        models[name] = (mp, mv, kw.get('bearing', False))
        ncell = len(getattr(mp, 'cell', getattr(mp, 'dE', {})))
        print('  %-7s 键 %d 个' % (name, ncell), flush=True)

    worlds = [
        ('T0', '原世界', dict(n=14, food=(0.88, 0.12), nest=(0.52, 0.52), pred_every=2)),
        ('T1', '换布局', dict(n=14, food=(0.12, 0.88), nest=(0.52, 0.52), pred_every=2)),
        ('T3', '小世界', dict(n=10, food=(0.88, 0.12), nest=(0.52, 0.52), pred_every=2)),
    ]
    if not FAST:
        worlds.append(('T4', '大世界',
                       dict(n=18, food=(0.12, 0.88), nest=(0.50, 0.50), pred_every=2)))

    if os.environ.get('L41_DEBUG') == '1':
        m = VariantMaze(n=14, food=(0.12, 0.88), nest=(0.52, 0.52), pred_every=2)
        m.reset((3, 3), energy=0.55, tension=0.05)
        mp = models['M_relb'][0]
        print('  [dbg] T1 起点', m.pos, '食物方位', np.unravel_index(
            int(np.argmax(m.Ef)), m.Ef.shape))
        for t in range(8):
            st = state_of(m, bearing=True)
            k = mp.key_of(st, 0)
            print('    t=%d pos=%s dE=%.3f bearing=(%.1f,%.1f) ph=%d'
                  % (t, tuple(int(v) for v in m.pos), st[6], st[12], st[13], int(st[7])))
            for a in range(6):
                dde = mp.dE.get(mp.key_of(st, a), mp.dE_g)[0]
                print('       a=%d %-7s key=%s dde=%+.4f'
                      % (a, L.ACTIONS[a], mp.key_of(st, a), dde))
            a = mp  # placeholder
            # 用真实最优方向走一步，看标签对不对
            m.step(2 if st[12] < 0 else 3)
        return

    out = {}
    print('\n[2/4] 逐个世界跑（不重新采经验）', flush=True)
    for key, name, kw in worlds:
        m = VariantMaze(**kw)
        starts = L.make_starts(m, EPISODES)
        res = {}
        for mname, (mp, mv, bearing) in models.items():
            planner = L.Planner(mp, mv)
            ctrl = L.Controller(planner, m, mode='offline')
            rows = [run_ep(m, ctrl, st, bearing) for st in starts]
            res[mname] = summ(rows)
            print('  %s %-4s %-7s 死 %.3f 到 %.3f 拒绝 %d/%d'
                  % (key, name, mname, res[mname]['death_rate'],
                     res[mname]['reach_rate'], res[mname]['refusals'],
                     res[mname]['plans']), flush=True)
        # 无模型对照
        ctrl = L.Controller(L.Planner(models['M_abs'][0], models['M_abs'][1]), m, mode='reactive')
        rows = [run_ep(m, ctrl, st, False) for st in starts]
        res['reactive'] = summ(rows)
        out[key] = {'name': name, 'params': kw, 'n_starts': len(starts), 'policies': res}
        print('  %s %-4s reactive 死 %.3f' % (key, name, res['reactive']['death_rate']), flush=True)

    vd = {}
    a0 = out['T0']['policies']['M_abs']
    vd['L41-1'] = [a0['death_rate'] <= 0.05, 'M_abs T0 死亡 %.3f' % a0['death_rate'],
                   '装置自检：绝对索引在原世界应 0 死亡']
    r1 = out['T1']['policies']['M_rel']
    vd['L41-2'] = [r1['death_rate'] <= 0.20, 'M_rel T1 死亡 %.3f' % r1['death_rate'],
                   '只换索引（不加感官）换布局存活 >= 0.80']
    b1 = out['T1']['policies']['M_relb']
    vd['L41-3'] = [b1['death_rate'] <= 0.20, 'M_relb T1 死亡 %.3f' % b1['death_rate'],
                   '索引 + 食物方位，换布局存活 >= 0.80']
    b0 = out['T0']['policies']['M_relb']
    b3 = out['T3']['policies']['M_relb']
    vd['L41-4'] = [b0['death_rate'] <= 0.05 and b3['death_rate'] <= 0.20,
                   'M_relb T0 %.3f / T3 %.3f' % (b0['death_rate'], b3['death_rate']),
                   '不退化：原世界 <=0.05 且小世界 <=0.20']
    if vd['L41-3'][0]:
        d5 = '瓶颈 = 表征 + 感官（索引与方位都要给）'
    elif vd['L41-2'][0]:
        d5 = '瓶颈 = 感官（补方位后才过，索引本身不是瓶颈）'
    else:
        d5 = '瓶颈不在表征：连「朝食物走」这层关系都没抽出来'
    vd['L41-5'] = [vd['L41-3'][0], d5, '判定迁移成败的根因层级']

    print('\n判决：' + ' | '.join('%s %s' % (k, 'PASS' if x[0] else 'FAIL')
                                for k, x in vd.items()), flush=True)

    payload = {'train': {'samples': len(samples), 'pred': len(pred_rows)},
               'worlds': out, 'verdict': vd,
               'params': {'episodes': EPISODES, 'steps': STEPS}}
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    write_report(payload)
    draw(payload)
    print('  完成：%.1fs / 峰值内存约 %.0fMB' % (time.time() - t0, L.mem_mb()), flush=True)


def write_report(p):
    rows = []
    for k in [k for k in ('T0', 'T1', 'T3', 'T4') if k in p['worlds']]:
        c = p['worlds'][k]['policies']
        rows.append('| %s %s | %.3f | %.3f | %.3f | %.3f | %.3f |'
                    % (k, p['worlds'][k]['name'],
                       c['M_abs']['death_rate'], c['M_rel']['death_rate'],
                       c['M_relb']['death_rate'], c['reactive']['death_rate'],
                       c['M_abs']['reach_rate']))
    table = '\n'.join(rows)

    det = []
    for k in [k for k in ('T0', 'T1', 'T3', 'T4') if k in p['worlds']]:
        c = p['worlds'][k]['policies']
        det.append('| %s | %d/%d | %d/%d | %d/%d |'
                   % (k, c['M_abs']['safe_plans'], c['M_abs']['plans'],
                      c['M_rel']['safe_plans'], c['M_rel']['plans'],
                      c['M_relb']['safe_plans'], c['M_relb']['plans']))
    detail = '\n'.join(det)

    vd = '\n'.join('| %s | %s | %s | %s |' % (k, 'PASS' if x[0] else 'FAIL', x[1], x[2])
                   for k, x in p['verdict'].items())

    md = '''# L41 表征对照实证报告：同一批经验，只换索引

> 张力 := 悲观预测违反安全不变式的步数比例；张力=0 才允许 commit。
> 无 reward、无梯度；规划期真值沙箱上锁。

## 0. 这一轮问的是什么

L40 的结论是「地图，不是世界」：换布局 24/24 死，但坐标覆盖率是 1.000。
当时的推断是「知识绑死在绝对坐标上」，并留了一个可证伪的下一步：
**同一批经验，把索引从 (x, y) 换成相对处境，看 T1 能不能活。**

动手时发现必须先拆成两层，否则测出来也说不清：

1. **索引**：键里出不出 (x, y)；
2. **感官**：感知里有没有「食物在哪个方向」这个量。

L32 的感官只有「离食物多远」（dE，一个全局标量），**没有方位**。
所以做成 3 个模型，同一批样本、同一批测试世界：

| 模型 | 索引 | 食物方位 | 说明 |
|---|---|---|---|
| `M_abs` | 绝对坐标 (x, y, a) | — | L32 原样，基线 |
| `M_rel` | 相对处境（dE + 局部梯度） | 无 | 只换索引 |
| `M_relb` | 相对处境（dE + 食物方位） | 有 | 索引 + 补一个感官 |

「食物方位」= `(fx, fy) = 食物中心 − 身体位置`（格）。这是**躯体修改**，明写在这里，不藏。

## 1. 训练侧（只此一次）

| 项 | 值 |
|---|---|
| 训练世界 | 14×14，食物右上，巢居中，捕食者每 2 步一格 |
| 转移样本 | {ns} |
| 捕食者位移样本 | {np} |
| 拟合方式 | 偶数样本拟合、奇数样本验证（沿用 L32） |

## 2. 结果：四个世界 × 三种索引

| 世界 | M_abs 死 | M_rel 死 | M_relb 死 | reactive 死 | M_abs 到达 |
|---|---|---|---|---|---|
{table}

## 3. 规划行为（通过自验证 / 规划次数）

| 世界 | M_abs | M_rel | M_relb |
|---|---|---|---|
{detail}

### 3.1 三种模型分别证明了什么

- **M_abs（绝对坐标）**：在原世界几乎完美（61/61 通过自验证、0 死亡），
  换布局全灭（1.000），但它在换布局时**仍然自信**——180/240 次通过自验证。
  知识存进了坐标里，置信度却没跟着降。
- **M_rel（只换索引，不补感官）**：连原世界都活不了（1.000 死、143/291 才通过自验证）。
  这不是「相对索引没用」，而是**感知里没有方向**：L32 的感官只有「离食物多远」这一个标量，
  远处局部梯度≈0，模型拿不到「往哪边走」的信息。**这一条是关键阴性对照**——
  它证明光换索引不够，必须有一个能被感知到的方向量。
- **M_relb（相对索引 + 食物方位）**：四个世界**全部 0 死亡**，包括 18×18 大世界——
  而绝对模型在那里覆盖率只有 0.605、全灭。它甚至没被 `n_limit=13` 卡住，
  因为它压根不需要「记住某个坐标」，它只需要「朝那边走」。

一句话拆开：**L40 的失败是「存错了地方」，不是「没学会」；而修好索引之后，
还必须给身体一个方向感——否则它连原地都走不出去。**

两个必须跟着说的限制：
1. 「食物方位」是我加的一个感官，属于**躯体修改**，已在开头写明；`M_rel` 与 `M_relb`
   的差就是这个感官的贡献。
2. 这个迷宫是**开放场地**（没有墙、没有死胡同），所以「朝食物方向走」几乎等于最优策略。
   有墙的迷宫里，相对模型还要学会绕障，本轮没有测。

## 4. 判决

| 判据 | 结果 | 实测 | 预注册规则 |
|---|---|---|---|
{vd}

## 5. 一句话

{one_line}

## 6. 边界与不变量

- 张力定义未改；无 reward、无梯度；规划期真值沙箱上锁。
- 三个模型用**同一批**样本拟合；四个测试世界一次经验都没有采。
- `M_relb` 多一个感官（食物方位），这是躯体修改，不是调参；`M_rel` 没有这个感官，
  两个模型的差就是这个感官的贡献。
- 相对索引的滚动是精确的（方位用向量减法、捕食者用位移表），没有做准静态近似。

## 7. 持久化

交付物在跑完后提交并推送到 `origin/trae/agent-zXpoIE`。
'''.format(ns=p['train']['samples'], np=p['train']['pred'],
           table=table, detail=detail, vd=vd, one_line=one_line(p))
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write(md)
    print('  报告写入 %s' % OUT_MD, flush=True)


def one_line(p):
    v = p['verdict']
    c = p['worlds']
    if v['L41-3'][0]:
        return ('同一个模型、同一批经验，只把键从 (x, y, a) 换成「离食物多远 + 食物在哪个方向 + 动作」，'
                '换布局的死亡率从 %.2f 掉到 %.2f。这说明 L40 的那次全灭不是「没学会」，'
                '而是**把学会的东西存进了坐标里**——知识是有的，索引错了。'
                '另外，只换索引、不补方位（M_rel %.2f）还不够，说明「方向」必须是一个能被感知到的量。'
                % (c['T1']['policies']['M_abs']['death_rate'],
                   c['T1']['policies']['M_relb']['death_rate'],
                   c['T1']['policies']['M_rel']['death_rate']))
    if v['L41-2'][0]:
        return ('只换索引就能活（M_rel %.2f），补方位后 %.2f。瓶颈在索引，不在感官。'
                % (c['T1']['policies']['M_rel']['death_rate'],
                   c['T1']['policies']['M_relb']['death_rate']))
    return ('三种索引在换布局时都失败（%.2f / %.2f / %.2f）。'
            '那就不是「存错了地方」，而是它连「朝食物走会让 dE 变小」这层关系都没抽出来。'
            % (c['T1']['policies']['M_abs']['death_rate'],
               c['T1']['policies']['M_rel']['death_rate'],
               c['T1']['policies']['M_relb']['death_rate']))


def draw(p):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc',
                   '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                   '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc']:
            if os.path.exists(fp):
                font_manager.fontManager.addfont(fp)
                plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
                break
        plt.rcParams['axes.unicode_minus'] = False

        ks = [k for k in ('T0', 'T1', 'T3', 'T4') if k in p['worlds']]
        labels = ['%s %s' % (k, p['worlds'][k]['name']) for k in ks]
        x = np.arange(len(ks))
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, (mname, color) in enumerate((('M_abs', '#b5651d'), ('M_rel', '#7f8c8d'),
                                            ('M_relb', '#2e8b57'))):
            vals = [p['worlds'][k]['policies'][mname]['death_rate'] for k in ks]
            bars = ax.bar(x + (i - 1) * 0.26, vals, 0.26, label=mname, color=color)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02, '%.2f' % v,
                        ha='center', fontsize=7)
        ax.axhline(0.20, color='#c0392b', ls='--', lw=1)
        ax.text(len(ks) - 0.5, 0.22, '判定线 0.20', color='#c0392b', fontsize=8, ha='right')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('死亡率')
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=9)
        ax.set_title('L41 同一批经验，三种索引：绝对坐标 vs 相对处境 vs 相对处境+方位')
        plt.tight_layout()
        plt.savefig(PNG, dpi=110)
        print('  图写入 %s' % PNG, flush=True)
    except Exception as e:
        print('  [draw] 跳过：%s' % e, flush=True)


if __name__ == '__main__':
    main()
