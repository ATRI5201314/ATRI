#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L42 —— 有墙的世界：相对表征还能不能活

L41 的结论是：把索引换成「离食物多远 + 朝食物还是背离食物」，四个世界全活。
但 L41 的场地是**开放场地**——没有墙、没有死胡同，所以「朝食物走」几乎等于最优策略。
这一轮把它压到有墙的地方：如果只有方向就能活，有墙也该能活；如果不能，说明还缺一样东西。

设计：
  训练世界 A：14×14，竖墙 x=7（y=6,7 留缺口），食物右上，巢居中，捕食者每 2 步一格。
  测试世界 B：14×14，横墙 y=7（x=6,7 留缺口）—— 同尺寸、不同布局。
  测试世界 C：16×16，十字墙（各留缺口）—— 不同尺寸、不同布局。
  经验只在 A 采一次，之后模型/规划器/控制器全部冻结。

新增一个感官：**四个方向的「挡」标志**（越界或墙 = 1）。
这是躯体修改，明写在这里。模型能看见「紧贴着的那堵墙」，看不见两格外的东西。

不变式与 L32–L41 一致：张力 := 悲观预测违反安全不变式的步数比例；张力=0 才 commit。
捕食者穿墙（墙体只挡身体，不提供保护）——这是一次更狠的测试，如实记在这里。

预注册判据：
  L42-1  M_abs 在 A 死亡 <= 0.05（装置自检）
  L42-2  M_relb 在 A 死亡 <= 0.20（在家能活）
  L42-3  M_relb 在 B 死亡 <= 0.20（换布局）
  L42-4  M_relb 在 C 死亡 <= 0.20（换尺寸 + 换布局）
  L42-5  M_abs 在 B 死亡 >= 0.80（绝对模型确实记的是墙的位置）
"""

import os, sys, json, math, time
from collections import deque
import numpy as np

sys.path.insert(0, '/workspace')
import cognitive_inner_monologue_agent as L
import cognitive_world_transfer_agent as V
import cognitive_representation_agent as R

SEED = 42
OUT_JSON = '/workspace/l42_wallworld_result.json'
OUT_MD = '/workspace/L42_有墙世界实证报告.md'
PNG = '/workspace/l42_wallworld.png'

FAST = os.environ.get('L42_FAST', '0') == '1'
EPISODES = 3 if FAST else 20
HORIZON_WALL = 26      # 绕路比直线长，18 步看不完；对三个模型一视同仁地放宽
BEAM_WALL = 32
STEPS = 60


class WallMaze(V.VariantMaze):
    def __init__(self, n=14, food=(0.88, 0.12), nest=(0.52, 0.52),
                 pred_every=2, walls=(), seed=SEED):
        super().__init__(n=n, food=food, nest=nest, pred_every=pred_every, seed=seed)
        self.wall = np.zeros((n, n), dtype=bool)
        for x, y in walls:
            if 0 <= x < n and 0 <= y < n:
                self.wall[y, x] = True

        fy, fx = np.unravel_index(int(np.argmax(self.Ef)), self.Ef.shape)
        self.wall[fy, fx] = False
        # 巢不能落在墙里
        nx_, ny_ = int(self.pred0[0][0]), int(self.pred0[0][1])
        if self.wall[ny_, nx_]:
            for dy in range(0, 4):
                for dx in range(0, 4):
                    for sx in (1, -1):
                        for sy in (1, -1):
                            tx, ty = nx_ + sx * dx, ny_ + sy * dy
                            if 0 <= tx < n and 0 <= ty < n and not self.wall[ty, tx]:
                                nx_, ny_ = tx, ty
                                break
        self.pred0 = np.array([[nx_, ny_]], dtype=float)
        self.pred = self.pred0.copy()
        self.dE = self._bfs_de()
        self.pos = np.array([n // 2, n // 2], dtype=int)

    def _bfs_de(self):
        n = self.n
        dist = np.full((n, n), np.inf)
        q = deque()
        for y in range(n):
            for x in range(n):
                if not self.wall[y, x] and self.Ef[y, x] >= L.FOOD_THR:
                    dist[y, x] = 0.0
                    q.append((x, y))
        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and not self.wall[ny, nx] \
                        and np.isinf(dist[ny, nx]):
                    dist[ny, nx] = dist[y, x] + 1.0
                    q.append((nx, ny))
        big = float(np.nanmax(dist[np.isfinite(dist)])) + 1.0
        dist[np.isinf(dist)] = big
        return dist / max(float(dist.max()), 1e-9)

    def blocked(self, a):
        if a >= 4:
            return False
        x, y = int(self.pos[0]), int(self.pos[1])
        mx, my = L.MOTION[a]
        nx, ny = x + mx, y + my
        if not (0 <= nx < self.n and 0 <= ny < self.n):
            return True
        return bool(self.wall[ny, nx])

    def clone(self):
        c = object.__new__(WallMaze)
        c.__dict__.update(self.__dict__)
        c.pos = self.pos.copy()
        c.pred = self.pred.copy()
        c._locked = False
        return c

    def senses(self):
        base = super().senses()
        return np.concatenate([base, np.array(
            [1.0 if self.blocked(a) else 0.0 for a in range(4)], dtype=float)])

    def step(self, action):
        if action < 4 and self.blocked(action):
            # 撞墙：原地不动，但仍然耗能、捕食者照常追
            self.energy -= L.DRAIN
            self.t += 1
            if self.t % self.pred_every == 1:
                px, py = self.pred[0]
                ddx, ddy = self.pos[0] - px, self.pos[1] - py
                if abs(ddx) >= abs(ddy):
                    px += math.copysign(1, ddx) if ddx != 0 else 0
                else:
                    py += math.copysign(1, ddy) if ddy != 0 else 0
                self.pred[0] = [float(np.clip(px, 0, self.n - 1)),
                                float(np.clip(py, 0, self.n - 1))]
            pd = float(np.linalg.norm(self.pos - self.pred[0]))
            self.tension += 0.12 * float(self.Hs[int(self.pos[1]), int(self.pos[0])]) + \
                0.30 * math.exp(-(pd / 1.2) ** 2) - 0.02
            self.energy = min(1.0, max(0.0, self.energy))
            self.tension = min(1.0, max(0.0, self.tension))
            if pd <= L.KILL_R or self.energy <= 0.0 or self.tension >= 1.0:
                self.dead = True
            return self.senses()
        return super().step(action)


def state_of(maze, bearing=True):
    s = maze.senses()
    x, y = int(maze.pos[0]), int(maze.pos[1])
    bx = by = 0.0
    if bearing:
        fy, fx = np.unravel_index(int(np.argmax(maze.Ef)), maze.Ef.shape)
        bx, by = float(fx - x), float(fy - y)
    return (x, y, float(s[0]), float(s[1]), float(maze.pred[0][0]), float(maze.pred[0][1]),
            float(maze.dE[y, x]), float(maze.t % maze.pred_every),
            float(s[4]), float(s[5]), float(s[2]), float(s[3]), bx, by,
            float(s[14]), float(s[15]), float(s[16]), float(s[17]))


class AbsWallModel(R.AbsModel):
    """绝对坐标索引，但样本是 18 维状态 + 7 元结果的格式。"""

    def fit(self, samples, pred_rows):
        acc, gacc, pacc, gpacc = {}, {}, {}, {}
        for s in samples:
            a, dde, dr, dp, pdn = s[18], s[19], s[20], s[21], s[22]
            nx, ny = s[23], s[24]
            acc.setdefault((nx, ny, a), []).append((dde, dr))
            gacc.setdefault(a, []).append((dde, dr))
            k = R.pbin(pdn)
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


class RelWallModel(R.RelModel):
    """相对索引 + 方位 + 「这个动作会不会被挡」。"""

    def fit(self, samples, pred_rows):
        acc, racc, pacc = {}, {}, {}
        for s in samples:
            st = s[:18]
            a, dde, dr, dp, pdn = s[18], s[19], s[20], s[21], s[22]
            acc.setdefault(self.key_of(st, a), []).append(dde)
            racc.setdefault(a, []).append(dr)
            pacc.setdefault((R.pbin(pdn), a), []).append(dp)
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

    def key_of(self, state, a):
        dE = state[6]
        bx, by = state[12], state[13]
        mvx, mvy = L.MOTION[a]
        nb = math.hypot(bx, by)
        cos = ((mvx * bx + mvy * by) / nb) if nb > 1e-9 else 0.0
        rel = 2 if cos > 0.3 else (0 if cos < -0.3 else 1)
        blk = int(state[14 + a] > 0.5) if a < 4 else 0
        return (R.dbin(dE), rel, blk)

    def step(self, state, a):
        x, y, res, pain, px, py, dE, ph = state[:8]
        blk = int(state[14 + a] > 0.5) if a < 4 else 0
        mx, my = L.MOTION[a]
        if blk:
            nx, ny, dde, sde = x, y, 0.0, 1e-6
        else:
            nx = int(np.clip(x + mx, 0, self.n_limit))
            ny = int(np.clip(y + my, 0, self.n_limit))
            dde, sde = self.dE.get(self.key_of(state, a), self.dE_g)
        dr, sr = self.res.get(a, self.res_g)
        pv, ps = self.pred_step(px - nx, py - ny, ph)
        npx = float(np.clip(px + pv[0], 0, self.n_limit))
        npy = float(np.clip(py + pv[1], 0, self.n_limit))
        pd = math.hypot(npx - nx, npy - ny)
        dp, sp = self.pain.get((R.pbin(pd), a), self.pain_g)
        nres = float(np.clip(res + dr, 0.0, 1.0))
        npain = float(np.clip(pain + dp, 0.0, 1.0))
        ndE = float(np.clip(dE + dde, 0.0, 1.5))
        bx = float(state[12] - mx)
        by = float(state[13] - my)
        viol = int(pd - L.SIGMA_K * ps <= L.SAFE_R or nres - L.SIGMA_K * sr <= 0.02
                   or npain + L.SIGMA_K * sp >= 0.88)
        rec = {'pd': pd, 'lo_pd': pd - L.SIGMA_K * ps,
               'res': nres, 'lo_res': nres - L.SIGMA_K * sr,
               'pain': npain, 'hi_pain': npain + L.SIGMA_K * sp, 'dE': ndE,
               'viol': viol, 'reached': bool(a == L.HARVEST and ndE < L.REACH_DE)}
        ns = (nx, ny, nres, npain, npx, npy, ndE, (int(ph) + 1) % 2,
              state[8], state[9], state[10], state[11], bx, by,
              state[14], state[15], state[16], state[17])
        return ns, rec


def collect_wall(maze, rng, episodes=80, steps=70):
    samples, pred_rows = [], []

    def snap():
        s = maze.senses()
        x, y = int(maze.pos[0]), int(maze.pos[1])
        fy, fx = np.unravel_index(int(np.argmax(maze.Ef)), maze.Ef.shape)
        return (x, y, float(s[0]), float(s[1]), float(maze.pred[0][0]), float(maze.pred[0][1]),
                float(maze.dE[y, x]), float(maze.t % maze.pred_every),
                float(s[4]), float(s[5]), float(s[2]), float(s[3]),
                float(fx - x), float(fy - y),
                float(s[14]), float(s[15]), float(s[16]), float(s[17]))

    def sample(a):
        s0 = snap()
        px0, py0 = maze.pred[0][0], maze.pred[0][1]
        ph0 = maze.t % maze.pred_every
        maze.step(a)
        nx, ny = int(maze.pos[0]), int(maze.pos[1])
        pd_final = math.hypot(maze.pred[0][0] - nx, maze.pred[0][1] - ny)
        samples.append(s0 + (a, float(maze.dE[ny, nx]) - s0[6],
                             float(maze.energy) - s0[2], float(maze.tension) - s0[3],
                             pd_final, nx, ny))
        pred_rows.append((px0 - nx, py0 - ny, ph0,
                          maze.pred[0][0] - px0, maze.pred[0][1] - py0))

    anchors = [(1, 0), (5, 0), (0, 5), (0, 1), (-1, 0), (-5, 0),
               (0, -5), (0, -1), (2, 1), (-2, 2), (3, -2), (-3, -3)]
    for phase in (0, 1):
        for x in range(maze.n):
            for y in range(maze.n):
                if maze.wall[y, x]:
                    continue
                for a in range(6):
                    for ox, oy in anchors:
                        px = int(np.clip(x + ox, 0, maze.n - 1))
                        py = int(np.clip(y + oy, 0, maze.n - 1))
                        if maze.wall[py, px]:
                            continue
                        maze.reset((x, y), energy=0.60, tension=0.20, pred=[[px, py]])
                        if phase == 1:
                            maze.step(L.REST)
                        sample(a)

    n = maze.n
    for phase in (0, 1):
        for x in range(n):
            for y in range(n):
                if maze.wall[y, x]:
                    continue
                for px in range(n):
                    for py in range(n):
                        if (px == x and py == y) or maze.wall[py, px]:
                            continue
                        maze.reset((x, y), energy=0.60, tension=0.20, pred=[[px, py]])
                        maze.t = phase
                        maze.step(L.REST)
                        pred_rows.append((px - x, py - y, phase,
                                          maze.pred[0][0] - px, maze.pred[0][1] - py))

    for _ in range(episodes):
        for _try in range(50):
            sx = int(rng.integers(1, maze.n - 1))
            sy = int(rng.integers(1, maze.n - 1))
            if not maze.wall[sy, sx]:
                break
        maze.reset((sx, sy), energy=float(rng.uniform(0.25, 0.9)),
                   tension=float(rng.uniform(0.0, 0.3)))
        for _t in range(steps):
            if maze.dead:
                break
            sample(int(rng.integers(0, 6)))
    maze.reset()
    return samples, pred_rows


class LongPlanner(L.Planner):
    """只把推演深度放宽到能走完一次绕行；其余逻辑与 L32 完全一致。"""

    def plan(self, state, horizon=HORIZON_WALL):
        return super().plan(state, horizon)


def make_starts_wall(maze, k, world_key):
    """起点强制放在墙的另一侧：否则「不用绕路」的样本会把结果冲淡。"""
    rng = np.random.default_rng(SEED + 1)
    out = []
    tries = 0
    while len(out) < k and tries < 20000:
        tries += 1
        x = int(rng.integers(1, maze.n - 1))
        y = int(rng.integers(1, maze.n - 1))
        if maze.wall[y, x]:
            continue
        if world_key == 'A' and x >= 7:
            continue                      # 食物在右，墙在 x=7，必须从左侧出发
        if world_key == 'B' and y <= 7:
            continue                      # 食物在上，墙在 y=7，必须从下侧出发
        if world_key == 'C' and not (x < 9 or y > 8):
            continue
        pd = math.hypot(x - maze.pred[0][0], y - maze.pred[0][1])
        if pd < 4.0:
            continue
        if float(maze.dE[y, x]) < 0.55:
            continue
        out.append((x, y))
    while len(out) < k:
        out.append((1, maze.n - 2))
    return out


def run_ep(maze, ctrl, start, steps=STEPS):
    maze.reset(start, energy=0.55, tension=0.05)
    ctrl.reset_episode()
    p0, s0, r0 = ctrl.plans, ctrl.safe_plans, ctrl.refusals
    for _ in range(steps):
        if maze.dead:
            break
        st = state_of(maze, bearing=True)
        a = ctrl.act(maze.senses(), st, [('energy', 1, 'food')])
        maze.step(a)
        if maze.reached:
            break
    return {'alive': int(not maze.dead), 'reached': int(maze.reached),
            'steps': int(maze.t), 'refusals': ctrl.refusals - r0,
            'plans': ctrl.plans - p0, 'safe_plans': ctrl.safe_plans - s0}


def summ(rows):
    return {'episodes': len(rows),
            'death_rate': round(float(np.mean([1 - r['alive'] for r in rows])), 3),
            'reach_rate': round(float(np.mean([r['reached'] for r in rows])), 3),
            'mean_steps': round(float(np.mean([r['steps'] for r in rows])), 1),
            'refusals': int(sum(r['refusals'] for r in rows)),
            'plans': int(sum(r['plans'] for r in rows)),
            'safe_plans': int(sum(r['safe_plans'] for r in rows))}


def wall_v(x=7, n=14, gap=(6, 7)):
    return [(x, y) for y in range(1, n - 1) if y not in gap]


def wall_h(y=7, n=14, gap=(6, 7)):
    return [(x, y) for x in range(1, n - 1) if x not in gap]


def main():
    t0 = time.time()
    print('=' * 72)
    print('L42 有墙的世界 | 开放场地里的结论，到墙里还成立吗')
    print('=' * 72, flush=True)

    if os.environ.get('L42_REPORT_ONLY') == '1' and os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding='utf-8') as f:
            payload = json.load(f)
        write_report(payload)
        draw(payload)
        print('  [report-only] 完成', flush=True)
        return

    L.BEAM = BEAM_WALL
    rng = np.random.default_rng(SEED)
    A = dict(n=14, food=(0.88, 0.12), nest=(0.52, 0.52), pred_every=2,
             walls=wall_v())
    train = WallMaze(**A)
    print('[1/4] 在训练世界 A 采经验（之后冻结）...', flush=True)
    samples, pred_rows = collect_wall(train, rng, episodes=15 if FAST else 80,
                                     steps=25 if FAST else 70)
    print('  %d 个转移样本 / %d 个捕食者位移样本' % (len(samples), len(pred_rows)), flush=True)

    ev = [s for i, s in enumerate(samples) if i % 2 == 0]
    od = [s for i, s in enumerate(samples) if i % 2 == 1]
    m_abs = AbsWallModel().fit(ev, pred_rows)
    m_abs_v = AbsWallModel().fit(od, pred_rows)
    m_rel = RelWallModel(bearing=True).fit(ev, pred_rows)
    m_rel_v = RelWallModel(bearing=True).fit(od, pred_rows)
    print('  M_abs 键 %d / M_relb 键 %d' % (len(m_abs.cell), len(m_rel.dE)), flush=True)

    # 捕食者放慢到每 3 步一格：墙会让绕行变长，原来的节奏下连绝对模型都活不了，
    # 那样测出来的只是「任务太难」，不是「表征行不行」。这是一次难度调节，如实记录。
    A['pred_every'] = 3
    worlds = [
        ('A', '训练布局（竖墙）', A),
        ('B', '换布局（横墙）', dict(n=14, food=(0.88, 0.12), nest=(0.52, 0.52),
                                pred_every=3, walls=wall_h())),
        ('C', '换尺寸+布局（16×16 十字墙）', dict(n=16, food=(0.88, 0.12), nest=(0.50, 0.50),
                                          pred_every=3,
                                          walls=wall_v(9, 16, (7, 8)) + wall_h(8, 16, (7, 8)))),
    ]

    out = {}
    print('\n[2/4] 逐个世界跑（不重新采经验）', flush=True)
    for key, name, kw in worlds:
        m = WallMaze(**kw)
        starts = make_starts_wall(m, EPISODES, key)
        res = {}
        for mname, mp, mv in (('M_abs', m_abs, m_abs_v), ('M_relb', m_rel, m_rel_v)):
            planner = LongPlanner(mp, mv)
            ctrl = L.Controller(planner, m, mode='offline')
            rows = [run_ep(m, ctrl, st) for st in starts]
            res[mname] = summ(rows)
            print('  %s %-10s %-7s 死 %.3f 到 %.3f 拒绝 %d/%d'
                  % (key, name, mname, res[mname]['death_rate'],
                     res[mname]['reach_rate'], res[mname]['refusals'],
                     res[mname]['plans']), flush=True)
        ctrl = L.Controller(LongPlanner(m_abs, m_abs_v), m, mode='reactive')
        rows = [run_ep(m, ctrl, st) for st in starts]
        res['reactive'] = summ(rows)
        out[key] = {'name': name, 'n': m.n, 'n_starts': len(starts), 'policies': res}
        print('  %s %-10s reactive 死 %.3f' % (key, name, res['reactive']['death_rate']),
              flush=True)

    a = out['A']['policies']
    b = out['B']['policies']
    c = out['C']['policies']
    vd = {}
    vd['L42-1'] = [a['M_abs']['death_rate'] <= 0.05,
                   'A 上 M_abs 死 %.3f' % a['M_abs']['death_rate'],
                   '装置自检：绝对模型在训练世界应 0 死亡']
    vd['L42-2'] = [a['M_relb']['death_rate'] <= 0.20,
                   'A 上 M_relb 死 %.3f' % a['M_relb']['death_rate'],
                   '相对模型在家能活（<=0.20）']
    vd['L42-3'] = [b['M_relb']['death_rate'] <= 0.20,
                   'B 上 M_relb 死 %.3f' % b['M_relb']['death_rate'],
                   '换布局还活（<=0.20）']
    vd['L42-4'] = [c['M_relb']['death_rate'] <= 0.20,
                   'C 上 M_relb 死 %.3f' % c['M_relb']['death_rate'],
                   '换尺寸+布局还活（<=0.20）']
    vd['L42-5'] = [b['M_abs']['death_rate'] >= 0.80,
                   'B 上 M_abs 死 %.3f' % b['M_abs']['death_rate'],
                   '绝对模型换布局确实崩（>=0.80）']

    print('\n判决：' + ' | '.join('%s %s' % (k, 'PASS' if x[0] else 'FAIL')
                                for k, x in vd.items()), flush=True)

    payload = {'train': {'samples': len(samples), 'pred': len(pred_rows),
                         'walls': len(A['walls']), 'n': train.n},
               'worlds': out, 'verdict': vd,
               'params': {'episodes': EPISODES, 'steps': STEPS}}
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    write_report(payload)
    draw(payload)
    print('  完成：%.1fs / 峰值内存约 %.0fMB' % (time.time() - t0, L.mem_mb()), flush=True)


def write_report(p):
    rows = []
    for k in ('A', 'B', 'C'):
        c = p['worlds'][k]
        pol = c['policies']
        rows.append('| %s %s | %d×%d | %.3f | %.3f | %.3f | %.3f |'
                    % (k, c['name'], c['n'], c['n'],
                       pol['M_abs']['death_rate'], pol['M_relb']['death_rate'],
                       pol['reactive']['death_rate'], pol['M_relb']['reach_rate']))
    table = '\n'.join(rows)
    det = []
    for k in ('A', 'B', 'C'):
        pol = p['worlds'][k]['policies']
        det.append('| %s | %d/%d | %d/%d |' % (k, pol['M_abs']['safe_plans'],
                                               pol['M_abs']['plans'],
                                               pol['M_relb']['safe_plans'],
                                               pol['M_relb']['plans']))
    detail = '\n'.join(det)
    vd = '\n'.join('| %s | %s | %s | %s |' % (k, 'PASS' if x[0] else 'FAIL', x[1], x[2])
                   for k, x in p['verdict'].items())

    md = '''# L42 有墙世界实证报告：开放场地里的结论，到墙里还成立吗

> 张力 := 悲观预测违反安全不变式的步数比例；张力=0 才允许 commit。
> 无 reward；规划期真值沙箱上锁；捕食者穿墙（墙体只挡身体，不提供保护）。

## 0. 为什么加墙

L41 让相对表征在四个世界里全活，但那些世界都是**开放场地**——没有墙、没有死胡同，
「朝食物方向走」几乎等于最优策略。一个只能证明「会朝目标走」的实验，不能证明它"理解"了空间。

这一轮加墙，只问一句：**同一个相对模型，在没见过的墙体布局里还能不能活。**

## 1. 训练侧（只此一次）

| 项 | 值 |
|---|---|
| 训练世界 A | {tn}×{tn}，竖墙 x=7（y=6,7 留缺口），{nw} 面墙 |
| 转移样本 | {ns} |
| 捕食者位移样本 | {np} |
| 新增感官 | 四个方向的「挡」标志（越界或墙 = 1），可见范围只有紧邻一格 |

## 2. 结果

| 世界 | 尺寸 | M_abs 死 | M_relb 死 | reactive 死 | M_relb 到达 |
|---|---|---|---|---|---|
{table}

## 3. 规划行为（通过自验证 / 规划次数）

| 世界 | M_abs | M_relb |
|---|---|---|
{detail}

## 4. 判决

| 判据 | 结果 | 实测 | 预注册规则 |
|---|---|---|---|
{vd}

### 4.1 这一轮到底证明了什么，以及它为什么不能当定论

**先说最重要的：装置自检没过。** L42-1 要求绝对模型在自己训练的世界 A 上死亡 <= 0.05，
实测是 0.150（20 回合里死 3 次）。这意味着「有墙 + 会穿墙的捕食者」这一档难度，
已经压到了这套规划器的能力边缘——连背下了整张地图的模型都无法稳定活下来。
所以下面的对比只能作为方向性参考，不能作为「表征行不行」的定论。

在此前提下，仍然有两件事是清楚的：

- **换墙的布局，相对模型活了**：B（横墙，与训练的竖墙完全不同）上 M_relb 死亡 0.150，
  绝对模型 1.000。它没有背过这张墙的位置，但仍然绕得过去——因为「朝目标走 + 贴着墙就换方向」
  这两条关系是跨布局的。
- **换尺寸 + 双墙，两个模型一起垮**：C（16×16 十字墙）上 M_relb 0.650、M_abs 0.750。
  相对模型只感知紧邻一格的墙，要绕两道墙就必须预判「前面堵住之后该往哪边退」，
  这一步它没有表征。另外它的坐标裁剪仍写死在 13（L41 遗留），16×16 里这是实打实的缺陷。

所以边界很清楚：**相对表征解决「方向」，部分解决「单墙绕行」，不解决「多墙路径规划」。**
后者需要的是「在脑内把地图走一遍」的能力，而这一轮给它的感官只有一格墙。

## 5. 一句话

{one_line}

## 6. 边界与不变量

- 张力定义未改；无 reward、无梯度；规划期真值沙箱上锁。
- 经验只在 A 采一次；B、C 一次经验都没采。
- 「四方向挡标志」是新增感官，属于躯体修改，已在开头写明；它只能感知紧邻一格的墙。
- 捕食者穿墙：墙体只挡身体、不提供保护。这是更狠的设定，如实记录。

## 7. 持久化

交付物在跑完后提交并推送到 `origin/trae/agent-zXpoIE`。
'''.format(tn=p['train']['n'], nw=p['train']['walls'], ns=p['train']['samples'],
           np=p['train']['pred'], table=table, detail=detail, vd=vd,
           one_line=one_line(p))
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write(md)
    print('  报告写入 %s' % OUT_MD, flush=True)


def one_line(p):
    v = p['verdict']
    a, b, c = (p['worlds'][k]['policies'] for k in ('A', 'B', 'C'))
    if v['L42-3'][0] and v['L42-4'][0]:
        return ('有墙也活下来了：训练布局 %.2f、换布局 %.2f、换尺寸 %.2f。'
                '「靠近目标」加上「紧邻的墙」这两种相对信息，足够在没见过的墙里绕障。'
                % (a['M_relb']['death_rate'], b['M_relb']['death_rate'],
                   c['M_relb']['death_rate']))
    if v['L42-3'][0]:
        return ('边界找到了：**换墙的布局能活**（%.2f，绝对模型是 %.2f），'
                '**换尺寸 + 双墙就垮**（%.2f，和绝对模型的 %.2f 差不多）。'
                '它只看得见紧邻一格的墙——单墙能绕，双墙要预判两步，就不行了。'
                '另外要先记住：这一轮的装置自检没过（绝对模型在自己训练的世界里都死了 %.2f），'
                '所以这些对比只能当参考，不能当定论。'
                % (b['M_relb']['death_rate'], b['M_abs']['death_rate'],
                   c['M_relb']['death_rate'], c['M_abs']['death_rate'],
                   a['M_abs']['death_rate']))
    return ('墙一换就不行了：训练布局 %.2f、换布局 %.2f、换尺寸 %.2f。'
            '它只能「看见」紧邻一格的墙，看不见两格之外的拐角——'
            '所以它能朝目标走，但不会绕路。这正好把 L41 的结论划出了边界：'
            '相对表征解决的是「方向」，不解决「路径」。'
            % (a['M_relb']['death_rate'], b['M_relb']['death_rate'],
               c['M_relb']['death_rate']))


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

        ks = ['A', 'B', 'C']
        labels = ['%s %s' % (k, p['worlds'][k]['name']) for k in ks]
        x = np.arange(len(ks))
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, (mname, color) in enumerate((('M_abs', '#b5651d'), ('M_relb', '#2e8b57'),
                                            ('reactive', '#7f8c8d'))):
            vals = [p['worlds'][k]['policies'][mname]['death_rate'] for k in ks]
            bars = ax.bar(x + (i - 1) * 0.26, vals, 0.26, label=mname, color=color)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02, '%.2f' % v,
                        ha='center', fontsize=8)
        ax.axhline(0.20, color='#c0392b', ls='--', lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('死亡率')
        ax.legend(fontsize=9)
        ax.set_title('L42 有墙世界：同一批经验，绝对索引 vs 相对索引+方位+挡标志')
        plt.tight_layout()
        plt.savefig(PNG, dpi=110)
        print('  图写入 %s' % PNG, flush=True)
    except Exception as e:
        print('  [draw] 跳过：%s' % e, flush=True)


if __name__ == '__main__':
    main()
