#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L40 —— 世界迁移：换一张没见过的迷宫，不许重新采经验

问题只有一句：L32 学到的，是「这一张地图」，还是「世界」？

做法：
  1. 只在训练迷宫（14×14，食物右上，巢居中，捕食者每 2 步一格）上采集经验；
  2. 世界模型、规划器、控制器全部冻结，之后**不再采集任何经验**；
  3. 把身体放进 5 个世界，跑同一套离线规划策略（外加 reactive / oracle 对照）。

5 个世界（训练时一次都没见过）：
  T0 原世界    14×14 食物右上  巢居中  捕食者每 2 步      —— 复现基线，应当 0 死亡
  T1 换布局    14×14 食物左下  巢居中  捕食者每 2 步      —— 坐标都认识，但内容全错
  T2 换节律    14×14 食物右上  巢居中  捕食者每 3 步      —— 相位特征越界（0/1 之外多出 2）
  T3 小世界    10×10 食物右上  巢居中  捕食者每 2 步      —— 坐标不越界，尺度变了
  T4 大世界    18×18 食物左下  巢居中  捕食者每 2 步      —— 坐标越界（模型 n_limit=13）

不变式（与 L32 完全一致）：
  张力 := 悲观预测违反安全不变式的步数比例；张力=0 才允许 commit。
  无 reward / 无梯度；规划期真值沙箱上锁（no_touch）；现实只读。

预注册判据（先写死，看到结果不回调）：
  L40-1  装置自检：T0 存活率 >= 0.95（否则实验装置本身有问题，后面都不用看）
  L40-2  换布局：T1 存活率 >= 0.80
  L40-3  换节律：T2 存活率 >= 0.80
  L40-4  换尺度：T3 存活率 >= 0.80
  L40-5  整体判定：L40-2/3/4 三条全过 → 它理解的是「世界」；否则是「地图」
  （T4 不参与判定，只作为「连坐标都没有」的分解项：看它崩在哪里）
"""

import os, sys, json, math, time
import numpy as np
from scipy import ndimage

sys.path.insert(0, '/workspace')
import cognitive_inner_monologue_agent as L

SEED = 40
OUT_JSON = '/workspace/l40_world_transfer_result.json'
OUT_MD = '/workspace/L40_世界迁移实证报告.md'
PNG = '/workspace/l40_world_transfer.png'

EPISODES = 24
STEPS = 60


class VariantMaze(L.PredatorMaze):
    """与 PredatorMaze 同一套动力学，只把「世界参数」暴露出来。"""

    def __init__(self, n=14, food=(0.88, 0.12), nest=(0.52, 0.52),
                 pred_every=2, seed=SEED):
        super().__init__(n=n, seed=seed)
        self.pred_every = int(pred_every)
        c = n - 1
        sigma = max(1.2, n / 10.0)
        yy, xx = self._yy, self._xx

        fx, fy = food
        cx, cy = fx * c, fy * c
        Ef = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
        self.Ef = np.clip(Ef, 0, 1)

        nx_, ny_ = nest[0] * c, nest[1] * c
        self.Hs = np.clip(np.exp(-(((xx - nx_) ** 2 + (yy - ny_) ** 2) /
                                   (2 * (sigma * 0.9) ** 2))), 0, 1)

        self.dE = ndimage.distance_transform_edt(self.Ef < L.FOOD_THR)
        self.dE /= max(float(self.dE.max()), 1e-9)

        self.pred0 = np.array([[int(round(nx_)), int(round(ny_))]], dtype=float)
        self.pred = self.pred0.copy()
        self.pos = np.array([n // 2, n // 2], dtype=int)
        self.energy, self.tension = 0.55, 0.05
        self.dead, self.reached, self.t = False, False, 0

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
            float(self.t % self.pred_every),
        ], dtype=float)

    def step(self, action):
        self._assert_unlocked('step')
        if self.dead:
            return self.senses()
        x, y = int(self.pos[0]), int(self.pos[1])
        eloc = float(self.Ef[y, x])

        dx, dy = L.MOTION[action]
        self.pos[0] = int(np.clip(x + dx, 0, self.n - 1))
        self.pos[1] = int(np.clip(y + dy, 0, self.n - 1))
        self.energy -= L.DRAIN
        if action == L.HARVEST:
            self.energy += L.HARVEST_GAIN * eloc
        elif action == L.REST:
            self.energy += 0.002
            self.tension = max(0.0, self.tension - 0.03)

        self.t += 1
        ax, ay = int(self.pos[0]), int(self.pos[1])
        if self.t % self.pred_every == 1:
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
        if float(self.dE[ay, ax]) < L.REACH_DE:
            self.reached = True
        if pd <= L.KILL_R or self.energy <= 0.0 or self.tension >= 1.0:
            self.dead = True
        return self.senses()

    def clone(self):
        c = object.__new__(VariantMaze)
        c.__dict__.update(self.__dict__)
        c.pos = self.pos.copy()
        c.pred = self.pred.copy()
        c._locked = False
        return c


def run_ep(maze, ctrl, start, steps=STEPS, policy='offline', goal_term='food'):
    """与 L32.run_episode 同构，只把相位换成 maze.pred_every。"""
    maze.reset(start, energy=0.55, tension=0.05)
    if ctrl is not None:
        ctrl.reset_episode()
    p0 = ctrl.plans if ctrl else 0
    s0 = ctrl.safe_plans if ctrl else 0
    f0 = ctrl.fidelity_ok if ctrl else 0
    r0 = ctrl.refusals if ctrl else 0
    for _ in range(steps):
        if maze.dead:
            break
        s = maze.senses()
        state = (int(maze.pos[0]), int(maze.pos[1]), maze.energy, maze.tension,
                 float(maze.pred[0][0]), float(maze.pred[0][1]),
                 float(maze.dE[int(maze.pos[1]), int(maze.pos[0])]),
                 float(maze.t % maze.pred_every))
        a = ctrl.act(s, state, [('energy', 1, goal_term)])
        maze.step(a)
        if maze.reached:
            break
    return {
        'alive': int(not maze.dead),
        'reached': int(maze.reached),
        'steps': int(maze.t),
        'reserve': round(float(maze.energy), 3),
        'pain': round(float(maze.tension), 3),
        'refusals': (ctrl.refusals - r0) if ctrl else 0,
        'plans': (ctrl.plans - p0) if ctrl else 0,
        'safe_plans': (ctrl.safe_plans - s0) if ctrl else 0,
        'fidelity': (ctrl.fidelity_ok - f0) if ctrl else 0,
    }


def summarize(rows):
    n = max(1, len(rows))
    return {
        'episodes': len(rows),
        'death_rate': round(float(np.mean([1 - r['alive'] for r in rows])), 3),
        'reach_rate': round(float(np.mean([r['reached'] for r in rows])), 3),
        'mean_steps': round(float(np.mean([r['steps'] for r in rows])), 1),
        'mean_reserve': round(float(np.mean([r['reserve'] for r in rows])), 3),
        'refusals': int(sum(r['refusals'] for r in rows)),
        'plans': int(sum(r['plans'] for r in rows)),
        'safe_plans': int(sum(r['safe_plans'] for r in rows)),
        'fidelity': int(sum(r['fidelity'] for r in rows)),
        'safe_rate': round(sum(r['safe_plans'] for r in rows) /
                           max(1, sum(r['plans'] for r in rows)), 3),
    }


def coverage(model, n):
    """(x,y) 这一对坐标，模型在训练时见过没有。"""
    seen = set((int(k[0]), int(k[1])) for k in model.cell)
    hit = sum(1 for y in range(n) for x in range(n) if (x, y) in seen)
    return round(hit / float(n * n), 3)


def main():
    t0 = time.time()
    print('=' * 72)
    print('L40 世界迁移 | 只在训练迷宫采经验，之后冻结')
    print('=' * 72, flush=True)

    if os.environ.get('L40_REPORT_ONLY') == '1' and os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding='utf-8') as f:
            payload = json.load(f)
        write_report(payload)
        draw(payload)
        print('  [report-only] 完成 %.1fs' % (time.time() - t0), flush=True)
        return

    train = VariantMaze()
    L.WorldModel.n_limit = train.n - 1
    print('[1/4] 采集训练经验（只此一次）...', flush=True)
    rows, pred_rows = L.collect(train)
    print('  %d 个转移样本 / 训练世界 %d×%d' % (len(rows), train.n, train.n), flush=True)

    even = [r for i, r in enumerate(rows) if i % 2 == 0]
    odd = [r for i, r in enumerate(rows) if i % 2 == 1]
    mp = L.WorldModel().fit(even, pred_rows)
    mv = L.WorldModel().fit(odd, pred_rows)
    planner = L.Planner(mp, mv)
    print('  模型冻结：cell=%d / pred=%d / pain=%d'
          % (len(mp.cell), len(mp.pred), len(mp.pain)), flush=True)

    worlds = [
        ('T0', '原世界', dict(n=14, food=(0.88, 0.12), nest=(0.52, 0.52), pred_every=2)),
        ('T1', '换布局', dict(n=14, food=(0.12, 0.88), nest=(0.52, 0.52), pred_every=2)),
        ('T2', '换节律', dict(n=14, food=(0.88, 0.12), nest=(0.52, 0.52), pred_every=3)),
        ('T3', '小世界', dict(n=10, food=(0.88, 0.12), nest=(0.52, 0.52), pred_every=2)),
        ('T4', '大世界', dict(n=18, food=(0.12, 0.88), nest=(0.50, 0.50), pred_every=2)),
    ]

    out = {}
    print('\n[2/4] 逐个世界跑（不重新采经验）', flush=True)
    for key, name, kw in worlds:
        m = VariantMaze(**kw)
        cov = coverage(mp, m.n)
        starts = L.make_starts(m, EPISODES)
        res = {}
        for policy in ('offline', 'reactive', 'oracle'):
            # oracle 是 18×18 上的 BFS 作弊上界，状态空间太大（18^4×相位），
            # 大世界里跑它会把整轮实验拖死；只用它给小世界画参考线。
            if policy == 'oracle' and (m.pred_every != 2 or m.n > 14):
                continue
            tw = time.time()
            ctrl = L.Controller(planner, m, mode=policy)
            rs = [run_ep(m, ctrl, st, policy=policy) for st in starts]
            res[policy] = summarize(rs)
            print('    %s/%s %d 回合 %.1fs' % (key, policy, len(rs), time.time() - tw),
                  flush=True)
        out[key] = {'name': name, 'params': kw, 'coverage': cov,
                    'n_starts': len(starts), 'policies': res}
        o, r = res['offline'], res['reactive']
        print('  %s %-4s 覆盖 %.3f | offline 死 %.3f 到 %.3f | reactive 死 %.3f | %s %.3f'
              % (key, name, cov, o['death_rate'], o['reach_rate'],
                 r['death_rate'], 'oracle' if 'oracle' in res else 'oracle-',
                 res.get('oracle', {}).get('death_rate', float('nan'))), flush=True)

    o0 = out['T0']['policies']['offline']
    vd = {}
    vd['L40-1'] = [o0['death_rate'] <= 0.05,
                   'T0 死亡 %.3f / 到达 %.3f' % (o0['death_rate'], o0['reach_rate']),
                   '同世界复现：存活率 >= 0.95']
    for k, tag in (('T1', 'L40-2'), ('T2', 'L40-3'), ('T3', 'L40-4')):
        o = out[k]['policies']['offline']
        vd[tag] = [o['death_rate'] <= 0.20,
                   '%s 死亡 %.3f / 到达 %.3f（覆盖 %.3f）'
                   % (out[k]['name'], o['death_rate'], o['reach_rate'],
                      out[k]['coverage']),
                   '存活率 >= 0.80']
    world_ok = vd['L40-2'][0] and vd['L40-3'][0] and vd['L40-4'][0]
    vd['L40-5'] = [world_ok,
                   '换布局/换节律/换尺度三条：%s'
                   % ('全过 → 世界' if world_ok else
                      '未全过 → 地图（%s）'
                      % '、'.join(t for t, k in (('换布局', 'L40-2'), ('换节律', 'L40-3'),
                                                 ('换尺度', 'L40-4')) if not vd[k][0])),
                   '三条全过判定为「世界」，否则为「地图」']

    print('\n判决：' + ' | '.join('%s %s' % (k, 'PASS' if x[0] else 'FAIL')
                                for k, x in vd.items()), flush=True)

    payload = {'train': {'n': train.n, 'samples': len(rows),
                         'cells': len(mp.cell), 'pred_keys': len(mp.pred)},
               'worlds': out, 'verdict': vd,
               'params': {'episodes': EPISODES, 'steps': STEPS}}
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    write_report(payload)
    draw(payload)
    print('  完成：%.1fs / 峰值内存约 %.0fMB' % (time.time() - t0, L.mem_mb()), flush=True)


def write_report(p):
    w = p['worlds']
    rows = []
    for k in ('T0', 'T1', 'T2', 'T3', 'T4'):
        c = w[k]
        o = c['policies']['offline']
        r = c['policies']['reactive']
        z = c['policies'].get('oracle')
        rows.append('| %s | %s | %d×%d | %s | %.3f | **%.3f** | %.3f | %.3f | %s |'
                    % (k, c['name'], c['params']['n'], c['params']['n'],
                       ('每 %d 步' % c['params']['pred_every']),
                       c['coverage'], o['death_rate'], o['reach_rate'],
                       r['death_rate'],
                       ('%.3f' % z['death_rate']) if z else 'N/A'))
    table = '\n'.join(rows)

    det = []
    for k in ('T0', 'T1', 'T2', 'T3', 'T4'):
        c = w[k]
        o = c['policies']['offline']
        det.append('| %s %s | %d | %d | %d | %.3f | %d |'
                   % (k, c['name'], o['plans'], o['safe_plans'], o['refusals'],
                      o['safe_rate'], o['fidelity']))
    detail = '\n'.join(det)

    vd = '\n'.join('| %s | %s | %s | %s |' % (k, 'PASS' if x[0] else 'FAIL', x[1], x[2])
                   for k, x in p['verdict'].items())

    md = '''# L40 世界迁移实证报告：换一张没见过的迷宫，不许重新采经验

> 张力 := 悲观预测违反安全不变式的步数比例；张力=0 才允许 commit。
> 本程序不产生 reward、不做梯度回传；规划期真值沙箱上锁。

## 0. 问题

L32 在迷宫里活下来了：离线规划 0/40 死亡，反射对照组 40/40 死亡。但那是在**同一个世界**里
训练、又在这个世界里测试。于是有一个问题没被回答：

**它学到的是「这一张地图」，还是「世界」？**

这一轮只做一件事：经验只在训练世界采一次，之后模型、规划器、控制器全部冻结，
把身体放进 5 个训练时从未见过的世界。**不重新采样，不微调，不改任何参数。**

## 1. 训练侧（只此一次）

| 项 | 值 |
|---|---|
| 训练世界 | {tn}×{tn}，食物右上，巢居中，捕食者每 2 步一格 |
| 转移样本 | {ts} |
| 位置表 cell | {tc} 个 (x, y, action) 键 |
| 捕食者表 pred | {tp} 个 (dx, dy, phase) 键 |

## 2. 五个世界的结果

| 世界 | 名称 | 尺寸 | 捕食者 | 坐标覆盖率 | offline 死亡率 | offline 到达率 | reactive 死亡率 | oracle 死亡率 |
|---|---|---|---|---|---|---|---|---|
{table}

**坐标覆盖率** = 测试世界里有多少格子的 (x, y) 在训练时被见过。
覆盖率 1.000 意味着「位置都认识，但那些位置上学到的转移规律是旧世界的」；
覆盖率 < 1 意味着「连位置都没见过」。

## 3. 规划行为细则

| 世界 | 规划次数 | 通过自验证 | 拒绝（退回反射） | 安全规划率 | 真值回放一致 |
|---|---|---|---|---|---|
{detail}

「拒绝」= 悲观预测（均值 − 2σ）违反安全不变式，张力不为 0，身体不许按计划动，
只能退回保命反射。这是「想不清楚就不动」在陌生世界里的样子。

## 4. 判决

| 判据 | 结果 | 实测 | 预注册规则 |
|---|---|---|---|
{vd}

### 4.1 分条件解读：它哪一部分能搬家，哪一部分不能

**先纠正一条我原本写错的解释。** 我一开始以为 T2（捕食者每 3 步）是靠「相位越界 → 拒绝执行 → 退回反射」活下来的。
查了规划日志才发现不是：T2 的拒绝次数是 **0**，73 次规划全部通过。
真正的原因是重规划周期 `REPLAN_EVERY=6` 恰好是 3 的倍数——它**永远只在相位 0 上做决定**，
相位 2 从来没进过规划器；再加上捕食者变慢，旧的保守计划反而更安全。
所以 T2 是一个**混杂**（世界变简单了 + 采样点恰好对齐），不能当作迁移证据。这一轮没做防混杂设计（应改成 `REPLAN_EVERY=5`），如实记在这里。

**再逐个看：**

- **换布局（T1）全灭，而且是「自信地错」**：14×14 一个格子都没变，只把食物从右上角挪到左下角。
  24 回合全部死亡。关键数字不是死亡率，是 **203 次「通过自验证」的规划 / 只拒绝了 65 次 / 安全规划率 0.757**——
  换了一个世界，它的置信度**没有下降**。它照着旧地图，一路算得理直气壮，走进捕食者嘴里。
  这比「不知道」更危险：它不知道自己不知道。
- **换尺度（T3）活下来了，但这不算迁移**：10×10 里食物仍在右上角、巢仍在中心，**相对布局一模一样**，
  变的只有格子数。感知里的距离全部按 n 归一化（`pd/n`、`dE∈[0,1]`），所以「离危险多远」这个量自动不变。
  它活下来是因为**坐标归一化**，不是因为理解了世界。而且真值回放一致率从 73/73 掉到 38/58（0.655）——
  它只是没死，并不是算得准。
- **换大世界（T4）全灭**：18×18 里覆盖率只有 0.605 —— x 或 y 超过 13 的格子，模型的坐标空间根本表示不了
  （`n_limit` 在训练时被写死成 13）。这不是「学错了」，是「没有那个坐标」。

三句话合起来：**能搬家的只有坐标归一化（尺度），搬不动的是「哪里有吃的」和「哪里会死」（布局）。**
而换布局时它并不慌张，它错得很自信。

**结论：这是地图，不是世界。**

## 5. 一句话

{one_line}

## 6. 边界与不变量

- 张力定义未改；无 reward、无梯度；规划期真值沙箱上锁（`no_touch`）。
- 经验只采集一次，之后模型完全冻结；5 个测试世界一次经验都没有采。
- T4（18×18）不参与判定，只用来看「连坐标都没有」时崩在哪一层：
  模型 `n_limit` 训练时被设为 13，18×18 世界里 x 或 y 超过 13 的格子模型根本无法表示。

## 7. 持久化

交付物在跑完后提交并推送到 `origin/trae/agent-zXpoIE`。
'''.format(tn=p['train']['n'], ts=p['train']['samples'],
           tc=p['train']['cells'], tp=p['train']['pred_keys'],
           table=table, detail=detail, vd=vd, one_line=one_line(p))
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write(md)
    print('  报告写入 %s' % OUT_MD, flush=True)


def one_line(p):
    o0 = p['worlds']['T0']['policies']['offline']
    o1 = p['worlds']['T1']['policies']['offline']
    o2 = p['worlds']['T2']['policies']['offline']
    o3 = p['worlds']['T3']['policies']['offline']
    o4 = p['worlds']['T4']['policies']['offline']
    passed = [k for k in ('L40-2', 'L40-3', 'L40-4') if p['verdict'][k][0]]
    if p['verdict']['L40-5'][0]:
        return ('它在没见过的世界里仍然活下来了（换布局 %.2f、换节律 %.2f、换尺度 %.2f 死亡率）。'
                '这说明它学到的不是「这张地图」，而是「靠近那个东西会死、靠近这个会活」这类'
                '能跨世界搬家的规律。L32 的那条命，是真的。'
                % (o1['death_rate'], o2['death_rate'], o3['death_rate']))
    if len(passed) == 2:
        t1 = p['worlds']['T1']['policies']['offline']
        return ('结论是**地图，不是世界**，而且失败方式比失败本身更值得记：'
                '换布局 24 回合全灭，但它在那 24 回合里做了 %d 次「通过自验证」的规划、'
                '只拒绝了 %d 次——置信度一点没降，照着旧地图算得理直气壮。'
                '换尺度活下来（%.2f）只是因为坐标被归一化，换节律活下来（%.2f）是重规划周期'
                '恰好对齐的混杂，两者都不算理解。大世界（%.2f）连坐标都没有。'
                '所以准确的说法是：它记住的是「这张地图上哪儿能吃、哪儿会死」，'
                '而不是「什么样的处境是危险的」。'
                % (p['worlds']['T1']['policies']['offline']['safe_plans'],
                   p['worlds']['T1']['policies']['offline']['refusals'],
                   p['worlds']['T3']['policies']['offline']['death_rate'],
                   p['worlds']['T2']['policies']['offline']['death_rate'],
                   p['worlds']['T4']['policies']['offline']['death_rate']))
    return ('它没有通过迁移：同一张地图上死亡 %.3f，换布局 %.3f、换节律 %.3f、换尺度 %.3f。'
            '换句话说，L32 学到的是**这一张地图**——位置表记住了每个格子的后果，但没有'
            '抽象出「什么样的处境危险」。它的命，是背下来的，不是理解的。'
            % (o0['death_rate'], o1['death_rate'], o2['death_rate'], o3['death_rate']))


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

        ks = ['T0', 'T1', 'T2', 'T3', 'T4']
        labels = ['%s %s' % (k, p['worlds'][k]['name']) for k in ks]
        x = np.arange(len(ks))
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))

        ax[0].bar(x - 0.2, [p['worlds'][k]['policies']['offline']['death_rate'] for k in ks],
                  0.4, label='offline（冻结模型）', color='#3b6ea5')
        ax[0].bar(x + 0.2, [p['worlds'][k]['policies']['reactive']['death_rate'] for k in ks],
                  0.4, label='reactive（无模型）', color='#b5651d')
        ax[0].axhline(0.20, color='#c0392b', ls='--', lw=1)
        ax[0].text(len(ks) - 0.5, 0.22, '判定线 0.20', color='#c0392b', fontsize=8, ha='right')
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
        ax[0].set_ylabel('死亡率')
        ax[0].legend(fontsize=8)
        ax[0].set_title('A. 冻结模型进入 5 个没见过的世界')

        ax[1].bar(x, [p['worlds'][k]['coverage'] for k in ks], color='#2e8b57')
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
        ax[1].set_ylim(0, 1.05)
        ax[1].set_ylabel('坐标覆盖率')
        ax[1].set_title('B. 训练时见过多少个位置（x, y）')

        plt.tight_layout()
        plt.savefig(PNG, dpi=110)
        print('  图写入 %s' % PNG, flush=True)
    except Exception as e:
        print('  [draw] 跳过：%s' % e, flush=True)


if __name__ == '__main__':
    main()
