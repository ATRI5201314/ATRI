#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认知主体 · 对话版（第三步：真语言体系的涌现 / 交流的诞生）

目标（对应用户「第三步」指令 + 已确认的最小规格）：
  - 两个认知主体 A(说方) / B(听方) 处在同一张 64x64 像素世界里，各自独立账本。
  - 世界里放 K 个「信标」作为指称物（referent）。
  - 每回合：随机挑一个指称 r；A 看得见(靠像素里的高亮环) 它，B 看不见；
    A 从自己的符号表发一个符号 s（初始随机、无先验语义）；B 收到 s 后动一步。
  - B 走到 r -> 双方的预测被验证；走不到 -> 预测误差（张力）。
  - 零 reward：A 只按「(对哪个指称, 发哪个符号) 历史上让 B 走到了吗」的频次记账；
    B 只按「(这个符号) 历史上对应哪个位置」的频次记账。唯一内生信号 = 预测误差。

守住的六条不变量（与 L0-L13 一致）：
  I1 唯一信号=预测误差（无 reward、无 loss 目标）；
  I2 语言先验须可证伪（符号初始随机、无词表；跨种子会收敛到"不同但都对"的编码）；
  I3 无硬编码自我/反思/规划模块；I4 发/忍住是判断不是定时；I5 无搜索树；
  I6 感知来自像素（A 必须从自己的画面里"找"到指称，不许拿真值索引）。

与旧文件关系：cognitive_agent.py / cognitive_language_agent.py /
cognitive_pixel_agent.py 一律冻结；本文件只复用其几何与渲染，独立成篇。
依赖：numpy（无 CNN 重训，纯频次账本，秒级可跑）。
"""

import os
import sys
import math
import random
from collections import defaultdict, deque

import numpy as np

# 复用 L13 已冻结的连续世界几何与渲染工具（保持"同一张世界"的连续性）
from cognitive_pixel_agent import (
    IMG, DOM, WALLS, CHAOS, AGENT_R, STEP_LEN, MOVES, DELTA,
    C_FLOOR, C_WALL, C_CHAOS, C_AGENT, C_AGENT_CORE,
    write_png, upscale, set_seed,
)

# ============================================================ 常量
SEED = 0
K = 4                        # 信标数（指称物个数）
N = 6                        # 符号表大小（"词"的容量；初始无任何语义）
T_MOVE = 12                  # 每回合 B 最多移动步数
REACH_R = 0.10               # 进入此半径 -> 认为"走到了"该信标
P_CHAOS_LOCAL = 0.75         # 身处混沌区时动作被篡改的概率（沿用 L13）
SPAWN = [0.45, 0.45]         # B 每回合的固定出发点
SENDER_POS = [0.16, 0.16]    # A 的固定观察位（说方不动）

# 信标位置（开敞区，且从出发点(0.45,0.45)到每个信标的直线路径都绕开混沌区）
BEACONS = [(0.28, 0.82), (0.84, 0.16), (0.90, 0.42), (0.38, 0.16)]
BEACON_COLORS = [(225, 70, 70), (70, 190, 90), (80, 120, 240), (235, 205, 60)]
C_RING = (220, 0, 220)       # 只有"当前指称"才有的高亮环（A 靠它在像素里认出 r）

# 学习调度
ROUNDS = 3000
EPS0, EPS1 = 0.35, 0.02      # 说方的 ε（各自独立的探索欲）
HOLD_FRAC = 0.80             # L14-3：后 20% 才把"留出的指称"引入（零/一次样本）
WINDOWS = 12                 # 指标曲线分段数
DECAY = 0.99                 # 听方账本的时间遗忘：没被反复使用的旧约定会淡出（新指称因此能被快速接纳）
KNOW_MIN = 0.5               # 账本总量低于此 -> 视为"没听过这个词"（不动作 -> 触发揭晓校正）


def _fill(img, rect, color):
    x0, y0, x1, y1 = rect
    c0 = max(0, int(round(x0 * IMG))); c1 = min(IMG, int(round(x1 * IMG)))
    r0 = max(0, int(round(y0 * IMG))); r1 = min(IMG, int(round(y1 * IMG)))
    img[r0:r1, c0:c1] = color


def _disc(img, cx, cy, r, color, core=None):
    px, py = cx * IMG, cy * IMG
    pr = max(2.0, r * IMG)
    y0 = max(0, int(py - pr - 1)); y1 = min(IMG, int(py + pr + 2))
    x0 = max(0, int(px - pr - 1)); x1 = min(IMG, int(px + pr + 2))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
    img[y0:y1, x0:x1][d <= pr] = color
    if core is not None:
        img[y0:y1, x0:x1][d <= pr * 0.55] = core


def _dim(color, f=0.30):
    return tuple(int(c * f) for c in color)


# ============================================================ 像素世界（共享几何，含信标）
class World:
    def __init__(self):
        self.b = max(1, int(round(AGENT_R * IMG / DOM)))

    def blocked(self, x, y):
        if x < AGENT_R or x > DOM - AGENT_R or y < AGENT_R or y > DOM - AGENT_R:
            return True
        for (x0, y0, x1, y1) in WALLS:
            if x0 - AGENT_R <= x <= x1 + AGENT_R and y0 - AGENT_R <= y <= y1 + AGENT_R:
                return True
        return False

    def in_chaos(self, x, y):
        x0, y0, x1, y1 = CHAOS
        return x0 <= x <= x1 and y0 <= y <= y1

    def render(self, recv, show_beacons=False, target=None):
        """recv: B 的位置；show_beacons: 是否画出信标(A 视角)；target: 高亮哪个指称。"""
        img = np.empty((IMG, IMG, 3), dtype=np.uint8)
        img[:, :] = C_FLOOR
        x0, y0, x1, y1 = CHAOS
        _fill(img, (x0, y0, x1, y1), C_CHAOS)
        for w in WALLS:
            _fill(img, w, C_WALL)
        img[:self.b, :] = C_WALL; img[-self.b:, :] = C_WALL
        img[:, :self.b] = C_WALL; img[:, -self.b:] = C_WALL
        if show_beacons:
            for i, (bx, by) in enumerate(BEACONS):
                if i == target:
                    continue
                _disc(img, bx, by, 0.045, _dim(BEACON_COLORS[i]))
            if target is not None:
                bx, by = BEACONS[target]
                _disc(img, bx, by, 0.078, C_RING)               # 环底
                _disc(img, bx, by, 0.050, BEACON_COLORS[target])  # 亮芯
        _disc(img, recv[0], recv[1], AGENT_R, C_AGENT, C_AGENT_CORE)
        return img


def perceive_target(frame):
    """A 的感知：只从自己的像素画面里找那个"高亮环"（品红），反推是哪个指称。
    绝不使用真值索引 —— 这是 I6（感知来自像素）的守门。"""
    r = frame[:, :, 0].astype(int); g = frame[:, :, 1].astype(int); bl = frame[:, :, 2].astype(int)
    m = (r > 170) & (bl > 170) & (g < 100)
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    cx = xs.mean() / IMG; cy = ys.mean() / IMG
    return min(range(K), key=lambda i: (BEACONS[i][0] - cx) ** 2 + (BEACONS[i][1] - cy) ** 2)


# ============================================================ 两个主体（各自独立账本）
class Sender:
    """说方 A：唯一驱动=「我发这个符号，听方历史上走到了吗」的预测误差账本。无 reward。"""
    def __init__(self):
        self.succ = defaultdict(lambda: np.zeros(N))
        self.trial = defaultdict(lambda: np.zeros(N))
        self.usage = np.zeros(N)      # 全局：每个符号被用过多少次
        self.active = {}              # r -> 正在试的符号（该指称尚未解决时坚持它）

    def _solve_active(self, r, rng):
        """未解决的指称：认领一个"还没被别人占用的、最冷门"的符号并坚持发。
        这是把 L13「一次只问一个问题」搬到语言上：新人/新物出现时，先给它一个
        没被用过的词，坚持对着它问，让对方先建立起这个映射，再谈别的。
        在等冷门的符号里随机破平 -> 编码仍然"被谈出来"，不做种子固化。"""
        if r in self.active:
            return self.active[r]
        taken = {s for s in self.active.values()}
        cand = [s for s in range(N) if s not in taken] or list(range(N))
        lo = min(self.usage[s] for s in cand)
        cold = [s for s in cand if self.usage[s] <= lo + 1e-9]
        s = rng.choice(cold)
        self.active[r] = s
        return s

    def policy(self, r, rng, eps):
        p = (self.succ[r] + 1.0) / (self.trial[r] + 2.0)
        if self.succ[r].max() < 2:
            return self._solve_active(r, rng)          # 没打通(需≥2次确认) -> 认领冷门词并坚持
        if rng.random() < eps:
            return rng.randrange(N)
        best = p.max()
        cand = [s for s in range(N) if p[s] >= best - 1e-9]
        return rng.choice(cand)

    def predict_reach(self, r, s):
        return (self.succ[r][s] + 1.0) / (self.trial[r][s] + 2.0)

    def observe(self, r, s, reached):
        self.trial[r][s] += 1
        self.succ[r][s] += (1.0 if reached else 0.0)
        self.usage[s] += 1
        if self.succ[r][s] >= 2:
            self.active.pop(r, None)                   # 连续打通 -> 释放，交给常规 ε-greedy

    def dominant(self, r):
        p = (self.succ[r] + 1.0) / (self.trial[r] + 2.0)
        return int(np.argmax(p))


class Receiver:
    """听方 B：唯一驱动=「听到 s，它指向哪个指称」的预测误差。以「符号->指称」频次记账。
    无 reward、无词表先验。

    关键：B 看不见"哪个是目标"（画面里没有高亮环），但它知道候选信标的位置（共同知识）。
    s 到底指谁，只能靠回合结束时世界"揭晓"的真实指称来校正 —— 这就是 B 的预测误差来源。
    记账用频次（而非位置均值）：均值会把"同一符号被用在多个指称上"的历史抹成一个质心，
    质心离任何信标都≥REACH_R，导致 B 永远到不了、说方永远拿不到成功信号（冷启动死锁）。
    频次法在经典 Lewis 指称博弈里可自举收敛。"""
    def __init__(self):
        self.cnt = np.zeros((N, K))     # cnt[s][r]：发出 s 时，真实指称为 r 的频次

    def tick(self):
        self.cnt *= DECAY              # 每回合淡出：冷门的旧约定自行消退

    def know(self, s):
        return self.cnt[s].sum() > KNOW_MIN

    def infer(self, s):
        if not self.know(s):
            return None                  # 未知 -> 不动 -> 必然错过 -> 触发揭晓校正
        return list(BEACONS[int(np.argmax(self.cnt[s]))])

    def predict_err(self, s, true_r):
        """预测误差：我(据 s)以为它指向哪个指称 vs 世界揭晓的真值。"""
        if not self.know(s):
            return 1.0
        j = int(np.argmax(self.cnt[s]))
        return math.hypot(BEACONS[j][0] - BEACONS[true_r][0],
                          BEACONS[j][1] - BEACONS[true_r][1])

    def observe(self, s, true_r):
        self.cnt[s][true_r] += 1.0


# ============================================================ 物理：朝目标走 + 混沌篡改
def move_toward(world, pos, aim, rng):
    """朝 aim 迈一步；选让 dot(步向量, 朝向) 最大的可走方向。返回 (新位置, 是否撞墙)。"""
    if aim is None:
        return list(pos), False
    dx, dy = aim[0] - pos[0], aim[1] - pos[1]
    order = sorted(MOVES, key=lambda a: -(DELTA[a][0] * dx + DELTA[a][1] * dy))
    for a in order:
        if world.in_chaos(pos[0], pos[1]) and rng.random() < P_CHAOS_LOCAL:
            a = rng.choice(MOVES)                 # 混沌区：动作被不可观测地篡改
        ddx, ddy = DELTA[a]
        nx, ny = pos[0] + ddx * STEP_LEN, pos[1] + ddy * STEP_LEN
        if not world.blocked(nx, ny):
            pos = [nx, ny]
            return pos, False
    return list(pos), True


# ============================================================ 一个回合
def play_round(world, sender, receiver, rng, eps, closed=False, n_active=K):
    receiver.tick()                               # 每回合：听方账本先淡出一档（时间遗忘）
    r = rng.randrange(n_active)                   # n_active<K：留出指称 K-1 暂不出现（L14-3）
    # A 看世界（含信标 + 目标高亮环），从像素里认指称
    sframe = world.render(SPAWN, show_beacons=True, target=r)
    r_hat = perceive_target(sframe)
    if r_hat is None:
        r_hat = rng.randrange(K)                  # 感知失败 -> 只能说错话（真实存在的失败）
    if closed:
        # 闭麦基线：通道被切断。A 的信号与指称无关（s 随机），B 收不到信息 -> 固定瞎猜一个位置。
        s = rng.randrange(N)
        aim = list(BEACONS[0])
        pos = list(SPAWN)
        for _ in range(T_MOVE):
            if math.hypot(aim[0] - pos[0], aim[1] - pos[1]) < STEP_LEN * 0.5:
                break
            pos, _ = move_toward(world, pos, aim, rng)
        reached = math.hypot(pos[0] - BEACONS[r][0], pos[1] - BEACONS[r][1]) <= REACH_R
        return dict(r=r, r_hat=r_hat, s=s, reached=reached, tension=0.0,
                    percp=(r_hat == r), aim=aim, pos=pos)
    s = sender.policy(r_hat, rng, eps)
    # B 的视角里没有信标(看不见指称)，只有信号 s
    aim = receiver.infer(s)
    pos = list(SPAWN)
    pred_err = receiver.predict_err(s, r)
    for _ in range(T_MOVE):
        if aim is not None and math.hypot(aim[0] - pos[0], aim[1] - pos[1]) < STEP_LEN * 0.5:
            break
        pos, _ = move_toward(world, pos, aim, rng)
    reached = math.hypot(pos[0] - BEACONS[r][0], pos[1] - BEACONS[r][1]) <= REACH_R
    # ---- 回合结束：双方各自观测（不是 reward，只是"世界揭晓了什么"）
    receiver.observe(s, r)                         # B 校正"这个符号指向哪个指称"
    sender.observe(r_hat, s, reached)              # A 校正"这个符号管不管用"
    # 张力：A 对自己"预言 B 会不会走到"的误差 + B 对"指称在哪"的误差
    tension = abs(sender.predict_reach(r_hat, s) - (1.0 if reached else 0.0)) + pred_err
    return dict(r=r, r_hat=r_hat, s=s, reached=reached, tension=tension,
                percp=(r_hat == r), aim=None if aim is None else list(aim), pos=list(pos))


# ============================================================ 指标
def mutual_information(pairs):
    """I(s;r)：符号到底带了多少"指哪个东西"的信息。"""
    if not pairs:
        return 0.0
    import collections
    cnt = collections.Counter(pairs)
    ps = collections.Counter(s for s, _ in pairs)
    pr = collections.Counter(r for _, r in pairs)
    tot = len(pairs)
    mi = 0.0
    for (s, r), c in cnt.items():
        p = c / tot
        mi += p * math.log((p) / ((ps[s] / tot) * (pr[r] / tot)) + 1e-12)
    return max(0.0, mi)


def run(closed_mouth=False, rounds=ROUNDS, seed=SEED):
    """closed_mouth=True: 闭麦基线——B 收不到信号，只能固定瞎猜一个位置（信息上限≈1/K）。"""
    set_seed(seed)
    rng = random.Random(seed)
    world = World()
    sender = Sender()
    receiver = Receiver()
    hold_start = int(rounds * HOLD_FRAC)

    win_len = max(1, rounds // WINDOWS)
    reach_w, mi_w, ten_w, closed_w = [], [], [], []
    win_reach, win_mi, win_ten = [], [], []

    hold_first_hit = None
    hold_hits = hold_tries = 0
    percp_ok = 0

    for t in range(rounds):
        eps = EPS1 + (EPS0 - EPS1) * max(0.0, 1.0 - t / rounds)
        n_active = (K - 1) if t < hold_start else K   # 前80% 留出指称 K-1
        res = play_round(world, sender, receiver, rng, eps, closed=closed_mouth, n_active=n_active)
        if res['percp']:
            percp_ok += 1
        win_reach.append(1.0 if res['reached'] else 0.0)
        win_mi.append((res['s'], res['r']))
        win_ten.append(res['tension'])
        if t >= hold_start and res['r'] == K - 1:      # 留出指称的测试段
            hold_tries += 1
            if res['reached']:
                hold_hits += 1
                if hold_first_hit is None:
                    hold_first_hit = t - hold_start
        if (t + 1) % win_len == 0:
            reach_w.append(float(np.mean(win_reach))); win_reach = []
            mi_w.append(mutual_information(win_mi)); win_mi = []
            ten_w.append(float(np.mean(win_ten))); win_ten = []

    # 最终编码：A 的 r->s 与 B 的 s->(落到哪个信标)，一致性
    code = {}
    consist = 0
    for r in range(K):
        s = sender.dominant(r)
        if not receiver.know(s):
            code[r] = (s, None); continue
        j = int(np.argmax(receiver.cnt[s]))
        code[r] = (s, j)
        if j == r:
            consist += 1

    return dict(reach_w=reach_w, mi_w=mi_w, ten_w=ten_w, code=code,
                consist=consist / K, hold_hits=hold_hits, hold_tries=hold_tries,
                hold_first_hit=hold_first_hit, seed=seed, closed=closed_mouth,
                percp=percp_ok / rounds)


# ============================================================ 出图：指标曲线
def draw_chart(path, series, W=620, H=340):
    img = np.full((H, W, 3), 250, np.uint8)
    ml, mr, mt, mb = 58, 16, 18, 36
    img[mt:H - mb, ml:ml + 1] = (140, 140, 140)
    img[H - mb - 1:H - mb, ml:W - mr] = (140, 140, 140)
    vmax = 1e-6
    for _, _, vals in series:
        vmax = max(vmax, max(vals) if vals else 0.0)
    for name, color, vals in series:
        n = len(vals)
        for k in range(1, n):
            x0 = ml + int((W - ml - mr) * (k - 1) / (n - 1))
            y0 = H - mb - int((H - mt - mb) * vals[k - 1] / vmax)
            x1 = ml + int((W - ml - mr) * k / (n - 1))
            y1 = H - mb - int((H - mt - mb) * vals[k] / vmax)
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            for i in range(steps + 1):
                xx = x0 + (x1 - x0) * i // steps
                yy = y0 + (y1 - y0) * i // steps
                img[max(0, yy - 1):yy + 2, max(0, xx - 1):xx + 2] = color
    write_png(path, upscale(img, 1))
    return vmax


# ============================================================ 主实验
def main():
    sd = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else SEED
    print("=" * 74)
    print("认知主体 · 对话版 —— 第三步：真语言体系的涌现（双主体指称游戏）")
    print("=" * 74)
    print(f"  K(指称)={K}  N(符号表)={N}  rounds={ROUNDS}  每回合B最多走{T_MOVE}步")

    op = run(closed_mouth=False, seed=sd)
    cl = run(closed_mouth=True, seed=sd)

    print("\n[L14-1] 指称互信息 I(signal; referent) —— 符号是否真带上了『指哪个』的信息")
    print("   开麦 MI 轨迹: " + " ".join(f"{v:.2f}" for v in op['mi_w']))
    print("   闭麦 MI 轨迹: " + " ".join(f"{v:.2f}" for v in cl['mi_w']))
    print(f"   [I6守门] 说方仅凭像素认出指称的正确率 = {op['percp']:.2f}（不许用真值索引）")

    print("\n[L14-2] 约定收敛度 —— A 的 r->s 与 B 的 s->位置 是否收敛一致")
    print(f"   开麦 到达率轨迹: " + " ".join(f"{v:.2f}" for v in op['reach_w']))
    print(f"   闭麦 到达率轨迹: " + " ".join(f"{v:.2f}" for v in cl['reach_w']))
    print(f"   开麦 最终编码一致度 = {op['consist']:.2f}   (每种指称都被B正确指向的比例)")
    print("   开麦 最终词典 r->s : " + "  ".join(
        f"{r}->s{s}(得{j})" for r, (s, j) in op['code'].items()))
    print(f"   张力轨迹(应随语言建立而下降): " + " ".join(f"{v:.2f}" for v in op['ten_w']))

    print("\n[L14-3] 泛化/零样本 —— 前80%不出现的留出指称，后20%才引入")
    if op['hold_tries'] > 0:
        print(f"   留出指称 首次命中延迟={op['hold_first_hit']} 回合   测试段到达率="
              f"{op['hold_hits']}/{op['hold_tries']} = {op['hold_hits']/op['hold_tries']:.2f}")
    else:
        print("   留出指称未被抽样中（步数不够）")

    print("\n[基线对照] 闭麦(无通信) vs 开麦")
    print(f"   平均到达率  闭麦={np.mean(cl['reach_w']):.2f}  开麦={np.mean(op['reach_w']):.2f}"
          f"   (随机瞎猜理论上限≈1/K={1.0/K:.2f})")
    print(f"   平均互信息  闭麦={np.mean(cl['mi_w']):.2f}  开麦={np.mean(op['mi_w']):.2f}")

    # 跨种子：语言是"被发明"的（不同种子应收敛到不同但都管用的编码）
    print("\n[可证伪] 语言先验检验：跨种子是否收敛到'不同但都有效'的任意编码")
    maps = []
    for sd2 in (1, 2, 3):
        r2 = run(closed_mouth=False, seed=sd2)
        maps.append(tuple(sorted(r2['code'].items())))
        print(f"   seed={sd2}  到达率={np.mean(r2['reach_w']):.2f}  一致度={r2['consist']:.2f}  "
              f"词典=" + " ".join(f"{r}->s{s}" for r, (s, j) in r2['code'].items()))
    same = sum(1 for m in maps[1:] if m == maps[0])
    print(f"   与 seed1 完全相同的编码数 = {same}/{len(maps) - 1} "
          f"({'任意性成立：编码是被谈出来的，不是写死的' if same == 0 else '存在固化，需继续观察'})")

    draw_chart(os.path.join('/workspace', 'dialogue_emergence.png'),
               [("open reach", (40, 170, 70), op['reach_w']),
                ("closed reach", (200, 60, 60), cl['reach_w']),
                ("mutual info", (60, 100, 220), op['mi_w']),
                ("tension", (230, 140, 40), op['ten_w'])])
    print("\n[产物] dialogue_emergence.png -> /workspace")

    # ---------- 判定 ----------
    print("\n[判定]")
    ok1 = (np.mean(op['mi_w'][-3:]) > 0.5) and (np.mean(op['mi_w'][-3:]) > np.mean(cl['mi_w'][-3:]) + 0.3)
    ok2 = (np.mean(op['reach_w'][-3:]) > 0.8) and (np.mean(op['reach_w'][-3:]) > np.mean(cl['reach_w']) + 0.3)
    ok3 = (op['hold_tries'] > 0 and (op['hold_first_hit'] is not None and op['hold_first_hit'] <= 30))
    ok4 = (op['consist'] >= 0.75)
    print(f"   P14-1 互信息涌现={'✓' if ok1 else '✗'}  "
          f"P14-2 到达率/收敛={'✓' if ok2 else '✗'}")
    print(f"   P14-3 零样本新指称={'✓' if ok3 else '✗'}  "
          f"P14-4 编码一致度={'✓' if ok4 else '✗'}")
    if all([ok1, ok2, ok3, ok4]):
        print("   结论：无 reward、无词表的双主体，自己谈出了一套可用的指称语言。")
    else:
        print("   结论：部分成立，需继续调参/加强（见各 ✗ 项）。")


if __name__ == '__main__':
    main()
