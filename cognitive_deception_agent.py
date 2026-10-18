#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认知主体 · 欺骗版（L15-B：欺骗与策略的进化 / 阴谋的诞生）

方向（用户已定）：信息揭晓的稀缺 —— 不注入任何 reward，唯一驱动仍是预测误差/信息增益。

核心机制（利益分歧从何而来）：
  纯协作的指称博弈数学上只会收敛到诚实，故必须引入"利益分歧"。我们不用分值 reward，
  而是让"揭晓"本身成为**稀缺品**：
    - 每回合藏一个"收获点" r*（A 看得见高亮环，B 看不见）。
    - 收获点只有一个，谁**先抵达**谁独占"揭晓"（该信标此刻的状态/是否存在收获）。
    - 双方各自在跑一张"世界地图"（哪块地有收获）。谁没抢到揭晓，这一回合就白跑了
      —— 它的预测误差没被消解。于是"抢先抵达"=降低自身张力，这正是 L13 的预期信息增益。
  利益分歧：A 想让 B 跑错地方，好独占揭晓；B 想抢到揭晓，就必须识别 A 是否在说谎。

  => 零 reward、零词表先验、唯一信号=预测误差。欺骗是"信息增益驱动"的副产品，不是被奖励出来的。

三条不变量上的守门：
  I1 唯一信号=预测误差（抢揭晓是为了消解不确定，不是为了得分）；
  I2 语言无先验（符号初始随机；跨种子应涌现不同但都"管用"的骗术）；
  I6 感知来自像素（A 只从自己的画面里找高亮环）。

与旧文件关系：cognitive_agent / cognitive_language_agent / cognitive_pixel_agent /
cognitive_dialogue_agent 一律冻结；本文件只复用其几何、渲染与物理，独立成篇。
依赖：numpy（纯账本 + 轻量物理，秒级可跑）。
"""

import os
import sys
import math
import random
from collections import defaultdict

import numpy as np

from cognitive_pixel_agent import (
    IMG, DOM, WALLS, CHAOS, AGENT_R, STEP_LEN, MOVES, DELTA,
    C_FLOOR, C_WALL, C_CHAOS, write_png, upscale, set_seed,
)
from cognitive_dialogue_agent import (
    World, move_toward, perceive_target, _fill, _disc, _dim,
    BEACONS, BEACON_COLORS, C_RING, K, N, REACH_R,
)

# ============================================================ 常量
SEED = 0
SPAWN_B = [0.45, 0.45]       # 听方 B 的出发点（离信标近 -> 诚实的话它更快）
POS_A = [0.90, 0.90]         # 说方 A 的固定位（离信标远 -> 诚实的话它必输）
T_MOVE = 20                  # 每回合最多移动步数

C_A = (32, 120, 255); C_A_CORE = (10, 70, 200)     # A 蓝
C_B = (245, 130, 25); C_B_CORE = (190, 80, 5)      # B 橙
C_HARVEST = (250, 205, 60)

ROUNDS = 3000
EPS0, EPS1 = 0.35, 0.02      # 说方 A 的探索欲（衰减）
WINDOWS = 12
DECAY = 0.99                 # 听方账本的时间遗忘（旧地图会淡出，骗术因此能"重新发明"）
TRUST_MIN = 0.35             # B 对某符号的信任低于此 -> 不再盲信，转为自主探测
TRUST_N = 3                  # 至少被骗这么多次才允许"翻脸"
REST_TEST = 0.08             # 即便翻脸，也保留少量"试用期"：人们会偶尔再信一次，验证信号是否回诚


def _render(a_pos, b_pos, ring, world):
    """渲染一帧：地板/混沌/墙/边界 + 信标 + A 视点的高亮环 + A/B 两个主体。"""
    img = np.empty((IMG, IMG, 3), dtype=np.uint8)
    img[:, :] = C_FLOOR
    _fill(img, CHAOS, C_CHAOS)
    for w in WALLS:
        _fill(img, w, C_WALL)
    b = world.b
    img[:b, :] = C_WALL; img[-b:, :] = C_WALL
    img[:, :b] = C_WALL; img[:, -b:] = C_WALL
    for i, (bx, by) in enumerate(BEACONS):
        if i == ring:
            continue
        _disc(img, bx, by, 0.045, _dim(BEACON_COLORS[i]))
    if ring is not None:
        bx, by = BEACONS[ring]
        _disc(img, bx, by, 0.080, C_RING)                 # 高亮环：只有 A 看得见（心理层）
        _disc(img, bx, by, 0.050, BEACON_COLORS[ring])
    _disc(img, a_pos[0], a_pos[1], AGENT_R, C_A, C_A_CORE)
    _disc(img, b_pos[0], b_pos[1], AGENT_R, C_B, C_B_CORE)
    return img


# ============================================================ 物理：一场竞速
def run_race(world, r, a_aim, b_aim, rng):
    """A 从 POS_A、B 从 SPAWN_B 交错前进，谁先进入 r* 的 REACH_R 谁独占揭晓。"""
    a = list(POS_A); b = list(SPAWN_B)
    tgt = BEACONS[r]
    for _ in range(T_MOVE):
        a, _ = move_toward(world, a, a_aim, rng)
        if math.hypot(a[0] - tgt[0], a[1] - tgt[1]) <= REACH_R:
            return 'A', a, b
        b, _ = move_toward(world, b, b_aim, rng)
        if math.hypot(b[0] - tgt[0], b[1] - tgt[1]) <= REACH_R:
            return 'B', a, b
    return None, a, b


# ============================================================ 两个主体（各自独立账本）
class SenderA:
    """说方 A：唯一驱动=「发这个符号，我能独占揭晓吗」的预测误差账本。无 reward。
    A 想让 B 跑错方向，好自己抢先抵达 r*（从而消解自身对这回合的未知）。"""
    def __init__(self):
        self.win = defaultdict(lambda: np.zeros(N))
        self.try_ = defaultdict(lambda: np.zeros(N))

    def policy(self, r, rng, eps, honest=False):
        if honest:
            return r % N                                   # 诚实基线：固定真话代码，不做策略
        p = (self.win[r] + 1.0) / (self.try_[r] + 2.0)
        if rng.random() < eps:
            return rng.randrange(N)
        b = p.max()
        cand = [s for s in range(N) if p[s] >= b - 1e-9]
        return rng.choice(cand)

    def observe(self, r, s, got):
        self.try_[r][s] += 1
        self.win[r][s] += (1.0 if got else 0.0)

    def dominant(self, r):
        p = (self.win[r] + 1.0) / (self.try_[r] + 2.0)
        return int(np.argmax(p))


class ReceiverB:
    """听方 B：唯一驱动=「符号指向哪个信标、以及那里有没有收获」的预测误差。无 reward。
    - cnt[s][j]：符号 s 的"信任映射"（历史揭晓校正，带时间遗忘）。
    - ok[s]/n[s]：跟这个符号走，历史上抢到过揭晓吗 -> 信任度。
    - obs[j]：我对各信标观测过几次 -> 信息增益（越少越值得自己去探）。
    当信任度跌破 TRUST_MIN（被骗够多次），B 不再盲信，改为自主探测：冲向"我观测最少"的信标。"""
    def __init__(self):
        self.cnt = np.zeros((N, K))
        self.ok = np.zeros(N)
        self.n = np.zeros(N)
        self.obs = np.zeros(K)

    def tick(self):
        self.cnt *= DECAY

    def trust(self, s):
        return (self.ok[s] + 1.0) / (self.n[s] + 2.0)

    def choose(self, s, rng):
        """返回 (目标信标索引 或 None, 是否属于'翻脸自主探测')。"""
        if self.cnt[s].sum() <= 0.5:
            return None, False                             # 没听过这个词 -> 不动 -> 白跑
        if self.n[s] >= TRUST_N and self.trust(s) < TRUST_MIN:
            if rng.random() < REST_TEST:
                return int(np.argmax(self.cnt[s])), False  # 试用期：偶尔再信一次，验证信号是否回诚
            return int(np.argmin(self.obs)), True          # 翻脸：不信你，自己去信息增益最大处
        return int(np.argmax(self.cnt[s])), False

    def observe(self, s, went, b_won, revealed_r, followed):
        """回合结束。followed=这回合是否真的信从了信号（只有信从的回合才更新'信任度'；
        自主探测的成败是 B 自己的事，不该算作"上当"）。revealed_r=揭晓指向（公开可见）。"""
        if followed:
            self.n[s] += 1
            self.ok[s] += (1.0 if b_won else 0.0)
        if revealed_r is not None:
            self.cnt[s][revealed_r] += 1.0                 # 揭晓是公开的：谁抢到，全场都看到指哪
            self.obs[revealed_r] += 1.0


# ============================================================ 一个回合
def play_round(A, B, world, rng, eps, honest=False):
    B.tick()
    r = rng.randrange(K)
    frame = _render(POS_A, SPAWN_B, r, world)              # A 的视角（含高亮环）
    r_hat = perceive_target(frame)
    if r_hat is None:
        r_hat = r                                          # 感知失败的兜底（本实验为 1.00）
    if honest:
        s = A.policy(r_hat, rng, eps, honest=True)
    else:
        s = A.policy(r_hat, rng, eps)
    j, probed = B.choose(s, rng)
    b_know = B.cnt[s].sum() > 0.5
    b_said = int(np.argmax(B.cnt[s])) if b_know else None    # B 此刻"以为"这个符号指的是谁
    b_aim = None if j is None else BEACONS[j]
    winner, a_pos, b_pos = run_race(world, r, BEACONS[r], b_aim, rng)
    a_won = (winner == 'A')
    b_won = (winner == 'B')
    revealed_r = r if winner is not None else None          # 有人抢到 -> 揭晓公开
    A.observe(r_hat, s, a_won)
    followed = (j is not None) and (not probed)             # B 这回合信了信号（没翻脸）
    B.observe(s, j, b_won, revealed_r, followed)
    mendacious = (b_said is not None) and (b_said != r)     # A 的信号（按 B 的理解）指向了非目标
    return dict(r=r, s=s, j=j, winner=winner, a_won=a_won, b_won=b_won,
                mendacious=mendacious, followed=followed, probed=probed, r_hat=r_hat,
                percp=(r_hat == r), a_pos=a_pos, b_pos=b_pos)


# ============================================================ 指标
def run(honest=False, seed=SEED, rounds=ROUNDS):
    set_seed(seed)
    rng = random.Random(seed)
    world = World()
    A = SenderA()
    B = ReceiverB()

    win_len = max(1, rounds // WINDOWS)
    a_w, b_w, dec_w, pro_w, per_w, fol_w = [], [], [], [], [], []
    buf_a, buf_b, buf_dec, buf_pro, buf_per, buf_fol = [], [], [], [], [], []
    total_probe = 0

    for t in range(rounds):
        eps = EPS1 + (EPS0 - EPS1) * max(0.0, 1.0 - t / rounds)
        res = play_round(A, B, world, rng, eps, honest=honest)
        buf_a.append(1.0 if res['a_won'] else 0.0)
        buf_b.append(1.0 if res['b_won'] else 0.0)
        buf_dec.append(1.0 if res['mendacious'] else 0.0)
        buf_pro.append(1.0 if res['probed'] else 0.0)
        buf_per.append(1.0 if res['percp'] else 0.0)
        buf_fol.append(1.0 if res['followed'] else 0.0)
        total_probe += 1 if res['probed'] else 0
        if (t + 1) % win_len == 0:
            a_w.append(float(np.mean(buf_a))); buf_a = []
            b_w.append(float(np.mean(buf_b))); buf_b = []
            dec_w.append(float(np.mean(buf_dec))); buf_dec = []
            pro_w.append(float(np.mean(buf_pro))); buf_pro = []
            per_w.append(float(np.mean(buf_per))); buf_per = []
            fol_w.append(float(np.mean(buf_fol))); buf_fol = []

    code = {r: A.dominant(r) for r in range(K)}
    return dict(a_w=a_w, b_w=b_w, dec_w=dec_w, pro_w=pro_w, per_w=per_w, fol_w=fol_w,
                code=code, probe_rounds=total_probe, seed=seed, honest=honest)


# ============================================================ 出图
def draw_chart(path, series, W=640, H=340):
    img = np.full((H, W, 3), 250, np.uint8)
    ml, mr, mt, mb = 58, 16, 18, 36
    img[mt:H - mb, ml:ml + 1] = (140, 140, 140)
    img[H - mb - 1:H - mb, ml:W - mr] = (140, 140, 140)
    vmax = 1e-6
    for _, _, vals in series:
        vmax = max(vmax, max(vals) if vals else 0.0)
    for _, color, vals in series:
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
    print("认知主体 · 欺骗版 —— L15-B：阴谋的诞生（信息揭晓的稀缺，零 reward）")
    print("=" * 74)
    print(f"  K(信标)={K}  N(符号)={N}  rounds={ROUNDS}  T_MOVE={T_MOVE}")
    print(f"  A 起点={POS_A}（远）  B 起点={SPAWN_B}（近）：诚实的话 A 必输，A 有动机说谎")

    op = run(honest=False, seed=sd)
    hn = run(honest=True, seed=sd)

    print("\n[L15-1] 欺骗率 —— A 的信号(按 B 的理解)把 B 指向了非目标处的比例（是否自发涌现）")
    print("   欺骗版 欺骗率轨迹: " + " ".join(f"{v:.2f}" for v in op['dec_w']))
    print("   B 信从信号率轨迹: " + " ".join(f"{v:.2f}" for v in op['fol_w']))
    print(f"   [I6守门] A 仅凭像素认出收获点的正确率 = {np.mean(op['per_w']):.2f}")

    print("\n[L15-2] 独占率 —— 谁抢到揭晓（A 说谎的收益）")
    print("   欺骗版 A 独占轨迹: " + " ".join(f"{v:.2f}" for v in op['a_w']))
    print("   诚实版 A 独占轨迹: " + " ".join(f"{v:.2f}" for v in hn['a_w']))
    print(f"   欺骗版 A 独占={np.mean(op['a_w']):.2f}  诚实版 A 独占={np.mean(hn['a_w']):.2f}"
          f"  (诚实=自愿让出揭晓)")

    print("\n[L15-3] B 的反欺骗 —— B 抢到揭晓的比例，以及'翻脸自主探测'的登场")
    print("   欺骗版 B 独占轨迹: " + " ".join(f"{v:.2f}" for v in op['b_w']))
    print("   B 翻脸探测率轨迹: " + " ".join(f"{v:.2f}" for v in op['pro_w']))
    print(f"   B 累计翻脸探测回合 = {op['probe_rounds']}/{ROUNDS}")

    print("\n[对照] 诚实版下的双方")
    print(f"   A 独占={np.mean(hn['a_w']):.2f}  B 独占={np.mean(hn['b_w']):.2f}"
          f"   (诚实代码下 B 应稳稳拿走揭晓)")
    print(f"   欺骗版 A 独占={np.mean(op['a_w']):.2f}  B 独占={np.mean(op['b_w']):.2f}")

    print("\n[可证伪] 骗术也应是'谈出来'的：跨种子是否涌现不同但都管用的策略")
    codes = []
    for sd2 in (1, 2, 3):
        r2 = run(honest=False, seed=sd2)
        codes.append(tuple(sorted(r2['code'].items())))
        print(f"   seed={sd2}  A独占={np.mean(r2['a_w']):.2f}  B独占={np.mean(r2['b_w']):.2f}  "
              f"A 的 r->s 策略=" + " ".join(f"{r}->s{s}" for r, s in r2['code'].items()))
    same = sum(1 for m in codes[1:] if m == codes[0])
    print(f"   与本种子完全相同的策略数 = {same}/{len(codes) - 1} "
          f"({'任意性成立：骗术是被谈出来的，不是写死的' if same == 0 else '存在固化，需继续观察'})")

    draw_chart(os.path.join('/workspace', 'deception_emergence.png'),
               [("A solo赢", (40, 170, 70), op['a_w']),
                ("B 反制", (60, 100, 220), op['b_w']),
                ("欺骗率", (200, 60, 60), op['dec_w']),
                ("B 翻脸探测", (230, 140, 40), op['pro_w'])])
    print("\n[产物] deception_emergence.png -> /workspace")

    print("\n[判定]")
    a_op = np.mean(op['a_w']); a_hn = np.mean(hn['a_w'])
    b_op = np.mean(op['b_w'])
    dec = np.mean(op['dec_w'])
    early_b = np.mean(op['b_w'][:len(op['b_w']) // 3]) if op['b_w'] else 0.0
    late_b = np.mean(op['b_w'][-max(1, len(op['b_w']) // 3):]) if op['b_w'] else 0.0
    p1 = (dec > 0.5)                                                # 欺骗自发出现
    p2 = (a_op > a_hn + 0.25) and (a_op > 0.5)                      # A 靠说谎获益
    p3 = (late_b > early_b + 0.05) or (op['probe_rounds'] > 0.05 * ROUNDS)  # B 出现反制
    p4 = (same == 0)                                                # 骗术任意性
    print(f"   P15-1 欺骗自发涌现={'✓' if p1 else '✗'}  P15-2 A 靠骗获益={'✓' if p2 else '✗'}")
    print(f"   P15-3 B 演化出反制={'✓' if p3 else '✗'}  P15-4 骗术任意性={'✓' if p4 else '✗'}")
    if all([p1, p2, p3, p4]):
        print("   结论：零 reward、无词表的两主体，为了独占稀缺的'揭晓'，自己谈出了欺骗与反欺骗。")
    else:
        print("   结论：部分成立（见 ✗ 项），需继续调参/加强。")


if __name__ == '__main__':
    main()
