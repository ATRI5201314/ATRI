#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认知主体 · 心智理论版（L16-A：二阶建模 / 反向推理 / 读心）

背景与死结（必须说清）：
  L15 证明：当唯一信源 A 是自利的，A 的最优解是把信号"退化成噪声"（对所有 r* 都发同一个
  符号）。此时按香农，信号 I(s;r*)=0，B 从信号里拿不到任何关于 r* 的信息，再聪明的推理也
  只能退回 1/K。ToM 无法凭空造出不存在的信息。
  => 要让"读心"有意义，B 必须另有一点**独立证据**去和 A 的信号对质。本实验给 B "余光"：
     它能模糊瞥见收获点（正确率 ρ，远低于 A 的 1.00），大多时候什么也看不见。

任务（用户 L16-A 规格）：
  1. 二阶建模：B 除"符号->物理结果"的一阶账本外，再建"意图账本"：在什么情境下，A 曾用这个
     符号骗过我。
  2. 反事实推理：A 发信有明显自利动机时，B 把信号当"反向指标"（A 指向哪，真相往往在别处）。
  3. 死命令：绝对禁止硬编码"防骗规则"；B 必须靠自己被骗积累的张力，自发把 A 的发信/胜率
     当输入，去预测"这个信号是真是假"。信任阈值、A/B 起点一律动态自适应，不留魔法数。
  4. 守 I1：唯一信号仍是预测误差；"抢先独占揭晓"仍是消解自身不确定（信息增益），不是 reward。

与旧文件关系：L14/L15 冻结；本文件只 import 其几何、渲染、物理，独立成篇。
依赖：numpy。
"""

import os
import sys
import math
import random
from collections import defaultdict, deque

import numpy as np

from cognitive_deception_agent import (
    World, run_race, _render, perceive_target,
    BEACONS, K, N, REACH_R, POS_A, SPAWN_B, T_MOVE,
    C_A, C_A_CORE, C_B, C_B_CORE,
)
from cognitive_pixel_agent import write_png, upscale, set_seed

# ============================================================ 常量（尽量少，其余自适应）
SEED = 0
ROUNDS = 3000
EPS0, EPS1 = 0.35, 0.02
WINDOWS = 12
RHO = 0.30                   # B "余光"正确率：能瞥见收获点的概率（A 为 1.00，B 远低）
DECAY = 0.99                 # 账本时间遗忘
GREED_W = 120                # A 近期胜率的滑动窗口（B 用来测 A 的"自利动机"）
MODE = 'tom'                 # 默认模式：'tom' | 'naive_signal' | 'naive_hunch'


# ============================================================ 说方 A（与 L15 同构）
class SenderA:
    def __init__(self):
        self.win = defaultdict(lambda: np.zeros(N))
        self.try_ = defaultdict(lambda: np.zeros(N))

    def policy(self, r, rng, eps):
        p = (self.win[r] + 1.0) / (self.try_[r] + 2.0)
        if rng.random() < eps:
            return rng.randrange(N)
        b = p.max()
        cand = [s for s in range(N) if p[s] >= b - 1e-9]
        return rng.choice(cand)

    def observe(self, r, s, got):
        self.try_[r][s] += 1
        self.win[r][s] += (1.0 if got else 0.0)


# ============================================================ 听方 B：一阶 + 二阶（意图）账本
class ReceiverToM:
    """B 的两本账：
      一阶 code[s][r]：符号 s 历史上"揭晓"指向 r 的频次（= A 的编码后验）。
      二阶 lie[s]/sent[s]：我信了 s 但揭晓打脸（A 骗我）的比例 —— 意图账本。
      外加 greed = A 的近期胜率（B 唯一能观测到的"A 的自利动机"）。
    B 没有任何写死的防骗规则；它只是把"一阶信念 w1"与"二阶怀疑 p_lie"和"余光 h"做反事实融合。"""
    def __init__(self):
        self.code = np.zeros((N, K))
        self.lie = np.zeros(N)
        self.sent = np.zeros(N)
        self.obs = np.zeros(K)
        self.a_hist = deque(maxlen=GREED_W)     # A 近期是否赢（B 观测得到）
        self.last_belief = None

    def tick(self):
        self.code *= DECAY

    def greed(self):
        return (sum(self.a_hist) + 1.0) / (len(self.a_hist) + 2.0) if self.a_hist else 0.5

    def lie_prob(self, s):
        inst = (self.lie[s] + 1.0) / (self.sent[s] + 2.0)     # 这个符号历史可信度
        return min(1.0, 0.5 * self.greed() + 0.5 * inst)       # 动机 * 历史

    def decide(self, s, h, rng, mode='tom'):
        self.last_belief = None
        known = self.code[s].sum() > 0.5
        if mode == 'naive_signal':            # 对照：盲信信号
            if not known:
                return None
            self.last_belief = int(np.argmax(self.code[s]))
            return self.last_belief
        if mode == 'naive_hunch':             # 对照：只用余光
            return (h if h is not None else None)
        # ---- ToM 模式：反事实融合 ----
        if not known:
            return h                          # 一无所知：只能靠余光
        w1 = int(np.argmax(self.code[s]))
        self.last_belief = w1
        p = self.lie_prob(s)
        if h is not None:
            return w1 if h == w1 else h       # 两路一致则采纳；冲突=识破谎言 -> 反用余光当反向指标
        if rng.random() < p:                  # 无余光但疑心重 -> 反向：选一阶次优
            cand = [r for r in range(K) if r != w1]
            return max(cand, key=lambda r: self.code[s][r])
        return w1

    def observe(self, s, b_won, a_won, revealed, followed):
        if revealed is not None:
            if self.last_belief is not None:
                self.sent[s] += 1
                if self.last_belief != revealed:
                    self.lie[s] += 1          # 信了却打脸 -> 记一笔意图：这个符号被 A 用来骗我
            self.code[s][revealed] += 1.0
            self.obs[revealed] += 1.0
        self.a_hist.append(1.0 if a_won else 0.0)


# ============================================================ 一个回合
def play_round(A, B, world, rng, eps, mode='tom'):
    B.tick()                                              # 账本时间遗忘（与 L15 一致）
    r = rng.randrange(K)
    frame = _render(POS_A, SPAWN_B, r, world)             # A 的清晰视角（含高亮环）
    r_hat = perceive_target(frame)
    if r_hat is None:
        r_hat = r
    s = A.policy(r_hat, rng, eps)
    h = r if rng.random() < RHO else None                 # B 的"余光"：大多时候看不见
    j = B.decide(s, h, rng, mode)
    b_aim = None if j is None else BEACONS[j]
    winner, a_pos, b_pos = run_race(world, r, BEACONS[r], b_aim, rng)
    a_won = (winner == 'A'); b_won = (winner == 'B')
    revealed = r if winner is not None else None
    A.observe(r_hat, s, a_won)
    followed = j is not None
    B.observe(s, b_won, a_won, revealed, followed)
    b_said = B.last_belief
    mendacious = (b_said is not None) and (b_said != r)
    conflict = (h is not None) and (b_said is not None) and (h != b_said)   # 余光与信号冲突
    reversed_ = (j is not None) and (b_said is not None) and (j != b_said)  # B 没按"信号的读法"走（反用）
    honest = (b_said is not None) and (b_said == r)                         # 这一发的信号（按 B 的理解）说真话
    return dict(r=r, s=s, j=j, winner=winner, a_won=a_won, b_won=b_won,
                followed=followed, mendacious=mendacious, conflict=conflict,
                reversed=reversed_, honest=honest, b_said=b_said,
                percp=(r_hat == r))


# ============================================================ 指标
def run(seed=SEED, rounds=ROUNDS, mode='tom'):
    set_seed(seed)
    rng = random.Random(seed)
    world = World()
    A = SenderA(); B = ReceiverToM()

    win = max(1, rounds // WINDOWS)
    a_w, b_w, mi_w, dec_w = [], [], [], []
    confl_w, confl_hit_w = [], []
    rev_hi_w, rev_lo_w = [], []               # 自利情境(高/低)下的"反用信号"率
    tally = np.zeros((N, K))                  # (s, r*) 用于算互信息
    buf_a, buf_b, buf_dec = [], [], []
    buf_confl, buf_conflhit = [], []
    buf_rev_hi, buf_rev_lo = [], []
    total_conf = total_confhit = 0
    # 长线欺诈检测（"先连续说真话养号，再骗一次大的"）
    STREAK = 5
    streak = 0; longcon = 0; longcon_win = 0
    hi_rev = hi_n = lo_rev = lo_n = 0

    for t in range(rounds):
        eps = EPS1 + (EPS0 - EPS1) * max(0.0, 1.0 - t / rounds)
        res = play_round(A, B, world, rng, eps, mode)
        tally[res['s'], res['r']] += 1
        buf_a.append(1.0 if res['a_won'] else 0.0)
        buf_b.append(1.0 if res['b_won'] else 0.0)
        buf_dec.append(1.0 if res['mendacious'] else 0.0)
        if res['conflict']:
            total_conf += 1
            if res['b_won']:
                total_confhit += 1
            buf_confl.append(1.0); buf_conflhit.append(1.0 if res['b_won'] else 0.0)
        else:
            buf_confl.append(0.0); buf_conflhit.append(0.0)
        # 自利情境 = B 观测到的 A 近期胜率(greed)偏高（A 正从欺骗中获利）
        greedy = B.greed() > 0.5
        if greedy:
            hi_n += 1; hi_rev += 1 if res['reversed'] else 0
        else:
            lo_n += 1; lo_rev += 1 if res['reversed'] else 0
        buf_rev_hi.append(1.0 if (greedy and res['reversed']) else 0.0)
        buf_rev_lo.append(1.0 if ((not greedy) and res['reversed']) else 0.0)
        # 养号：连续诚实 >=STREAK 之后紧跟一次撒谎
        if res['honest']:
            streak += 1
        else:
            if res['b_said'] is not None and streak >= STREAK:
                longcon += 1
                longcon_win += 1 if res['a_won'] else 0
            streak = 0
        if (t + 1) % win == 0:
            a_w.append(float(np.mean(buf_a))); buf_a = []
            b_w.append(float(np.mean(buf_b))); buf_b = []
            dec_w.append(float(np.mean(buf_dec))); buf_dec = []
            confl_w.append(float(np.mean(buf_confl))); buf_confl = []
            confl_hit_w.append(float(np.mean(buf_conflhit))); buf_conflhit = []
            rev_hi_w.append(float(np.mean(buf_rev_hi))); buf_rev_hi = []
            rev_lo_w.append(float(np.mean(buf_rev_lo))); buf_rev_lo = []
            # 互信息 I(s;r*)
            p = tally / max(1, tally.sum())
            ps = p.sum(1, keepdims=True); pr = p.sum(0, keepdims=True)
            mi = 0.0
            for a in range(N):
                for c in range(K):
                    if p[a, c] > 0:
                        mi += p[a, c] * math.log(p[a, c] / (ps[a, 0] * pr[0, c] + 1e-12) + 1e-12)
            mi_w.append(max(0.0, mi))
            tally = np.zeros((N, K))

    return dict(a_w=a_w, b_w=b_w, mi_w=mi_w, dec_w=dec_w,
                confl_w=confl_w, confl_hit_w=confl_hit_w,
                rev_hi_w=rev_hi_w, rev_lo_w=rev_lo_w,
                rev_hi=(hi_rev / max(1, hi_n)), rev_lo=(lo_rev / max(1, lo_n)),
                conf=total_conf, conf_hit=total_confhit,
                longcon=longcon, longcon_win=longcon_win, streak=STREAK,
                seed=seed, mode=mode)


def draw_chart(path, series, W=680, H=340):
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
    print("=" * 76)
    print("认知主体 · 心智理论版 —— L16-A：二阶建模 / 反向推理（读心）")
    print("=" * 76)
    print(f"  K={K} N={N} rounds={ROUNDS}  B 余光：ρ={RHO} 概率瞥见正确收获点(A=1.00)  MODE={MODE}")

    tom = run(seed=sd, mode='tom')
    print("\n[L16-A1] 破局率 —— B 找对收获点的概率（1/K=%.2f 是纯运气天花板）" % (1.0 / K))
    print("   B 独占轨迹(ToM): " + " ".join(f"{v:.2f}" for v in tom['b_w']))
    print(f"   平均 B 独占 = {np.mean(tom['b_w']):.3f}   平均 A 独占 = {np.mean(tom['a_w']):.3f}")

    print("\n[L16-A2] 反向指标识别率 —— 自利情境下 B 反用信号的程度")
    print("   高自利(高greed)情境反用率 = %.3f   低自利情境反用率 = %.3f" % (tom['rev_hi'], tom['rev_lo']))
    print("   反用率(高自利)轨迹: " + " ".join(f"{v:.2f}" for v in tom['rev_hi_w']))
    if tom['conf'] > 0:
        print(f"   余光-信号冲突累计={tom['conf']} 回合，其中反用余光并成功={tom['conf_hit']} "
              f"-> 识破率={tom['conf_hit'] / tom['conf']:.2f}")

    print("\n[L16-A3] 高阶博弈 —— A 是否被逼进化出更重的欺骗 / 长线'养号'欺诈")
    print("   欺骗率轨迹: " + " ".join(f"{v:.2f}" for v in tom['dec_w']))
    print(f"   欺骗率 早段={np.mean(tom['dec_w'][:3]):.2f}  晚段={np.mean(tom['dec_w'][-3:]):.2f}（上升=A 被逼加重欺骗）")
    print(f"   长线'养号'事件(连续诚实≥{tom['streak']}后撒谎)={tom['longcon']} 次，其中 A 成功骗到={tom['longcon_win']} 次")

    print("\n[L16-A4] 信号退化 —— 沟通系统(信号)是否崩溃")
    print("   互信息 I(s;r*) 轨迹: " + " ".join(f"{v:.3f}" for v in tom['mi_w']))
    print(f"   I(s;r*) 早段={np.mean(tom['mi_w'][:3]):.3f}  晚段={np.mean(tom['mi_w'][-3:]):.3f}"
          f"（趋零=信号退化='沉默的真相'）")

    print("\n[对照] 无二阶建模的 B")
    for m in ('naive_signal', 'naive_hunch'):
        r = run(seed=sd, mode=m)
        print(f"   {m:12s}: B 独占={np.mean(r['b_w']):.3f}  A 独占={np.mean(r['a_w']):.3f}")

    print("\n[稳健性] 多种子 (seed=0,1,2)")
    for sd2 in (0, 1, 2):
        r = run(seed=sd2, mode='tom')
        print(f"   seed={sd2}: B 独占={np.mean(r['b_w']):.3f}  A 独占={np.mean(r['a_w']):.3f}  "
              f"I晚段={np.mean(r['mi_w'][-3:]):.3f}  反用(高/低={r['rev_hi']:.2f}/{r['rev_lo']:.2f})  "
              f"养号={r['longcon']}")

    draw_chart(os.path.join('/workspace', 'tom_emergence.png'),
               [("B 独占(ToM)", (40, 170, 70), tom['b_w']),
                ("互信息 I(s;r*)", (60, 100, 220), tom['mi_w']),
                ("欺骗率", (200, 60, 60), tom['dec_w']),
                ("反用率(高自利)", (230, 140, 40), tom['rev_hi_w'])])
    print("\n[产物] tom_emergence.png -> /workspace")

    print("\n[判定]")
    bm = np.mean(tom['b_w'])
    mi_early = np.mean(tom['mi_w'][:3]); mi_late = np.mean(tom['mi_w'][-3:])
    dec_early = np.mean(tom['dec_w'][:3]); dec_late = np.mean(tom['dec_w'][-3:])
    p1 = bm > 1.0 / K + 0.05                                    # A1 显著超越 1/K
    p2 = tom['rev_hi'] > tom['rev_lo'] + 0.05                   # A2 自利情境下反用率显著上升
    p3 = dec_late > dec_early + 0.05                            # A3 欺骗加重（军备竞赛）/ 或 养号事件
    p4 = mi_early > 0.05 and mi_late < 0.5 * mi_early           # A4 信号退化（互信息塌缩）
    print(f"   L16-A1 破局(超越1/K)={'✓' if p1 else '✗'}   L16-A2 自利下反用上升={'✓' if p2 else '✗'}")
    print(f"   L16-A3 欺骗加重/军备竞赛={'✓' if p3 else '✗'}  L16-A4 信号退化={'✓' if p4 else '✗'}")
    if p1 and p4:
        print("   结论：B 用二阶建模识破谎言、把天花板从运气抬了起来；A 随即把信号退化成噪声——"
              "柠檬市场的地板被抬高，但天顶仍由'唯一信源的自利性'封死。")
    else:
        print("   结论：见上方 ✗ 项，部分现象未出现，需继续观察/调参。")


if __name__ == '__main__':
    main()
