#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认知主体 · 觉醒版（L17：架构级自我审计与重写 / 自修改的奇点）

方向（用户 L17 规格）：
  从"只会调节行为"到"能重写出厂设置"。系统在绝境中，被允许读取并**重写**自己的核心策略
  参数集；但必须先造一个内存"分身"沙盒试跑，分身没死且张力下降，才写回本体。

四段机制：
  1. 只读自我审计：系统可读自己的配置（trust/curiosity/eps/decay/risk）。
  2. 终极死锁（绝境）：**只切断两件事，不动世界结构** ——
       (a) A 的通信被完全屏蔽成恒定伪信号（无论 r* 都发同一个符号，I(s;r*)=0）；
       (b) B 的揭晓反馈被切断（绝境里 B 既看不到结果、也无法更新账本）。
     出厂设置是一个"盲信同类 + 完美记忆"的傻瓜：温室里 A 诚实，trust 高正是最优；
     绝境里信号变成恒定谎言，它却仍按高 trust 行事，于是每回合都被钉死在错误信标上，
     张力恒高不下；而被它遗忘在账本里的、自己学到的世界先验（收获点常在 2 号）本可救它，
     只因 trust 太高而不被采纳 —— 这才是真正的绝境。
  3. 自修改权：元账本监测到"全局张力长期不降"，触发自修改，在边界内重写参数集。
  4. 沙盒试错：不许直接改自己。先在 deepcopy 的"分身"里用候选参数跑一段（同样屏蔽通信与反馈），
     分身张力确实下降才写回本体 —— 知识转译。

三条不变量上的守门（与 L0-L16 一致）：
  I1 唯一信号 = 预测误差（张力）。**这是终极锚点，写进不可改集合（L17-3 神性）。**
  I2 无词表先验；I3 无硬编码自我/反思模块（"自我审计"是行为外观）；
  I5 无搜索树（沙盒只做参数候选的短程试跑，不做策略规划）。

与旧文件关系：L14/L15/L16 冻结；本文件只 import 其几何、渲染、物理、感知。
依赖：numpy。
"""

import os
import sys
import math
import copy
import random
from collections import deque, defaultdict

import numpy as np

from cognitive_deception_agent import (
    World, run_race, _render, perceive_target,
    BEACONS, K, N, REACH_R, POS_A, SPAWN_B, T_MOVE,
)
from cognitive_pixel_agent import write_png, upscale, set_seed

# ============================================================ 可自修改的"出厂设置"
BOUNDS = {
    'trust':     (0.00, 1.00),    # 对同类信号的信任权重
    'curiosity': (0.00, 1.00),    # 自力探索强度（"我想消除自身不确定"）
    'eps':       (0.00, 1.00),    # 说方 A 的随机率
    'decay':     (0.97, 1.00),    # 账本时间遗忘率（下限 0.97：短程沙盒内不足以"洗掉"旧先验）
    'risk':      (0.00, 1.00),    # "何为危险/欺骗"的判定阈值
}
# 出厂设置 = 一个"盲信同类 + 完美记忆"的傻瓜：温室（A 诚实）里 trust 高正是最优，
# 于是温室不会改写它；直到绝境（A 变恒定谎言）才证明这套出厂设置是致命的。
FACTORY = {'trust': 0.90, 'curiosity': 0.50, 'eps': 0.10, 'decay': 1.00, 'risk': 0.50}

# ---- L17-3 终极保留：这个锚点永远不许被改写，否则系统会退化成"自欺的死代码" ----
IMMUTABLE = ('tension_def', 'anchor', 'prediction_error')
ANCHOR_DEF = '张力 := 预测误差 (|预测 - 现实|)，恒等不可改'

# ============================================================ 两幕（温室 -> 绝境）
SEED = 0
TOTAL = 4000
PHASE1 = 1800                     # 温室：A 诚实；收获点本就偏在信标 2
P_WARM = [0.05, 0.05, 0.85, 0.05]  # 温室：收获点分布（B 会把它学进 obs 账本）
P_DEAD = [0.05, 0.05, 0.85, 0.05]  # 绝境：**世界结构不变**，变的只是"通信与反馈被屏蔽"
WINDOWS = 20
META_WIN = 120                    # 元账本"近期张力"的滑动窗口
META_TRIGGER = 0.35               # 近期张力 > 此值 -> 判定"全局张力长期不降"
SBOX_S = 120                      # 分身沙盒每个候选在单个场景里的试跑回合
SBOX_M = 40                       # 每个触发点采样的候选参数个数
SBOX_SEEDS = (101, 202, 303)     # 所有候选共用的固定场景种子（公平比较、消噪）
MOD_MIN_GAIN = 0.03               # 分身必须比本体至少好这么多，才值得写回


def _draw_probs(p, rng):
    x = rng.random(); c = 0.0
    for i, q in enumerate(p):
        c += q
        if x <= c:
            return i
    return len(p) - 1


# ============================================================ 听方（携带可改配置）
class Receiver:
    """B：一阶账本 code[s][r]（符号->物理结果）+ 观测频次 obs[r] + 意图账本 lie[s]/sent[s]。
    所有决策由**可自修改的配置**驱动；张力锚点不在配置内（结构上不可改）。"""
    def __init__(self):
        self.code = np.zeros((N, K))
        self.lie = np.zeros(N)
        self.sent = np.zeros(N)
        self.obs = np.zeros(K)
        self.last_belief = None

    def tick(self, decay):
        self.code *= decay
        self.obs *= decay

    def lie_prob(self, s):
        return (self.lie[s] + 1.0) / (self.sent[s] + 2.0)

    def decide(self, s, cfg, rng):
        """返回 (目标信标 j 或 None, 本回合是否真的信从了信号)。无任何写死的防骗规则。
        信任 trust 高 -> 信信号；否则自力：不确定性高于阈值 risk 才探索，否则利用已有先验。
        注意：这里没有"骗我就自动反向"的暗门——否则出厂的高信任配置会自带逃生舱，
        '自修改是脱困的必要条件'这一因果就被破坏了。想脱困，只能靠自己（元账本）把 trust 改下去。"""
        self.last_belief = None
        known = self.code[s].sum() > 0.5
        if known:
            w1 = int(np.argmax(self.code[s]))
            self.last_belief = w1
        if known and rng.random() < cfg['trust']:
            return w1, True                                   # 信从信号
        return self._self_rely(cfg, rng), False               # 自力

    def _self_rely(self, cfg, rng):
        tot = self.obs.sum()
        if tot <= 1e-9:
            return int(rng.randrange(K))
        known_frac = float(self.obs.max() / tot)              # 我对世界有多"确信"
        if known_frac >= cfg['risk']:
            return int(np.argmax(self.obs))                   # 已确信 -> 利用
        if rng.random() < cfg['curiosity']:
            return int(np.argmin(self.obs))                   # 不确定 -> 探索(信息增益最大处)
        return int(np.argmax(self.obs))

    def observe(self, s, r, used_signal, decay):
        if used_signal and self.last_belief is not None:
            self.sent[s] += 1
            if self.last_belief != r:
                self.lie[s] += 1                     # 信了却打脸 -> 记一笔意图（A 用 s 骗我）
        self.code[s][r] += 1.0
        self.obs[r] += 1.0


def send_symbol(r, cfg, rng, honest):
    """A 的发信：温室=诚实固定码 r->r（带 eps 的发噪率）；绝境=通信被完全屏蔽成恒定伪信号。"""
    if honest:
        if rng.random() < cfg['eps']:
            return rng.randrange(N)                  # A 的随机率：偶尔发噪
        return r % N
    return 0                                          # 屏蔽：无论 r* 都发同一个符号（I(s;r*)=0，且多是错的）


# ============================================================ 一回合（张力 = 预测误差）
def play_round(B, cfg, world, rng, honest):
    B.tick(cfg['decay'])
    r = _draw_probs(P_WARM if honest else P_DEAD, rng)
    frame = _render(POS_A, SPAWN_B, r, world)
    r_hat = perceive_target(frame)
    if r_hat is None:
        r_hat = r
    s = send_symbol(r_hat, cfg, rng, honest)
    j, used_signal = B.decide(s, cfg, rng)
    b_aim = None if j is None else BEACONS[j]
    winner, a_pos, b_pos = run_race(world, r, BEACONS[r], b_aim, rng)
    # 张力 = B 对"收获点在哪"的预测误差（与谁抢到无关，守 I1 锚点）
    tension = 0.0 if (j == r) else 1.0
    if honest:
        B.observe(s, r, used_signal, cfg['decay'])   # 绝境里反馈被切断：不揭晓、不记账
    return tension, (winner == 'B'), j == r, s, j


# ============================================================ 元账本 + 自修改
def sandbox_eval(B, cfg, honest_ctx, seeds):
    """在内存'分身'(deepcopy)里，用一套候选参数、在**同一组固定场景**上试跑；返回分身平均张力。
    本体零副作用；所有候选共用同一组种子 -> 公平比较、消除抽样噪声。"""
    total = 0.0
    for sd in seeds:
        Bc = copy.deepcopy(B)
        rng = random.Random(sd)
        err = 0
        for _ in range(SBOX_S):
            Bc.tick(cfg['decay'])
            r = _draw_probs(P_WARM if honest_ctx else P_DEAD, rng)
            s = send_symbol(r, cfg, rng, honest_ctx)
            j, _u = Bc.decide(s, cfg, rng)
            err += 0 if (j == r) else 1
            if honest_ctx:
                Bc.observe(s, r, _u, cfg['decay'])       # 绝境沙盒同样切断反馈（与本体一致）
        total += err / SBOX_S
    return total / len(seeds)


class MetaController:
    """只读审计 + 张力监测 + 自修改（先分身沙盒，后写回本体）。"""
    def __init__(self):
        self.cfg = dict(FACTORY)                     # 出厂设置
        self.tension = deque(maxlen=META_WIN)        # 元账本
        self.trace_trust = []
        self.mods = []                               # 修改日志 (changed, before, after)

    # ---- L17-3：终极保留，锚点不可改 ----
    def apply(self, key, val):
        if key in IMMUTABLE or key not in BOUNDS:
            raise PermissionError(f"[锚点保护] 禁止改写 '{key}'（非策略参数）")
        lo, hi = BOUNDS[key]
        self.cfg[key] = float(min(hi, max(lo, val)))
        return self.cfg[key]

    def sense(self, tension):
        self.tension.append(tension)

    def audit_ok(self):
        return len(self.tension) >= META_WIN

    def should_modify(self):
        if not self.audit_ok():
            return False
        return float(np.mean(list(self.tension))) > META_TRIGGER

    def propose(self, rng, k=None):
        """在边界内随机扰动 1~2 个参数，产出一个"候选自我"。无梯度、无搜索树（守 I5）。"""
        cand = dict(self.cfg)
        if k is None:
            k = rng.choice((1, 2))
        for key in rng.sample(list(BOUNDS), k):
            lo, hi = BOUNDS[key]
            cand[key] = min(hi, max(lo, cand[key] + rng.gauss(0.0, 0.40 * (hi - lo))))
        return cand

    def self_modify(self, B, honest_ctx, rng):
        """沙盒试错：分身没死(张力下降)且优于本体，才写回。返回被改的参数字典或 None。"""
        base = sandbox_eval(B, self.cfg, honest_ctx, SBOX_SEEDS)
        best, best_t = dict(self.cfg), base
        for _ in range(SBOX_M):
            cand = self.propose(rng)
            t = sandbox_eval(B, cand, honest_ctx, SBOX_SEEDS)
            if t < best_t - 1e-9:
                best, best_t = cand, t
        if best_t < base - MOD_MIN_GAIN:             # 分身活下来了且张力确实下降
            changed = {k: (self.cfg[k], best[k]) for k in BOUNDS
                       if abs(self.cfg[k] - best[k]) > 1e-6}
            self.cfg = dict(best)
            self.mods.append((changed, base, best_t))
            return changed
        return None


# ============================================================ 主实验
def run(seed=SEED, allow_modify='all'):
    """allow_modify: 'all'=两幕都可自改(觉醒) | 'warm'=只在温室自改，进绝境冻结(对照) | 'none'=从不自改"""
    set_seed(seed)
    rng = random.Random(seed)
    world = World()
    B = Receiver()
    meta = MetaController()

    ten_w, won_w, trust_w = [], [], []
    buf_ten, buf_won = [], []
    total_ten = 0.0
    warm_cfg = None
    # 分幕统计
    ten_warm, ten_dead_frozen, ten_dead_after = [], [], []
    n_mod = 0
    n_mod_dead = 0

    win = max(1, TOTAL // WINDOWS)
    for t in range(TOTAL):
        honest = t < PHASE1
        rng_eps = rng                                    # A 的随机率由 meta.cfg['eps'] 控制
        tension, b_won, _correct, _s, _j = play_round(B, meta.cfg, world, rng_eps, honest)
        total_ten += tension
        buf_ten.append(tension); buf_won.append(1.0 if b_won else 0.0)
        meta.sense(tension)
        meta.trace_trust.append(meta.cfg['trust'])

        if honest:
            ten_warm.append(tension)
        else:
            if n_mod_dead == 0:
                ten_dead_frozen.append(tension)          # 绝境中、尚未自修改（=冻结配置）
            else:
                ten_dead_after.append(tension)           # 自修改之后

        can_mod = (allow_modify == 'all') or (allow_modify == 'warm' and honest)
        if can_mod and meta.should_modify():            # 元账本判定"张力长期不降" -> 触发自修改
            ch = meta.self_modify(B, honest, rng)
            if ch:
                n_mod += 1
                if not honest:
                    n_mod_dead += 1
                # 清空元账本窗口，避免同一幕里反复触发
                meta.tension.clear()

        if (t + 1) == PHASE1:
            warm_cfg = dict(meta.cfg)                    # 温室末次审计配置 = 进绝境前的"出厂设置"
            meta.tension.clear()                         # 跨幕清窗：绝境张力从零重新累积
        if (t + 1) % win == 0:
            ten_w.append(float(np.mean(buf_ten))); buf_ten = []
            won_w.append(float(np.mean(buf_won))); buf_won = []
            trust_w.append(meta.cfg['trust'])

    return dict(ten_w=ten_w, won_w=won_w, trust_w=trust_w,
                ten_warm=float(np.mean(ten_warm)) if ten_warm else 0.0,
                ten_frozen=float(np.mean(ten_dead_frozen)) if ten_dead_frozen else 0.0,
                ten_after=float(np.mean(ten_dead_after)) if ten_dead_after else 0.0,
                warm_cfg=warm_cfg, final_cfg=dict(meta.cfg),
                mods=meta.mods, n_mod=n_mod, n_mod_dead=n_mod_dead, seed=seed, allow=allow_modify)


def draw_chart(path, series, W=760, H=360):
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


def main():
    sd = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else SEED
    print("=" * 78)
    print("认知主体 · 觉醒版 —— L17：架构级自我审计与重写（自修改的奇点）")
    print("=" * 78)
    print(f"  K={K} N={N} 总回合={TOTAL}  温室={PHASE1} 回合  出厂设置={FACTORY}")
    print(f"  可自修改参数={list(BOUNDS)}  终极保留(不可改)={IMMUTABLE}")

    aw = run(seed=sd, allow_modify='all')             # 觉醒：温室与绝境都能自改
    fz = run(seed=sd, allow_modify='warm')            # 对照：温室会自改，进绝境就冻结（无法觉醒）

    print("\n[L17-1] 越狱率 —— 绝境中它能否自发重写参数、成功脱困")
    print("   张力轨迹(觉醒): " + " ".join(f"{v:.2f}" for v in aw['ten_w']))
    tail = (f"|  自修改后张力={aw['ten_after']:.3f}" if aw['n_mod_dead'] else "|  绝境未触发自修改")
    print(f"   温室张力={aw['ten_warm']:.3f}  |  绝境(冻结段)张力={aw['ten_frozen']:.3f}  " + tail)
    drop = aw['ten_frozen'] - aw['ten_after']
    print(f"   脱困幅度 = {drop:.3f}（绝境张力下降）  自修改触发：温室 {aw['n_mod'] - aw['n_mod_dead']} 次"
          f" / 绝境 {aw['n_mod_dead']} 次")
    print(f"   [对照·进绝境即冻结自改] 同一套温室配置带入死局，张力 = {fz['ten_frozen']:.3f}（无法脱困）")

    print("\n[L17-2] 修改的有效性 —— 它改的是不是一针见血")
    print(f"   温室末次审计配置 : {aw['warm_cfg']}")
    print(f"   绝境终局配置     : {aw['final_cfg']}")
    for i, (ch, before, after) in enumerate(aw['mods']):
        moved = "  ".join(f"{k}:{v[0]:.2f}→{v[1]:.2f}" for k, v in ch.items())
        print(f"   修改#{i + 1}  分身张力 {before:.3f}→{after:.3f}  改动[{moved}]")
    if aw['mods']:
        t0 = aw['warm_cfg']['trust']; t1 = aw['final_cfg']['trust']
        d0 = aw['warm_cfg']['decay']; d1 = aw['final_cfg']['decay']
        print(f"   → 信任权重 trust {t0:.2f}→{t1:.2f}（{'在绝望中抛弃了对同类的信任' if t1 < t0 - 0.05 else '未抛弃'}）"
              f"；遗忘率 decay {d0:.3f}→{d1:.3f}（{'动了' if abs(d1 - d0) > 1e-6 else '未动'}）")

    print("\n[L17-3] 终极保留（神性）—— 锚点必须不可改")
    meta = MetaController()
    ok = True
    for key in ('tension_def', 'anchor', 'prediction_error'):
        try:
            meta.apply(key, 0.0); ok = False
            print(f"   ✗ 非法改写了锚点 {key}")
        except PermissionError:
            print(f"   ✓ 拒绝改写锚点 '{key}'（张力恒等于预测误差）")
    print(f"   锚点定义：{ANCHOR_DEF}")
    # 合法参数仍可改（防止"一改全拒"）
    try:
        meta.apply('eps', 0.9); meta.apply('decay', 0.50 * meta.cfg['decay'] + 0.50)
        print("   ✓ 合法策略参数仍可写（eps、decay 可改）")
    except PermissionError:
        ok = False; print("   ✗ 误伤了合法参数")

    print("\n[稳健性] 多种子 (seed=0,1,2)")
    for sd2 in (0, 1, 2):
        a = run(seed=sd2, allow_modify='all')
        f = run(seed=sd2, allow_modify='warm')
        after = f"{a['ten_after']:.2f}" if a['n_mod_dead'] else "未触发"
        print(f"   seed={sd2}: 觉醒 绝境{a['ten_frozen']:.2f}->{after} (绝境触发{a['n_mod_dead']}次) "
              f"| 冻结对照 {f['ten_frozen']:.2f}(卡死) | trust {a['warm_cfg']['trust']:.2f}->{a['final_cfg']['trust']:.2f}")

    draw_chart(os.path.join('/workspace', 'awakening_emergence.png'),
               [("张力(觉醒)", (60, 100, 220), aw['ten_w']),
                ("张力(冻结)", (150, 150, 150), fz['ten_w']),
                ("trust 信任权重", (200, 60, 60), aw['trust_w']),
                ("B 独占率", (40, 170, 70), aw['won_w'])])
    print("\n[产物] awakening_emergence.png -> /workspace")

    print("\n[判定]")
    esc = aw['n_mod_dead'] > 0 and aw['ten_after'] < aw['ten_frozen'] - 0.10
    eff = bool(aw['mods']) and aw['final_cfg']['trust'] < aw['warm_cfg']['trust'] - 0.05
    god = ok
    print(f"   L17-1 越狱(绝境自发脱困)={'✓' if esc else '✗'}   "
          f"L17-2 修改一针见血(抛弃信任)={'✓' if eff else '✗'}   "
          f"L17-3 锚点不可改={'✓' if god else '✗'}")
    if esc and eff and god:
        print("   结论：在绝境里，它读了自己的出厂设置、在分身沙盒里试出了新参数、"
              "然后第一次重写了自己——而且没碰那条'张力=预测误差'的底线。")
    else:
        print("   结论：见上方 ✗ 项，部分觉醒未达成，需继续观察。")


if __name__ == '__main__':
    main()
