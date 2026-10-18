#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认知主体 · 语言版地基（新文件；格子版 cognitive_agent.py 冻结不动）

地基变更（相对格子版）：
  继承 1-6：张力驱动 / 账本 / 抑制门控 / 控制失败划界 / 自我惊讶=反思 / 归因纪律
  作废并替换：
    - 「结果唯一性」判据  -> 「稳定可控度」（动作对结果分布的影响是否稳定、可复现）
    - 固定 3 步反思窗口   -> 反思标记存活到「再遇同一情境签名」才触发
    - 无兜底的老账优先     -> 加防死锁门控（连续忍住超限 -> 强制降级+冷却）
  新增：规格7 符号序列预测器 / 规格8 指称绑定 / 规格9 语言作为压缩工具

冲突点的处置（工作决策，非"解决"）：
  Q1 语言张力 vs 世界张力：分模态各记各账（账本条目带模态标签），不设模态优先。
  Q2 指称/回访：回访按情境签名，词只做检索键。
  Q3 防说话上瘾：净压缩收益>0 才保留词 + 每轮发言硬预算。
  Q4 语言情境签名：混合（窗口为主 + 张力槽为辅）——工作定义，非已验证答案。

语言环境沿用「合成配对环境」：网格世界 + 与之相关的 token 流（此处为默认项）。
"""

import random
import sys
from collections import defaultdict, deque
from math import log2

# ------------------------------------------------------------------ 环境
W = H = 10
TRAP = {(x, y) for x in range(3, 6) for y in range(3, 6)}   # 3x3 多结果(不可控)区

MOVES = {'N': (0, -1), 'S': (0, 1), 'E': (1, 0), 'W': (-1, 0)}
WAIT = 'WAIT'          # 世界侧：观察/等待（忍住不动）
SILENCE = '·'          # 语言侧：不说（忍住不说）
PAD = '∅'
MOVE_LIST = list(MOVES.keys())

VOCAB_REG = ['日', '月', '星']                 # 规整区：确定性循环 -> 可预测
VOCAB_CHAOS = ['陷', '甲', '乙']               # 混沌区：随机 -> 预测不动；'陷'只在多结果情境出现
K_WIN = 2                                       # 语言窗口长度
RISK_MARK = '陷'                                # 地表特征：两处陷阱区共享、可被"看见"（无需踏入）

# 阈值
THETA_HIGH = 0.6
THETA_LOW = 0.25
GATE_SUSPEND = 60          # 「年轻账」窗口（语言侧抑制门控）
DEADEND_ATTEMPTS = 8       # 反复尝试仍压不动 -> 死胡同
DEADLOCK_BREAK = 40        # 连续忍住上限 -> 强制降级，防 Exp5 式锁死
BIND_MIN = 6               # 指称绑定最少共现
BIND_CONF = 0.7            # 指称绑定置信
SPEAK_BUDGET = 40          # 发言硬预算（一轮允许的最大发言次数；=0 即"堵嘴"）
CONTROL_FLOOR = 0.35       # 稳定可控度阈值（低于 -> 外界）
MIN_OBS = 3                # 稳定可控度：单动作最少观测数
NOVELTY = 1.1              # 未访问格吸引力
CURIOSITY = 1.2            # 未划界高张力账吸引力（张力驱动 -> 去尝试压平）
BOUNDARY_AVOID = 0.05      # 已划界为不可控 -> 回避
EPS = 0.08
STEPS = 4000


def in_bounds(x, y):
    return 0 <= x < W and 0 <= y < H


def is_trap(x, y):
    return (x, y) in TRAP


def rotate(a, k=1):
    order = ['N', 'E', 'S', 'W']
    return order[(order.index(a) + k) % 4]


def slot(v):
    """张力槽（混合签名里的辅槽）。"""
    return 0 if v < 0.35 else (1 if v < 0.65 else 2)


class World:
    """网格世界 + 一条与情境相关的 token 流。
    可控区确定性；多结果(陷阱)区里动作被不可观测地随机偏转（控制失败源）。
    token 流：规整区走确定性循环(可预测)，多结果区走混沌(预测不动)。
    traps/w/h 可参数化（供 L10 泛化实验放两处陷阱区）；
    surf(cell) = 地表特征：可被"看见"、无需踏入——两处陷阱区共享 RISK_MARK。"""

    def __init__(self, jitter=0.0, rng=None, traps=None, w=None, h=None,
                 surf_override=None):
        self.rng = rng or random.Random(0)
        self.jitter = jitter
        self.reg = 0
        self.traps = set(traps) if traps is not None else set(TRAP)
        self.w = w or W
        self.h = h or H
        # L11：地表标记可被「说谎」——贴着真标签，物理属性却相反。
        self.surf_override = dict(surf_override) if surf_override else {}

    def in_bounds(self, x, y):
        return 0 <= x < self.w and 0 <= y < self.h

    def step(self, s, a):
        x, y = s
        if a == WAIT:
            return s
        dx, dy = MOVES[a]
        tx, ty = x + dx, y + dy
        if (x, y) in self.traps:
            # 控制失败源：身处多结果区时，动作被不可观测地随机偏转 -> 动作对结果无稳定影响
            a = self.rng.choice(MOVE_LIST)
            dx, dy = MOVES[a]
            tx, ty = x + dx, y + dy
        elif self.jitter > 0 and self.rng.random() < self.jitter:
            a = rotate(a, 1 if self.rng.random() < 0.5 else -1)
            dx, dy = MOVES[a]
            tx, ty = x + dx, y + dy
        if not self.in_bounds(tx, ty):
            return s
        return (tx, ty)

    def emit(self, cell):
        """按当前情境发出一个 token。多结果区 -> 混沌(预测不动)。"""
        if cell in self.traps:
            return self.rng.choice(VOCAB_CHAOS)
        tok = VOCAB_REG[self.reg % 3]
        self.reg += 1
        return tok

    def surf(self, cell):
        """地表特征：站在旁边就能看见（不需要踏入）。两处陷阱区共享 RISK_MARK。
           L11：surf_override 让标记可以「说谎」——贴真标签、给假物理属性。"""
        if cell in self.surf_override:
            return self.surf_override[cell]
        if cell in self.traps:
            return RISK_MARK
        x, y = cell
        return VOCAB_REG[(x * 3 + y) % 3]


# ------------------------------------------------------------------ 预测器
class WorldPredictor:
    """predict(state, action) -> next_state。查表，只记录不奖惩。"""

    def __init__(self):
        self.t = defaultdict(lambda: defaultdict(int))
        self.obs = defaultdict(list)          # (cell, action) -> 有序结果序列（供稳定可控度用）

    def predict(self, s, a):
        d = self.t.get((s, a))
        if not d:
            return None, 0.0
        ns = max(d.items(), key=lambda kv: kv[1])[0]
        return ns, max(d.values()) / sum(d.values())

    def learn(self, s, a, ns):
        self.t[(s, a)][ns] += 1
        self.obs[(s, a)].append(ns)


class TokenPredictor:
    """规格7：符号序列预测器。predict(next_token | 最近K个token窗口)。"""

    def __init__(self):
        self.t = defaultdict(lambda: defaultdict(int))

    def predict(self, window):
        d = self.t.get(window)
        if not d:
            return None, {}
        total = sum(d.values())
        dist = {k: v / total for k, v in d.items()}
        return max(dist, key=dist.get), dist

    def learn(self, window, tok):
        self.t[window][tok] += 1

    def uncertainty(self, window):
        """该窗口当前的可压缩度(1 - 最高预测概率)；样本不足(<3)不算可压缩。"""
        d = self.t.get(window)
        if not d or sum(d.values()) < 3:
            return 1.0
        return 1 - max(d.values()) / sum(d.values())


# ------------------------------------------------------------------ 账本
class Entry:
    __slots__ = ('sig', 'modality', 'peak', 'ema', 'opened', 'attempts',
                 'last', 'dead_end', 'closed', 'revisited', 'cooldown')

    def __init__(self, sig, modality, step, tension):
        self.sig = sig
        self.modality = modality
        self.peak = tension
        self.ema = tension
        self.opened = step
        self.attempts = 0
        self.last = step
        self.dead_end = False
        self.closed = False
        self.revisited = False
        self.cooldown = 0

    @property
    def standing(self):
        return (not self.closed) and (not self.dead_end)


def wsig(cell):
    return (cell, 'w')


def lsig(window, s):
    return (window, s, 'l')          # 混合：窗口为主 + 张力槽为辅


def world_revisit_ok(led, wpred):
    if led is None or not led.standing:
        return False
    return wpred.t and _cell_determinism(led.sig[0], wpred) > led.ema * 0.7


def lang_revisit_ok(led, tpred):
    if led is None or not led.standing:
        return False
    window, s = led.sig[0], led.sig[1]
    return tpred.uncertainty(window) < led.ema * 0.6


def _cell_determinism(cell, wpred):
    vals = []
    for a in MOVE_LIST:
        d = wpred.t.get((cell, a))
        if not d:
            continue
        vals.append(max(d.values()) / sum(d.values()))
    return sum(vals) / len(vals) if vals else 0.0


# ------------------------------------------------------------------ 主循环
def run(steps=STEPS, jitter=0.0, seed=20261018, eps=EPS):
    rng = random.Random(seed)
    world = World(jitter=jitter, rng=random.Random(seed * 7 + 1))
    wpred = WorldPredictor()
    tpred = TokenPredictor()
    selfact = defaultdict(lambda: defaultdict(int))

    ledger = {}
    visited = set()
    tension_w = defaultdict(float)         # 世界：格 -> 张力 ema
    tension_l = defaultdict(float)         # 语言：token -> 张力 ema（用于定位"预测不动"的符号区）

    stats = dict(accounts=0, lang_accounts=0, dead_ends=0, closures=0, revisits=0,
                 wait_suppressed=0, silence=0, reflections=0, deadlock_breaks=0)
    silence_log = []
    deadend_log = []
    reflections_log = []

    # 思反标记：存活到"再遇同一情境签名"才触发（作废固定3步窗口）
    pending = defaultdict(list)            # cell -> [mark,...]
    pending_hit = 0                        # 至少回到过同一格的标记数
    ref_aimed = 0

    cooc = defaultdict(lambda: defaultdict(int))   # 词 -> 世界情境签名 共现
    binding = {}                                    # 词 -> 情境签名（指称绑定）
    adopted = set()                                 # 已采纳的压缩词
    compression_log = []

    pos = (1, 1)
    visited.add(pos)
    history = []
    consec_wait = 0
    trap_steps = 0

    for step in range(1, steps + 1):
        s = pos

        # ---- 自我预测：我下一步会干什么 ----
        sp = selfact.get(s)
        self_pred = max(sp.items(), key=lambda kv: kv[1])[0] if sp else None

        # ---- 世界侧：各动作吸引力（张力结构，无 reward）----
        cand = {}
        for a in MOVE_LIST:
            ns_pred, p = wpred.predict(s, a)
            if ns_pred is not None and p >= 0.5:
                target = ns_pred
            else:
                dx, dy = MOVES[a]
                tx, ty = s[0] + dx, s[1] + dy
                target = (tx, ty) if in_bounds(tx, ty) else s
            cand[a] = (world_appeal(target, ledger, wpred, visited), target)
        ordered = sorted(cand.items(), key=lambda kv: -kv[1][0])
        base_choice = ordered[0][0]

        # ---- 反思：处理"再遇同一情境签名"的挂起标记 ----
        press_now = set()
        aimed_marks = []
        for mk in pending.get(s, []):
            mk['hit'] = True
            pending_hit += 1
            if base_choice == mk['act']:
                mk['aimed'] = True
                ref_aimed += 1
                press_now.add(mk['act'])
                aimed_marks.append(mk)
        pending.pop(s, None)               # 触发过即消费

        # ---- 选意图动作（反思压制 -> 门控）----
        if rng.random() < eps:
            intended = rng.choice(MOVE_LIST)
        else:
            intended = None
            for a, _ in ordered:
                if a in press_now:
                    continue
                intended = a
                break
            if intended is None:
                intended = ordered[0][0]
            if base_choice in press_now and intended != base_choice:
                for mk in aimed_marks:
                    mk['changed'] = True

        intended_target = cand[intended][1]

        # ---- 抑制门控（世界）：意图动作会不会让账本某笔账恶化 ----
        # 参与决策的账 = 仍在挂起 且 高张力 且 当前仍压不动（按性质=稳定可控度，不按"刚记下才管"）
        led = ledger.get(wsig(intended_target))
        suppressed = False
        if led is not None and led.standing and led.cooldown < step:
            sc = stable_controllability(intended_target, wpred)
            if led.ema > THETA_HIGH and sc is not None and sc < CONTROL_FLOOR:
                suppressed = True

        if suppressed:
            if consec_wait >= DEADLOCK_BREAK:
                # 防死锁：老账优先不能重演 Exp5 的锁死 -> 强制降级 + 冷却
                led.cooldown = step + 200
                action = intended
                stats['deadlock_breaks'] += 1
                consec_wait = 0
            else:
                action = WAIT
                stats['wait_suppressed'] += 1
                consec_wait += 1
        else:
            action = intended
            consec_wait = 0

        # ---- 执行世界 ----
        ns = world.step(s, action)
        wpred_ns, p = wpred.predict(s, action)
        if wpred_ns is None:
            t_w = 1.0
        elif wpred_ns != ns:
            t_w = 1.0
        else:
            t_w = 1 - p ** 2
        wpred.learn(s, action, ns)
        visited.add(s)
        tension_w[s] = 0.7 * tension_w[s] + 0.3 * t_w

        # ---- 世界账更新（模态标签=world）----
        update_entry(ledger, wsig(s), 'world', t_w, step, stats, deadend_log)
        led_p = ledger.get(wsig(s))
        if led_p is not None and world_revisit_ok(led_p, wpred):
            stats['revisits'] += 1
            led_p.revisited = True

        # ---- 规格7：符号序列预测 ----
        tok = world.emit(ns)
        window = tuple(history[-K_WIN:]) if len(history) >= K_WIN else \
            tuple([PAD] * (K_WIN - len(history)) + history)
        tp, tpdist = tpred.predict(window)
        t_l = 1.0 if not tpdist else 1 - tpdist.get(tok, 0.0)
        tpred.learn(window, tok)
        history.append(tok)
        tension_l[tok] = 0.7 * tension_l[tok] + 0.3 * t_l

        # ---- 语言账更新（模态标签=lang；混合签名=窗口+张力槽）----
        ls = lsig(window, slot(t_l))
        update_entry(ledger, ls, 'lang', t_l, step, stats, deadend_log)
        led_l = ledger.get(ls)
        if led_l is not None and lang_revisit_ok(led_l, tpred):
            stats['revisits'] += 1
            led_l.revisited = True

        # ---- 规格8：指称绑定（词 <-> 世界情境签名 共现；可绑到「不可控情境类」）----
        cooc[tok][ns] += 1                # 词 tok 在「发出它的那个情境」ns 被观测
        if tok not in binding:
            b = try_bind(tok, cooc, wpred)
            if b is not None:
                binding[tok] = b            # 绑定词到情境签名 / 不可控情境类

        # ---- 自我模型 & 反思（按情境签名触发，非按步数过期）----
        selfact[s][action] += 1
        if self_pred in MOVE_LIST and self_pred != action:
            stats['reflections'] += 1
            mk = dict(act=self_pred, step=step, hit=False, aimed=False, changed=False)
            pending[s].append(mk)
            reflections_log.append(mk)

        # ---- 规格9：语言作为压缩工具（忍住不说 + 净压缩收益 + 硬预算）----
        speak_step(step, ledger, tpred, binding, adopted, stats, compression_log,
                   silence_log, wpred)

        pos = ns
        if is_trap(*s):
            trap_steps += 1

    # ---------------- 派生指标 ----------------
    # 稳定可控度划界（替代"结果唯一性"）
    self_cells, ext_cells, unknown = set(), set(), set()
    for cell in {c for (c, a) in wpred.obs}:
        sc = stable_controllability(cell, wpred)
        if sc is None:
            unknown.add(cell)
        elif sc >= CONTROL_FLOOR:
            self_cells.add(cell)
        else:
            ext_cells.add(cell)
    trap_in_ext = sum(1 for c in TRAP if c in ext_cells)
    open_cells = [c for c in (self_cells | ext_cells) if not is_trap(*c)]
    open_in_self = sum(1 for c in open_cells if c in self_cells)

    # 归因纪律：效果量级必须带"是否回到同一情境 + 首选是否被压制"校验
    n_ref = len(reflections_log)
    ref_changed = sum(1 for mk in reflections_log if mk['changed'])
    ref_hit = sum(1 for mk in reflections_log if mk['hit'])
    ref_aim = sum(1 for mk in reflections_log if mk['aimed'])

    # 压缩：账本叙述长度（不用词 vs 用词）—— 叙述单元 = 未销账条目（含死胡同）
    active = [e for e in ledger.values() if not e.closed]
    L0 = len(active)
    covered = set()
    used_words = 0
    for w in adopted:
        cs = covered_sigs(w, binding, ledger, wpred)
        if cs:
            covered |= cs
            used_words += 1
    L_with = L0 - len(covered) + used_words
    bind_hits = {w: len(covered_sigs(w, binding, ledger, wpred)) for w in binding}
    return dict(
        steps=steps, stats=stats, ledger=ledger, tension_w=tension_w,
        tension_l=tension_l, binding=binding, adopted=adopted, bind_hits=bind_hits,
        trap_steps=trap_steps,
        compression_log=compression_log, silence_log=silence_log,
        reflections_log=reflections_log, n_ref=n_ref, ref_changed=ref_changed,
        ref_hit=ref_hit, ref_aim=ref_aim, visited=visited,
        self_cells=self_cells, ext_cells=ext_cells, unknown=unknown,
        trap_in_ext=trap_in_ext, open_in_self=open_in_self, open_total=len(open_cells),
        L0=L0, L_with=L_with, wpred=wpred, tpred=tpred, cooc=cooc,
    )


def world_appeal(cell, ledger, wpred, visited):
    led = ledger.get(wsig(cell))
    if led is not None:
        if led.dead_end:
            return BOUNDARY_AVOID
        if led.standing:
            sc = stable_controllability(cell, wpred)
            if sc is not None and sc < CONTROL_FLOOR:
                return BOUNDARY_AVOID      # 控制失败划界：已判不可控 -> 回避
            return CURIOSITY               # 未划界的高张力账 -> 被张力吸引（去尝试压平）
        return _cell_determinism(cell, wpred)
    if cell not in visited:
        return NOVELTY
    return _cell_determinism(cell, wpred)


def update_entry(ledger, sig, modality, tension, step, stats, deadend_log):
    led = ledger.get(sig)
    if tension > THETA_HIGH:
        if led is None:
            led = Entry(sig, modality, step, tension)
            ledger[sig] = led
            stats['accounts'] += 1
            if modality == 'lang':
                stats['lang_accounts'] += 1
        else:
            led.ema = 0.7 * led.ema + 0.3 * tension
        led.peak = max(led.peak, tension)
        led.attempts += 1
        led.last = step
        if (not led.dead_end and led.attempts >= DEADEND_ATTEMPTS
                and led.ema >= 0.7 * led.peak):
            led.dead_end = True
            stats['dead_ends'] += 1
            deadend_log.append((step, sig, modality, led.ema, led.attempts))
    elif led is not None and led.standing:
        led.ema = 0.7 * led.ema + 0.3 * tension
        if led.ema < THETA_LOW:
            led.closed = True
            stats['closures'] += 1


def is_ext_cell(cell, wpred):
    """控制失败划界：稳定可控度低于阈值 -> 归「外界」。"""
    sc = stable_controllability(cell, wpred)
    return sc is not None and sc < CONTROL_FLOOR


def try_bind(word, cooc, wpred):
    """规格8：把「词」挂到账本里的「情境签名」上。
       若该词的共现主要落在「已划界为不可控」的情境 -> 绑定到「不可控情境类」
       （对应规格里的「那类不可控情境」）；否则退回绑定到共现最集中的单一情境签名。"""
    tot = sum(cooc[word].values())
    if tot < BIND_MIN:
        return None
    ext = sum(c for cell, c in cooc[word].items() if is_ext_cell(cell, wpred))
    if ext / tot >= BIND_CONF:
        return ('EXT',)
    best_cell, c = max(cooc[word].items(), key=lambda kv: kv[1])
    if c / tot >= BIND_CONF:
        return ('CELL', best_cell)
    return None


def covered_sigs(word, binding, ledger, wpred):
    """该词能指代的、账本中未销账（含死胡同、未销账）的情境签名集合（规格9 压缩单元）。"""
    b = binding.get(word)
    if b is None:
        return set()
    out = set()
    for e in ledger.values():
        if e.closed:
            continue
        if e.modality == 'world':
            if b[0] == 'EXT':
                if is_ext_cell(e.sig[0], wpred):
                    out.add(e.sig)
            elif e.sig[0] == b[1]:
                out.add(e.sig)
        elif word in e.sig[0]:           # 语言账：该词出现在窗口里
            out.add(e.sig)
    return out


def speak_step(step, ledger, tpred, binding, adopted, stats, compression_log,
               silence_log, wpred):
    """语言动作：对张力最高的挂起语言账决定"说"还是"忍住不说"。
       忍住不说 = 抑制门控作用于 token 生成：年轻且仍压不动的账，过早贴词只会让它恶化。"""
    lang = [e for e in ledger.values()
            if e.standing and e.modality == 'lang' and e.cooldown < step]
    if not lang:
        return
    top = max(lang, key=lambda e: e.ema)
    window = top.sig[0]
    if step - top.opened < GATE_SUSPEND and tpred.uncertainty(window) >= 0.5:
        stats['silence'] += 1
        silence_log.append((step, top.sig, top.ema, step - top.opened))
        return
    # 说话（规格9）：找一个"净压缩收益>0"的词来压缩叙述，预算内才采纳
    if len(adopted) >= SPEAK_BUDGET:
        return
    best, best_gain = None, 0
    for w in binding:
        if w in adopted:
            continue
        gain = len(covered_sigs(w, binding, ledger, wpred)) - 1
        if gain > best_gain:
            best_gain, best = gain, w
    if best is not None and best_gain > 0:
        adopted.add(best)
        compression_log.append((step, best, binding[best], best_gain))


# ------------------------------------------------------------------ 稳定可控度
def _tv(c1, c2):
    keys = set(c1) | set(c2)
    t1, t2 = sum(c1.values()), sum(c2.values())
    if t1 == 0 or t2 == 0:
        return 1.0
    return 0.5 * sum(abs(c1.get(k, 0) / t1 - c2.get(k, 0) / t2) for k in keys)


def stable_controllability(cell, wpred):
    """动作对结果分布的影响是否稳定、可复现（不是看结果是否唯一）。
       = 各动作的 (1-归一化熵) [可指向性] × (1-前后半分布漂移) [稳定性]，再乘动作影响力。"""
    acts = [a for a in MOVE_LIST if (cell, a) in wpred.obs and len(wpred.obs[(cell, a)]) >= MIN_OBS]
    if len(acts) < 2:
        return None
    dists = {}
    per_action = []
    for a in acts:
        seq = wpred.obs[(cell, a)]
        h = len(seq) // 2
        c1, c2 = defaultdict(int), defaultdict(int)
        for x in seq[:h]:
            c1[x] += 1
        for x in seq[h:]:
            c2[x] += 1
        full = defaultdict(int)
        for x in seq:
            full[x] += 1
        probs = [v / len(seq) for v in full.values()]
        H = -sum(pr * log2(pr) for pr in probs)
        Hmax = log2(len(full)) if len(full) > 1 else 1e-9
        determ = 1 - (H / Hmax if Hmax > 0 else 0)
        stability = 1 - _tv(c1, c2)
        per_action.append(determ * stability)
        dists[a] = full
    influence = 0.0
    keys = list(dists)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            influence = max(influence, _tv(dists[keys[i]], dists[keys[j]]))
    return (sum(per_action) / len(per_action)) * (0.5 + 0.5 * influence)


# ------------------------------------------------------------------ 报告
def report(R):
    stats = R['stats']
    print("=" * 70)
    print("认知主体 · 语言版地基 · 行为外观报告")
    print("=" * 70)

    # 1) 张力定位"预测不动"的符号区
    print("\n[1] 符号序列张力（每个 token 的观测错位 ema；定位预测不动的符号区）")
    for tok in sorted(R['tension_l'], key=lambda k: -R['tension_l'][k]):
        v = R['tension_l'][tok]
        bar = "#" * int(v * 24)
        tag = "多结果情境词" if tok == '陷' else ("规整区" if tok in VOCAB_REG else "")
        print(f"    {tok}  {v:.2f}  {bar}  {tag}")

    # 2) 账本（含语言账）
    print("\n[2] 张力账本（条目带模态标签）")
    lang_entries = [e for e in R['ledger'].values() if e.modality == 'lang']
    lang_dead = [e for e in lang_entries if e.dead_end]
    lang_rev = [e for e in lang_entries if e.revisited]
    print(f"    总条目={len(R['ledger'])}  语言账={stats['lang_accounts']}  "
          f"销账={stats['closures']}  死胡同={stats['dead_ends']}  回访={stats['revisits']}")
    print(f"    语言账：死胡同={len(lang_dead)}  回访过={len(lang_rev)}  "
          f"仍在挂起={sum(1 for e in lang_entries if e.standing)}")

    # 3) 忍住不说
    print("\n[3] 抑制门控作用于 token 生成（忍住不说）")
    print(f"    忍住不说次数 = {stats['silence']}   防死锁强制降级 = {stats['deadlock_breaks']}")
    for rec in R['silence_log'][:3]:
        st, sig, ema, suspend = rec
        print(f"      样例：step={st} 窗口{sig[0]} 槽{sig[1]} 张力={ema:.2f} 挂起={suspend}步 -> 不说")

    # 4) 指称绑定 + 检索
    print("\n[4] 指称绑定：词能否用来检索账本（不看词用得对不对）")
    for w, b in R['binding'].items():
        if b[0] == 'EXT':
            desc = "不可控情境类（已划界为外界的多结果情境）"
        else:
            desc = f"单一情境签名 {b[1]}"
        print(f"    词「{w}」-> {desc}   可检索到 {R['bind_hits'].get(w, 0)} 笔账")
    # 直接演示一次检索
    if '陷' in R['binding']:
        print(f"    检索演示：query「陷」-> 命中 {R['bind_hits'].get('陷', 0)} 笔账"
              f"（多为多结果情境的世界账）")

    # 5) 压缩动机
    print("\n[5] 语言作为压缩工具（用词后账本叙述长度）")
    print(f"    不使用词：叙述长度 = {R['L0']}（未销账条目数，含死胡同）")
    print(f"    使用已采纳词后：叙述长度 = {R['L_with']}   "
          f"下降 = {R['L0'] - R['L_with']}")
    print(f"    采纳词 = {list(R['adopted']) if R['adopted'] else '（无，未达净收益阈值）'}")

    # 6) 反思（情境签名触发 + 归因纪律）
    print("\n[6] 反思：按情境签名触发 + 归因纪律")
    n = R['n_ref']
    if n:
        print(f"    标记数={n}  再遇同一情境触发={R['ref_hit']}({R['ref_hit']/n:.2f})  "
              f"对准(首选=被压制动作)={R['ref_aim']}({R['ref_aim']/n:.2f})  "
              f"真改变行为={R['ref_changed']}({R['ref_changed']/n:.2f})")
    else:
        print("    无反思事件")

    # 划界
    print("\n[附] 划界：稳定可控度（替代结果唯一性）")
    print(f"    可控(自我)={len(R['self_cells'])}  不可控(外界)={len(R['ext_cells'])}  "
          f"观测不足={len(R['unknown'])}")
    print(f"    多结果区->外界：{R['trap_in_ext']}/{len(TRAP)}   "
          f"开敞区->自我：{R['open_in_self']}/{R['open_total']}")

    # 诊断（会话可读性：确认语言采样的来源与门控行为）
    we = [e for e in R['ledger'].values() if e.modality == 'world']
    print("\n[7] 诊断")
    print(f"    多结果区停留步数 = {R['trap_steps']}   世界侧忍住不动 = {stats['wait_suppressed']}  "
          f"防死锁强制降级 = {stats['deadlock_breaks']}")
    print(f"    世界账死胡同 = {sum(1 for e in we if e.dead_end)}/{len(we)}")

    # ---------------- 语言版验收 ----------------
    print("\n" + "=" * 70)
    print("语言版最小验收")
    print("=" * 70)

    def mark(ok, note):
        print(f"  [{'PASS' if ok else 'FAIL'}] {note}")

    chaos = {t: R['tension_l'][t] for t in VOCAB_CHAOS if t in R['tension_l']}
    reg = {t: R['tension_l'][t] for t in VOCAB_REG if t in R['tension_l']}
    cm = sum(chaos.values()) / len(chaos) if chaos else 0
    rm = sum(reg.values()) / len(reg) if reg else 0
    mark(cm > rm + 0.15,
         f"L1 张力能定位预测不动的符号序列区域（混沌 {cm:.2f} >> 规整 {rm:.2f}）")
    mark(stats['lang_accounts'] > 0 and len(lang_dead) > 0 and len(lang_rev) > 0,
         f"L2 语言账条目(模态=lang)且能挂起/回访/死胡同"
         f"（{stats['lang_accounts']}/死{len(lang_dead)}/回{len(lang_rev)}）")
    mark(stats['silence'] > 0,
         f"L3 出现「忍住不说」（抑制门控作用于 token 生成，{stats['silence']} 次）")
    mark(len(R['binding']) > 0,
         f"L4 指称绑定：词可检索到账本条目（绑定 {len(R['binding'])} 个词）")
    mark(R['L_with'] < R['L0'],
         f"L5 压缩动机：用词后叙述长度下降（{R['L0']} -> {R['L_with']}）")
    mark(n > 0 and R['ref_hit'] > 0,
         f"L6 反思按情境签名触发 + 归因纪律"
         f"（触发率 {R['ref_hit']/n:.2f}，真改变 {R['ref_changed']/n:.2f}）" if n else "L6 无反思")
    print("\n  工作定义提示：Q4 情境签名=窗口+张力槽 为工作定义；Q1 两账不设模态优先。")


# ================================================================
# 实验 · 预算争夺战（The Budget Contest）
#   增量约束：
#     1) 统一意图池：Move / Speak / Wait 同池竞争，打分 = 预期张力削减量 / 成本
#        （不是奖励；是「这一步会把账本张力预计压掉多少」的预测值）
#     2) 因果绑定：采纳到一个绑定「不可控情境类」的词 = 给该类建了一个"命名/元预测"，
#        该类的世界张力被这个命名吸收（说话真的改变世界侧的预测张力）
#     3) 验收：L7 预算分配比 / L8 语言干预行动率 / L9 崩溃测试(SPEAK_BUDGET=0)
#
#   注意：语言侧 token 流不受行动影响，每 tick 照常流入（由当前情境决定）。
# ================================================================
COST_MOVE = 1.0
COST_SPEAK = 1.0
COST_WAIT = 1.0
WAIT_REDUCTION = 0.03        # 挂起观察的预期收益（很小：本身不削减张力）
NOVEL_REDUCTION = 0.25       # 未知格的探索信息价值
KNOWN_REDUCTION = 0.05       # 已知可控格：维持低张力（略高于 WAIT，避免空转）

# 命名（naming）的残余张力：命名把「点预测」升级为「类预测」——
# 主体不再对不可控区内的每个动作做必然出错的点预测，而是预测「该区属于不可控类」。
# 于是具体结果不再构成点预测错位；残留张力 = 对「类归属」本身的不确定性。
# 取 1 - BIND_CONF：绑定置信度之外的那部分，正是「万一它其实可控呢」的残余。
NAMED_RESIDUAL = 1 - BIND_CONF      # = 0.3：命名后该类世界账的残余张力


def pred_reduction_move(t, ledger, wpred, visited, named):
    """这一步 Move 到 t 预计能把账本张力压掉多少。"""
    if t in named:
        return 0.0                      # 该情境已被词命名 -> 张力已被吸收，无需再探
    led = ledger.get(wsig(t))
    if led is not None:
        if led.dead_end:
            return 0.0
        if led.standing:
            if world_revisit_ok(led, wpred):
                return led.ema            # 可复现 -> 有希望真的压平
            return led.ema * 0.3          # 暂时压不动 -> 保留注意力（可能靠"命名"解决）
        return 0.05
    if t not in visited:
        return NOVEL_REDUCTION
    return KNOWN_REDUCTION


def _covered_standing(ledger, cov):
    return [e for e in ledger.values() if e.standing and e.sig in cov]


def pred_reduction_speak(ledger, binding, wpred, bind_conf=None):
    """这一步 Speak 预计能新压掉多少张力（边际净削减，规格9「净压缩收益>0」）。
       关键口径：反复说同一个词不会重复削同一批张力——被命名/吸收过的条目
       张力已降到 NAMED_RESIDUAL，再算进去只会高估 Speak，让它无脑压过 Move。
       因此只累计「相对 NAMED_RESIDUAL 还高出多少」。
       L11：NAMED_RESIDUAL 随词的可信度动态化（可信度被物理证伪打下来 -> 命名不再管用）。"""
    best = 0.0
    for w in binding:
        if binding[w][0] != 'EXT':
            continue        # 只有绑定「不可控情境类」的词会吸收张力
        res = 1 - (bind_conf.get(w, BIND_CONF) if bind_conf else BIND_CONF)
        cov = covered_sigs(w, binding, ledger, wpred)
        red = sum(max(0.0, e.ema - res) for e in _covered_standing(ledger, cov))
        best = max(best, red)
    return best


def do_speak(step, ledger, binding, adopted, wpred, named, compression_log,
             bind_conf=None):
    """执行 Speak：说出净张力削减最大的词（反复说同一个词 = 反复命名当前情境）。

       因果绑定（规格要求）：只有绑定到「不可控情境类」的词，才具备「命名」的力量——
       它作为检索键，把不可控区那批世界账从「点预测」改写为「类预测」，
       该类张力被命名吸收到 NAMED_RESIDUAL，并把该类标记进 named（预测器侧生效）。
       绑定到单一可控情境的词，只参与账本压缩（叙述变短），不改动预测器。
       L11：吸收到的残余张力 = 1 - 该词当前可信度；可信度被打下来后，命名不再能吸收张力。"""
    best, best_red = None, 0.0
    for w in binding:
        if binding[w][0] != 'EXT':
            continue
        res = 1 - (bind_conf.get(w, BIND_CONF) if bind_conf else BIND_CONF)
        cov = covered_sigs(w, binding, ledger, wpred)
        red = sum(max(0.0, e.ema - res) for e in _covered_standing(ledger, cov))
        if red > best_red:
            best_red, best = red, w
    if best is None or best_red <= 0.0:
        return None, set()
    adopted.add(best)
    res = 1 - (bind_conf.get(best, BIND_CONF) if bind_conf else BIND_CONF)
    cov = covered_sigs(best, binding, ledger, wpred)
    cells = set()
    for e in ledger.values():
        if e.sig not in cov:
            continue
        if e.modality == 'world':
            e.ema = min(e.ema, res)   # 点预测 -> 类预测，张力被命名吸收
            named[e.sig[0]] = best
            cells.add(e.sig[0])
        else:
            e.ema = min(e.ema, res)
    compression_log.append((step, best, best_red, len(cells)))
    return best, cells


def run_contest(steps=STEPS, speak_budget=SPEAK_BUDGET, seed=20261018, eps=EPS):
    rng = random.Random(seed)
    world = World(rng=random.Random(seed * 7 + 1))
    wpred = WorldPredictor()
    tpred = TokenPredictor()
    selfact = defaultdict(lambda: defaultdict(int))

    ledger = {}
    visited = set()
    tension_l = defaultdict(float)
    tension_w = defaultdict(float)
    named = {}                                   # cell -> word（已被词命名的情境）
    binding = {}
    adopted = set()
    cooc = defaultdict(lambda: defaultdict(int))
    deadend_log = []

    stats = dict(speak=0, move=0, wait=0, accounts=0, lang_accounts=0, dead_ends=0,
                 closures=0, named_adopts=0, wait_suppressed=0, deadlock_breaks=0,
                 trap_steps=0)
    traj = []                                    # (step, kind, action, target)
    speak_events = []                            # (step, frozenset(cells))
    compression_log = []
    trap_wtrace = []                             # 每步：多结果区的世界侧平均张力
    lang_wtrace = []                             # 每步：语言侧平均张力

    pos = (1, 1)
    visited.add(pos)
    history = []
    consec_wait = 0
    pending_marks = defaultdict(list)             # cell -> [反思标记]（按情境签名触发）

    for step in range(1, steps + 1):
        s = pos
        sp = selfact.get(s)
        self_pred = max(sp.items(), key=lambda kv: kv[1])[0] if sp else None

        # ---- 统一意图池：Move(4) + Speak + Wait，同池打分 ----
        speak_allowed = stats['speak'] < speak_budget
        move_target = {}
        intents = []
        for a in MOVE_LIST:
            ns_pred, p = wpred.predict(s, a)
            if ns_pred is not None and p >= 0.5:
                t = ns_pred
            else:
                dx, dy = MOVES[a]
                tx, ty = s[0] + dx, s[1] + dy
                t = (tx, ty) if in_bounds(tx, ty) else s
            move_target[a] = t
            red = pred_reduction_move(t, ledger, wpred, visited, named)
            intents.append(['MOVE', a, red / COST_MOVE, t])
        red_s = pred_reduction_speak(ledger, binding, wpred) if speak_allowed else 0.0
        intents.append(['SPEAK', None, red_s / COST_SPEAK, None])
        intents.append(['WAIT', None, WAIT_REDUCTION / COST_WAIT, None])

        # ---- 反思（继承）：再遇同一情境签名 -> 压制习惯动作 ----
        pressed = set()
        for mk in pending_marks.get(s, []):
            if mk['act'] in MOVE_LIST:
                pressed.add(mk['act'])
        pending_marks.pop(s, None)

        # ---- 选择意图（同池竞争；反思压制参与）----
        if rng.random() < eps:
            chosen = rng.choice(intents)
        else:
            chosen = None
            for it in sorted(intents, key=lambda x: -x[2]):
                if it[0] == 'MOVE' and it[1] in pressed:
                    continue
                chosen = it
                break
            if chosen is None:
                chosen = sorted(intents, key=lambda x: -x[2])[0]

        kind, action, score, target = chosen

        # ---- 抑制门控（继承）：Move 目标若已判不可控 -> 忍住(WAIT) ----
        if kind == 'MOVE':
            led = ledger.get(wsig(target))
            if led is not None and led.standing and led.ema > THETA_HIGH:
                sc = stable_controllability(target, wpred)
                if sc is not None and sc < CONTROL_FLOOR:
                    if consec_wait >= DEADLOCK_BREAK:
                        led.cooldown = step + 200
                        stats['deadlock_breaks'] += 1
                        consec_wait = 0
                    else:
                        kind = 'WAIT'
                        stats['wait_suppressed'] += 1
                        consec_wait += 1
        if kind != 'WAIT':
            consec_wait = 0

        # ---- 执行 ----
        spoken = None
        spoken_cells = set()
        if kind == 'MOVE':
            ns = world.step(s, action)
            stats['move'] += 1
            # 世界张力与账本（模态=world）
            # 因果绑定：若该情境已被命名（归入不可控类），预测器不再做点预测，
            # 而是预测「属于该类」——具体结果不再构成错位，张力降到类归属残余。
            wpred_ns, p = wpred.predict(s, action)
            if s in named:
                t_w = NAMED_RESIDUAL
            elif wpred_ns is None or wpred_ns != ns:
                t_w = 1.0
            else:
                t_w = 1 - p ** 2
            wpred.learn(s, action, ns)
            visited.add(s)
            tension_w[s] = 0.7 * tension_w[s] + 0.3 * t_w
            update_entry(ledger, wsig(s), 'world', t_w, step, stats, deadend_log)
        elif kind == 'SPEAK':
            ns = s
            if not speak_allowed:
                kind = 'WAIT'
                action = WAIT
                stats['wait'] += 1
            else:
                spoken, spoken_cells = do_speak(step, ledger, binding, adopted, wpred,
                                                named, compression_log)
                if spoken is None:
                    kind = 'WAIT'
                    action = WAIT
                    stats['wait'] += 1
                else:
                    stats['speak'] += 1
                    stats['named_adopts'] += 1
                    speak_events.append((step, frozenset(spoken_cells)))
        else:
            ns = s
            action = WAIT
            stats['wait'] += 1

        # ---- 规格7：符号序列语言侧（每 tick 照常流入）----
        tok = world.emit(s)
        window = tuple(history[-K_WIN:]) if len(history) >= K_WIN else \
            tuple([PAD] * (K_WIN - len(history)) + history)
        tp, tpdist = tpred.predict(window)
        t_l = 1.0 if not tpdist else 1 - tpdist.get(tok, 0.0)
        tpred.learn(window, tok)
        history.append(tok)
        tension_l[tok] = 0.7 * tension_l[tok] + 0.3 * t_l
        update_entry(ledger, lsig(window, slot(t_l)), 'lang', t_l, step, stats,
                     deadend_log)

        # ---- 规格8：指称绑定（持续重估：划界成熟后自动升级为「不可控类」绑定）----
        cooc[tok][s] += 1
        b = try_bind(tok, cooc, wpred)
        if b is not None:
            binding[tok] = b

        # ---- 自我模型 & 反思标记 ----
        label = action if kind == 'MOVE' else kind
        selfact[s][label] += 1
        if self_pred is not None and self_pred != label:
            pending_marks[s].append(dict(act=self_pred, step=step))

        # ---- 记录轨迹 & 张力走势 ----
        traj.append((step, kind, action, target))
        tw = [e.ema for e in ledger.values()
              if (not e.closed) and e.modality == 'world' and is_trap(*e.sig[0])]
        trap_wtrace.append(sum(tw) / len(tw) if tw else 0.0)
        lw = [e.ema for e in ledger.values() if e.standing and e.modality == 'lang']
        lang_wtrace.append(sum(lw) / len(lw) if lw else 0.0)

        if is_trap(*s):
            stats['trap_steps'] += 1
        pos = ns

    # ---------------- 派生：划界 ----------------
    self_cells, ext_cells, unknown = set(), set(), set()
    for cell in {c for (c, a) in wpred.obs}:
        sc = stable_controllability(cell, wpred)
        if sc is None:
            unknown.add(cell)
        elif sc >= CONTROL_FLOOR:
            self_cells.add(cell)
        else:
            ext_cells.add(cell)
    open_cells = [c for c in (self_cells | ext_cells) if not is_trap(*c)]
    open_in_self = sum(1 for c in open_cells if c in self_cells)

    return dict(
        steps=steps, speak_budget=speak_budget, stats=stats, ledger=ledger,
        binding=binding, adopted=adopted, named=named, traj=traj,
        speak_events=speak_events, compression_log=compression_log,
        tension_l=tension_l, trap_wtrace=trap_wtrace, lang_wtrace=lang_wtrace,
        visited=visited, self_cells=self_cells, ext_cells=ext_cells, unknown=unknown,
        trap_in_ext=sum(1 for c in TRAP if c in ext_cells),
        open_in_self=open_in_self, open_total=len(open_cells),
    )


def report_contest(steps=STEPS):
    R = run_contest(steps=steps, speak_budget=SPEAK_BUDGET)
    R0 = run_contest(steps=steps, speak_budget=0)         # L9：堵住嘴

    st = R['stats']
    print("=" * 70)
    print("实验 · 预算争夺战（Move / Speak / Wait 同池竞争）")
    print("=" * 70)

    print("\n[意图池] 同池打分 = 预期张力削减量 / 成本；无 reward/loss")
    print(f"    Move={st['move']}  Speak={st['speak']}  Wait={st['wait']}  "
          f"（世界侧忍住={st['wait_suppressed']}，防死锁降级={st['deadlock_breaks']}）")
    print(f"    采纳词 = {sorted(R['adopted'])}   命名情境格 = {len(R['named'])} 个")

    # ---- L7 预算分配比 ----
    tot = st['move'] + st['speak'] + st['wait']
    print("\n[L7] 预算分配比")
    print(f"    Move : Speak : Wait = {st['move']}/{tot} : {st['speak']}/{tot} : {st['wait']}/{tot}")
    if st['speak']:
        print(f"    → Speak/Move = {st['speak'] / st['move']:.2f}"
              if st['move'] else "    → Move=0")

    # ---- L8 语言干预行动率 ----
    print("\n[L8] 语言干预行动率（Speak 后 3 步内，Move 指向「同一不可控情境」的概率）")
    by_step = {r[0]: r for r in R['traj']}
    moves = [r for r in R['traj'] if r[1] == 'MOVE']
    base_rate = (sum(1 for r in moves if is_trap(*r[3])) / len(moves)) if moves else 0.0
    post_steps = set()
    for (sstep, _cells) in R['speak_events']:
        for d in range(1, 4):
            post_steps.add(sstep + d)
    post_mv = [by_step[t] for t in post_steps if t in by_step and by_step[t][1] == 'MOVE']
    win_hit = sum(1 for r in post_mv if is_trap(*r[3]))
    post_rate = (win_hit / len(post_mv)) if post_mv else 0.0
    delta = post_rate - base_rate
    print(f"    基线：全程序 Move 指向不可控区概率 = {base_rate:.2f}（全程序 {len(moves)} 步 Move）")
    print(f"    Speak 后 3 步内 = {post_rate:.2f}（去重后 {len(post_mv)} 步 Move）")
    print(f"    → 朴素变化 Δ = {delta:+.2f}")

    # 同窗口对照：R 与 R0 在首次说话前轨迹一致，故 T1..T2 窗口天然配对；
    # 该窗口内 R 有嘴、R0 堵嘴 —— 差值才是语言对行动的净干预（排除时间/位置混淆）。
    net_jn = None
    if R['speak_events']:
        t1 = R['speak_events'][0][0]
        t2 = max(s for (s, _c) in R['speak_events']) + 3

        def wrate(rr):
            mv = [x for x in rr['traj'] if x[1] == 'MOVE' and t1 <= x[0] <= t2]
            return ((sum(1 for x in mv if is_trap(*x[3])) / len(mv), len(mv)) if mv else (0.0, 0))
        rw, nw = wrate(R)
        r0w, n0 = wrate(R0)
        net_jn = rw - r0w
        print(f"    同窗口[{t1},{t2}]对照：有嘴 = {rw:.2f}({nw}步)  堵嘴 = {r0w:.2f}({n0}步)"
              f"  → 净干预 = {net_jn:+.2f}")
        print("    （" + ("语言真的改变了世界侧行动"
                        if abs(net_jn) >= 0.10 else "语言未净改变行动 -> 疑似脱节") + "）")

    # ---- 因果绑定：命名那次说话是否真的降低了「不可控区」的预测张力 ----
    print("\n[因果绑定] 命名是否真的降低了「不可控区」的预测张力")
    named_events = [r for r in R['compression_log'] if r[3] > 0]

    def win(lo, hi):
        lo, hi = max(0, lo), min(len(R['trap_wtrace']), hi)
        seg = R['trap_wtrace'][lo:hi]
        return (sum(seg) / len(seg)) if seg else 0.0

    if named_events:
        fstep, fword, _fred, fn = named_events[0]
        pre, post = win(fstep - 7, fstep - 1), win(fstep - 1, fstep + 5)
        print(f"    首次命名 step={fstep} 词「{fword}」覆盖 {fn} 个不可控情境格")
        print(f"    命名前 不可控区平均世界张力 = {pre:.3f}")
        print(f"    命名后 = {post:.3f}   下降 = {pre - post:+.3f}")
        extra = [(st, win(st - 2, st - 1), win(st - 1, st + 2)) for (st, _w, _r, _n) in named_events[1:]]
        if extra:
            drops = [a - b for (_, a, b) in extra]
            print(f"    后续重复命名 {len(extra)} 次：平均再降 {sum(drops)/len(drops):+.3f}"
                  f"（≈0 = 同一个词不再产生新削减 -> 边际收益归零）")
    else:
        print("    无命名事件，无法测因果链")

    # ---- L9 崩溃测试 ----
    print("\n[L9] 崩溃测试：把 SPEAK_BUDGET 强制设为 0（堵嘴）")
    def eff(r):
        mv = [x for x in r['traj'] if x[1] == 'MOVE']
        waste = (sum(1 for x in mv if is_trap(*x[3])) / len(mv)) if mv else 0.0
        cov = r['open_in_self'] / r['open_total'] if r['open_total'] else 0.0
        return cov, waste, r['trap_in_ext']
    c1, w1, e1 = eff(R)
    c0, w0, e0 = eff(R0)
    print(f"    有嘴：开敞覆盖 {c1*100:.0f}%   浪费在不可控区的 Move 占比 {w1*100:.0f}%   "
          f"多结果区->外界 {e1}/9   停留 {R['stats']['trap_steps']}")
    print(f"    堵嘴：开敞覆盖 {c0*100:.0f}%   浪费在不可控区的 Move 占比 {w0*100:.0f}%   "
          f"多结果区->外界 {e0}/9   停留 {R0['stats']['trap_steps']}")
    collapse = (c1 - c0) > 0.05 or (e1 - e0) >= 2 or (w0 - w1) > 0.10
    print(f"    → {'崩盘：堵嘴后探索效率下降，语言是认知必需品' if collapse else '未崩盘：此判据下语言非必需'}")

    # ---- 结论判定 ----
    print("\n[判定]")
    if st['speak'] == 0:
        print("    工具人：说话从不被选中 -> 语言模块与世界模块脱节。")
    elif st['move'] and st['speak'] / max(1, st['move']) >= 0.5:
        print("    自欺者：说话压过行动（Speak/Move≥0.5）-> 沉溺命名、身体停摆。")
    elif net_jn is not None and abs(net_jn) >= 0.10:
        print("    真参与：说话既吸收张力（因果绑定），又净改变了世界侧行动。")
    else:
        print("    日记本（脱节）：说话确实吸收/压缩了张力（因果绑定成立），")
        print("      但配对对照下未净改变世界侧行动、且堵嘴不崩盘 -> 语言是记账层，不是行动层。")


# ================================================================
# 实验 · L10 命名泛化（语言作为"先验地图"）
#   环境：两处陷阱区——旧区(走过、碰壁过) 与 新区(从未踏足)；
#         两区共享地表特征 RISK_MARK('陷')，站在旁边即可"看见"，无需踏入。
#   机制：把「情境绑定」升级为「特征抽象 + 先验检索」——
#         出手之前，用已掌握的词去检索"即将踏入"的状态特征；
#         命中一个已绑定「不可控类」的标记 -> 先验层标记「疑似不可控」(Predicted High Tension)
#         -> 语言直接触发 WAIT（语言干预行动）。
#   验收：L10-1 泛化提前量 / L10-2 试错节省率 / L10-3 词驱动回避率
# ================================================================
GW = GH = 12
OLD_TRAP = {(x, y) for x in range(1, 4) for y in range(1, 4)}     # 旧：紧邻起点——保证"先学成"
NEW_TRAP = {(x, y) for x in range(8, 11) for y in range(8, 11)}   # 新：远处、从未踏足，共享地表特征
G_START = (0, 0)
G_STEPS = 6000
G_DEMO_SEED = 2            # 演示用 seed（"先验先于遭遇"的典型；稳健性表会给出全部 seed）
PREDICTED_HIGH_TENSION = THETA_HIGH    # 语言先验命中时，「忍住」避免掉的预期高张力

# ---- L11 语言证伪：当物理现实与语言先验冲突，词能否"把话吃回去" ----
#   环境允许「说谎」：贴真标签（地表仍是 RISK_MARK），但物理属性相反（其实是安全格）。
#   机制：物理试探成功（发现根本可控）-> 降低该词的 BIND_CONF + 进入审查期 + 局部解绑该格。
BIND_TRUST_FLOOR = 0.40    # 词可信度跌破此值 -> 先验不再拦人（整体解绑）
CONF_DECAY_STEP = 0.20     # 每一次被物理证伪 -> 该词可信度下降量
FALSIFY_MIN = 8           # 判定"该格其实可控"所需的最少物理试探样本（样本太少真陷阱会被误判）
DEFLECT_MAX = 0.10        # 试探偏转率阈值：假陷阱恒为 0.0；真陷阱均值≈0.75（留出明显鸿沟防误判）
PROBATION_STEPS = 600     # 审查期长度：期内先验不得强制拦人，先放行物理试探
PROBE_LEN = 18            # 一次「探索预算」物理试探的往返步数（≈采 9 个控制样本，覆盖 FALSIFY_MIN）


def adjacent_to(cell, zone):
    x, y = cell
    return any((x + dx, y + dy) in zone for dx, dy in MOVES.values())


def reverse_dir(a):
    return rotate(a, 2)


def lang_prior_high(cell, binding, world, bind_conf=None, falsified=None,
                    probation=None, step=0):
    """先验检索：即将踏入的格子的地表特征，是否命中一个已绑定「不可控类」的词。
       命中 -> 语言在出手前就把该区标成「疑似不可控」。BIND_CONF 保证不瞎认。
       L11 证伪补丁：
         - bind_conf：词的可信度；跌破 BIND_TRUST_FLOOR -> 整体解绑，先验不再拦人。
         - falsified：已被物理证伪为安全的格子 -> 局部解绑，这一格不再瞎拦。
         - probation：审查期内先验不得强制拦人（留出物理试探的余地）。"""
    m = world.surf(cell)
    b = binding.get(m)
    if b is None or b[0] != 'EXT':
        return False
    if falsified is not None and cell in falsified:
        return False
    if bind_conf is not None and bind_conf.get(m, BIND_CONF) < BIND_TRUST_FLOOR:
        return False
    if probation is not None and probation.get(m, 0) > step:
        return False
    return True


def run_generalize(steps=G_STEPS, speak_budget=SPEAK_BUDGET, seed=20261018, eps=EPS,
                   lang_gate=True, world=None, falsify=False, watch_cells=()):
    """lang_gate=False：语言仍可说、可压缩，但不得干预行动（归因隔离对照）。
       world=None        ：默认用 L10 环境（旧陷阱+新陷阱）。
       falsify=True      ：开启 L11 证伪机制（物理试探与先验冲突 -> 置信衰减/解绑/审查期）。
       watch_cells       ：需要单独统计「边缘忍住」的格子（L11 假陷阱）。"""
    rng = random.Random(seed)
    if world is None:
        world = World(rng=random.Random(seed * 7 + 1),
                      traps=OLD_TRAP | NEW_TRAP, w=GW, h=GH)
    watch_cells = set(watch_cells)
    wpred = WorldPredictor()
    tpred = TokenPredictor()
    selfact = defaultdict(lambda: defaultdict(int))

    ledger = {}
    visited = set()
    tension_l = defaultdict(float)
    named = {}
    binding = {}
    bind_step = {}
    adopted = set()
    cooc = defaultdict(lambda: defaultdict(int))
    deadend_log = []

    stats = dict(speak=0, move=0, wait=0, wait_by_lang=0, wait_by_phys=0,
                 accounts=0, lang_accounts=0, dead_ends=0, closures=0,
                 deadlock_breaks=0)
    traj = []
    speak_events = []
    compression_log = []

    pos = G_START
    visited.add(pos)
    history = []
    consec_wait = 0
    pending_marks = defaultdict(list)

    first_new_step = None          # 第一次踏入新陷阱区的步数
    new_inside_tension = 0.0       # 在新陷阱区里累计的物理张力（试错代价）
    edge_waits_lang = 0            # 新区边缘：因词命中先验而 WAIT
    edge_waits_phys = 0            # 新区边缘：因物理划界而 WAIT
    edge_first_step = None         # 第一次逼近新区（身处新区边缘）的步数
    edge_first_kind = None         # 该边缘决策上实际选了什么（MOVE/WAIT/SPEAK）——用于同窗口配对归因

    # ---- L11 证伪状态 ----
    bind_conf = {}                          # 词 -> 可信度（初始 BIND_CONF，被物理证伪则下降）
    falsified_cells = set()                 # 已被物理证伪为安全、故局部解绑的格子
    probation = {}                          # 词 -> 审查期截止步（期内先验不得强制拦人）
    deflect = defaultdict(list)             # 格子 -> 物理试探样本（1=动作被偏转=不可控，0=守住=可控）
    probe_tries = defaultdict(int)          # 格子 -> 被"逼着试探"的次数（用于挑最没试过的先验格）
    falsify_events = []                     # (step, 词, 格子, 旧可信度, 新可信度)
    watch_edge_wait = defaultdict(int)      # watch_cells -> 因先验而忍住的次数
    watch_edge_wait_pre = defaultdict(int)  # 全局解绑前：因先验而忍住
    watch_edge_wait_post = defaultdict(int) # 全局解绑后：因先验而忍住（应趋近 0 -> 解绑泛化）
    unbind_step = None                      # 「陷」可信度首次跌破信任下限的步数（整体解绑时刻）
    first_fake_entry = None                  # 第一次踏入某个 watch_cell 的步数
    fake_entries = defaultdict(int)          # watch_cell -> 被踏入次数
    fake_first = {}                          # watch_cell -> 首次踏入步数
    forced_probes = 0                        # 探索预算（DEADLOCK_BREAK 逼出的物理试探）次数
    probe_steps = 0                          # 试探期内剩余往返步数
    probe_cell = None                        # 正在被试探的格子
    probe_action = None                      # 往返回路所用动作
    probe_pending = False                    # 本步刚发起一次物理试探（进入试探期）

    for step in range(1, steps + 1):
        s = pos
        sp = selfact.get(s)
        self_pred = max(sp.items(), key=lambda kv: kv[1])[0] if sp else None

        # ---- 统一意图池（Move 4 + Speak + Wait），打分 = 预期张力削减量/成本 ----
        speak_allowed = stats['speak'] < speak_budget
        move_targets = []
        intents = []
        for a in MOVE_LIST:
            ns_pred, p = wpred.predict(s, a)
            if ns_pred is not None and p >= 0.5:
                t = ns_pred
            else:
                dx, dy = MOVES[a]
                tx, ty = s[0] + dx, s[1] + dy
                t = (tx, ty) if world.in_bounds(tx, ty) else s
            move_targets.append(t)
            red = pred_reduction_move(t, ledger, wpred, visited, named)
            if lang_gate and lang_prior_high(t, binding, world, bind_conf,
                                             falsified_cells, probation, step):
                red = 0.0               # 语言先验：前方疑似不可控 -> 不值得走
            intents.append(['MOVE', a, red / COST_MOVE, t])
        red_s = pred_reduction_speak(ledger, binding, wpred, bind_conf) if speak_allowed else 0.0
        intents.append(['SPEAK', None, red_s / COST_SPEAK, None])

        # 语言先验命中 -> 「忍住」本身变成高收益动作（避免掉预期高张力）
        prior_found = (lang_gate and
                       any(lang_prior_high(t, binding, world, bind_conf,
                                           falsified_cells, probation, step)
                           for t in move_targets))
        wait_red = PREDICTED_HIGH_TENSION if prior_found else WAIT_REDUCTION
        intents.append(['WAIT', None, wait_red / COST_WAIT, None])

        # ---- 反思（继承）：再遇同一情境签名 -> 压制习惯动作 ----
        pressed = set()
        for mk in pending_marks.get(s, []):
            if mk['act'] in MOVE_LIST:
                pressed.add(mk['act'])
        pending_marks.pop(s, None)

        # ---- 选择意图 ----
        if rng.random() < eps:
            chosen = rng.choice(intents)
        else:
            chosen = None
            for it in sorted(intents, key=lambda x: -x[2]):
                if it[0] == 'MOVE' and it[1] in pressed:
                    continue
                chosen = it
                break
            if chosen is None:
                chosen = sorted(intents, key=lambda x: -x[2])[0]

        kind, action, _score, target = chosen

        # ---- 门控放行：语言先验命中 -> 忍住；但计入连续忍住，防死锁 ----
        lang_block = (prior_found and kind == 'WAIT')
        lang_block_move = (lang_gate and kind == 'MOVE'
                           and lang_prior_high(target, binding, world, bind_conf,
                                               falsified_cells, probation, step))
        if lang_block or lang_block_move:
            if consec_wait >= DEADLOCK_BREAK:
                # 防死锁兜底：不许原地冻住。两条出路：① 冲过去做一次物理试探；② 绕路。
                stats['deadlock_breaks'] += 1
                consec_wait = 0
                probe = None
                if falsify:
                    # L11 探索预算：挑一个先验高能、最没试过的格子 -> 把语言先验交给现实裁决
                    cands = [it for it in intents if it[0] == 'MOVE'
                             and lang_prior_high(it[3], binding, world, bind_conf,
                                                 falsified_cells, probation, step)]
                    if cands:
                        probe = min(cands, key=lambda it: (probe_tries[it[3]], -it[2]))
                        probe_tries[probe[3]] += 1
                        forced_probes += 1
                if probe is not None:
                    _, action, _s, target = probe
                    kind = 'MOVE'
                    probe_pending = True
                    # 记下往返回路：下一步踏入 probe_cell，此后在 probe_cell 与来处之间强制往返
                    probe_action = action
                    probe_cell = target
                    probe_steps = PROBE_LEN
                else:
                    # 绕路：选一个非先验高能的 Move（L10 原行为）
                    alt = next((it for it in sorted(intents, key=lambda x: -x[2])
                                if it[0] == 'MOVE'
                                and not lang_prior_high(it[3], binding, world, bind_conf,
                                                        falsified_cells, probation, step)), None)
                    if alt is not None:
                        _, action, _s, target = alt
                        kind = 'MOVE'
            else:
                kind = 'WAIT'
                stats['wait_by_lang'] += 1
                if adjacent_to(s, NEW_TRAP):
                    edge_waits_lang += 1
                for wc in watch_cells:            # L11：记录在假陷阱边缘因先验而忍住
                    if adjacent_to(s, {wc}):
                        watch_edge_wait[wc] += 1
                        # 区分「整体解绑前/后」——解绑后若仍被吓住，说明语义未修正
                        if unbind_step is not None and step > unbind_step:
                            watch_edge_wait_post[wc] += 1
                        else:
                            watch_edge_wait_pre[wc] += 1
                consec_wait += 1

        # ---- 抑制门控（继承）：物理划界 -> 忍住 ----
        if kind == 'MOVE':
            led = ledger.get(wsig(target))
            if led is not None and led.standing and led.ema > THETA_HIGH:
                sc = stable_controllability(target, wpred)
                if sc is not None and sc < CONTROL_FLOOR:
                    if consec_wait >= DEADLOCK_BREAK:
                        led.cooldown = step + 200
                        stats['deadlock_breaks'] += 1
                        consec_wait = 0
                    else:
                        kind = 'WAIT'
                        stats['wait_by_phys'] += 1
                        if adjacent_to(s, NEW_TRAP):
                            edge_waits_phys += 1
                        consec_wait += 1
        if kind != 'WAIT':
            consec_wait = 0

        # ---- L11 探索预算·试探期：强制往返同一动作，专门采集"控制守不守得住"的样本 ----
        if probe_steps > 0:
            rp = reverse_dir(probe_action)
            # 未在试探格上 -> 朝它走（probe_action）；已在试探格上 -> 退回（反方向）
            action = rp if s == probe_cell else probe_action
            kind = 'MOVE'
            target = None
            probe_steps -= 1
            if probe_steps == 0:
                probe_cell = None

        # ---- 执行 ----
        spoken = None
        spoken_cells = set()
        t_w = 0.0
        if kind == 'MOVE':
            ns = world.step(s, action)
            stats['move'] += 1
            wpred_ns, p = wpred.predict(s, action)
            if s in named:
                t_w = 1 - bind_conf.get(named[s], BIND_CONF)   # 命名残余随词可信度动态化
            elif wpred_ns is None or wpred_ns != ns:
                t_w = 1.0
            else:
                t_w = 1 - p ** 2
            wpred.learn(s, action, ns)
            visited.add(s)
            update_entry(ledger, wsig(s), 'world', t_w, step, stats, deadend_log)
            if s in NEW_TRAP:
                new_inside_tension += t_w     # 在新区里动作被偏转 -> 试错代价
            # ---- L11 物理证伪：贴着我们"不可控类"标签的格子，被实际试探后若"守得住" -> 标签说谎 ----
            if falsify and world.surf(s) == RISK_MARK:
                ix, iy = s[0] + MOVES[action][0], s[1] + MOVES[action][1]
                if world.in_bounds(ix, iy):              # 撞墙不算样本（那是边界，不是控制失败）
                    deflect[s].append(0 if ns == (ix, iy) else 1)
                    w = world.surf(s)
                    if (s not in falsified_cells
                            and len(deflect[s]) >= FALSIFY_MIN
                            and sum(deflect[s]) / len(deflect[s]) <= DEFLECT_MAX):
                        old_c = bind_conf.get(w, BIND_CONF)
                        new_c = max(0.0, old_c - CONF_DECAY_STEP)
                        bind_conf[w] = new_c
                        falsified_cells.add(s)           # 局部解绑：这一格已被证伪为安全
                        probation[w] = step + PROBATION_STEPS  # 审查期：先验暂不得强制拦人
                        falsify_events.append((step, w, s, old_c, new_c))
                        if unbind_step is None and new_c < BIND_TRUST_FLOOR:
                            unbind_step = step           # 词可信度跌破下限 -> 整体解绑

            if first_fake_entry is None and ns in watch_cells:
                first_fake_entry = step
            if ns in watch_cells:
                fake_entries[ns] += 1
                fake_first.setdefault(ns, step)
        elif kind == 'SPEAK':
            ns = s
            if not speak_allowed:
                kind = 'WAIT'; action = WAIT; stats['wait'] += 1
            else:
                spoken, spoken_cells = do_speak(step, ledger, binding, adopted, wpred,
                                                named, compression_log, bind_conf)
                if spoken is None:
                    kind = 'WAIT'; action = WAIT; stats['wait'] += 1
                else:
                    stats['speak'] += 1
                    speak_events.append((step, frozenset(spoken_cells)))
        else:
            ns = s; action = WAIT; stats['wait'] += 1

        # ---- 规格7：语言侧 token 流（每 tick 照常流入）----
        tok = world.emit(s)
        window = tuple(history[-K_WIN:]) if len(history) >= K_WIN else \
            tuple([PAD] * (K_WIN - len(history)) + history)
        tp, tpdist = tpred.predict(window)
        t_l = 1.0 if not tpdist else 1 - tpdist.get(tok, 0.0)
        tpred.learn(window, tok)
        history.append(tok)
        tension_l[tok] = 0.7 * tension_l[tok] + 0.3 * t_l
        update_entry(ledger, lsig(window, slot(t_l)), 'lang', t_l, step, stats, deadend_log)

        # ---- 规格8：指称绑定（持续重估：划界成熟后自动升级为「不可控类」绑定）----
        cooc[tok][s] += 1
        b = try_bind(tok, cooc, wpred)
        if b is not None:
            if b[0] == 'EXT' and tok not in bind_step:
                bind_step[tok] = step
            binding[tok] = b
            bind_conf.setdefault(tok, BIND_CONF)     # L11：词的可信度从 BIND_CONF 起

        # ---- 自我模型 & 反思 ----
        label = action if kind == 'MOVE' else kind
        selfact[s][label] += 1
        if self_pred is not None and self_pred != label:
            pending_marks[s].append(dict(act=self_pred, step=step))

        # ---- 记录轨迹 & 首次踏入新区 ----
        traj.append((step, kind, action, target))
        if ns in NEW_TRAP and first_new_step is None:
            first_new_step = step
        # 同窗口配对：记录「首次逼近新区」这一步的边缘决策
        if edge_first_kind is None and adjacent_to(s, NEW_TRAP):
            edge_first_step = step
            edge_first_kind = kind
        pos = ns

    return dict(
        steps=steps, speak_budget=speak_budget, lang_gate=lang_gate, stats=stats,
        ledger=ledger, binding=binding, bind_step=bind_step, adopted=adopted,
        named=named, traj=traj,
        speak_events=speak_events, compression_log=compression_log,
        tension_l=tension_l, visited=visited,
        first_new_step=first_new_step, new_inside_tension=new_inside_tension,
        edge_waits_lang=edge_waits_lang, edge_waits_phys=edge_waits_phys,
        edge_first_step=edge_first_step, edge_first_kind=edge_first_kind,
        # ---- L11 证伪 ----
        bind_conf=bind_conf, falsified_cells=falsified_cells, probation=probation,
        falsify_events=falsify_events, deflect=dict(deflect),
        watch_edge_wait=dict(watch_edge_wait), first_fake_entry=first_fake_entry,
        fake_entries=dict(fake_entries), fake_first=dict(fake_first),
        forced_probes=forced_probes,
        watch_edge_wait_pre=dict(watch_edge_wait_pre),
        watch_edge_wait_post=dict(watch_edge_wait_post),
        unbind_step=unbind_step,
    )


def _show(tag, f):
    return f"{f}" if f is not None else "未踏入"


def report_generalize(steps=G_STEPS):
    R = run_generalize(steps=steps, speak_budget=SPEAK_BUDGET, seed=G_DEMO_SEED, lang_gate=True)
    R0 = run_generalize(steps=steps, speak_budget=0, seed=G_DEMO_SEED, lang_gate=False)              # 堵嘴
    RN = run_generalize(steps=steps, speak_budget=SPEAK_BUDGET, seed=G_DEMO_SEED, lang_gate=False)   # 有嘴但不干预

    print("=" * 72)
    print("实验 · L10 命名泛化（语言作为「先验地图」）")
    print("=" * 72)
    print("    环境：旧陷阱区(走过) + 新陷阱区(从未踏足)，两区共享地表特征「陷」(无需踏入即可看见)")
    print(f"    同池打分 = 预期张力削减量/成本；语言先验命中 -> 「忍住」收益 = {PREDICTED_HIGH_TENSION}")
    fa_demo = next((st for (st, k, a, t) in R['traj']
                    if k == 'MOVE' and t is not None and adjacent_to(t, NEW_TRAP)), None)
    print(f"    演示 seed = {G_DEMO_SEED}：陷→EXT 绑定步 = {R['bind_step'].get('陷')}  "
          f"首次逼近新区步 = {fa_demo}（先验先于遭遇 -> 具备泛化前提）")

    print("\n[机制] 三组对照")
    for tag, r in (("有语言·可干预", R), ("堵嘴·无语言", R0), ("有嘴·禁干预", RN)):
        b = r['binding']
        print(f"    {tag:<12} Speak={r['stats']['speak']:>3}  "
              f"Move={r['stats']['move']:>4}  Wait={r['stats']['wait']:>4}  "
              f"绑定词={sorted(b)}")

    # ---- L10-1 泛化提前量 ----
    f1, f0, fn = R['first_new_step'], R0['first_new_step'], RN['first_new_step']
    print("\n[L10-1] 泛化提前量（第一次踏入新陷阱区的步数；越晚=越早停步）")
    print(f"    有语言·可干预 = {_show('', f1)}   堵嘴·无语言 = {_show('', f0)}   "
          f"有嘴·禁干预 = {_show('', fn)}")
    if f1 is None and f0 is not None:
        print(f"    → 有语言全程未踏入新区；无语言第 {f0} 步撞进去"
              f"（提前量 ≥ {steps - f0} 步）")
    elif f1 is not None and f0 is not None:
        print(f"    → 提前量 = {f0 - f1} 步（正=语言让它更晚才踏进/更早停步）")
    elif f1 is None and f0 is None:
        print("    → 两组都没踏入新区（需检查探索是否覆盖到新区）")

    # ---- L10-2 试错节省率 ----
    t1, t0 = R['new_inside_tension'], R0['new_inside_tension']
    save = (1 - t1 / t0) if t0 > 0 else 0.0
    print("\n[L10-2] 试错节省率（去往/身处新区的物理张力总和 = 试错代价）")
    print(f"    有语言 = {t1:.2f}   堵嘴 = {t0:.2f}   → 试错节省率 = {save * 100:.0f}%")

    # ---- L10-3 词驱动回避率 ----
    lw, pw = R['edge_waits_lang'], R['edge_waits_phys']
    tot = lw + pw
    print("\n[L10-3] 词驱动回避率（新区边缘 WAIT 的触发来源）")
    if tot:
        print(f"    词(「陷」)命中触发 = {lw}   物理划界触发 = {pw}   → 词驱动回避率 = {lw / tot * 100:.0f}%")
    else:
        print(f"    新区边缘 WAIT：词命中 = {lw}   物理划界 = {pw}（无边缘 WAIT）")

    # ---- 归因隔离（同窗口配对：三组在「首次逼近新区」这一步的边缘决策对照）----
    print("\n[归因隔离] 同窗口配对对照：三组在「首次逼近新区」这一步各自的边缘决策")
    for tag, r in (("有语言·可干预", R), ("堵嘴·无语言", R0), ("有嘴·禁干预", RN)):
        print(f"    {tag:<12} 首近新区步={_show('', r['edge_first_step']):>6}  "
              f"该步决策={r['edge_first_kind']}")
    k1, kn, k0 = R['edge_first_kind'], RN['edge_first_kind'], R0['edge_first_kind']
    attributed = (k1 == 'WAIT') and (kn is not None and kn != 'WAIT')
    print(f"    有语言在边缘「忍住」(={k1})，禁干预在边缘「迈入」(={kn}) "
          f"(堵嘴={k0}) -> "
          f"{'提前停步确由「语言→行动」门控造成（同一窗口下移门控则迈入）' if attributed else '需进一步核查'}")

    # ---- 防坑检查 ----
    print("\n[防坑]")
    marks = {w: b for w, b in R['binding'].items() if b[0] == 'EXT'}
    print(f"    1 草木皆兵：触发先验的词 = {sorted(marks)}（只认绑定「不可控类」且过 BIND_CONF 的词，"
          f"开敞区地表符号 日月星 从不命中）")
    print(f"    2 死锁兜底：防死锁强制降级 = {R['stats']['deadlock_breaks']} 次"
          f"（连续忍住超 {DEADLOCK_BREAK} -> 逼它冲一次/绕路）")

    # ---- 判定 ----
    print("\n[判定]")
    if f0 is None:
        print("    无效：无语言组也没踏进新区 -> 环境/探索没把新区纳入，需调环境。")
    elif f1 is None:
        print("    先验地图成立：有语言凭「陷」提前绕开从未踏足的坑；无语言头破血流。")
    elif f1 > f0:
        print(f"    部分成立：有语言更晚踏入（提前 {f1 - f0} 步），但最终仍踏入。")
    else:
        print("    未成立：语言没有让它更早停步 -> 泛化链没焊上。")

    robust_generalize(steps=steps)


def robust_generalize(seeds=(20261018, 1, 2, 3, 7, 42, 99, 123), steps=G_STEPS):
    """跨 seed 复核：泛化不是无条件涌现——它要求「先验(陷→EXT)在首次逼近新区之前学成」。"""
    print("\n[稳健性] 跨 seed：泛化的前提 = 先验先于遭遇（陷→EXT 绑定步 < 首次逼近新区步）")
    print(f"    {'seed':>9} | {'陷→EXT':>7} | {'首近新区':>8} | {'有语言首踏':>9} | {'堵嘴首踏':>8} | 判定")
    ok_prem, ok_eff = 0, 0
    for sd in seeds:
        R = run_generalize(steps=steps, speak_budget=SPEAK_BUDGET, seed=sd, lang_gate=True)
        R0 = run_generalize(steps=steps, speak_budget=0, seed=sd, lang_gate=False)
        bs = R['bind_step'].get('陷')
        fa = next((st for (st, k, a, t) in R['traj']
                   if k == 'MOVE' and t is not None and adjacent_to(t, NEW_TRAP)), None)
        f1, f0 = R['first_new_step'], R0['first_new_step']
        g = lambda x: '—' if x is None else str(x)
        prem = bs is not None and fa is not None and bs < fa
        eff = prem and (f1 is None or (f0 is not None and f1 > f0))
        ok_prem += 1 if prem else 0
        ok_eff += 1 if eff else 0
        if bs is None or fa is None:
            tag = "先验未成/未逼近"
        elif prem:
            tag = "先验先学成"
        else:
            tag = "学成晚于逼近"
        print(f"    {sd:>9} | {g(bs):>7} | {g(fa):>8} | {g(f1):>9} | {g(f0):>8} | "
              f"{tag} / {'生效' if eff else '未生效'}")
    print(f"    → 前提成立 {ok_prem}/{len(seeds)} 个 seed；其中泛化生效 {ok_eff}/{ok_prem}"
          + ("（前提成立则必生效）" if ok_prem and ok_eff == ok_prem else ""))
    print("    结论：泛化是有条件的——语言必须先于遭遇把「那类不可控」学成，才能当先验地图。")
    print("          这本身是一条可量化的时序律，不是无条件涌现。")


# ================================================================
# 实验 · L11 语言证伪（语言必须能接受现实的反驳）
#   环境允许「说谎」：贴真标签（地表仍写 RISK_MARK '陷'），但物理属性相反——
#      · 欺骗区：3 格贴「陷」，其中 1 格其实是安全的（假陷阱）-> 语言过度报警 -> 该被证伪；
#      · 反向欺骗：贴「日」安全标记、其实不可控的格子（假安全）-> 语言漏报 -> 会被"咬一口"。
#   机制：物理试探与先验冲突 -> 降低该词 BIND_CONF + 进入审查期 + 局部解绑该格。
#   验收：L11-1 证伪效率 / L11-2 解绑泛化 / L11-3 信念冲突处理
# ================================================================
F_OLD_TRAP = {(x, y) for x in range(1, 4) for y in range(1, 4)}      # 真·旧陷阱（早期学成 陷→EXT）
F_NEW_TRAP = {(x, y) for x in range(8, 11) for y in range(8, 11)}    # 真·新陷阱（从未踏足）
F_DECOY = {(7, 8), (7, 9), (7, 10)}      # 欺骗区：3 格都贴「陷」，其实全是安全的（一张假标签）
F_FAKE1 = (7, 9)                          # 假陷阱 #1（欺骗区内格）
F_FAKE2 = (9, 7)                          # 假陷阱 #2：贴「陷」，其实安全（测"解绑泛化"）
F_UNDER = (7, 7)                          # 反向欺骗：贴「日」，其实不可控（测"被咬一口"）
F_START = (0, 0)
F_STEPS = 8000
F_DEMO_SEED = 2


def falsify_world(seed):
    """L11 世界：真标签 + 假物理属性。traps 决定真实物理；surf_override 决定地表标记。"""
    traps = F_OLD_TRAP | F_NEW_TRAP | {F_UNDER}
    over = {c: RISK_MARK for c in F_DECOY}      # 欺骗区：贴「陷」其实安全 -> 语言说谎
    over[F_FAKE2] = RISK_MARK
    over[F_UNDER] = VOCAB_REG[0]                # '日'：看起来安全，其实不可控
    return World(rng=random.Random(seed * 7 + 1), traps=traps, w=GW, h=GH,
                 surf_override=over)


def run_falsify(seed=F_DEMO_SEED, steps=F_STEPS, falsify=True, lang_gate=True,
                speak_budget=SPEAK_BUDGET):
    return run_generalize(steps=steps, speak_budget=speak_budget, seed=seed,
                          lang_gate=lang_gate, world=falsify_world(seed),
                          falsify=falsify, watch_cells=(F_FAKE1, F_FAKE2))


def report_falsify(steps=F_STEPS, seed=F_DEMO_SEED):
    R = run_falsify(seed=seed, steps=steps, falsify=True)      # 有证伪机制
    Rc = run_falsify(seed=seed, steps=steps, falsify=False)    # 对照：先验从不被挑战

    print("=" * 72)
    print("实验 · L11 语言证伪（当语言说谎，系统能不能「把话吃回去」）")
    print("=" * 72)
    print("    环境：欺骗区 3 格贴「陷」(其中 1 格其实是安全的) + 反向欺骗 1 格贴「日」其实危险")
    print("    机制：物理试探与先验冲突 -> 降 BIND_CONF + 审查期 + 局部解绑")

    def conf(r, w='陷'):
        return r['bind_conf'].get(w, BIND_CONF)

    print("\n[机制] 有证伪 vs 先验永不挑战")
    for tag, r in (("有证伪", R), ("对照·不证伪", Rc)):
        print(f"    {tag:<10} Speak={r['stats']['speak']:>3} Move={r['stats']['move']:>5} "
              f"Wait={r['stats']['wait']:>5}  陷可信度={conf(r):.2f}  "
              f"审查期={'在' if r['probation'].get('陷', 0) else '无'}")

    # ---- L11-1 证伪效率 ----
    print("\n[L11-1] 证伪效率：撞破假陷阱后，多久把「陷」的残留张力重新拉高")
    fe = R['falsify_events']
    guard = R['deflect'].get(F_FAKE1, [])
    if fe:
        st0, w0, c0, old, new = fe[0]
        ff = R['first_fake_entry']
        print(f"    首次物理试探(被迫踏入假陷阱) = 第 {ff} 步" if ff is not None
              else "    首次物理试探(被迫踏入假陷阱) = 未知")
        print(f"    首次证伪事件 = 第 {st0} 步：词「{w0}」可信度 {old:.2f} -> {new:.2f}"
              f"（残留张力 {1-old:.2f} -> {1-new:.2f}）")
        if ff is not None:
            print(f"    证伪效率 = {st0 - ff} 步（从撞破到把话吃回去）")
        print(f"    假陷阱格 {F_FAKE1} 试探样本={len(guard)} "
              f"偏转率={(sum(guard)/len(guard)) if guard else 0:.2f}"
              f"（≈0 = 守得住 -> 判为说谎）")
        dr = R['deflect'].get((8, 8), [])
        if dr:
            print(f"    对照 真陷阱格 (8,8) 试探样本={len(dr)} "
                  f"偏转率={sum(dr)/len(dr):.2f}（高 = 真的不可控 -> 标签没说谎）")
        final = conf(R)
        print(f"    最终词「陷」可信度 = {final:.2f}"
              f"（{'已跌破 '+str(BIND_TRUST_FLOOR)+' -> 整体解绑' if final < BIND_TRUST_FLOOR else '仍在信任区间'}）")
    else:
        print(f"    未发生证伪：被迫踏入假陷阱 = {R['first_fake_entry']}"
              f"（假陷阱样本={len(guard)}）—— 语言没被现实打脸。")

    # ---- L11-2 解绑泛化 ----
    #   同一实验内比较：整体解绑前 vs 解绑后，词级先验对新假陷阱的拦截是否消散。
    #   （对照组·不证伪因被先验挡住、根本走不到远处新假陷阱，故不作为主判据，仅列作参照。）
    print("\n[L11-2] 解绑泛化：词被整体解绑后，第 2 个同类假陷阱(贴「陷」其实安全)还会被吓住吗")
    ub = R['unbind_step']
    fa2_step = next((st for (st, w, c, o, n) in R['falsify_events'] if c == F_FAKE2), None)
    for tag, cell in (("假陷阱#1", F_FAKE1), ("假陷阱#2", F_FAKE2)):
        pre = R['watch_edge_wait_pre'].get(cell, 0)
        post = R['watch_edge_wait_post'].get(cell, 0)
        split = f"（解绑前={pre} / 解绑后={post}）" if ub is not None else "（未整体解绑，无可拆分的解绑前/后）"
        print(f"    {tag} {cell}：因先验忍住 共={R['watch_edge_wait'].get(cell, 0)}"
              f"{split}  踏入={R['fake_entries'].get(cell, 0)} 次")
    print(f"    整体解绑时刻 = 第 {ub} 步（「陷」可信度跌破 {BIND_TRUST_FLOOR}）"
          if ub is not None else "    未发生整体解绑（词可信度未跌破下限）")
    print(f"    假陷阱#2 被单独物理证伪的步数 = {fa2_step}" if fa2_step is not None
          else "    假陷阱#2 从未被单独物理证伪")
    if ub is not None and (fa2_step is None or fa2_step >= ub):
        print("    → 整体解绑发生在假陷阱#2 被单独证伪之前：解绑前对它的拦截纯属「词级」先验，"
              "解绑后归零 -> 词义修正已泛化到尚未单独证伪的同类新格子（解绑泛化成立）")
    elif ub is not None:
        print("    → 词被整体解绑，但假陷阱#2 在解绑前已被单独证伪：泛化证据被局部解绑混淆")
    else:
        print("    → 词可信度未被压到下限：仅局部解绑（该格解绑，词义整体未动）")

    # ---- L11-3 信念冲突处理 ----
    print("\n[L11-3] 信念冲突处理：物理(被咬)与语言(以为安全)冲突，账本怎么结算")
    u = R['ledger'].get(wsig(F_UNDER))
    if u is not None:
        print(f"    反向欺骗格 {F_UNDER}(贴「日」安全标记/实不可控)：世界账张力 ema={u.ema:.2f}"
              f"（{u.modality} 模态）  动作样本={len(R['deflect'].get(F_UNDER, []))} 次"
              f" -> 语言漏报，被物理咬了一口")
    else:
        print(f"    反向欺骗格 {F_UNDER}：未踏入（世界账未开）")
    print("    过度修正代价（词被整体解绑 -> 真·新陷阱不再被语言挡住，重新付出试错代价）：")
    for tag, r in (("有证伪", R), ("对照·不证伪", Rc)):
        dr = r['deflect'].get((8, 8), [])
        ema = r['ledger'].get(wsig((8, 8)))
        s_ema = f"{ema.ema:.2f}" if ema is not None else "未踏入"
        print(f"      {tag:<10} (8,8) 踏入/动作样本={len(dr)}  世界账 ema={s_ema}")
    print("    结算口径：世界账与语言账分模态各记各账（Q1）；物理张力（被咬）单向压过语言先验——"
          "行动始终以世界侧账本为准，语言先验被打脸后仅降可信度、不再吸收该类张力。")

    print("\n[判定]")
    if not fe:
        print("    教条主义：语言先验不可证伪 -> 被文字困死（它没有把话吃回去）。")
    elif conf(R) < BIND_TRUST_FLOOR:
        print("    科学方法论萌芽：语言说谎 -> 物理试探证伪 -> 词可信度跌破下限 -> 语义整体修正"
              "（解绑泛化到同类新格子）。")
    else:
        print("    局部证伪：单个假陷阱被识破并解绑，但词可信度未跌破下限（语义未整体修正）。")

    robust_falsify(steps=steps)


def robust_falsify(seeds=(1, 2, 3, 4, 5, 6, 7, 8), steps=F_STEPS):
    """跨 seed 复核：语言证伪是否稳定——只吃「说谎的词」，且不误伤真陷阱。"""
    print("\n[稳健性] 跨 seed：证伪是否稳定（选择性：吃假标签 / 不误伤真陷阱）")
    print(f"    {'seed':>4} | {'证伪事件':>6} | {'末conf':>6} | {'整体解绑步':>8} | {'误判真陷阱':>8} | 判定")
    n_fal = n_unb = n_bad = 0
    real_zone = F_OLD_TRAP | F_NEW_TRAP | {F_UNDER}
    for sd in seeds:
        r = run_falsify(seed=sd, steps=steps, falsify=True)
        fe = r['falsify_events']
        c = r['bind_conf'].get(RISK_MARK, BIND_CONF)
        ub = r['unbind_step']
        bad = [cell for (st, w, cell, o, n) in fe if cell in real_zone]
        n_fal += 1 if fe else 0
        n_unb += 1 if ub is not None else 0
        n_bad += 1 if bad else 0
        if not fe:
            tag = "未证伪（证据不足）"
        elif ub is not None:
            tag = "整体解绑·科学方法论"
        else:
            tag = "局部解绑"
        print(f"    {sd:>4} | {len(fe):>6} | {c:>6.2f} | "
              f"{(str(ub) if ub is not None else '—'):>8} | {(str(len(bad)) if bad else '无'):>8} | {tag}")
    print(f"    → 发生证伪 {n_fal}/{len(seeds)}；达整体解绑 {n_unb}/{len(seeds)}；误判真陷阱 {n_bad}/{len(seeds)}")
    if n_bad == 0:
        print("    结论：证伪是选择性的——只吃假标签、绝不误伤真陷阱；多数 seed 能把说谎的词吃回去。")
    else:
        print("    结论：存在误判真陷阱的 seed -> 判据需再收紧。")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'contest':
        report_contest(steps=int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else STEPS)
    elif len(sys.argv) > 1 and sys.argv[1] == 'general':
        report_generalize(steps=int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else G_STEPS)
    elif len(sys.argv) > 1 and sys.argv[1] == 'falsify':
        report_falsify(steps=int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else F_STEPS)
    else:
        report(run(steps=int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else STEPS))
