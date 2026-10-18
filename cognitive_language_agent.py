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
    token 流：规整区走确定性循环(可预测)，多结果区走混沌(预测不动)。"""

    def __init__(self, jitter=0.0, rng=None):
        self.rng = rng or random.Random(0)
        self.jitter = jitter
        self.reg = 0

    def step(self, s, a):
        x, y = s
        if a == WAIT:
            return s
        dx, dy = MOVES[a]
        tx, ty = x + dx, y + dy
        if is_trap(x, y):
            # 控制失败源：身处多结果区时，动作被不可观测地随机偏转 -> 动作对结果无稳定影响
            a = self.rng.choice(MOVE_LIST)
            dx, dy = MOVES[a]
            tx, ty = x + dx, y + dy
        elif self.jitter > 0 and self.rng.random() < self.jitter:
            a = rotate(a, 1 if self.rng.random() < 0.5 else -1)
            dx, dy = MOVES[a]
            tx, ty = x + dx, y + dy
        if not in_bounds(tx, ty):
            return s
        return (tx, ty)

    def emit(self, cell):
        """按当前情境发出一个 token。多结果区 -> 混沌(预测不动)。"""
        if is_trap(*cell):
            return self.rng.choice(VOCAB_CHAOS)
        tok = VOCAB_REG[self.reg % 3]
        self.reg += 1
        return tok


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


def pred_reduction_speak(ledger, binding, wpred):
    """这一步 Speak 预计能新压掉多少张力（边际净削减，规格9「净压缩收益>0」）。
       关键口径：反复说同一个词不会重复削同一批张力——被命名/吸收过的条目
       张力已降到 NAMED_RESIDUAL，再算进去只会高估 Speak，让它无脑压过 Move。
       因此只累计「相对 NAMED_RESIDUAL 还高出多少」。"""
    best = 0.0
    for w in binding:
        if binding[w][0] != 'EXT':
            continue        # 只有绑定「不可控情境类」的词会吸收张力
        cov = covered_sigs(w, binding, ledger, wpred)
        red = sum(max(0.0, e.ema - NAMED_RESIDUAL) for e in _covered_standing(ledger, cov))
        best = max(best, red)
    return best


def do_speak(step, ledger, binding, adopted, wpred, named, compression_log):
    """执行 Speak：说出净张力削减最大的词（反复说同一个词 = 反复命名当前情境）。

       因果绑定（规格要求）：只有绑定到「不可控情境类」的词，才具备「命名」的力量——
       它作为检索键，把不可控区那批世界账从「点预测」改写为「类预测」，
       该类张力被命名吸收到 NAMED_RESIDUAL，并把该类标记进 named（预测器侧生效）。
       绑定到单一可控情境的词，只参与账本压缩（叙述变短），不改动预测器。"""
    best, best_red = None, 0.0
    for w in binding:
        if binding[w][0] != 'EXT':
            continue
        cov = covered_sigs(w, binding, ledger, wpred)
        red = sum(max(0.0, e.ema - NAMED_RESIDUAL) for e in _covered_standing(ledger, cov))
        if red > best_red:
            best_red, best = red, w
    if best is None or best_red <= 0.0:
        return None, set()
    adopted.add(best)
    is_ext_word = True
    cov = covered_sigs(best, binding, ledger, wpred)
    cells = set()
    for e in ledger.values():
        if e.sig not in cov:
            continue
        if e.modality == 'world':
            e.ema = min(e.ema, NAMED_RESIDUAL)   # 点预测 -> 类预测，张力被命名吸收
            named[e.sig[0]] = best
            cells.add(e.sig[0])
        else:
            e.ema = min(e.ema, NAMED_RESIDUAL)
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

        # ---- 规格8：指称绑定 ----
        cooc[tok][s] += 1
        if tok not in binding:
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


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'contest':
        report_contest(steps=int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else STEPS)
    else:
        report(run(steps=int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else STEPS))
