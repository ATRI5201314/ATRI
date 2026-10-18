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
SPEAK_BUDGET = 4           # 发言硬预算（最多采纳几个词）
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


if __name__ == '__main__':
    report(run(steps=int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else STEPS))
