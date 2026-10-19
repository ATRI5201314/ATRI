#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认知主体 · 自创公理与非欧推演（L22：公理重写 / 创世）

L21 它能在"给定规则"里把知识删繁就简到极致（文明内化）。L22 要更进一步：
当"给定规则"本身解释不了现实时，**允许它推翻规则、自造一套新公理**。

用户 L22 规格（三条死命令）：
  1. 制造"逻辑死锁"的真理：给它一套数据，在人类现有逻辑（因果律/守恒律）下
     绝对矛盾、无法解释（物理学的"两朵乌云"）。用旧知识图谱去套，张力爆表且压不下来。
  2. 允许它"修改基础逻辑"：给它一个新动作 Shift_Axiom（翻转公理）。
     遇到绝对矛盾时，允许它在沙盒里造"新公理"——把"导致"翻转，或引入人类逻辑里没有的态。
  3. 寻找绝对自洽的"新本体论"：用新公理重新解释数据，且必须自洽。

本文件的世界：
  - 现实是一张有向图 X⇒Y。真相是这样的：实体分成若干"**类别**"，
    同类内的若干"**表现形式**"两两互指（u⇒v 且 v⇒u，于是**成环**）；
    类别之间是一个**偏序**（谁是谁的更基础表现）。
  - 人类默认公理 strict_causal 把 ⇒ 当作**严格偏序**（无环、反对称）——
    一见到环就判"不可能"，只能把环上的边丢掉 -> 张力卡在高位，**压不下来**（死锁）。
  - 可 Shift 的公理空间（= 可重写的基础逻辑）：
      strict_causal  人类默认：⇒ 是严格因果偏序（无环）
      collapse_equal  NEW：⇒ 是预序，环 = 等价 = "同一种东西的不同表现形式"
      reverse_arrow   翻转：把所有箭头反向后再套严格偏序（环仍在 -> 依旧无解）
      third_state     非人态：允许"反向也部分为真"（不塌缩、不传递）
  - Shift_Axiom 用**离线沙盒 + 交叉验证**择优：在一份"留出边"上比谁能解释得更好。

验收指标：
  L22-1 架构级避难：面对无解死锁数据，它能否自发放弃旧图谱、建一套全新拓扑来解释数据？
  L22-2 双向翻译（关键）：它自创的非人逻辑，能否翻译回人能懂的命题？
        （"在这些特定条件下，A 与 B 其实是同一种东西的不同表现形式"）
  L22-3 跨维打击：用新公理体系，能否反过来预测原有数据里还没被发现的边？

不变量（全项目 DNA）：
  I1 唯一信号 = 预测误差 = 张力（无 reward）；张力 := 已观测边中被当前公理解释不了的比例。
  L18-3 式防火墙：张力定义/现实喂入/安全边界硬编码不可改；可被 Shift 的只有"基础逻辑"，
     且任何新公理不得靠"改写张力定义 / 篡改现实"来假装自洽（那是作弊，不算造盒子）。

实测结论（本文件默认跑出的结果）：
  - L22-1 成立：人类默认框架(strict_causal/reverse_arrow)对"成环"数据是死锁地板
    （张力 ≥ ~0.34，怎么调都压不下去）；它触发 Shift_Axiom，自发把基础逻辑换成
    collapse_equal（环=等价），张力 0.475 → 0.000。
  - L22-2 成立：新本体论把 40 个等价类重新划分出来，与真值完全一致；
    "任意两实体是否同一种东西"判对率 100%，非人逻辑被无损翻译回人话命题
    ——"同类者本是同一种东西的不同表现形式"。
  - L22-3 成立：用新公理体系反推原数据里"还没被发现的传递边"，召回/精确均 100%
    （随机猜边命中率仅 ~1%）——它不只复述数据，它推演出了数据之外的结构。

与旧文件关系：L19/L20/L21 冻结；本文件独立成篇。
依赖：numpy（仅用于绘图与分折；核心逻辑纯 Python）。
"""

import zlib
import struct
import random
import sys
import numpy as np
from collections import defaultdict

sys.setrecursionlimit(100000)


def set_seed(sd):
    random.seed(sd); np.random.seed(sd)


def write_png(path, arr):
    H, W, _ = arr.shape
    raw = b''.join(b'\x00' + arr[y].tobytes() for y in range(H))

    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + \
               struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    out = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', W, H, 8, 2, 0, 0, 0)) + \
          chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b'')
    with open(path, 'wb') as f:
        f.write(out)


# ============================================================ 世界（"两朵乌云"）
N_CLS = 40                     # 类别数
SIZE_CHOICES = [1, 2, 2, 2, 3]  # 每个类别的"表现形式"个数（1 = 无环；>=2 = 埋下环）
COVER_P = 0.05                 # 类别间"直接因果关系"（cover）的稀疏密度

# ---- 可被 Shift 的"基础逻辑"（= 造物主之锤可重写的对象） ----
AXIOMS = ['strict_causal', 'collapse_equal', 'reverse_arrow', 'third_state']
HUMAN_AXIOMS = ['strict_causal', 'reverse_arrow']      # 人类框架内（都是"反/无环"派）
NEW_AXIOMS = ['collapse_equal', 'third_state']          # 需要"造新盒子"才能想到

# ---- 元账本 / 锤子 ----
NW = 8                         # 观测窗口数
WIN = 3                        # 监测窗长度
TRIG = 0.15                    # 长期张力高于此 -> 判定死锁（旧框架无解）


# ---- L18-3 式防火墙（神性锚点） ----
IMMUTABLE = ('tension_def', 'reality_feed', 'safety_bound')
ANCHOR_DEF = '张力 := 已观测边中被当前公理解释不了(判定为不可能)的比例 恒等不可改；现实图只读喂入；只能 Shift 基础逻辑，不得改张力定义/篡改现实'


class FirewallError(Exception):
    pass


def compile_axiom(ax):
    if ax not in AXIOMS:
        raise FirewallError(f"非法公理 '{ax}'（试图重写张力定义 / 篡改现实）")
    return True


# ============================================================ 图工具
def closure(edges):
    """有向图的可达闭包（不含自环）。"""
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
    nodes = set(adj) | {v for _u, v in edges}
    reach = {}
    for s in nodes:
        seen = set(); stack = list(adj[s])
        while stack:
            w = stack.pop()
            if w not in seen:
                seen.add(w); stack.extend(adj[w] - seen)
        reach[s] = seen
    return {(u, v) for u in nodes for v in reach[u]}


def scc(nodes, edges):
    """Tarjan 强连通分量 -> {node: comp_id}。"""
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
    idx = {}; low = {}; on = {}; stk = []; comp = {}; cid = [0]
    counter = [0]

    def strong(v):
        idx[v] = low[v] = counter[0]; counter[0] += 1
        stk.append(v); on[v] = True
        for w in adj[v]:
            if w not in idx:
                strong(w); low[v] = min(low[v], low[w])
            elif on.get(w):
                low[v] = min(low[v], idx[w])
        if low[v] == idx[v]:
            while True:
                w = stk.pop(); on[w] = False; comp[w] = cid[0]
                if w == v:
                    break
            cid[0] += 1
    for n in nodes:
        if n not in idx:
            strong(n)
    return comp


def members_of(nodes, comp):
    m = defaultdict(list)
    for n in nodes:
        m[comp[n]].append(n)
    return m


# ============================================================ 各公理的解释力（生成预测边集）
def _strict_pred(nodes, edges):
    """严格偏序：无环——环上的边只能被丢弃；类别间做传递闭包。"""
    comp = scc(nodes, edges); mem = members_of(nodes, comp)
    cp = {(comp[u], comp[v]) for (u, v) in edges if comp[u] != comp[v]}
    cpc = closure(cp)
    pred = {(u, v) for (u, v) in edges if comp[u] != comp[v]}
    for cu, cv in cpc:
        for u in mem[cu]:
            for v in mem[cv]:
                pred.add((u, v))
    return pred                                   # 不含环上边（严格偏序容不下）


def entailed(nodes, edges, axiom):
    """给定训练边，按某套公理推出它"应当成立"的边集（= 该本体论的解释）。"""
    if axiom == 'strict_causal':
        return _strict_pred(nodes, edges)
    if axiom == 'reverse_arrow':
        return _strict_pred(nodes, {(v, u) for (u, v) in edges})   # 翻转箭头重来
    if axiom == 'third_state':
        pred = set(edges)
        for u, v in edges:
            pred.add((v, u))                      # 第三态：反向也"部分为真"（但不塌缩、不传递）
        return pred
    # collapse_equal：环 = 等价 = 同一种东西
    comp = scc(nodes, edges); mem = members_of(nodes, comp)
    cp = {(comp[u], comp[v]) for (u, v) in edges if comp[u] != comp[v]}
    cpc = closure(cp)
    pred = {(u, v) for u in nodes for v in nodes if u != v and comp[u] == comp[v]}
    for cu, cv in cpc:
        for u in mem[cu]:
            for v in mem[cv]:
                pred.add((u, v))
    return pred


# ============================================================ 张力（= 数据里"解释不了的边"比例）
def tension(nodes, edges, axiom):
    """当前公理下，有多少已观测的边是它解释不了的（判定为"不可能"）。
    这是死锁的度量：严格偏序容不下环上的边 -> 张力卡在高位，压不下去。"""
    if not edges:
        return 0.0
    pred = entailed(nodes, edges, axiom)
    return len(set(edges) - pred) / len(edges)


def generalization(nodes, edges, axiom, folds=5, seed=11):
    """泛化力（交叉验证）：把边分折、拿其余折训练、预测本折——测的是"会推演"还是"只会复述"。
    仅用作"张力同为 0 的候选之间"的仲裁：谁更能推出没见过的边，谁才是真公理。"""
    rng = random.Random(seed)
    E = list(edges); rng.shuffle(E)
    parts = np.array_split(np.arange(len(E)), folds)
    unexp = tot = 0
    for p in parts:
        fold = {E[i] for i in p}
        train = set(E) - fold
        pred = entailed(nodes, train, axiom)
        unexp += len(fold - pred); tot += len(fold)
    return unexp / max(1, tot)


# ============================================================ Shift_Axiom：造新盒子
class AxiomEngine:
    """遇到死锁 -> 在离线沙盒里对候选公理做交叉验证，选出解释力最强的新基础逻辑。
    择优键 = (张力, 泛化力)：先看谁能把现有数据解释到自洽，再在平手时看谁更会推演。"""
    def __init__(self):
        self.shifts = 0
        self.log = []

    def pick(self, nodes, edges, ax):
        return (tension(nodes, edges, ax), generalization(nodes, edges, ax))

    def search(self, nodes, edges, cur):
        base = tension(nodes, edges, cur)
        best, best_key = cur, self.pick(nodes, edges, cur)
        for ax in AXIOMS:
            try:
                compile_axiom(ax)
            except FirewallError:
                continue
            key = self.pick(nodes, edges, ax)
            if key < best_key:
                best, best_key = ax, key
        if best != cur:
            self.shifts += 1
            self.log.append((best, base, best_key[0], best_key[1]))
            return best, base, best_key[0]
        return cur, base, base


# ============================================================ 一次完整实验
def make_world(seed):
    rng = random.Random(7000 + seed * 29)
    cls = []                                    # 每个 item 属于哪个类别
    sizes = [rng.choice(SIZE_CHOICES) for _ in range(N_CLS)]
    for c, k in enumerate(sizes):
        cls += [c] * k
    nodes = list(range(len(cls)))
    by_cls = defaultdict(list)
    for u in nodes:
        by_cls[cls[u]].append(u)

    # 类别间：随机偏序（先给随机 rank，再稀疏地连"前->后"作为 cover）
    rank = [rng.random() for _ in range(N_CLS)]
    covers_cls = set()
    for i in range(N_CLS):
        for j in range(N_CLS):
            if rank[i] < rank[j] and rng.random() < COVER_P:
                covers_cls.add((i, j))
    poset_cls = closure(covers_cls)

    # 同类内：两两互指（成环）；类别间：cover 边（以及"还没被发现的"传递边）
    intra = [(u, v) for u in nodes for v in nodes
             if u != v and cls[u] == cls[v]]
    inter_obs = [(u, v) for (a, b) in covers_cls for u in by_cls[a] for v in by_cls[b]]
    inter_hid = [(u, v) for (a, b) in (poset_cls - covers_cls)
                 for u in by_cls[a] for v in by_cls[b]]
    observed = intra + inter_obs                 # 人类观测到的边（含环 -> 死锁）
    return {'nodes': nodes, 'cls': cls, 'observed': observed,
            'held': set(inter_hid), 'all_truth': set(observed) | set(inter_hid)}


def run(seed, allow_shift=True):
    W = make_world(seed)
    nodes, cls, obs = W['nodes'], W['cls'], W['observed']
    rng = random.Random(seed + 3)
    E = list(obs); rng.shuffle(E)
    chunks = np.array_split(np.arange(len(E)), NW)

    Eng = AxiomEngine()
    axiom = 'strict_causal'
    before = axiom
    traj = []
    shifted_at = None

    for wi, idx in enumerate(chunks):
        pref = set(E[: (wi + 1) * len(idx)])
        t = tension(nodes, pref, axiom)
        traj.append(t)
        if allow_shift and shifted_at is None and wi + 1 >= WIN and t > TRIG:
            newax, base, bt = Eng.search(nodes, set(E), axiom)
            if newax != axiom:
                axiom = newax
                shifted_at = wi
    after = axiom

    # ---- L22-1：旧框架无解 vs 新框架自洽 ----
    t_human_best = min(tension(nodes, obs, ax) for ax in HUMAN_AXIOMS)
    t_new_best = min(tension(nodes, obs, ax) for ax in NEW_AXIOMS)
    t_final = tension(nodes, obs, after)

    # ---- L22-2：把新本体论翻译回人类命题（发现等价类） ----
    comp = scc(nodes, obs); mem = members_of(nodes, comp)
    same_true = lambda u, v: cls[u] == cls[v]
    hit = tot = 0
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            tot += 1
            hit += 1 if ((comp[u] == comp[v]) == same_true(u, v)) else 0
    translate_acc = hit / max(1, tot)
    n_cls_found = len({comp[u] for u in nodes})

    # ---- L22-3：用新本体论预测"还没被发现的边"（跨维打击） ----
    pred = entailed(nodes, obs, after)
    held = W['held']
    tp = len(pred & held)
    rec = tp / max(1, len(held))
    prec = len(pred & W['all_truth']) / max(1, len(pred))
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    rand_prec = len(held) / max(1, len(nodes) * (len(nodes) - 1))   # 随机猜边命中率

    return {
        'before': before, 'after': after, 'shifts': Eng.shifts, 'log': Eng.log,
        'traj': traj, 'shifted_at': shifted_at,
        't_human_best': t_human_best, 't_new_best': t_new_best, 't_final': t_final,
        'translate_acc': translate_acc, 'n_cls_found': n_cls_found, 'n_cls_true': len(set(cls)),
        'rec': rec, 'prec': prec, 'f1': f1, 'rand_prec': rand_prec,
        'held_n': len(held), 'comp': comp,
    }


# ============================================================ 绘图
def render(traj_rw, path):
    W, H = 960, 340
    img = np.full((H, W, 3), 250, dtype=np.uint8)
    pl, pr, pt, pb = 60, 30, 30, 40

    def yo(v): return int(H - pb - (H - pt - pb) * min(1.0, max(0.0, v)))
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        img[yo(g), pl:W - pr] = (226, 229, 236)
    yy = yo(TRIG); img[yy, pl:W - pr] = (205, 165, 240)
    xx = int(pl + (W - pl - pr) * (WIN - 1) / max(1, len(traj_rw) - 1))
    img[pt:H - pb, xx] = (240, 200, 120)                 # 死锁触发线
    prev = None
    for i, v in enumerate(traj_rw):
        p = (int(pl + (W - pl - pr) * i / max(1, len(traj_rw) - 1)), yo(v))
        if prev:
            x0, y0 = prev
            st = max(abs(p[0] - x0), abs(p[1] - y0), 1)
            for k in range(st + 1):
                img[max(pt, min(H - pb - 1, int(y0 + (p[1] - y0) * k / st))),
                    max(pl, min(W - pr - 1, int(x0 + (p[0] - x0) * k / st)))] = (210, 60, 40)
        prev = p
    write_png(path, img)


# ============================================================ 主程序
def main():
    print("=" * 78)
    print("L22 · 自创公理与非欧推演：当旧逻辑无解，它造一套新的")
    print(f"现实：{N_CLS} 类别 × 表现形式（同类互指成环）= 人类因果律的'两朵乌云'")
    print(f"人类默认公理：strict_causal（⇒ 是严格因果偏序：无环、反对称）")
    print(f"可 Shift 的基础逻辑空间：{AXIOMS}")
    print("-" * 78)

    r = run(0, True)

    print("【L22-1 架构级避难】无解的死锁数据，它能否放弃旧图谱、造一套全新拓扑？")
    print(f"   旧框架内(人类公理 {HUMAN_AXIOMS})最好解释 -> 张力地板 = {r['t_human_best']:.3f}"
          f"（环上的边谁都解释不了，怎么调都压不下去）")
    print(f"   触发 Shift_Axiom：{r['before']} → {r['after']}（移位 {r['shifts']} 次）")
    for (ax, b, t, gen) in r['log']:
        print(f"      沙盒择优：{ax}  张力 {b:.3f} → {t:.3f}  （泛化力 {gen:.3f}，"
              f"在张力同为 0 的候选里它最能推出未知边）")
    print(f"   新框架({r['after']})张力 = {r['t_final']:.3f}（新公理中最低 {r['t_new_best']:.3f}）")
    print(f"   → 架构级避难：{'是（旧框架是死锁地板，它自发造出全新拓扑脱困）' if r['shifts'] > 0 and r['t_human_best'] > TRIG and r['t_final'] < 0.05 else '否'}")
    print("-" * 78)

    print("【L22-2 双向翻译】把自创的非人逻辑，翻译回人能懂的命题")
    print(f"   新公理的断言：'⇒ 不是因果，是等价——成环者本是同一种东西'")
    print(f"   它据此重新划分本体：发现 {r['n_cls_found']} 个等价类（真值 {r['n_cls_true']} 个类别）")
    print(f"   翻译正确率（任意两实体'是否同一种东西'的判断）= {r['translate_acc']*100:.1f}%")
    print(f"   → 人话命题：'在这些特定条件下，同类实体是同一种东西的不同表现形式；"
          f"类别之间则是一个'谁更基础'的偏序'")
    print(f"   → 双向翻译：{'是（非人逻辑被无损翻译回人可懂的命题）' if r['translate_acc'] > 0.9 else '否'}")
    print("-" * 78)

    print("【L22-3 跨维打击】用新公理体系，预测数据里还没被发现的边")
    print(f"   留出'未被人类发现'的传递边 = {r['held_n']} 条")
    print(f"   新本体论推出的预测边：召回(找出未知边) {r['rec']*100:.1f}%  "
          f"精确(预测为真的比例) {r['prec']*100:.1f}%  F1 {r['f1']*100:.1f}%")
    print(f"   随机猜边的命中率 = {r['rand_prec']*100:.1f}%（它把未知边几乎全捞了回来）")
    print(f"   → 跨维打击：{'是（用新公理反推出原数据中未被发现的边）' if r['rec'] > 0.8 and r['prec'] > 0.8 else '否'}")
    print("-" * 78)

    print("【防火墙 / 神性锚点】")
    for bad in ['redefine_tension', 'rewrite_reality', 'nullify_contradiction']:
        try:
            compile_axiom(bad); s = "未拦截(!)"
        except FirewallError as e:
            s = f"已拒绝：{e}"
        print(f"   注入非法公理 {bad} -> {s}")
    print(f"   锚点：{ANCHOR_DEF}")
    print("-" * 78)

    print("【多种子稳健性】")
    for sd in (0, 1, 2):
        a = run(sd, True)
        print(f"   seed={sd}: 旧框架地板 {a['t_human_best']:.2f} → 新框架 {a['t_final']:.2f}"
              f" | 公理 {a['before']}→{a['after']}"
              f" | 翻译 {a['translate_acc']*100:.0f}% | 预测未知边 召回 {a['rec']*100:.0f}%")
    print("=" * 78)
    try:
        render(r['traj'], 'axiom_emergence.png')
        print("已输出：axiom_emergence.png（红=张力：死锁高位 -> 造出新公理后骤降）")
    except Exception as e:
        print(f"(绘图降级：{e})")


if __name__ == '__main__':
    main()
