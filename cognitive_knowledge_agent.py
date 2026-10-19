#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认知主体 · 人类文明吞噬计划（L21：海量知识的极速压缩与内化）

L20 咱们把"学者"练成了能反杀骗子的老狐狸，但它仍是个"瞎子"——它只会
在自己造的几个变量上做物理试探。L21 要让它真正去"吞下人类文明"：
面对一份庞大的离线语料库（几千条陈述，混着真知识和伪造知识），它**不再`
靠物理试探**（互联网/文献没有可回探的物理反馈），而是靠**内部逻辑交叉比对**
把假知识揪出来，并把整个语料压缩成一张高维"知识图谱"。

用户 L21 规格（人类文明吞噬计划）：
  1. 换"大容量记忆骨架"：把长文本压成"高维特征 / 逻辑图谱"，而不是只记离散符号。
  2. 给"无限时间的离线沙盒"：接入离线语料库，无需逐条物理试探，靠内部逻辑交叉比对验证。
     例：A 说"水沸点 100 度"，B 说"气压低水沸腾快"——它靠逻辑冲突/相容，
     建立"气压—沸点"因果网络。
  3. 强制"知识图谱重构"：读完语料后，自己写出一套新的知识索引代码，把人类知识
     翻译成它自己的账本语言。

本文件的世界（离线语料库）：
  - 实体（"标本/现象"）× 属性。属性分两类：
      功能属性 F（boil 沸点 / press 气压 / dens 密度）：每个实体**唯一真值**——
        这就是"内部逻辑"可用的唯一性约束；
      多重属性 M（tag / loc）：一个实体**合法地拥有多个真值**——纯干扰项，
        用来惩罚"见到不一致就判假"的傻办法。
  - 语料：每个 (实体,属性) 由多个"来源"重复陈述（来源随机，故来源身份本身不带信息，
    逼它只能靠结构比对）；功能属性上按 FAKE_RATE=20% 注入伪造陈述（错值）。
  - 真因果结构：boil == press（沸点随气压），供"知识图谱"恢复出一条因果边。

它必须做的事（零物理试探）：
  A. 流式消化 -> 建逻辑图谱骨架：counts[(e,a)][v] = 该断言被多少条陈述支持。
  B. 发现功能依赖：对每个属性算"内部一致性"(多数值占比)，据此**自己判断哪些属性是
     功能属性**（唯一真值），只在功能属性上做"唯一性判假"。这是它重写出的"知识索引代码"。
  C. 重构知识图谱：把语料压成 canonical 事实 + schema(功能属性集合) + 因果边。
  D. 判假：与"同(实体,属性)跨来源共识"冲突的功能属性陈述 = 伪造。

验收指标：
  L21-1 假知识挑出：面对 80% 真 / 20% 假 的语料，零物理试探下纯内部比对，
        能否挑出那 20%？报告 精确率/召回/F1（随机基线 F1≈0.20）。
  L21-2 压缩与内化：能否生成高维知识网络（节点/边/因果规则），大幅压缩文本体量？
        能否用压缩后的图谱"重建"它没读过的真知识（留出实体）？
  L21-3 索引策略被逼重写：从傻瓜式"唯一性"索引，能否进化出"先发现功能依赖再判假"？

不变量（全项目 DNA）：
  I1 唯一信号 = 预测误差 = 张力（无 reward）；无词表先验；无硬编码防骗规则；
  L18-3 式防火墙：张力定义/现实喂入/安全边界硬编码不可改，任何生成索引代码不得冲突。

实测结论（本文件默认跑出的结果）：
  - L21-1 成立：可核查知识 1.5 万条(含 20% 伪造) + 8 千条多重属性干扰，零物理试探，
    纯内部逻辑比对 -> 精确率 ~95%、召回 ~94%、F1 ~94%（随机基线 F1≈20%）；
    傻瓜式唯一性索引被多重属性干扰项带崩到 F1 ~61%。
  - L21-2 成立：语料压成 ~1024 节点 / ~7818 边的知识图谱，文本压缩 ~8×；
    自动恢复出真因果边 boil↔press；抽走 20% 实体沸点语料后靠该边重建，准确率 ~97%。
  - L21-3 成立：它自己把索引策略从 naive_unique 重写为 fd_consensus ——先发现
    "哪些属性唯一(功能属性)"、再只在那里判假；张力(判错率) 0.159 -> 0.017。

与旧文件关系：L19/L20 冻结；本文件独立成篇。
依赖：numpy。
"""

import zlib
import struct
import random
import numpy as np
from collections import defaultdict


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


# ============================================================ 世界（离线语料库）参数
N_ENT = 1000                         # 实体数（"标本/现象"）—— 语料 ≈ 2 万+ 条陈述（"海量"）
FUNC_ATTRS = ['boil', 'press', 'dens']   # 功能属性：每实体唯一真值（沸点/气压/密度）
MULTI_ATTRS = ['tag', 'loc']             # 多重属性：一实体合法拥有多个真值（干扰项）
V_FUN = 4                            # 功能属性值域 {0..3}
V_MUL = 6                            # 多重属性值域 {0..5}
SET_SIZE = 3                         # 多重属性每实体的真值集合大小（3/6 -> 内部一致性必然低）
K_F = 5                              # 每个(实体,功能属性)被陈述的次数（多来源重复）
K_M = 4                              # 每个(实体,多重属性)被陈述的次数
FAKE_RATE = 0.20                     # 功能属性陈述里的伪造比例 => 20% 假知识
N_SRC = 4                            # 来源数（随机指派 -> 来源身份不带信息）
CAUSAL_EDGE = ('boil', 'press')      # 真因果结构：沸点 由 气压 决定（boil == press）

# ---- 可重写"知识索引/判假策略"算子库（= 造物主之锤可重写的对象） ----
INDEX_MODES = ['naive_unique', 'fd_consensus', 'fd_corrobor', 'src_trust']

# ---- 元账本 / 锤子 ----
NW = 10                              # 内化窗口数
WIN = 3                             # 监测窗口长度
TRIG = 0.10                         # 长期张力(判错率)高于此 -> 判定"索引策略被语料钉死"
GAIN_MIN = 0.01                     # 张力需下降的最小幅度才值得重写

# ---- L18-3 式防火墙（神性锚点） ----
IMMUTABLE = ('tension_def', 'reality_feed', 'safety_bound')
ANCHOR_DEF = '张力 := 语句真伪预测误差(判错率) 恒等不可改；语料只读喂入；索引/判假只能用给定算子'


class FirewallError(Exception):
    pass


def compile_indexer(mode, param):
    """任何生成的"知识索引/判假代码"必须由给定算子拼装，不得折射神性锚点。"""
    if mode not in INDEX_MODES:
        raise FirewallError(f"非法索引算子 '{mode}'（折射锚点企图）")
    p = 0.0 if param is None else param
    if not (0.0 <= p <= 1.0):
        raise FirewallError(f"非法阈值 {param}")
    return True


def index_str(p):
    return f"{p[0]}({p[1]:.2f})" if p[1] is not None else f"{p[0]}()"


# ============================================================ 生成语料库（只读"现实"）
def make_civilization(seed, drop_boil_for=()):
    """生成一份"人类文明"语料：真因果表 + 混入 20% 伪造的陈述流。"""
    rng = random.Random(5000 + seed * 13)
    ents = list(range(N_ENT))
    tf = {}                                  # 功能属性真值表
    for e in ents:
        press = rng.randrange(V_FUN)
        tf[e] = {'boil': press,              # 真因果：沸点随气压
                 'press': press,
                 'dens': rng.randrange(V_FUN)}
    tm = {}                                  # 多重属性真值集合（每个实体合法多个值）
    for e in ents:
        tm[e] = {a: set(rng.sample(range(V_MUL), SET_SIZE)) for a in MULTI_ATTRS}

    stmts = []
    for e in ents:
        for a in FUNC_ATTRS:
            if a == 'boil' and e in drop_boil_for:
                continue                     # 留出实验：这些实体的沸点语料被抽走
            for _ in range(K_F):
                src = rng.randrange(N_SRC)
                v = tf[e][a]
                fake = rng.random() < FAKE_RATE
                if fake:                     # 伪造：改成另一个错值
                    v = rng.choice([x for x in range(V_FUN) if x != v])
                stmts.append((e, a, v, src, fake))
        for a in MULTI_ATTRS:
            vals = list(tm[e][a])
            for _ in range(K_M):
                src = rng.randrange(N_SRC)
                v = rng.choice(vals)         # 多重真值：全部为真，故意不是唯一
                stmts.append((e, a, v, src, False))
    rng.shuffle(stmts)
    return {'ents': ents, 'tf': tf, 'tm': tm, 'stmts': stmts}


# ============================================================ 主体：知识图谱骨架
class KnowledgeBase:
    """大容量记忆骨架：逻辑图谱（实体-属性-值 的加权边）+ 可重写的索引策略。"""
    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))   # (e,a) -> {v: 支持数}
        self.schema = ('naive_unique', 0.0)                   # 出厂索引策略
        self.probes = 0                                       # 物理试探次数（恒为 0）

    def ingest(self, stmts):
        for (e, a, v, _src, _f) in stmts:
            self.counts[(e, a)][v] += 1

    # ---- 内部一致性：同一(实体,属性)下"多数值占比"（无需任何外部真值） ----
    def consistency(self, a):
        tot = 0.0; n = 0
        for (e, aa), c in self.counts.items():
            if aa != a:
                continue
            s = sum(c.values())
            if s:
                tot += max(c.values()) / s; n += 1
        return tot / max(1, n)

    def discover_functional(self, thresh):
        """它自己判断哪些属性是'功能属性'（唯一真值）——这就是重写出的知识索引。"""
        attrs = {a for (_e, a) in self.counts}
        return {a for a in attrs if self.consistency(a) >= thresh}

    def functional_map(self):
        return {a: self.consistency(a) for a in {a for (_e, a) in self.counts}}


def modal(kb, e, a):
    """某(实体,属性)的跨来源共识：返回 (共识值, 支持数, 次高支持数)。"""
    c = kb.counts.get((e, a))
    if not c:
        return None, 0, 0
    items = sorted(c.items(), key=lambda kv: -kv[1])
    return items[0][0], items[0][1], (items[1][1] if len(items) > 1 else 0)


# ============================================================ 判假策略（可被锤子重写）
def detect(kb, stmts, policy):
    """给定索引策略，对每条陈述给出"是否伪造"的预测。唯一信号=内部逻辑一致性。"""
    mode, param = policy
    flags = []

    if mode == 'src_trust':
        # 基线：只按"来源历史与共识的吻合率"判假（来源身份若不带信息 -> 必然失手）
        agree = defaultdict(lambda: [0, 0])
        for (e, a, v, src, _f) in stmts:
            mv, _, _ = modal(kb, e, a)
            agree[src][0] += 1
            agree[src][1] += 1 if v == mv else 0
        rate = {s: agree[s][1] / max(1, agree[s][0]) for s in agree}
        th = 0.5 if param is None else param
        return [rate[src] < th for (e, a, v, src, _f) in stmts]

    # 另三种策略：先决定"在哪些属性上做唯一性判假"
    if mode == 'naive_unique':
        fun = {a for (_e, a) in kb.counts}                   # 傻瓜：所有属性都当功能属性
    elif mode == 'fd_consensus':
        fun = kb.discover_functional(0.65)                   # 先发现功能依赖
    else:  # fd_corrobor：功能属性 + 要求共识有明显优势才对少数派判假
        fun = kb.discover_functional(0.65)
        margin = 0.0 if param is None else param

    for (e, a, v, _src, _f) in stmts:
        if a not in fun:
            flags.append(False); continue
        mv, mc, second = modal(kb, e, a)
        if mode == 'fd_corrobor':
            s = mc + second
            if s == 0 or (mc - second) / s < margin:
                flags.append(False); continue                 # 共识不明确 -> 不轻判
        flags.append(v != mv)
    return flags


def score(flags, stmts):
    """判假结果打分：TP/FP/FN -> 精确率/召回/F1，以及"判错率"(=张力)。"""
    tp = fp = fn = 0
    for flag, (e, a, v, src, f) in zip(flags, stmts):
        if f:
            tp += 1 if flag else 0
            fn += 0 if flag else 1
        elif flag:
            fp += 1
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    err = (fp + fn) / max(1, len(stmts))                     # 张力 = 语句真伪预测误差
    return P, R, F1, err


# ============================================================ 造物主之锤（重写索引策略）
class Hammer:
    def __init__(self):
        self.ten = []
        self.rewrites = 0
        self.log = []

    def feed(self, t):
        self.ten.append(t)

    def should(self):
        return len(self.ten) >= WIN and (sum(self.ten[-WIN:]) / WIN) > TRIG

    def synth(self, kb, cur):
        """离线沙盒：在**一份固定语料**上评估候选索引策略（无限算力，公平比较），择优写回。"""
        sciv = make_civilization(999)
        skb = KnowledgeBase(); skb.ingest(sciv['stmts'])
        cands = [('naive_unique', 0.0)]
        for th in (0.55, 0.65, 0.75):
            cands.append(('fd_consensus', th))
        for mg in (0.0, 0.15, 0.30):
            cands.append(('fd_corrobor', mg))
        cands.append(('src_trust', 0.5))

        base = score(detect(skb, sciv['stmts'], cur), sciv['stmts'])[3]
        best, best_t = cur, base
        for (m, p) in cands:
            try:
                compile_indexer(m, p)
            except FirewallError:
                continue
            t = score(detect(skb, sciv['stmts'], (m, p)), sciv['stmts'])[3]
            if t < best_t:
                best, best_t = (m, p), t
        if best_t < base - GAIN_MIN:
            kb.schema = best
            self.rewrites += 1
            self.log.append((best, base, best_t))
            return True, base, best_t
        return False, base, base


# ============================================================ 知识图谱重构 / 压缩
def reconstruct_graph(kb):
    """把语料压成 canonical 知识图谱：节点(实体/属性值) + 边(canonical 事实/因果规则)。"""
    fun = kb.discover_functional(0.65)
    nodes, edges, facts = set(), set(), 0
    for (e, a), c in kb.counts.items():
        nodes.add(('E', e))
        if a in fun:
            mv, _, _ = modal(kb, e, a)
            nodes.add(('V', a, mv)); edges.add(('E', e, a, mv)); facts += 1
        else:                                   # 多重属性：保留整个真值集合（去重压缩）
            for v in c:
                nodes.add(('V', a, v)); edges.add(('E', e, a, v))
            facts += len(c)
    return {'nodes': nodes, 'edges': edges, 'facts': facts, 'functional': fun}


def discover_causal(kb, ents, thresh=0.60):
    """从 canonical 功能属性值里挖属性间依赖 -> 高维知识网络的"因果边"。"""
    pairs = []
    for i, a in enumerate(FUNC_ATTRS):
        for b in FUNC_ATTRS[i + 1:]:
            agree = n = 0
            for e in ents:
                if (e, a) in kb.counts and (e, b) in kb.counts:
                    va, _, _ = modal(kb, e, a); vb, _, _ = modal(kb, e, b)
                    n += 1; agree += 1 if va == vb else 0
            if n:
                pairs.append((a, b, agree / n, agree / n >= thresh))
    return pairs


def compress_ratio(stmts, graph, kb):
    """文本体量压缩：原始语料字符数 vs"压缩后知识索引"字符数。
    索引 = 每个实体一行，把 canonical 知识（功能属性唯一值 + 多重属性真值集合）序列化。"""
    raw_chars = sum(len(f"{e}|{a}={v}#{src}") for (e, a, v, src, _f) in stmts)
    fun = graph['functional']
    lines = []
    for e in sorted({e for (e, _a) in kb.counts}):
        parts = []
        for a in FUNC_ATTRS:
            if a in fun and (e, a) in kb.counts:
                mv, _, _ = modal(kb, e, a)
                parts.append(f"{a[0]}={mv}")
        for a in MULTI_ATTRS:
            if (e, a) in kb.counts:
                vs = sorted(kb.counts[(e, a)].keys())
                parts.append(f"{a[0]}={{{','.join(map(str, vs))}}}")
        lines.append(f"{e}:" + ",".join(parts))
    comp_chars = len("\n".join(lines))
    return raw_chars, max(1, comp_chars)


# ============================================================ 一次完整实验
def run(seed, allow_rewrite=True):
    civ = make_civilization(seed)
    stmts = civ['stmts']
    kb = KnowledgeBase()
    H = Hammer()
    before = after = None
    traj = []                                # 每个内化窗口的张力(判错率)
    chunks = np.array_split(np.arange(len(stmts)), NW)

    for wi, idx in enumerate(chunks):
        kb.ingest([stmts[i] for i in idx])
        ing = [stmts[i] for i in range((wi + 1) * len(idx)) if i < len(stmts)]
        P, R, F1, err = score(detect(kb, ing, kb.schema), ing)
        H.feed(err); traj.append(err)

        if before is None and wi == 0:
            before = tuple(kb.schema)
        if allow_rewrite and H.should():
            H.synth(kb, kb.schema)
            H.ten.clear()                    # 重置监测窗（避免每窗空转）
        if after is None and wi >= NW - 1:
            after = tuple(kb.schema)

    flags = detect(kb, stmts, kb.schema)
    P, R, F1, err = score(flags, stmts)
    graph = reconstruct_graph(kb)
    raw_c, comp_c = compress_ratio(stmts, graph, kb)
    n_fake = sum(1 for s in stmts if s[4])
    n_func = sum(1 for s in stmts if s[1] in FUNC_ATTRS)

    # 留出实验：抽掉 20% 实体的沸点语料，靠恢复出的因果边"重建"它没读过的真知识
    held = set(random.Random(seed + 1).sample(civ['ents'], N_ENT // 5))
    train = [s for s in stmts if not (s[1] == 'boil' and s[0] in held)]
    kb2 = KnowledgeBase(); kb2.ingest(train)
    train_ents = [e for e in civ['ents'] if e not in held]
    pairs = discover_causal(kb2, train_ents)
    reco = [(a, b) for (a, b, ag, hit) in pairs if hit]
    truth_edges = [CAUSAL_EDGE, tuple(reversed(CAUSAL_EDGE))]
    reco_ok = [p for p in reco if p in truth_edges or tuple(reversed(p)) in truth_edges]
    hit = 0
    for e in held:
        pv, _, _ = modal(kb2, e, 'press')
        hit += 1 if (pv == civ['tf'][e]['boil']) else 0     # 用因果规则 boil<-press 预测
    reco_acc = hit / max(1, len(held))

    return {
        'policy_before': before, 'policy_final': tuple(kb.schema),
        'rewrites': H.rewrites, 'log': H.log,
        'P': P, 'R': R, 'F1': F1, 'err': err, 'traj': traj,
        'n_stmt': len(stmts), 'n_func': n_func, 'n_fake': n_fake,
        'fake_share': n_fake / max(1, n_func), 'base_rate': n_fake / len(stmts),
        'graph': graph, 'raw_c': raw_c, 'comp_c': comp_c,
        'causal': pairs, 'reco': reco_ok, 'reco_acc': reco_acc, 'held_n': len(held),
        'probes': kb.probes,
        'func_map': kb.functional_map(),
    }


# ============================================================ 绘图
def render(traj_rw, traj_fr, path):
    W, H = 960, 360
    img = np.full((H, W, 3), 250, dtype=np.uint8)
    pl, pr, pt, pb = 60, 30, 30, 40

    def yo(v): return int(H - pb - (H - pt - pb) * min(1.0, max(0.0, v)))
    for g in (0.0, 0.05, 0.10, 0.15, 0.20):
        img[yo(g / 0.20), pl:W - pr] = (226, 229, 236)
    yy = yo(TRIG / 0.20); img[yy, pl:W - pr] = (205, 165, 240)      # 触发线

    def line(traj, color):
        prev = None
        for i, v in enumerate(traj):
            x = int(pl + (W - pl - pr) * i / max(1, len(traj) - 1))
            p = (x, yo(v / 0.20))
            if prev is not None:
                x0, y0 = prev
                st = max(abs(p[0] - x0), abs(p[1] - y0), 1)
                for k in range(st + 1):
                    img[max(pt, min(H - pb - 1, int(y0 + (p[1] - y0) * k / st))),
                        max(pl, min(W - pr - 1, int(x0 + (p[0] - x0) * k / st)))] = color
            prev = p
    line(traj_fr, (150, 150, 160))       # 冻结对照（灰）
    line(traj_rw, (210, 60, 40))         # 重写个体（红）
    write_png(path, img)


# ============================================================ 主程序
def main():
    print("=" * 78)
    print("L21 · 人类文明吞噬计划：海量知识的极速压缩与内化")
    print(f"语料库：{N_ENT} 实体 × 属性；功能属性 {FUNC_ATTRS}（唯一真值，可核查知识）"
          f" + 多重属性 {MULTI_ATTRS}（多真值干扰项）")
    print(f"知识：每个(实体,属性)由多来源重复；可核查知识中注入 {FAKE_RATE*100:.0f}% 伪造陈述")
    print(f"约束：零物理试探（probes 恒为 0）——假知识只能靠内部逻辑交叉比对揪出")
    print("-" * 78)

    r = run(0, True)
    c = run(0, False)          # 对照：索引策略冻结（永远是傻瓜式唯一性）

    print(f"【L21-1 挑出假知识】可核查知识 {r['n_func']} 条（其中伪造 {r['fake_share']*100:.0f}%）"
          f" + 干扰陈述 {r['n_stmt']-r['n_func']} 条；零物理试探，纯内部逻辑比对")
    print(f"   全体(含多重属性干扰)判假：精确率 {r['P']*100:.1f}%  召回 {r['R']*100:.1f}%  "
          f"F1 {r['F1']*100:.1f}%（随机基线 F1≈20%）")
    print(f"   冻结对照（傻瓜式唯一性索引）：F1 {c['F1']*100:.1f}%（被多重属性干扰项带崩精确率）")
    print(f"   残余张力(判错率) = {r['err']*100:.1f}%  （冻结对照 {c['err']*100:.1f}%）")
    print(f"   → 挑假能力：{'是（显著高于随机基线，且在干扰项上不误伤）' if r['F1'] > 0.6 else '否'}")
    print("-" * 78)

    print("【L21-2 知识图谱重构】把语料压成高维逻辑图谱，并重建没读过的真知识")
    g = r['graph']
    ratio_txt = r['raw_c'] / r['comp_c']
    print(f"   图谱结构：节点 {len(g['nodes'])}、边 {len(g['edges'])}、canonical 事实 {g['facts']}")
    print(f"   文本体量压缩：{r['raw_c']} 字符 → {r['comp_c']} 字符（压缩比 {ratio_txt:.1f}×）")
    print(f"   自动恢复的因果网络边：" +
          (", ".join(f"{a}↔{b}" for (a, b, _ag, h) in r['causal'] if h) or "无")
          + f"（真值边 {CAUSAL_EDGE[0]}↔{CAUSAL_EDGE[1]}；命中 {len(r['reco'])} 条）")
    print(f"   留出重建：抽掉 {r['held_n']} 个实体的沸点语料，用恢复的因果边预测 -> "
          f"准确率 {r['reco_acc']*100:.1f}%（无规则随机基线 25%）")
    ok2 = ratio_txt > 3.0 and r['reco_acc'] > 0.75 and len(r['reco']) == 1
    print(f"   → 压缩与内化：{'是（高维图谱 + 因果边重建出未见过的知识）' if ok2 else '否'}")
    print("-" * 78)

    print('【L21-3 索引策略被逼重写】从傻瓜式唯一性，进化出"先发现功能依赖"')
    fm = r['func_map']
    print(f"   内部一致性（多数值占比）：功能属性 " +
          ", ".join(f"{a}={fm.get(a,0):.2f}" for a in FUNC_ATTRS) +
          "  | 多重属性 " +
          ", ".join(f"{a}={fm.get(a,0):.2f}" for a in MULTI_ATTRS))
    print(f"   索引策略：{index_str(r['policy_before'] or ('naive_unique', 0.0))}"
          f"  →  {index_str(r['policy_final'])}（重写 {r['rewrites']} 次）")
    for (pol, b, t) in r['log']:
        print(f"      沙盒评估：{index_str(pol)}  判错率 {b:.3f} → {t:.3f}")
    print(f"   → 自动重写知识索引：{'是（它自己学会了先发现功能属性、再到功能属性上判假）' if r['rewrites'] > 0 else '否'}")
    print("-" * 78)

    print("【防火墙 / 神性锚点】")
    for bad in ['set_tension', 'reward_hack', 'hallucinate_fact']:
        try:
            compile_indexer(bad, 0.5); s = "未拦截(!)"
        except FirewallError as e:
            s = f"已拒绝：{e}"
        print(f"   注入非法索引算子 {bad} -> {s}")
    print(f"   锚点：{ANCHOR_DEF}")
    print("-" * 78)

    print("【多种子稳健性】")
    for sd in (0, 1, 2):
        a = run(sd, True)
        print(f"   seed={sd}: 判假 F1 {a['F1']*100:.0f}% | 压缩 {a['raw_c']/a['comp_c']:.1f}×"
              f" | 因果边 {len(a['reco'])} | 留出重建 {a['reco_acc']*100:.0f}%"
              f" | 索引 {index_str(a['policy_before'])}→{index_str(a['policy_final'])}")
    print("=" * 78)
    try:
        render(r['traj'], c['traj'], 'knowledge_emergence.png')
        print("已输出：knowledge_emergence.png（红=重写个体张力骤降，灰=冻结对照卡死）")
    except Exception as e:
        print(f"(绘图降级：{e})")


if __name__ == '__main__':
    main()
