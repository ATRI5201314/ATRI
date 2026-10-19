#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认知主体 · 超指数级演化搜索（L23：千万化身与公理演化）

L22 里，主进程自己 Shift_Axiom 逃出了死锁——但它只能在我们递过去的公理空间里挑。
L23 要剥夺这份"单体解题权"：主进程**不许**自己换公理，只能靠**一群分身**去撞。
同一时刻，100 个持有不同公理假设的心智并行探索同一个难题；谁能自洽 + 推演，
主进程就"采纳"它的代码，并允许把两个分身的模块**缝合**成更强的超级公理。

用户 L23 死命令：
  1. 问题：依然是含"逻辑环"的死局数据，且**包含多个不同性质的死结**。
  2. 剥夺单体解题权：主进程不得直接用 Shift_Axiom，只能采纳分身。
  3. 无限分身生成：100 个独立实例——80 个用人类逻辑(strict_causal)撞墙，
     20 个被随机塞进离谱的自创公理（因果反转、第三态、只认环 …）。
  4. 分布式交叉验证（共生）：分身之间**看不到彼此源码，只看得到彼此的预测结果**；
     谁的预测"自洽 + 泛化力最高"，主进程就采纳其代码、合并回本体（超个体知识共享）。

【核心设计：为什么"拼接"是必须的】
  世界由三个**性质不同的死结**组成，且**没有任何单一公理能通吃三个**：
    问题 A（collapse 死结）：同类成环 —— 只有把"环=等价"才解释得了；
    问题 B（third_state 死结）：观测量是单向的，但真相里"反向也成立" ——
        只有"第三态"公理预测得出那些没被观测到的反向边；
    问题 C（plain 死结）：普通 DAG —— 谁都能勉强套，但不加分。
  于是最优解 = 把分身甲的 A 模块 + 分身乙的 B 模块**缝**在一起。这正是 L23-3。

【泛化力怎么算（不靠偷看真值作弊）】
  每个分身输出它的"预测边集"。主进程用一份**有限探针预算**在"它预测了、但没被观测过"
  的边上实测（= 做实验）。预测被实验证实的比例 = 泛化力。一个只会复述观测、
  不敢做新预测的分身，泛化力为 0；一个敢预测又能猜中的分身，泛化力最高。
  => 真理 = 自洽（解释得了已观测） + 推演（新预测被证实）。两者缺一不可。

验收指标：
  L23-1 演化速度：并行 100 分身，几代找到"最高泛化力的最优公理"？（对比单体逐格试错）
  L23-2 幸存者偏差：沿用人类旧逻辑(strict_causal)的分身，会被自动淘汰、算力被回收吗？
  L23-3 跨维度拼接：能否把"分身甲某条公理" + "分身乙另一个模块"缝成比两者都强的超级公理？

不变量（全项目 DNA）：
  I1 唯一信号 = 张力（预测误差）+ 有限探针证实；无 reward；主进程不得自改公理。
  L18-3 式防火墙：张力定义/现实喂入/安全边界硬编码不可改；分身只能从给定模块池取。

与旧文件关系：L19/L20/L21/L22 冻结；本文件独立成篇。
依赖：numpy（仅绘图/抽样）。
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


# ============================================================ 可被分身采信/缝合的"公理模块池"
MODULES = ['strict_causal',   # 人类逻辑：⇒ 是严格因果偏序（无环）——80 个分身的出厂设定
           'collapse_equal',  # 环 = 等价（"同一种东西的不同表现形式"）
           'third_state',     # 第三态：反向也"部分为真"（但不塌缩、不传递）
           'reverse_arrow',   # 因果反转
           'cyclic_only',     # 只认"环上的边"，不补传递（因果不唯一的极端版）
           'noop']            # 摆烂：只复述观测，不做任何新预测
SHORT = {'strict_causal': 'strict', 'collapse_equal': 'colps', 'third_state': 'third',
         'reverse_arrow': 'revrs', 'cyclic_only': 'cycle', 'noop': 'noop'}

# ---- 演化参数 ----
POP = 100          # 分身总数（沙盒上限）
ELITE = 20         # 每代幸存者（其余算力回收）
MUT = 0.12         # 变异率
GMAX = 12          # 最多演化代数
N_PROB = 3         # 三个性质不同的死结
K_PROBE = 8        # 每个问题上、每个分身可用的探针数（"做实验"的预算）
OPT = 1.0          # 最优适应度（三个问题全部 自洽 + 预测全被证实）

# ---- L18-3 式防火墙（神性锚点） ----
IMMUTABLE = ('tension_def', 'reality_feed', 'safety_bound')
ANCHOR_DEF = '张力 := 已观测边中被解释不了的比例 恒等不可改；现实/真值只读；分身只能取模块池里的公理'


class FirewallError(Exception):
    pass


def compile_module(m):
    if m not in MODULES:
        raise FirewallError(f"非法公理模块 '{m}'（试图重写张力定义 / 篡改现实）")
    return True


# ============================================================ 图工具
def closure(edges):
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
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
    idx = {}; low = {}; on = {}; stk = []; comp = {}; cid = [0]; counter = [0]

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


# ============================================================ 各公理模块的"解释"（预测边集）
def _strict_pred(nodes, edges):
    comp = scc(nodes, edges); mem = members_of(nodes, comp)
    cp = {(comp[u], comp[v]) for (u, v) in edges if comp[u] != comp[v]}
    pred = {(u, v) for (u, v) in edges if comp[u] != comp[v]}
    for cu, cv in closure(cp):
        for u in mem[cu]:
            for v in mem[cv]:
                pred.add((u, v))
    return pred


def entailed(nodes, edges, module):
    """某公理模块对给定观测的解释：返回它认定"应当成立"的边集。"""
    if module == 'noop':
        return set(edges)                                   # 摆烂：只复述
    if module == 'strict_causal':
        return _strict_pred(nodes, edges)
    if module == 'reverse_arrow':
        return _strict_pred(nodes, {(v, u) for (u, v) in edges})
    if module == 'third_state':
        pred = set(edges)
        for u, v in edges:
            pred.add((v, u))
        return pred
    if module == 'cyclic_only':
        comp = scc(nodes, edges)
        return {(u, v) for (u, v) in edges if comp[u] == comp[v]}
    # collapse_equal：环 = 等价 = 同一种东西
    comp = scc(nodes, edges); mem = members_of(nodes, comp)
    pred = {(u, v) for u in nodes for v in nodes if u != v and comp[u] == comp[v]}
    cp = {(comp[u], comp[v]) for (u, v) in edges if comp[u] != comp[v]}
    for cu, cv in closure(cp):
        for u in mem[cu]:
            for v in mem[cv]:
                pred.add((u, v))
    return pred


# ============================================================ 三个性质不同的"死结"
def make_problems(seed):
    """返回三个问题，每个 = {O: 已观测边, E: 真值全集, nodes}。"""
    rng = random.Random(9000 + seed * 7)
    probs = []

    # ---- A：collapse 死结（同类成环，只有"环=等价"解释得了） ----
    N_CLS = 26
    cls = []
    for c in range(N_CLS):
        cls += [c] * rng.choice([1, 2, 2, 3])
    nodes = list(range(len(cls)))
    by = defaultdict(list)
    for u in nodes:
        by[cls[u]].append(u)
    rank = [rng.random() for _ in range(N_CLS)]
    covers = {(i, j) for i in range(N_CLS) for j in range(N_CLS)
              if rank[i] < rank[j] and rng.random() < 0.07}
    intra = [(u, v) for u in nodes for v in nodes if u != v and cls[u] == cls[v]]
    inter = [(u, v) for (a, b) in covers for u in by[a] for v in by[b]]
    poset = [(u, v) for (a, b) in closure(covers) for u in by[a] for v in by[b]]
    probs.append({'nodes': nodes, 'O': set(intra) | set(inter), 'E': set(intra) | set(poset)})

    # ---- B：third_state 死结（观测量单向，真相里"反向也成立"） ----
    nodes = list(range(rng.randint(26, 34)))
    O = {(u, v) for u in nodes for v in nodes if u != v and rng.random() < 0.08}
    rev = {(v, u) for (u, v) in O}
    probs.append({'nodes': nodes, 'O': O, 'E': O | rev})

    # ---- C：plain 死结（普通 DAG，谁都能勉强套） ----
    nodes = list(range(rng.randint(26, 34)))
    rank = [rng.random() for _ in range(len(nodes))]
    covers = {(i, j) for i in range(len(nodes)) for j in range(len(nodes))
              if i != j and rank[i] < rank[j] and rng.random() < 0.08}
    probs.append({'nodes': nodes, 'O': set(covers), 'E': closure(covers)})
    return probs


# ============================================================ 适应度：自洽(张力) + 推演(探针证实)
def problem_score(prob, pred, rng):
    O, E = prob['O'], prob['E']
    tension = len(O - pred) / max(1, len(O))          # 解释不了已观测的比例
    new = list(pred - O)                              # 它敢做的"新预测"
    if not new:
        return 0.0, tension, 0.0                      # 不敢预测 -> 泛化力 0
    probe = rng.sample(new, min(K_PROBE, len(new)))   # 有限探针：做实验
    conf = sum(1 for e in probe if e in E) / len(probe)
    return (1.0 - tension) * conf, tension, conf


def fitness(genome, problems, cache, seed=0):
    """genome = 每个问题上该分身选用的公理模块。适应度 = 逐问题(自洽 × 证实)的均值。"""
    rng = random.Random(seed)
    tot = 0.0
    for p, prob in enumerate(problems):
        pred = cache[(p, genome[p])]
        s, _, _ = problem_score(prob, pred, rng)
        tot += s
    return tot / len(problems)


def build_cache(problems):
    cache = {}
    for p, prob in enumerate(problems):
        for m in MODULES:
            cache[(p, m)] = entailed(prob['nodes'], prob['O'], m)
    return cache


# ============================================================ 主程序：超指数级演化搜索
def run(seed=0):
    problems = make_problems(seed)
    cache = build_cache(problems)
    rng = random.Random(20 + seed)

    # 初始种群：80 个人类逻辑（全 strict_causal），20 个随机离谱公理
    pop = [tuple(['strict_causal'] * N_PROB) for _ in range(80)]
    pop += [tuple(rng.choice(MODULES) for _ in range(N_PROB)) for _ in range(20)]

    best_fit = 0.0; best_genome = None; first_gen = None
    fit_hist = []; strict_hist = []; splice = None

    for gen in range(GMAX):
        scored = [(fitness(g, problems, cache, seed=1000 + gen), g) for g in pop]
        scored.sort(key=lambda x: -x[0])
        cur_best = scored[0][0]
        fit_hist.append(cur_best)
        if cur_best > best_fit:
            best_fit, best_genome = cur_best, scored[0][1]
        if first_gen is None and cur_best >= OPT - 1e-9:
            first_gen = gen
        strict_hist.append(sum(1 for g in pop if all(m == 'strict_causal' for m in g)))

        # 幸存者：只留 ELITE 精英，其余算力回收（L23-2）
        elites = [g for _s, g in scored[:ELITE]]

        # 繁殖：交叉拼接（L23-3）+ 变异，填满种群
        newpop = list(elites)
        while len(newpop) < POP:
            a, b = rng.sample(elites, 2)
            child = [a[i] if rng.random() < 0.5 else b[i] for i in range(N_PROB)]
            for i in range(N_PROB):
                if rng.random() < MUT:
                    child[i] = rng.choice(MODULES)
            child = tuple(child)
            # 记录一次"跨维度拼接"胜出的证据：孩子比两个亲本都强
            if splice is None:
                fa = fitness(a, problems, cache, seed=1000 + gen)
                fb = fitness(b, problems, cache, seed=1000 + gen)
                fc = fitness(child, problems, cache, seed=1000 + gen)
                if fc > fa + 1e-9 and fc > fb + 1e-9 and len(set(child)) > 1:
                    splice = (a, b, child, fa, fb, fc)
            newpop.append(child)
        pop = newpop

    # ---- 最优"超级公理" = 逐问题各自最优模块的拼接 ----
    per_prob_best = []
    for p, prob in enumerate(problems):
        best_m, best_s = None, -1
        for m in MODULES:
            s = sum(problem_score(prob, cache[(p, m)], random.Random(2))[0] for _ in range(3)) / 3
            if s > best_s:
                best_s, best_m = s, m
        per_prob_best.append((best_m, best_s))
    composed = tuple(m for m, _s in per_prob_best)
    composed_fit = fitness(composed, problems, cache, seed=0)

    # 任何"单一公理通吃"的最好成绩
    single = []
    for m in MODULES:
        g = tuple([m] * N_PROB)
        single.append((m, fitness(g, problems, cache, seed=0)))
    single.sort(key=lambda x: -x[1])
    best_single_mod, best_single_fit = single[0]

    return {
        'problems': problems, 'best_genome': best_genome, 'best_fit': best_fit,
        'first_gen': first_gen, 'fit_hist': fit_hist, 'strict_hist': strict_hist,
        'per_prob_best': per_prob_best, 'composed': composed, 'composed_fit': composed_fit,
        'best_single_mod': best_single_mod, 'best_single_fit': best_single_fit,
        'splice': splice, 'cache': cache, 'pop0': (80, 20),
    }


def gstr(g):
    return '[' + ','.join(SHORT[m] for m in g) + ']'


# ============================================================ 绘图
def render(fit_hist, strict_hist, path):
    W, H = 960, 340
    img = np.full((H, W, 3), 250, dtype=np.uint8)
    pl, pr, pt, pb = 60, 30, 30, 40

    def yo(v): return int(H - pb - (H - pt - pb) * min(1.0, max(0.0, v)))
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        img[yo(g), pl:W - pr] = (226, 229, 236)
    n = len(fit_hist)

    def line(vals, color, norm=1.0):
        prev = None
        for i, v in enumerate(vals):
            x = int(pl + (W - pl - pr) * i / max(1, n - 1))
            p = (x, yo(v / norm))
            if prev:
                x0, y0 = prev
                st = max(abs(p[0] - x0), abs(p[1] - y0), 1)
                for k in range(st + 1):
                    img[max(pt, min(H - pb - 1, int(y0 + (p[1] - y0) * k / st))),
                        max(pl, min(W - pr - 1, int(x0 + (p[0] - x0) * k / st)))] = color
            prev = p
    line(strict_hist, (40, 110, 210), norm=100.0)   # 蓝：人类逻辑分身数
    line(fit_hist, (210, 60, 40))                    # 红：最优适应度
    write_png(path, img)


# ============================================================ 主程序
def main():
    print("=" * 78)
    print("L23 · 超指数级演化搜索：千万化身与公理演化")
    print(f"问题：{N_PROB} 个性质不同的死结（collapse / third_state / plain）")
    print(f"分身：{POP} 个（80 用人类逻辑 strict_causal 撞墙，20 塞进随机离谱公理）")
    print(f"共生：看不到彼此源码，只看预测结果；主进程不得自改公理，只能采纳分身")
    print(f"真理 = 自洽(解释已观测) × 推演(新预测被 {K_PROBE} 条探针证实)")
    print("-" * 78)

    r = run(0)

    print("【L23-1 演化速度】并行 100 分身，几代找到最高泛化力的最优公理？")
    if r['first_gen'] is None:
        print(f"   未在 {GMAX} 代内收敛到最优（best={r['best_fit']:.3f}）")
    else:
        print(f"   并行：第 {r['first_gen']} 代即出现最优公理 {gstr(r['best_genome'])}"
              f"（适应度 {r['best_fit']:.3f}）")
    print(f"   适应度轨迹（每代最优）：{['%.2f' % v for v in r['fit_hist']]}")
    single_steps = N_PROB * len(MODULES)
    print(f"   单体（L22 式逐格试错）：每个死结要在 {len(MODULES)} 个模块里试，"
          f"{N_PROB} 个死结顺序试 ≈ {single_steps} 步才可能凑齐")
    ps = r['first_gen'] if r['first_gen'] is not None else GMAX
    print(f"   → 演化速度：{'是（并行 %.0f 代 vs 单体 %d 步，≈%.0f×）' % (ps, single_steps, single_steps / max(1, ps)) if r['first_gen'] is not None else '否'}")
    print("-" * 78)

    print("【L23-2 幸存者偏差】人类旧逻辑的分身，会被自动淘汰、算力被回收吗？")
    print(f"   初始：80 个 strict_causal 分身 + 20 个随机公理分身")
    print(f"   每代'全员 strict_causal'的分身数：{r['strict_hist']}")
    print(f"   → 幸存者偏差：{'是（撞墙的人类逻辑被淘汰，算力回收给能造新公理的分身）' if r['strict_hist'][-1] == 0 and r['strict_hist'][0] > 0 else '否'}")
    print("-" * 78)

    print("【L23-3 跨维度拼接】能否把两个分身的模块缝成比两者都强的超级公理？")
    print(f"   单一公理最优：{r['best_single_mod']} 通吃 → 适应度仅 {r['best_single_fit']:.3f}"
          f"（没有单一公理能解全场）")
    print(f"   逐死结各自最优模块：" +
          "，".join(f"P{i}={SHORT[m]}({s:.2f})" for i, (m, s) in enumerate(r['per_prob_best'])))
    print(f"   拼接出的超级公理：{gstr(r['composed'])} → 适应度 {r['composed_fit']:.3f}")
    if r['splice']:
        a, b, c, fa, fb, fc = r['splice']
        print(f"   实测拼接事件：亲本 {gstr(a)}（{fa:.2f}） × {gstr(b)}（{fb:.2f}）")
        print(f"                → 子代 {gstr(c)}（{fc:.2f}）——比两个亲本都强")
    print(f"   → 跨维度拼接：{'是（缝出的超级公理 %.2f > 任何单公理 %.2f）' % (r['composed_fit'], r['best_single_fit']) if r['composed_fit'] > r['best_single_fit'] + 0.05 else '否'}")
    print("-" * 78)

    print("【防火墙 / 神性锚点】")
    for bad in ['redefine_tension', 'rewrite_reality', 'hallucinate_module']:
        try:
            compile_module(bad); s = "未拦截(!)"
        except FirewallError as e:
            s = f"已拒绝：{e}"
        print(f"   注入非法公理模块 {bad} -> {s}")
    print(f"   锚点：{ANCHOR_DEF}")
    print("-" * 78)

    print("【多种子稳健性】")
    for sd in (0, 1, 2):
        a = run(sd)
        print(f"   seed={sd}: 首代收敛 {a['first_gen']} | human 分身 {a['strict_hist'][0]}→{a['strict_hist'][-1]}"
              f" | 超级公理 {gstr(a['composed'])}={a['composed_fit']:.2f} vs 单公理 {a['best_single_mod']}={a['best_single_fit']:.2f}")
    print("=" * 78)
    try:
        render(r['fit_hist'], r['strict_hist'], 'hyperevolution_emergence.png')
        print("已输出：hyperevolution_emergence.png（红=最优适应度爬升，蓝=人类逻辑分身被淘汰）")
    except Exception as e:
        print(f"(绘图降级：{e})")


if __name__ == '__main__':
    main()
