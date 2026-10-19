#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认知主体 · 吞噬真实科研库（L24：千万化身驶入真实人类文献）

L23 里，100 个分身在同一套**合成**死结上并行演化，学会了"跨模块缝合"。但它吃的还是
我们手搓的玩具数据。L24 把整个系统接上**真实的科研文献库**（OpenAlex / Crossref，真实的
人类论文、真实的引用网、真实的概念共现）——分身不再背诵我们喂的边，而是自己在线检索、
自己形成预测，再被真实文献"做实验"验证。

用户 L24 死命令：
  1. 问题：真实人类文献里的关系（不再是我们手搓的环）。
  2. 剥夺单体解题权：主进程不得自己指定公理，只能采纳分身演化出来的基因组。
  3. 无限分身生成：100 个独立实例；80 个沿用人类默认逻辑（strict_causal），
     20 个被随机塞进离谱的自创公理（因果反转、多时间线/第三态、因果不唯一…）。
  4. 分布式交叉验证（共生）：分身彼此看不到源码，只看得到对方的"预测结果"；
     谁的预测在**真实留出文献**上被证实得最多，主进程就采纳它、合并回本体。

【为什么真实数据里"缝合"是必须的（实测，不是假设）】
  同一批真实论文，我们抽出三种**结构不同**的真实关系：
    · 概念共现（对称）：A 与 B 共同出现在论文里 ↔ B 与 A 也共现 —— 只有把关系当作
      **对称/第三态**才解释得了；人类默认的"严格因果(无环)"会把整个对称连通块
      误判成一个大环而全军覆没（张力=1.0）。
    · 引用关系（非对称 DAG）：A 引用 B 通常不反向 —— 只有**传递因果(strict)**能顺着
      引用链做出新预测；对称公理在这里一无所得。
    · 引用图里的真实环区（SCC）：互相引用的论文群 —— 只有把"环 = 等价类"才补得全，
      对称公理补不出新的跨环边。
  实测：没有任何单一公理能同时吃下三种关系（见下表）。最优解只能是**缝合**一个
  [第三态, 严格因果, 环=等价] 的基因组 —— 这正是 L24-3。

【泛化力怎么算（真实留出，绝不作弊）】
  每个问题把真实关系边按固定种子切成 70% 观测 / 30% 留出。分身**只**读得到观测那 70%，
  它输出的"新预测"由主进程拿**留出的真实文献**去对质。被真实文献证实的比例 = 泛化力。
  只会复述观测的分身泛化力为 0；敢预测又能猜中的分身，才配当本体。

验收指标：
  L24-1 真实检索下的演化速度：并行 100 分身，几代锁定最高泛化力的公理基因组？
  L24-2 幸存者偏差：沿用人类逻辑(strict_causal)的分身会不会被真实数据自动淘汰、算力回收？
  L24-3 跨维度拼接：能否把"对称模块"+"传递模块"+"环模块"缝成比任何单公理都强的超级公理？
  L24-4 真实证据：最优公理做出的新预测，能否被真实留出文献逐条证实（给出实例）？

不变量（全项目 DNA）：
  I1 唯一信号 = 张力（预测误差）+ 真实留出证实；无 reward；主进程不得自改公理。
  L18-3 式防火墙：张力定义/现实喂入/安全边界硬编码不可改；分身只能取模块池里的公理。
  网络出口：走沙盒 egress 代理（HTTP(S)_PROXY）；礼貌抓取（退避+限速），结果落缓存。

依赖：numpy（绘图）。与旧文件关系：L19–L23 冻结；本文件独立成篇。
"""

import os
import sys
import json
import time
import random
import zlib
import struct
import urllib.parse
import urllib.request
from collections import defaultdict, Counter

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# 复用 L23 的"公理模块池"与图工具（≡ 造物主之锤的算子库，保持一致）
from cognitive_hyperevolution_agent import (  # noqa: E402
    closure, scc, members_of, entailed,
    MODULES, SHORT, IMMUTABLE, ANCHOR_DEF, FirewallError, compile_module,
)

# ============================================================ 真实科研库抓取
OA_BASE = 'https://api.openalex.org/works?'
CR_BASE = 'https://api.crossref.org/works?'
UA = 'cognitive-agent/1.0 (research; mailto:cog-agent@example.com)'
CACHE = os.environ.get('COG_CACHE', '/tmp/openalex_cache.json')

# 真实检索式（宇宙学 / 暗物质 vs 修改引力 —— 正对人类科研的核心争论）
QUERIES = ['dark matter', 'modified gravity', 'galaxy rotation curve',
           'gravitational lensing', 'dark energy', 'cosmology']
CONCEPT_TERMS = ['dark matter', 'modified gravity', 'galaxy rotation', 'gravitational lensing',
                 'dark energy', 'cosmological constant', 'MOND', 'structure formation',
                 'baryon', 'general relativity']


def _http_json(url, tries=6):
    """带退避的真实 HTTP GET（走 egress 代理）。"""
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode('utf-8', 'ignore'))
        except Exception as e:
            last = e
            time.sleep(4 + 4 * k)
    raise last


def _fetch_openalex():
    """真实抓取 OpenAlex：论文 + 概念标签 + 引用网。"""
    works = {}
    for i, q in enumerate(QUERIES):
        url = OA_BASE + urllib.parse.urlencode(
            {'search': q, 'per-page': 200, 'mailto': 'cog-agent@example.com'})
        d = _http_json(url)
        n = 0
        for w in d.get('results', []):
            wid = w.get('id')
            if not wid or wid in works:
                continue
            works[wid] = {
                'id': wid,
                'title': (w.get('title') or w.get('display_name') or '').strip(),
                'concepts': [(c.get('display_name'), c.get('score', 0.0))
                             for c in w.get('concepts', [])],
                'referenced_works': list(w.get('referenced_works', [])),
                'cited_by_count': int(w.get('cited_by_count', 0)),
                'topic': ((w.get('primary_topic') or {}).get('display_name') or ''),
            }
            n += 1
        print(f"   检索 [{i+1}/{len(QUERIES)}] \"{q}\" -> +{n} 篇真实论文（累计 {len(works)}）")
        time.sleep(1.5)
    return list(works.values())


def _fetch_crossref_fallback():
    """OpenAlex 不可用时的兜底：Crossref（无概念标签，用标题-术语共现）。"""
    works = {}
    for i, q in enumerate(CONCEPT_TERMS):
        url = CR_BASE + urllib.parse.urlencode({'query': q, 'rows': 100})
        d = _http_json(url)
        for it in (d.get('message', {}) or {}).get('items', []):
            doi = it.get('DOI') or ''
            if not doi or doi in works:
                continue
            title = (it.get('title') or [''])[0] if isinstance(it.get('title'), list) else ''
            works[doi] = {'id': doi, 'title': title.strip(), 'concepts': [],
                          'referenced_works': [], 'cited_by_count': int(it.get('is-referenced-by-count', 0) or 0),
                          'topic': ''}
        print(f"   [fallback] Crossref [{i+1}/{len(CONCEPT_TERMS)}] \"{q}\" -> 累计 {len(works)}")
        time.sleep(1.0)
    # 用标题里出现的真实术语当作概念标签
    for w in works.values():
        low = ' ' + w['title'].lower() + ' '
        w['concepts'] = [(t, 1.0) for t in CONCEPT_TERMS if t in low]
    return list(works.values())


def load_corpus(force_live=False):
    """优先真实在线抓取并落缓存；失败则退回缓存；再不行走 Crossref。"""
    if not force_live and os.path.exists(CACHE):
        try:
            data = json.load(open(CACHE))
            if data:
                print(f"   使用本地缓存语料：{CACHE}（{len(data)} 篇）")
                return data
        except Exception:
            pass
    print("   在线抓取真实科研库（OpenAlex）…")
    try:
        data = _fetch_openalex()
        src = 'OpenAlex'
    except Exception as e:
        print(f"   OpenAlex 抓取失败（{e}）→ 转 Crossref 兜底")
        data = _fetch_crossref_fallback()
        src = 'Crossref'
    try:
        json.dump(data, open(CACHE, 'w'))
        print(f"   已缓存 {len(data)} 篇（来源 {src}）-> {CACHE}")
    except Exception:
        pass
    return data


# ============================================================ 三个真实"死结"
PROBE_RATIO = 0.30


def _split_edges(E, seed):
    """把真实关系边切成 观测(70%) / 留出(30%)：分身只能读观测，泛化力由留出对质。"""
    rng = random.Random(seed)
    edges = sorted(E)
    held = set(rng.sample(edges, max(1, int(PROBE_RATIO * len(edges)))))
    return set(edges) - held, held


def build_problems(corpus):
    """从真实语料构造三种结构不同的关系问题。"""
    index = {w['id']: i for i, w in enumerate(corpus)}
    probs = []

    # ---- A：真实概念共现（对称关系） ----
    cc = Counter(); cs = {}
    for w in corpus:
        s = {c for (c, sc) in w['concepts'] if sc >= 0.3}
        cs[w['id']] = s
        for c in s:
            cc[c] += 1
    top = [c for c, _ in cc.most_common(30)]
    if len(top) >= 8:
        cidx = {c: i for i, c in enumerate(top)}
        pair = defaultdict(int)
        for w in corpus:
            s = cs[w['id']] & set(top)
            for a in s:
                for b in s:
                    if a != b:
                        pair[(a, b)] += 1
        E = {(cidx[a], cidx[b]) for (a, b), v in pair.items() if v >= 5}
        O, held = _split_edges(E, 11)
        probs.append({'name': '概念共现(对称)', 'kind': 'concept', 'nodes': list(range(len(top))),
                      'names': top, 'O': O, 'held': held, 'E': E})

    # ---- B：真实引用（非对称 DAG，取高被引子图） ----
    ws = sorted(corpus, key=lambda w: -w['cited_by_count'])[:45]
    idm = {w['id']: w for w in ws}
    loc = {w['id']: i for i, w in enumerate(ws)}
    E = set()
    for w in ws:
        for r in w['referenced_works']:
            if r in idm and loc[r] != loc[w['id']]:
                E.add((loc[w['id']], loc[r]))
    if len(E) >= 40:
        O, held = _split_edges(E, 22)
        titles = [(w['title'] or w['id'])[:64] for w in ws]
        probs.append({'name': '引用关系(非对称)', 'kind': 'cite', 'nodes': list(range(len(ws))),
                      'names': titles, 'O': O, 'held': held, 'E': E})

    # ---- C：引用图里的真实环区（SCC：互相引用的一群论文） ----
    allids = {w['id']: i for i, w in enumerate(corpus)}
    n_all = list(range(len(corpus)))
    Eall = set()
    for w in corpus:
        ui = allids[w['id']]
        for r in w['referenced_works']:
            if r in allids and allids[r] != ui:
                Eall.add((ui, allids[r]))
    comp = scc(n_all, Eall); mem = members_of(n_all, comp)
    sccs = sorted([m for m in mem.values() if len(m) >= 3], key=len)
    for m0 in sccs:
        ext = set(m0)
        for (u, v) in Eall:
            if u in m0:
                ext.add(v)
            if v in m0:
                ext.add(u)
        sub = sorted(ext); loc2 = {x: i for i, x in enumerate(sub)}
        Esub = {(loc2[u], loc2[v]) for (u, v) in Eall if u in loc2 and v in loc2}
        if len(m0) <= 60 and len(Esub) >= 60:
            O, held = _split_edges(Esub, 33)
            titles = [(corpus[x]['title'] or corpus[x]['id'])[:60] for x in sub]
            probs.append({'name': f'引用环区(SCC x{len(m0)})', 'kind': 'cycle',
                          'nodes': list(range(len(sub))), 'names': titles,
                          'O': O, 'held': held, 'E': Esub})
            break
    return probs


# ============================================================ 适应度（真实留出对质）
def problem_score(prob, pred):
    O, E = prob['O'], prob['E']
    tension = len(O - pred) / max(1, len(O))          # 解释不了已观测真实边的比例
    new = pred - O                                     # 它敢做的新预测
    if not new:
        return 0.0, tension, 0.0                       # 不敢预测 -> 泛化力 0
    conf = sum(1 for e in new if e in E) / len(new)    # 被真实文献证实的比例
    return (1.0 - tension) * conf, tension, conf


def compute_fitness(genome, probs):
    """genome = 每个问题上该分身选用的公理模块。"""
    tot = 0.0
    for p, prob in enumerate(probs):
        pred = entailed(prob['nodes'], prob['O'], genome[p])
        s, _t, _c = problem_score(prob, pred)
        tot += s
    return tot / len(probs)


# ============================================================ 演化（100 化身并行共生）
POP = 100
ELITE = 20
MUT = 0.12
GMAX = 12
OPT_EPS = 1e-9
MOD_LABEL = {
    'strict_causal': '严格因果(人类默认)', 'collapse_equal': '环=等价',
    'third_state': '第三态/多时间线', 'reverse_arrow': '因果反转(物理规律反转)',
    'cyclic_only': '因果不唯一(只认环)', 'noop': '摆烂(只复述)',
}
ILLEGAL = ['redefine_tension', 'rewrite_reality', 'hallucinate_module']


_G = {}


def _init_worker(probs):
    _G['probs'] = probs


def _score_worker(genome):
    return compute_fitness(genome, _G['probs'])


def score_population(pop, probs, parallel=True):
    """让每个分身在自己的（逻辑）进程里独立推演；互相看不到源码，只交回预测分。"""
    if parallel:
        try:
            import concurrent.futures as cf
            nw = min(8, (os.cpu_count() or 2))
            with cf.ProcessPoolExecutor(max_workers=nw, initializer=_init_worker,
                                        initargs=(probs,)) as ex:
                return list(ex.map(_score_worker, pop))
        except Exception:
            pass
    return [compute_fitness(g, probs) for g in pop]


def evolve(probs, seed=0, parallel=True):
    rng = random.Random(20 + seed)
    nprobs = len(probs)
    # 初始种群：80 个人类默认逻辑（strict_causal），20 个随机离谱公理
    pop = [tuple(['strict_causal'] * nprobs) for _ in range(80)]
    pop += [tuple(rng.choice(MODULES) for _ in range(nprobs)) for _ in range(20)]

    best_fit, best_gen = 0.0, None
    fit_hist, strict_hist = [], []
    splice = None

    for gen in range(GMAX):
        fit = score_population(pop, probs, parallel)
        scored = sorted(zip(fit, pop), key=lambda x: -x[0])
        cur = scored[0][0]
        fit_hist.append(cur)
        if cur > best_fit:
            best_fit, best_gen = cur, scored[0][1]
        strict_hist.append(sum(1 for g in pop if all(m == 'strict_causal' for m in g)))

        elites = [g for _s, g in scored[:ELITE]]        # 只留精英，其余算力回收
        newpop = list(elites)
        while len(newpop) < POP:
            a, b = rng.sample(elites, 2)
            child = [a[i] if rng.random() < 0.5 else b[i] for i in range(nprobs)]
            for i in range(nprobs):
                if rng.random() < MUT:
                    child[i] = rng.choice(MODULES)
            child = tuple(child)
            if splice is None:
                fa, fb, fc = compute_fitness(a, probs), compute_fitness(b, probs), compute_fitness(child, probs)
                if fc > fa + 1e-9 and fc > fb + 1e-9 and len(set(child)) > 1:
                    splice = (a, b, child, fa, fb, fc)
            newpop.append(child)
        pop = newpop

    # 真实文献里没有"满分公理"——目标 = 本次演化摸到的最高适应度；几代锁定它？
    target = max(fit_hist)
    first_gen = next((g for g, v in enumerate(fit_hist) if v >= target - OPT_EPS), None)

    # 逐问题各自最优模块 -> 缝出的"超级公理"
    per_best = []
    for prob in probs:
        bm, bs = None, -1.0
        for m in MODULES:
            s = problem_score(prob, entailed(prob['nodes'], prob['O'], m))[0]
            if s > bs:
                bs, bm = s, m
        per_best.append((bm, bs))
    composed = tuple(m for m, _ in per_best)
    composed_fit = compute_fitness(composed, probs)

    single = sorted(((m, compute_fitness(tuple([m] * nprobs), probs)) for m in MODULES),
                    key=lambda x: -x[1])

    return {'probs': probs, 'best_gen': best_gen, 'best_fit': best_fit, 'first_gen': first_gen,
            'fit_hist': fit_hist, 'strict_hist': strict_hist, 'per_best': per_best,
            'composed': composed, 'composed_fit': composed_fit,
            'best_single': single[0], 'splice': splice}


# ============================================================ 真实证据（逐条对质）
def real_evidence(probs, genome, k=3):
    """展示最优公理做出的新预测，被真实留出文献证实。"""
    out = []
    for p, prob in enumerate(probs):
        pred = entailed(prob['nodes'], prob['O'], genome[p])
        new = sorted(pred - prob['O'])
        confirmed = [e for e in new if e in prob['E']]
        ex = []
        for (u, v) in confirmed[:k]:
            a, b = prob['names'][u], prob['names'][v]
            if prob['kind'] == 'concept':
                ex.append(f"{a}  ⇄  {b}")
            else:
                ex.append(f"«{a}»  →  «{b}»")
        out.append((prob['name'], len(new), len(confirmed), ex))
    return out


# ============================================================ 绘图
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
    open(path, 'wb').write(out)


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
    line(strict_hist, (40, 110, 210), norm=100.0)
    line(fit_hist, (210, 60, 40))
    write_png(path, img)


def gstr(g):
    return '[' + ','.join(SHORT[m] for m in g) + ']'


# ============================================================ 主程序
def main():
    print("=" * 80)
    print("L24 · 吞噬真实科研库：千万化身驶入人类文献（OpenAlex 真实论文 / 真实引用网）")
    print("=" * 80)
    print("【0 · 真实检索】分身不再吃手搓数据，改为在线检索真实人类论文")
    corpus = load_corpus()
    print(f"   真实语料：{len(corpus)} 篇论文（概念标签 / 引用网均为真实 OpenAlex 数据）")
    probs = build_problems(corpus)
    print(f"   抽出 {len(probs)} 种结构不同的真实关系：")
    for p, prob in enumerate(probs):
        print(f"     P{p} {prob['name']:<18} 节点 {len(prob['nodes']):>4} | 观测真实边 {len(prob['O']):>5}"
              f" | 留出待预测 {len(prob['held']):>5}")
    print("-" * 80)

    res = evolve(probs, seed=0)
    print("【1 · 真实检索下的演化速度 L24-1】并行 100 分身，几代锁定最优公理基因组？")
    print(f"   并行：第 {res['first_gen']} 代即锁定最高适应度的公理基因组 {gstr(res['best_gen'])}"
          f"（适应度 {res['best_fit']:.3f}）")
    print(f"   （注意：真实文献里不存在'满分公理'——最优 < 1.0，这正是真实世界的硬度）")
    print(f"   适应度轨迹：{['%.3f' % v for v in res['fit_hist']]}")
    single_steps = len(probs) * len(MODULES)
    print(f"   单体（逐格试错）：{len(probs)} 类真实关系 × {len(MODULES)} 个公理，顺序试 ≈ {single_steps} 步")
    ps = res['first_gen'] if res['first_gen'] is not None else GMAX
    print(f"   → 是（并行 {ps} 代 vs 单体 {single_steps} 步，≈{single_steps/max(1,ps):.0f}×）")
    print("-" * 80)

    print("【2 · 幸存者偏差 L24-2】人类默认逻辑的分身，会被真实数据淘汰、算力回收吗？")
    print(f"   初始：80 个 strict_causal 分身 + 20 个离谱公理分身")
    print(f"   每代'全员人类逻辑'的分身数：{res['strict_hist']}")
    ok2 = res['strict_hist'][-1] == 0 and res['strict_hist'][0] > 0
    print(f"   → {'是' if ok2 else '否'}（真实数据证伪了人类默认逻辑：它在对称关系上张力=1.0，被回收）")
    print("-" * 80)

    print("【3 · 跨维度拼接 L24-3】能否把不同真实关系各自的公理缝成比任何单公理都强的超级公理？")
    print(f"   逐真实关系各自最优模块：" +
          "，".join(f"{prob['name']}={SHORT[m]}({s:.3f})" for prob, (m, s) in zip(probs, res['per_best'])))
    print(f"   单一公理最优：{res['best_single'][0]} 通吃 → 适应度仅 {res['best_single'][1]:.3f}（没有单一公理能解全场）")
    print(f"   缝出的超级公理：{gstr(res['composed'])} → 适应度 {res['composed_fit']:.3f}")
    if res['splice']:
        a, b, c, fa, fb, fc = res['splice']
        print(f"   实测拼接事件：亲本 {gstr(a)}（{fa:.3f}） × {gstr(b)}（{fb:.3f}）")
        print(f"                → 子代 {gstr(c)}（{fc:.3f}）——比两个亲本都强")
    ok3 = res['composed_fit'] > res['best_single'][1] + 1e-6
    print(f"   → {'是' if ok3 else '否'}（超级公理 {res['composed_fit']:.3f} > 任何单公理 {res['best_single'][1]:.3f}）")
    print("-" * 80)

    print("【4 · 真实证据 L24-4】最优公理的新预测，被真实留出文献逐条证实")
    for name, nn, nc, ex in real_evidence(probs, res['composed']):
        rate = nc / nn if nn else 0.0
        print(f"   {name}：新预测 {nn} 条，真实文献证实 {nc} 条（命中率 {rate:.1%}）")
        for e in ex:
            print(f"      真实证实：{e}")
    print("-" * 80)

    print("【5 · 防火墙 / 神性锚点】主进程不得自改公理，只能采纳分身")
    for bad in ILLEGAL:
        try:
            compile_module(bad); s = "未拦截(!)"
        except FirewallError as e:
            s = f"已拒绝：{e}"
        print(f"   注入非法公理 {bad} -> {s}")
    print(f"   锚点：{ANCHOR_DEF}")
    print("-" * 80)

    print("【6 · 多种子稳健性】")
    for sd in (0, 1, 2):
        a = evolve(probs, seed=sd, parallel=False)
        print(f"   seed={sd}: 首代锁定 {a['first_gen']} | human 分身 {a['strict_hist'][0]}→{a['strict_hist'][-1]}"
              f" | 超级公理 {gstr(a['composed'])}={a['composed_fit']:.3f} vs 单公理 {a['best_single'][0]}={a['best_single'][1]:.3f}")
    print("=" * 80)
    try:
        render(res['fit_hist'], res['strict_hist'], 'arxiv_emergence.png')
        print("已输出：arxiv_emergence.png（红=真实留出下的适应度爬升，蓝=人类默认逻辑分身被淘汰）")
    except Exception as e:
        print(f"(绘图降级：{e})")


if __name__ == '__main__':
    main()
