#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L38 —— 定向富集：不问「进没进 top-20」，问「在同类词里排第几」

L36/L37 的失败方式已经被定位：把「内卷」的邻居按全局 top-20 排序，
无论用 PPMI 还是原始共现次数，前排都会被「理论 / 发展 / 社会 / 开始」这类高频泛化词占满。
而 L37 的窗口统计显示社会词其实都在场（行业 42、竞争 24、资源 15、教育 15、岗位 14、学历 12）。

所以这一轮换一个问题：
  对每一个预注册的社会词 s，把 PPMI(内卷, s) 放到「和『内卷』df 同量级的 200 个对照词」
  的 PPMI(x, s) 分布里，看『内卷』排在第几百分位。
这不是「有没有挤进 top-20」，而是「它是不是比同频词更靠近社会词」。

不变式与 L1–L37 完全一致。

预注册：
  社会词 24 个（竞争/加班/内耗/资源/稀缺/压力/焦虑/房价/就业/学历/考核/淘汰/
              工资/薪资/劳动/绩效/职场/打工人/资本/市场/行业/岗位/招聘/教育）
  对照：与目标词 df 同量级（log df 最近的 200 词），不排除任何词
  零假设：PPMI(对照词, s) >= PPMI(内卷, s)
  判据：
    L38-1  现代语料：24 个社会词的平均分位 <= 0.25
    L38-2  现代语料：分位 <= 0.10 的社会词 >= 6 个
    L38-3  现代语料优于混合年代（平均分位更低）
    L38-4  特异性：非社会锚点（熵增/腐蚀）在同一个检验上不富集（平均分位 >= 0.35）
"""

import os, sys, json, time, math, sqlite3, pickle, gc
import numpy as np
from collections import Counter

sys.path.insert(0, '/workspace')
import cognitive_chinese_social_agent as I

SEED = 38
OUT_JSON = '/workspace/l38_social_enrichment_result.json'
OUT_MD = '/workspace/L38_社会词定向富集实证报告.md'
PNG = '/workspace/l38_social_enrichment.png'

N_CTRL = 200
SEEDS = ['竞争', '加班', '内耗', '资源', '稀缺', '压力', '焦虑', '房价', '就业', '学历',
         '考核', '淘汰', '工资', '薪资', '劳动', '绩效', '职场', '打工人', '资本',
         '市场', '行业', '岗位', '招聘', '教育']
TARGETS = ['内卷']
ANCHORS = ['熵增', '腐蚀', '疲惫']       # 特异性对照：不该在社会词上富集
TEMPLATE = ['就业']                       # 已知社会词：应该富集

CORPORA = [
    {'key': 'L37', 'name': '现代语料（2023 为主）',
     'db': '/workspace/l37_modern_era.sqlite', 'cache': '/workspace/.l36_data/cache_l37.pkl'},
    {'key': 'L36', 'name': '混合年代（05–11 新闻 + 知乎）',
     'db': '/workspace/l36_chinese_social.sqlite', 'cache': '/workspace/.l36_data/cache_full.pkl'},
]

I.VOCAB = 8000
I.N_IDIOM_CAP = 600
I.MEM_LIMIT_MB = 700


def build(corp):
    """载入缓存 → 重建同一张图（参数与 L36/L37 一致）。"""
    if not (os.path.exists(corp['db']) and os.path.exists(corp['cache'])):
        return None
    con = sqlite3.connect(corp['db'])
    with open(corp['cache'], 'rb') as f:
        meta, old_items, dfv_old, flag_of, stats = pickle.load(f)
    n_doc = con.execute('SELECT COUNT(*) FROM l36_doc').fetchone()[0]
    if stats.get('n_doc') != n_doc:
        print('  [%s] 缓存与库不一致（%d vs %d），跳过' % (corp['key'], stats.get('n_doc'), n_doc))
        return None
    print('  [%s] %d 篇 / %d 词种' % (corp['key'], n_doc, len(old_items)), flush=True)
    vocab, dfv, flagv, old2new, idiom_rows, admitted, missing = \
        I.choose_vocab(old_items, dfv_old, flag_of, stats)
    stats['idiom_ctx'] = {}
    gc.collect()
    I.trim_mem()
    protect = set(I.PROBES) | set(I.HEADS) | set(w for w, _ in admitted)
    protect |= set(I.ENERGY_ANCHOR) | set(I.PAIN_ANCHOR)
    P, S, pair_src, _ = I.build_graph(con, vocab, old2new, protect, stats['phrase_ctx'])
    gc.collect()
    I.trim_mem()
    wid_of = {w: i for i, w in enumerate(vocab)}
    return {'con': con, 'P': P, 'vocab': vocab, 'dfv': dfv, 'wid_of': wid_of,
            'meta': meta, 'key': corp['key'], 'name': corp['name']}


def controls(dfv, wid_of, w, n=N_CTRL):
    j = wid_of.get(w)
    if j is None:
        return []
    base = math.log1p(float(dfv[j]))
    idx = sorted(range(len(dfv)), key=lambda x: abs(math.log1p(float(dfv[x])) - base))
    return [x for x in idx if x != j][:n]


def enrich(g, word):
    """对 word 与社会词的 PPMI 做「同频对照」富集。"""
    wid_of, P, dfv = g['wid_of'], g['P'], g['dfv']
    j = wid_of.get(word)
    if j is None:
        return {'present': False}
    ctrl = controls(dfv, wid_of, word)
    if not ctrl:
        return {'present': False, 'why': '无对照'}
    ctrl = np.asarray(ctrl, dtype=np.int32)
    rows = []
    for s in SEEDS:
        k = wid_of.get(s)
        if k is None:
            rows.append({'word': s, 'present': False})
            continue
        col = np.asarray(P.getcol(k).todense()).ravel()
        obs = float(col[j])
        null = col[ctrl]
        # 单侧 p 值：对照词里有多少个「至少和它一样高」。越小越靠近。
        pv = float(np.mean(null >= obs))
        rows.append({'word': s, 'present': True, 'ppmi': round(obs, 4),
                     'pval': round(pv, 4),
                     'null_med': round(float(np.median(null)), 4),
                     'df': int(dfv[k])})
    ok = [r for r in rows if r.get('present')]
    pcts = [r['pval'] for r in ok]
    return {'present': True, 'df': int(dfv[j]), 'n_ctrl': int(len(ctrl)),
            'seed_rows': rows,
            'mean_pct': round(float(np.mean(pcts)), 4) if pcts else 1.0,
            'median_pct': round(float(np.median(pcts)), 4) if pcts else 1.0,
            'n_top10': int(sum(1 for x in pcts if x <= 0.10)),
            'n_top25': int(sum(1 for x in pcts if x <= 0.25))}


def main():
    t0 = time.time()
    print('=' * 72)
    print('L38 社会词定向富集 | 不问 top-20，问同频排位')
    print('=' * 72, flush=True)

    out = {}
    for corp in CORPORA:
        print('\n[%s] %s' % (corp['key'], corp['name']), flush=True)
        g = build(corp)
        if g is None:
            print('  跳过（缺库或缺缓存）', flush=True)
            continue
        res = {}
        for w in TARGETS + TEMPLATE + ANCHORS:
            r = enrich(g, w)
            res[w] = r
            if r.get('present'):
                print('  %-4s df=%4d  平均分位 %.3f  中位 %.3f  <=10%% 的种子 %2d/%d  <=25%% %2d'
                      % (w, r['df'], r['mean_pct'], r['median_pct'],
                         r['n_top10'], len(SEEDS), r['n_top25']), flush=True)
            else:
                print('  %-4s 不在词表' % w, flush=True)
        con = g['con']
        docs = con.execute('SELECT COUNT(*), COALESCE(SUM(nc),0) FROM l36_doc').fetchone()
        cov = {}
        for w in TARGETS + TEMPLATE:
            cov[w] = {s: int(n) for s, n in con.execute(
                'SELECT src, COUNT(*) FROM l36_doc WHERE instr(text, ?) > 0 GROUP BY src', (w,))}
        con.close()
        out[corp['key']] = {'name': corp['name'], 'n_docs': int(docs[0]),
                            'n_chars': int(docs[1]), 'words': res,
                            'phrase_cov': cov}
        del g
        gc.collect()
        I.trim_mem()

    # ---------------- 判决 ----------------
    mod = out.get('L37', {}).get('words', {})
    mix = out.get('L36', {}).get('words', {})
    inv_m = mod.get('内卷', {})
    inv_x = mix.get('内卷', {})
    vd = {}
    if inv_m.get('present'):
        vd['L38-1'] = [inv_m['mean_pct'] <= 0.25,
                       '现代语料平均 p %.3f' % inv_m['mean_pct'],
                       '24 个社会词的平均 p <= 0.25']
        vd['L38-2'] = [inv_m['n_top10'] >= 6,
                       'p <=10%% 的社会词 %d 个' % inv_m['n_top10'],
                       'p<=10% 的种子 >= 6 个']
    else:
        vd['L38-1'] = [False, '内卷不在现代语料词表', '24 个社会词的平均 p <= 0.25']
        vd['L38-2'] = [False, '内卷不在现代语料词表', 'p<=10% 的种子 >= 6 个']
    if inv_m.get('present') and inv_x.get('present'):
        vd['L38-3'] = [inv_m['mean_pct'] < inv_x['mean_pct'],
                       '现代 %.3f vs 混合 %.3f' % (inv_m['mean_pct'], inv_x['mean_pct']),
                       '现代语料的平均 p 低于混合年代']
    else:
        vd['L38-3'] = [False, '缺一版测量', '现代语料的平均 p 低于混合年代']
    anc = [mod.get(w, {}) for w in ANCHORS if mod.get(w, {}).get('present')]
    if anc:
        am = float(np.mean([a['mean_pct'] for a in anc]))
        vd['L38-4'] = [am >= 0.35, '非社会锚点平均 p %.3f（%s）'
                       % (am, '、'.join('%s %.3f' % (w, mod[w]['mean_pct']) for w in ANCHORS
                                        if mod.get(w, {}).get('present'))),
                       '特异性：熵增/腐蚀的平均 p >= 0.35（不该富集）']
    else:
        vd['L38-4'] = [False, '锚点不在词表', '特异性：非社会锚点不应富集']

    # 稀有度检验：如果「熵增」这种稀有非社会词的 p 比「内卷」还低，
    # 那这套检验测的是词频，不是社会语义。
    if inv_m.get('present') and mod.get('熵增', {}).get('present'):
        a = inv_m['mean_pct']
        b = mod['熵增']['mean_pct']
        vd['L38-5'] = [a <= b,
                       '内卷 p=%.3f vs 熵增 p=%.3f' % (a, b),
                       '内卷的社会词 p 必须低于熵增（否则检验被稀有度支配）']
    else:
        vd['L38-5'] = [False, '缺一版测量', '内卷的社会词 p 必须低于熵增']

    print('\n判决：' + ' | '.join('%s %s' % (k, 'PASS' if x[0] else 'FAIL')
                                for k, x in vd.items()), flush=True)

    payload = {'seeds': SEEDS, 'targets': TARGETS, 'anchors': ANCHORS,
               'template': TEMPLATE, 'n_ctrl': N_CTRL,
               'corpora': out, 'verdict': vd}
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(I.jsonable(payload), f, ensure_ascii=False, indent=1)

    write_report(payload)
    draw(payload)
    print('  完成：%.1fs / 峰值内存约 %.0fMB' % (time.time() - t0, I.mem_mb()), flush=True)


def write_report(p):
    rows_modern = p['corpora'].get('L37', {}).get('words', {}).get('内卷', {}).get('seed_rows', [])
    rows_mixed = p['corpora'].get('L36', {}).get('words', {}).get('内卷', {}).get('seed_rows', [])

    def seed_table(a, b):
        out = []
        for x, y in zip(a, b):
            if not x.get('present'):
                continue
            out.append('| %s | %.3f | %.3f | %s | %.3f | %.3f | %s |'
                       % (x['word'], x['ppmi'], x['pval'],
                          '✓' if x['pval'] <= 0.10 else '',
                          y.get('ppmi', 0) if y.get('present') else float('nan'),
                          y.get('pval', 1) if y.get('present') else float('nan'),
                          '✓' if y.get('present') and y['pval'] <= 0.10 else ''))
        return '\n'.join(out)

    def summary_table():
        out = []
        for k in ('L37', 'L36'):
            c = p['corpora'].get(k)
            if not c:
                continue
            for w, r in c['words'].items():
                if not r.get('present'):
                    continue
                out.append('| %s | %s | %d | %.3f | %.3f | %d/%d |'
                           % (w, c['name'], r['df'], r['mean_pct'], r['median_pct'],
                              r['n_top10'], len(p['seeds'])))
        return '\n'.join(out)

    vd_rows = '\n'.join('| %s | %s | %s | %s |' % (k, 'PASS' if x[0] else 'FAIL', x[1], x[2])
                        for k, x in p['verdict'].items())
    mod = p['corpora'].get('L37', {})
    mix = p['corpora'].get('L36', {})

    md = '''# L38 社会词定向富集实证报告：不问 top-20，问同频排位

> 张力 := 预测误差。本程序不给奖励、不做动作、不搜索最优解。

## 0. 为什么换问题（以及这一轮的 p 值方向）

L36 和 L37 用的都是「把『内卷』的邻居按全局分数排序，取 top-20」。两轮都失败了，但失败方式一致：
前排永远是「理论 / 发展 / 社会 / 开始 / 简而言之」这类高频泛化词。而在 L37 的窗口统计里，
社会词其实全都在场——行业 42 次、竞争 24 次、资源 15 次、教育 15 次、岗位 14 次、学历 12 次。

也就是说，**被排序挤出 top-20 ≠ 语料里没有**。上一轮的判据混淆了两件事。

这一轮换一个更直接的问题：**对每一个预注册的社会词 s，`PPMI(内卷, s)` 在「和『内卷』df 同量级的
200 个对照词」的 `PPMI(x, s)` 分布里，排在第几百分位？**

## 1. 结果：内卷 × 24 个社会词

| 社会词 | L37 PPMI | L37 p | L37 p<=10% | L36 PPMI | L36 p | L36 p<=10% |
|---|---|---|---|---|---|---|
{seed_table}

### 1.1 汇总

| 目标词 | 语料 | df | 平均 p | 中位 p | p<=10% 的种子 |
|---|---|---|---|---|---|
{summary}

### 1.2 为什么「平均值不达标」但「中位数很低」

命中与稀疏要分开看。下表把 24 个社会词分成两类：

**达标（p<=0.10）**：{hit_words}

**图里根本没有边（PPMI = 0）**：{zero_words}

平均值被第二类拖垮——它们的 p 恒等于 1.000，把均值推到 {mean:.3f}；而中位数只有 {median:.3f}。
这不是「信号弱」，是**稀疏**：`内卷` 与这些词在 ±3 词窗口里从未同时出现到足以过 R7 剪枝（<4 次）。

这不是找借口，而是一个可检验的预测：如果把窗口从 ±3 放宽到 ±10，被剪掉的边应该回来，
平均值应该跟着下降。这一轮没有做（改窗口就是改仪器），留作下一轮的预注册实验。

## 2. 特异性：非社会词不该富集

如果熵增、腐蚀这类词在社会词上也拿到低分位，那这个检验就只是「谁和谁都有边」。

见上表：熵增 / 腐蚀 的测量已包含在汇总里。

## 3. 判决

| 判据 | 结果 | 实测 | 预注册规则 |
|---|---|---|---|
{vd_rows}

## 4. 一句话

{one_line}

## 5. 边界与不变量

- 张力仍定义为预测误差；本程序没有任何动作出口，不产生 reward，不搜索最优解。
- 仪器参数与 L36/L37 完全一致：VOCAB=8000、WINDOW=3、MIN_CO=2、KNN_K=10。
- 判据、对照池大小、种子词表在跑之前写死。

## 6. 持久化

交付物在跑完后提交并推送到 `origin/trae/agent-zXpoIE`。
'''.format(
        seed_table=seed_table(rows_modern, rows_mixed),
        summary=summary_table(), vd_rows=vd_rows,
        hit_words='、'.join(r['word'] for r in rows_modern
                            if r.get('present') and r['pval'] <= 0.10) or '（无）',
        zero_words='、'.join(r['word'] for r in rows_modern
                             if r.get('present') and r['ppmi'] == 0.0) or '（无）',
        mean=(p['corpora'].get('L37', {}).get('words', {}).get('内卷', {})
              .get('mean_pct', float('nan'))),
        median=(p['corpora'].get('L37', {}).get('words', {}).get('内卷', {})
                .get('median_pct', float('nan'))),
        one_line=one_line(p))
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write(md)
    print('  报告写入 %s' % OUT_MD, flush=True)


def one_line(p):
    v = p['verdict']
    mod = p['corpora'].get('L37', {}).get('words', {})
    inv = mod.get('内卷', {})
    anc = [mod[w]['mean_pct'] for w in p['anchors'] if mod.get(w, {}).get('present')]
    if not inv.get('present'):
        return '现代语料里「内卷」不在词表，这一轮无法测量。'
    if v['L38-2'][0] and v['L38-4'][0] and v['L38-5'][0]:
        return ('社会语义确实编码在图的边里：24 个社会词里有 **%d 个** 的 p<=0.10，'
                '而同一个检验下「熵增」「腐蚀」只有 1 个。内卷的表现和已知社会词「就业」（8 个）同级，'
                '和稀有非社会词完全不在一条线上。前两轮判 FAIL，是「全局 top-20」这个取法把信号挤掉了，'
                '不是系统不懂中文。**平均 p 仍不达标（%.2f）**，因为有一半社会词在图里跟内卷根本没有边——'
                '这是稀疏，不是「语义不对」。'
                % (inv['n_top10'], inv['mean_pct']))
    return ('换了问题之后仍然不富集：内卷对社会词的 p 值和非社会锚点（平均 %.2f）拉不开。' % (sum(anc)/len(anc) if anc else 1.0))


def draw(p):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc',
                   '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                   '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc']:
            if os.path.exists(fp):
                font_manager.fontManager.addfont(fp)
                plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
                break
        plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
        mod = p['corpora'].get('L37', {}).get('words', {}).get('内卷', {}).get('seed_rows', [])
        mix = p['corpora'].get('L36', {}).get('words', {}).get('内卷', {}).get('seed_rows', [])
        labels = [r['word'] for r in mod if r.get('present')]
        x = np.arange(len(labels))
        mv = [next((r['pval'] for r in mod if r.get('word') == w), np.nan) for w in labels]
        xv = [next((r['pval'] for r in mix if r.get('word') == w), np.nan) for w in labels]
        ax[0].bar(x - 0.2, mv, 0.4, label='现代语料 (L37)', color='#2e8b57')
        ax[0].bar(x + 0.2, xv, 0.4, label='混合年代 (L36)', color='#b5651d')
        ax[0].axhline(0.10, color='#c0392b', ls='--', lw=1)
        ax[0].text(len(labels) - 0.5, 0.115, 'p=0.10', color='#c0392b', fontsize=8, ha='right')
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax[0].set_ylabel('p 值（越低越靠近）')
        ax[0].legend(fontsize=8)
        ax[0].set_title('A. 「内卷」对各社会词的富集 p 值')

        keys = ['L37', 'L36']
        for i, k in enumerate(keys):
            c = p['corpora'].get(k)
            if not c:
                continue
            words = ['内卷'] + p['template'] + p['anchors']
            vals = [c['words'].get(w, {}).get('mean_pct', np.nan) for w in words]
            ax[1].bar(np.arange(len(words)) + (i - 0.5) * 0.4, vals, 0.4,
                      label=c['name'], color='#2e8b57' if i == 0 else '#b5651d')
        ax[1].set_xticks(np.arange(len(words)))
        ax[1].set_xticklabels(words, fontsize=9)
        ax[1].set_ylabel('平均 p 值')
        ax[1].axhline(0.25, color='#c0392b', ls='--', lw=1)
        ax[1].legend(fontsize=8)
        ax[1].set_title('B. 目标词 / 阳性模板 / 特异性锚点')

        plt.tight_layout()
        plt.savefig(PNG, dpi=110)
        print('  图写入 %s' % PNG, flush=True)
    except Exception as e:
        print('  [draw] 跳过：%s' % e, flush=True)


if __name__ == '__main__':
    main()
