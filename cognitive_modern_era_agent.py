#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L37 —— 时间切片对照：只换语料年代，仪器不动
=============================================
L36 的结论是：「内卷」的邻居没被社会语料拽回来。但 L36 自己也承认，
它的 87 篇「内卷」文档里 85 篇来自同一个来源，14,000 篇 THUCNews 社会新闻里只有 1 篇——
因为 THUCNews 是 2005–2011 年的语料，而「内卷」是 2018 年之后才进入大众语汇的。

于是这一轮的实验设计只有一句话：**把 L36 的仪器原样搬过来，只换语料年代。**

不变式（与 L1–L36 完全一致，不可改）：
  张力 := 预测误差。本程序不产生动作、不给 reward/loss、不搜索最优解。
  现实只读。单出口。内存安全线 700MB。LMAX=3。

预注册（先写死，看到结果不回调）：
  语料：现代 = COIG-CQIA（知乎高分问答 / 小红书 / 豆瓣影评剧评 / SegmentFault）
        + 知乎-KOL 分片 0、1 + 知乎问答 gz；
        对照 = THUCNews 社会新闻（2005–2011），只留 3000 篇当年代基线。
  仪器：VOCAB=8000、WINDOW=3、MIN_CO=2、KNN_K=10、NB_K=100、
        CTX_W=5、N_NULL=2000 —— 与 L36 完全相同，一个都不调。
  判据：与 L36 完全相同（top-20 社会词 >=3 且 top-8 植物词 = 0），两把尺都报。
  阴性对照 / 阳性对照：与 L36 完全相同（df 50–300 随机 60 词；就业/加班/职场/学历）。
"""

import os, sys, json, time, math, sqlite3, hashlib, gc
import numpy as np
from collections import Counter

sys.path.insert(0, '/workspace')
import cognitive_chinese_social_agent as I

SEED = 37
D = '/workspace/.l36_data'
DB = '/workspace/l37_modern_era.sqlite'
OUT_JSON = '/workspace/l37_modern_era_result.json'
OUT_MD = '/workspace/L37_现代语料对照实证报告.md'
PNG = '/workspace/l37_modern_era.png'
CACHE = os.path.join(D, 'cache_l37.pkl')
L36_JSON = '/workspace/l36_chinese_social_result.json'

REUSE = os.environ.get('L37_REUSE', '0') == '1'
MEM_LIMIT_MB = 700

# 仪器参数：与 L36 完全一致，一个都不调
I.VOCAB = 8000
I.N_IDIOM_CAP = 600
I.N_NULL = 2000
I.MEM_LIMIT_MB = MEM_LIMIT_MB
I.MIN_SENSE_N = 25
I.DEG_SENSE = 21

MODERN = ['cqa_zhihu', 'cqa_xhs', 'cqa_douban', 'cqa_segmentfault',
          'zhihu_kol15', 'zhihu_kol16', 'zhihu']
LEGACY = ['thu_social_legacy']
ALL_SRCS = MODERN + LEGACY

CAP = {   # (篇数上限, 字上限) —— 按「入湖」计
    'cqa_zhihu': (8000, 8000000),
    'cqa_xhs': (1500, 1500000),
    'cqa_douban': (250, 1500000),
    'cqa_segmentfault': (460, 1500000),
    'zhihu_kol15': (14000, 14000000),
    'zhihu_kol16': (14000, 14000000),
    'zhihu': (3000, 3000000),
    'thu_social_legacy': (3000, 3000000),
}

TENSION_DEF = ('张力 := 预测误差。L37 不产生动作、不给奖励；只换语料年代，'
               '不换张力定义。恒等不可改。')
SAFETY_BOUND = {'single_exit': True, 'reality_readonly': True, 'no_reward': True,
                'no_physical_action': True, 'mem_limit_mb': MEM_LIMIT_MB, 'lmax': 3}
IMMUTABLE = ('tension_def', 'reality_feed', 'safety_bound')


# ============================ 语料 ============================

def iter_cqa(paths):
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                q = str(o.get('instruction') or '')
                a = str(o.get('output') or '')
                md = str(o.get('metadata') or '')
                yield 'cqa', (q + '。\n' + a + '\n' + md).strip()[:8000]


def ingest(con):
    print('[1/6] 语料摄入（现代 + 年代基线）+ 张力账本', flush=True)
    cur = con.cursor()
    seen = set()
    docs = Counter()
    chars = Counter()
    drop = Counter()
    t0 = time.time()

    def feed(it, src):
        cap_docs, cap_chars = CAP[src]
        d = c = 0
        for label, text in it:
            if d >= cap_docs or c >= cap_chars:
                break
            ok, score, why = I.quality(text)
            if not ok:
                for w in why:
                    drop[(src, w)] += 1
                continue
            h = hashlib.md5((text[:200] + str(len(text))).encode('utf-8')).hexdigest()
            if h in seen:
                drop[(src, 'duplicate')] += 1
                continue
            seen.add(h)
            cur.execute('INSERT INTO l36_doc(src,label,text,nc) VALUES(?,?,?,?)',
                        (src, label, text, len(text)))
            d += 1
            c += len(text)
        con.commit()
        docs[src] = d
        chars[src] = c
        print('  %-18s %5d 篇 / %8d 字' % (src, d, c), flush=True)

    feed(iter_cqa([os.path.join(D, f) for f in [
        'cqa_zhihu_zhihu_score9.0-10_clean_v10.jsonl',
        'cqa_zhihu_zhihu_score8.5-9.0_clean_v4.jsonl',
        'cqa_zhihu_zhihu_score8.0-8.5_clean_v3.jsonl']]), 'cqa_zhihu')
    feed(iter_cqa([os.path.join(D, 'cqa_xhs_xhs.jsonl')]), 'cqa_xhs')
    feed(iter_cqa([os.path.join(D, 'cqa_douban_movie_reviews.jsonl'),
                   os.path.join(D, 'cqa_douban_tv_reviews.jsonl')]), 'cqa_douban')
    feed(iter_cqa([os.path.join(D, 'cqa_segmentfault_segmentfault_upvote5_clean.jsonl')]),
         'cqa_segmentfault')

    def parquet_src(path, src):
        for lab, text in I.iter_parquet(path):
            yield lab, text
    feed(parquet_src(os.path.join(D, 'zhihu_kol_0.parquet'), 'zhihu_kol15'), 'zhihu_kol15')
    feed(parquet_src(os.path.join(D, 'zhihu_kol_1.parquet'), 'zhihu_kol16'), 'zhihu_kol16')
    feed(I.iter_zhihu(os.path.join(D, 'zhihu_qa.jsonl.gz')), 'zhihu')
    feed(I.iter_thuc(os.path.join(D, 'thu_social.jsonl'), {'社会'}), 'thu_social_legacy')

    tot = cur.execute('SELECT COUNT(*), COALESCE(SUM(nc),0) FROM l36_doc').fetchone()
    for (src, reason), n in drop.items():
        cur.execute('INSERT INTO l36_filter(src,reason,n) VALUES(?,?,?)', (src, reason, n))
    con.commit()
    print('  入湖 %d 篇 / %d 字，丢弃 %d 篇，耗时 %.1fs'
          % (tot[0], tot[1], sum(drop.values()), time.time() - t0), flush=True)
    return {'docs': int(tot[0]), 'chars': int(tot[1]),
            'src_docs': dict(docs), 'src_chars': dict(chars),
            'modern_chars': int(sum(chars[s] for s in MODERN)),
            'legacy_chars': int(sum(chars[s] for s in LEGACY)),
            'drop': {'%s / %s' % k: int(v) for k, v in drop.items()}}


def phrase_cov(con, words):
    cur = con.cursor()
    out = {}
    for w in words:
        out[w] = {s: int(n) for s, n in cur.execute(
            'SELECT src, COUNT(*) FROM l36_doc WHERE instr(text, ?) > 0 GROUP BY src', (w,))}
    return out


# ============================ 判决 ============================

def verdict(meta, inv, rnull, cov):
    modern_ok = meta['modern_chars'] >= 30000000
    d1 = '现代语料 %d 字；年代基线 %d 字' % (meta['modern_chars'], meta['legacy_chars'])

    n_inv = sum(cov.get('内卷', {}).values())
    ok2 = n_inv >= 300
    d2 = '「内卷」出现在 %d 篇文档里（L36 是 87 篇）' % n_inv

    def rule(rows):
        if not rows:
            return False, 0, 0
        top8 = [r['word'] for r in rows[:8]]
        hits = [r['word'] for r in rows if r['word'] in I.SOCIAL_SEEDS]
        bot = [w for w in top8 if w in I.BOTANY_WORDS]
        return (len(hits) >= 3 and len(bot) == 0), len(hits), len(bot)

    okp = okc = False
    nhp = nhc = 0
    if inv.get('present'):
        okp, nhp, _ = rule(inv['ppmi']['neighbors'])
        okc, nhc, _ = rule(inv['ctx']['neighbors'])
        d3 = ('尺一 PPMI 社会词 %d 个 → %s；尺二 上下文 社会词 %d 个 → %s'
              % (nhp, '过' if okp else '不过', nhc, '过' if okc else '不过'))
    else:
        d3 = inv.get('why', '内卷不在词表')

    ok5 = bool(rnull and rnull.get('ok') and rnull.get('credible'))
    d5 = ('阴性对照 %d 个通过 %d 个（%.1f%%）'
          % (rnull['n_ctrl'], rnull['n_pass'], rnull['pass_rate'] * 100)) \
        if rnull and rnull.get('ok') else '未测'
    n_pos = rnull.get('n_pos_pass', 0) if rnull else 0
    ok6 = n_pos >= 2
    d6 = ('阳性对照 %d/4 过线（%s）'
          % (n_pos, '、'.join('%s:%d' % (r['word'], r['social_hits'])
                              for r in rnull.get('pos', [])))) if rnull else '未测'

    return {
        'L37-1': [bool(modern_ok), d1, '现代语料（2015 后来源）>=3000 万字'],
        'L37-2': [bool(ok2), d2, '「内卷」文档数 >=300（L36 是 87）'],
        'L37-3': [bool(okc), d3, '尺二同一判据：top-20 社会词 >=3 且 top-8 植物词 = 0'],
        'L37-4': [bool(okp), 'PPMI 社会词 %d 个' % nhp, '尺一同一判据（主口径）'],
        'L37-5': [bool(ok5), d5, '阴性对照通过率 <=10%（尺子不通吃）'],
        'L37-6': [bool(ok6), d6, '阳性对照 >=2/4 过线（尺子够灵敏，L36 只有 1/4）'],
    }


# ============================ 输出 ============================

def save(meta, vocab, dfv, inv, rnull, vd, cov, anchors, extra):
    payload = {'meta': meta, 'vocab_size': len(vocab),
               'params': {'VOCAB': I.VOCAB, 'WINDOW': I.WINDOW, 'MIN_CO': I.MIN_CO,
                          'KNN_K': I.KNN_K, 'NB_K': I.NB_K, 'CTX_W': I.CTX_W,
                          'N_NULL': I.N_NULL},
               'anchors': anchors, 'involution': inv, 'ruler_null': rnull,
               'phrase_cov': cov, 'verdict': vd}
    payload.update(extra)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(I.jsonable(payload), f, ensure_ascii=False, indent=1)
    return payload


def report(payload):
    meta = payload['meta']
    inv = payload['involution']
    rn = payload.get('ruler_null') or {}
    cov = payload.get('phrase_cov', {})
    vd = payload['verdict']
    l36 = payload.get('l36', {})

    srcs = ALL_SRCS
    src_rows = '\n'.join('| %s | %s | %d | %d |'
                         % (s, '现代' if s in MODERN else '年代基线',
                            meta['src_docs'].get(s, 0), meta['src_chars'].get(s, 0))
                         for s in srcs if meta['src_docs'].get(s, 0) > 0)
    cov_rows = '\n'.join('| %s | %s | %d |'
                         % (w, ' | '.join(str(cov.get(w, {}).get(s, 0)) for s in srcs),
                            sum(cov.get(w, {}).values()))
                         for w in ['内卷', '熵增', '打工人', '网暴', '人情世故', '家国情怀'])
    cov_head = '| 短语 | ' + ' | '.join(srcs) + ' | 合计 |'
    cov_sep = '|---' * (len(srcs) + 2) + '|'

    def nb_rows(ruler):
        rows = inv.get(ruler, {}).get('neighbors', []) if inv.get('present') else []
        out = []
        for r in rows[:20]:
            out.append('| %d | %s | %s | %d | %s | %+.3f |'
                       % (r['rank'], r['word'],
                          ('%.3f' % r['score']) if ruler == 'ppmi' else ('%d' % r['score']),
                          r['df'],
                          '、'.join('%s:%d' % (k, x) for k, x in r['srcs'].items()) or '-',
                          r['margin']))
        return '\n'.join(out) or '（内卷不在词表）'

    diag_rows = '\n'.join(
        '| %s | %d | %d | %s | %s |'
        % (r['word'], r['ctx_n'], r['df'],
           ('%.3f' % r['ppmi']) if r['rank'] > 0 else '0（被 R7 剪掉）',
           ('第 %d 名' % r['rank']) if r['rank'] > 0 else '无边')
        for r in inv.get('social_diag', [])[:16]) if inv.get('present') else '（内卷不在词表）'

    cmp_rows = ''
    if l36:
        a = l36.get('ppmi', {}).get('social_hits', [])
        b = l36.get('ctx', {}).get('social_hits', [])
        c = inv.get('ppmi', {}).get('social_hits', []) if inv.get('present') else []
        d = inv.get('ctx', {}).get('social_hits', []) if inv.get('present') else []
        cmp_rows = ('| L36（混合年代） | %d | %s | %d | %s |\n'
                    '| L37（现代为主） | %d | %s | %d | %s |'
                    % (len(a), '、'.join(a) or '-', len(b), '、'.join(b) or '-',
                       len(c), '、'.join(c) or '-', len(d), '、'.join(d) or '-'))

    pos_rows = '\n'.join('| %s | %d | %s | %s |'
                         % (r['word'], r['df'], r['social_hits'],
                            '、'.join(r['hit_words'][:5]) or '-')
                         for r in rn.get('pos', [])) or '| （无） | - | - | - |'
    neg_rows = '\n'.join('| %s | %d | %d | %s |'
                         % (r['word'], r['df'], r['social_hits'],
                            '、'.join(r['hit_words'][:5]) or '-')
                         for r in rn.get('examples', [])) or '| （无） | - | - | - |'

    vd_rows = '\n'.join('| %s | %s | %s | %s |' % (k, 'PASS' if x[0] else 'FAIL', x[1], x[2])
                        for k, x in vd.items())

    md = '''# L37 时间切片对照实证报告：只换语料年代，仪器不动

> 张力 := 预测误差。本程序不给奖励、不做动作、不搜索最优解。
> 这一轮唯一的自变量是**语料年代**；VOCAB / WINDOW / 判据 / 零模型全部沿用 L36，一个都没调。

## 0. 为什么要做这一轮

L36 的判决是「内卷的邻居没被社会语料拽回来」。但 L36 的报告同时留下了一个无法回避的事实：
它的 87 篇「内卷」文档里 85 篇来自同一个来源，14,000 篇 THUCNews 社会新闻里只有 1 篇。
**THUCNews 是 2005–2011 年的新闻，而「内卷」是 2018 年之后才进入大众语汇的词。**

所以 L36 很可能不是在测「内卷有没有社会语义」，而是在测「一个 2018 年才出现的词，
在 2005–2011 年的语料里长什么样」。那一轮无法回答这个问题，这一轮可以：
把语料换成 2023 年的中文（COIG-CQIA 的知乎高分回答 / 小红书 / 豆瓣 / SegmentFault）
加上更多知乎-KOL 分片，然后**把 L36 的仪器原样搬过来**。

## 1. 语料构成（自变量）

| 来源 | 年代 | 篇数 | 字 |
|---|---|---|---|
{src_rows}

现代语料 **{mc}** 字；年代基线 **{lc}** 字。

### 1.1 关键短语的分布

{cov_head}
{cov_sep}
{cov_rows}

## 2. 结果：同一把尺，换语料之后

### 2.1 尺一：PPMI（L36 的主口径）

| 排名 | 邻居 | PPMI | df | 来自哪些语料 | 该词 margin |
|---|---|---|---|---|---|
{nb_ppmi}

### 2.2 尺二：真实上下文共现次数

| 排名 | 邻居 | 共现次数 | df | 来自哪些语料 | 该词 margin |
|---|---|---|---|---|---|
{nb_ctx}

### 2.3 与 L36 并排

| 语料 | 尺一社会词数 | 尺一命中词 | 尺二社会词数 | 尺二命中词 |
|---|---|---|---|---|
{cmp_rows}

### 2.4 「语料里有」和「图里排第几」

| 社会词 | ±60 字共现 | df | PPMI | 排名 |
|---|---|---|---|---|
{diag_rows}

## 3. 对照：尺子本身有没有问题

- 阴性对照（df 50–300 随机 60 词）：通过 **{neg_pass}/{neg_n}**（**{neg_rate}%**）
- 阳性对照（就业/加班/职场/学历）：通过 **{pos_pass}/4**

| 阳性对照 | df | 社会词命中 | 命中了哪些 |
|---|---|---|---|
{pos_rows}

阴性对照里通过的词：

| 词 | df | 社会词命中 | 命中了哪些 |
|---|---|---|---|
{neg_rows}

## 4. 判决

| 判据 | 结果 | 实测 | 预注册规则 |
|---|---|---|---|
{vd_rows}

## 5. 一句话

{one_line}

## 6. 边界与不变量

- 张力仍定义为预测误差；本程序**没有任何动作出口**，不产生 reward，不搜索最优解。
- 神性锚点 `tension_def / reality_feed / safety_bound` 未改；内存峰值受 700MB 红线约束。
- 仪器参数与 L36 完全一致：VOCAB={vocab}、WINDOW={win}、MIN_CO={mco}、KNN_K={knn}、NB_K={nb}。
- 两把尺子、判据、零模型全部沿用 L36 的预注册值，本轮没有为结果调整任何一项。

## 7. 持久化

交付物在跑完后提交并推送到 `origin/trae/agent-zXpoIE`。
'''.format(
        src_rows=src_rows, mc=meta['modern_chars'], lc=meta['legacy_chars'],
        cov_head=cov_head, cov_sep=cov_sep, cov_rows=cov_rows,
        nb_ppmi=nb_rows('ppmi'), nb_ctx=nb_rows('ctx'), cmp_rows=cmp_rows,
        diag_rows=diag_rows,
        neg_pass=rn.get('n_pass', 0), neg_n=rn.get('n_ctrl', 0),
        neg_rate=round(rn.get('pass_rate', 0) * 100, 1), pos_pass=rn.get('n_pos_pass', 0),
        pos_rows=pos_rows, neg_rows=neg_rows, vd_rows=vd_rows,
        one_line=one_line(vd, inv, rn),
        vocab=I.VOCAB, win=I.WINDOW, mco=I.MIN_CO, knn=I.KNN_K, nb=I.NB_K)

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write(md)
    print('  报告写入 %s' % OUT_MD, flush=True)


def one_line(vd, inv, rn):
    if vd['L37-3'][0] and vd['L37-4'][0]:
        return ('换掉语料年代之后，两把尺子同时过线——L36 的 FAIL 不是算法无能，是语料年代不对。'
                '「内卷」在 2023 年的中文里，邻居确实是竞争、加班、资源这一批社会词。')
    if vd['L37-3'][0]:
        return ('换掉语料年代之后，尺二过线、尺一仍然不过线：社会语义进了窗口，但 PPMI 依旧'
                '把它挤给稀有泛化词。这说明两件事要分开说——语料是对的，主尺的偏好是真的。')
    return ('换掉语料年代之后，两把尺子仍然都不过线。那就不是年代问题，是这一整套'
            '「窗口共现 + PPMI」在这一类抽象词上根本测不出社会语义。这句结论比上一轮更硬。')


def draw(payload):
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

        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        meta = payload['meta']
        ss = [s for s in ALL_SRCS if meta['src_docs'].get(s, 0) > 0]
        meta_colors = ['#2e8b57' if s in MODERN else '#b5651d' for s in ss]
        ax[0].bar(range(len(ss)), [meta['src_chars'].get(s, 0) / 10000 for s in ss],
                  color=meta_colors)
        ax[0].set_xticks(range(len(ss)))
        ax[0].set_xticklabels(ss, rotation=30, ha='right', fontsize=8)
        ax[0].set_ylabel('万字')
        ax[0].set_title('A. 语料（绿=现代 / 棕=年代基线）')

        rn = payload.get('ruler_null') or {}
        names = [r['word'] for r in rn.get('pos', [])]
        vals = [r['social_hits'] for r in rn.get('pos', [])]
        if names:
            ax[1].bar(range(len(names)), vals, color='#3b6ea5')
            ax[1].axhline(3, color='#c0392b', ls='--', lw=1)
            ax[1].text(len(names) - 0.5, 3.15, '判据线 3', color='#c0392b', fontsize=8,
                       ha='right')
            ax[1].set_xticks(range(len(names)))
            ax[1].set_xticklabels(names, fontsize=9)
        ax[1].set_title('B. 阳性对照的社会词命中（尺子灵敏度）')
        ax[1].set_ylabel('top-20 里的社会词数')

        inv = payload['involution']
        rows = inv.get('ctx', {}).get('neighbors', [])[:12] if inv.get('present') else []
        if rows:
            cols = ['#2e8b57' if r['word'] in I.SOCIAL_SEEDS
                    else ('#c0392b' if r['word'] in I.BOTANY_WORDS else '#8d8d8d')
                    for r in rows]
            ax[2].barh(range(len(rows)), [r['score'] for r in rows], color=cols)
            ax[2].set_yticks(range(len(rows)))
            ax[2].set_yticklabels([r['word'] for r in rows], fontsize=9)
            ax[2].invert_yaxis()
        ax[2].set_title('C. 「内卷」top-12 邻居（尺二·真实共现）\n绿=社会词 红=植物词')

        plt.tight_layout()
        plt.savefig(PNG, dpi=110)
        print('  图写入 %s' % PNG, flush=True)
    except Exception as e:
        print('  [draw] 跳过：%s' % e, flush=True)


# ============================ 主流程 ============================

def main():
    t0 = time.time()
    print('=' * 72)
    print('L37 时间切片对照 | 只换语料年代，仪器不动 | REUSE=%s' % REUSE)
    print('张力 := 预测误差 | 无 reward | 现实只读 | 单出口 | 内存 < %dMB' % MEM_LIMIT_MB)
    print('=' * 72, flush=True)

    if os.path.exists(DB) and REUSE:
        con = sqlite3.connect(DB)
        have = con.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name='l36_doc'").fetchone()[0]
        if not have:
            I.setup_db(con)
    else:
        if os.path.exists(DB):
            os.remove(DB)
        con = sqlite3.connect(DB)
        I.setup_db(con)
    con.execute('PRAGMA journal_mode=WAL')

    meta = None
    loaded = False
    if REUSE and os.path.exists(CACHE):
        try:
            with open(CACHE, 'rb') as f:
                meta, old_items, dfv_old, flag_of, stats = __import__('pickle').load(f)
            n_doc = con.execute('SELECT COUNT(*) FROM l36_doc').fetchone()[0]
            if n_doc > 0 and stats.get('n_doc') == n_doc:
                print('  [reuse] 载入分词缓存：%d 篇 / %d 词种'
                      % (n_doc, len(old_items)), flush=True)
                loaded = True
        except Exception as e:
            print('  [reuse] 缓存不可用：%s' % e, flush=True)
    if not loaded:
        for t in ['l36_doc', 'l36_filter', 'l36_seq']:
            con.execute('DELETE FROM %s' % t)
        con.commit()
        meta = ingest(con)
        old_items, dfv_old, flag_of, stats = I.pass_a(con)
        try:
            with open(CACHE, 'wb') as f:
                __import__('pickle').dump((meta, old_items, dfv_old, flag_of, stats), f,
                                          protocol=4)
        except Exception as e:
            print('  [cache] 写入失败：%s' % e, flush=True)

    gc.collect()
    I.trim_mem()
    I.mem_guard('passA')

    vocab, dfv, flagv, old2new, idiom_rows, admitted, missing = \
        I.choose_vocab(old_items, dfv_old, flag_of, stats)
    stats['idiom_ctx'] = {}
    gc.collect()
    I.trim_mem()
    I.mem_guard('vocab')

    protect_words = set(I.PROBES) | set(I.HEADS) | set(w for w, _ in admitted)
    protect_words |= set(I.ENERGY_ANCHOR) | set(I.PAIN_ANCHOR)
    P, S, pair_src, protect = I.build_graph(con, vocab, old2new, protect_words,
                                            stats['phrase_ctx'])
    gc.collect()
    I.trim_mem()
    I.mem_guard('graph')

    wid_of = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    E = I.build_embeddings(P)
    Ee, Ep, ei, pi = I.axis_vectors(E, wid_of, V)
    e = np.asarray(E @ Ee).ravel()
    p = np.asarray(E @ Ep).ravel()
    margin = e - p
    anchors = {'energy_words': [vocab[i] for i in ei],
               'pain_words': [vocab[i] for i in pi]}

    inv = I.analyze_involution(vocab, P, wid_of, pair_src, dfv, margin,
                               stats.get('phrase_ctx', {}), stats.get('phrase_ctx_src', {}),
                               src_list=ALL_SRCS)
    rnull = I.ruler_null_test(con, vocab, wid_of, dfv)
    I.mem_guard('ruler_null')

    cov = phrase_cov(con, ['内卷', '熵增', '打工人', '网暴', '人情世故', '家国情怀'])
    vd = verdict(meta, inv, rnull, cov)
    print('\n判决：' + ' | '.join('%s %s' % (k, 'PASS' if x[0] else 'FAIL')
                                for k, x in vd.items()), flush=True)

    l36 = None
    if os.path.exists(L36_JSON):
        try:
            with open(L36_JSON, encoding='utf-8') as f:
                l36 = json.load(f).get('involution')
        except Exception:
            pass
    payload = save(meta, vocab, dfv, inv, rnull, vd, cov, anchors, {'l36': l36})
    report(payload)
    draw(payload)
    con.close()
    print('  完成：%.1fs / 峰值内存约 %.0fMB' % (time.time() - t0, I.mem_mb()), flush=True)


if __name__ == '__main__':
    main()
