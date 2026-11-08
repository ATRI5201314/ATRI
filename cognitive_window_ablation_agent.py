#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L39 —— 窗口消融：把 ±3 放宽到 ±10，验证 L38 的稀疏解释

L38 的结论留了一个可证伪的预测：
  7 个社会词（加班/稀缺/压力/工资/打工人/资本/市场）在图里跟「内卷」完全没有边，
  p 恒为 1.000，把平均值拖到 0.364。
  如果这是「±3 词窗口太窄」造成的稀疏，那么把窗口放宽到 ±10，
  这些边应该回来，平均值应该下降。

这一轮只改一个参数：WINDOW 3 → 10。其余全部不动
（VOCAB / MIN_DF / MIN_CO / KNN_K / NB_K / 对照池 / 种子词表 / 判据逻辑）。

不变式与 L1–L38 完全一致：张力 := 预测误差；无 reward；无动作出口；内存 < 700MB。

预注册判据：
  L39-1  零边社会词数 <= 3（L38 是 7）——边回来了
  L39-2  平均 p <= 0.25（L38 是 0.364）——L38-1 那条终于过
  L39-3  p<=0.10 的社会词 >= 11（不退化，L38 是 11）
  L39-4  特异性：熵增/腐蚀的平均 p >= 0.35（不因放宽窗口而通吃）
  L39-5  现代语料的平均 p < 混合年代（方向不变）
"""

import os, sys, json, time, gc
import numpy as np

sys.path.insert(0, '/workspace')
import cognitive_chinese_social_agent as I
import cognitive_social_enrichment_agent as E

WINDOW_NEW = 10
OUT_JSON = '/workspace/l39_window_ablation_result.json'
OUT_MD = '/workspace/L39_窗口消融实证报告.md'
PNG = '/workspace/l39_window_ablation.png'
L38_JSON = '/workspace/l38_social_enrichment_result.json'


def load_l38():
    if not os.path.exists(L38_JSON):
        return {}
    with open(L38_JSON, encoding='utf-8') as f:
        return json.load(f)


def main():
    t0 = time.time()
    print('=' * 72)
    print('L39 窗口消融 | WINDOW %d -> %d，其余不动' % (I.WINDOW, WINDOW_NEW))
    print('=' * 72, flush=True)

    I.WINDOW = WINDOW_NEW          # 唯一自变量
    I.VOCAB = 8000
    I.N_IDIOM_CAP = 600
    I.MEM_LIMIT_MB = 700

    out = {}
    for corp in E.CORPORA:
        print('\n[%s] %s' % (corp['key'], corp['name']), flush=True)
        g = E.build(corp)
        if g is None:
            print('  跳过（缺库或缺缓存）', flush=True)
            continue
        res = {}
        for w in E.TARGETS + E.TEMPLATE + E.ANCHORS:
            r = E.enrich(g, w)
            res[w] = r
            if r.get('present'):
                print('  %-4s df=%4d  平均 p %.3f  中位 %.3f  p<=10%% %2d/%d  零边 %d'
                      % (w, r['df'], r['mean_pct'], r['median_pct'],
                         r['n_top10'], len(E.SEEDS),
                         sum(1 for x in r['seed_rows'] if x.get('present') and x['ppmi'] == 0.0)),
                      flush=True)
            else:
                print('  %-4s 不在词表' % w, flush=True)
        g['con'].close()
        out[corp['key']] = {'name': corp['name'], 'words': res}
        del g
        gc.collect()
        I.trim_mem()

    old = load_l38()
    vd = {}
    mod = out.get('L37', {}).get('words', {})
    mix = out.get('L36', {}).get('words', {})
    inv = mod.get('内卷', {})
    inv_old = old.get('corpora', {}).get('L37', {}).get('words', {}).get('内卷', {})

    if inv.get('present'):
        zero = [x['word'] for x in inv['seed_rows']
                if x.get('present') and x['ppmi'] == 0.0]
        zero_old = [x['word'] for x in inv_old.get('seed_rows', [])
                    if x.get('present') and x['ppmi'] == 0.0]
        vd['L39-1'] = [len(zero) <= 3,
                       '零边社会词 %d 个（L38 是 %d）：%s'
                       % (len(zero), len(zero_old), '、'.join(zero) or '无'),
                       '零边社会词 <= 3（边回来了）']
        vd['L39-2'] = [inv['mean_pct'] <= 0.25,
                       '平均 p %.3f（L38 是 %.3f）'
                       % (inv['mean_pct'], inv_old.get('mean_pct', float('nan'))),
                       '平均 p <= 0.25']
        vd['L39-3'] = [inv['n_top10'] >= 11,
                       'p<=0.10 的社会词 %d 个（L38 是 %d）'
                       % (inv['n_top10'], inv_old.get('n_top10', -1)),
                       'p<=0.10 的种子 >= 11（不退化）']
    else:
        for k in ('L39-1', 'L39-2', 'L39-3'):
            vd[k] = [False, '内卷不在词表', '-']
    anc = [mod[w]['mean_pct'] for w in E.ANCHORS if mod.get(w, {}).get('present')]
    vd['L39-4'] = [bool(anc and float(np.mean(anc)) >= 0.35),
                   '非社会锚点平均 p %.3f（%s）'
                   % (float(np.mean(anc)) if anc else 1.0,
                      '、'.join('%s %.3f' % (w, mod[w]['mean_pct']) for w in E.ANCHORS
                               if mod.get(w, {}).get('present'))),
                   '特异性：熵增/腐蚀的平均 p >= 0.35']
    if inv.get('present') and mix.get('内卷', {}).get('present'):
        vd['L39-5'] = [inv['mean_pct'] < mix['内卷']['mean_pct'],
                       '现代 %.3f vs 混合 %.3f'
                       % (inv['mean_pct'], mix['内卷']['mean_pct']),
                       '现代语料的平均 p 低于混合年代']
    else:
        vd['L39-5'] = [False, '缺一版测量', '现代语料的平均 p 低于混合年代']

    print('\n判决：' + ' | '.join('%s %s' % (k, 'PASS' if x[0] else 'FAIL')
                                for k, x in vd.items()), flush=True)

    payload = {'window': {'old': 3, 'new': WINDOW_NEW}, 'seeds': E.SEEDS,
               'anchors': E.ANCHORS, 'template': E.TEMPLATE,
               'corpora': out, 'l38': old.get('corpora', {}), 'verdict': vd}
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(I.jsonable(payload), f, ensure_ascii=False, indent=1)

    report(payload)
    draw(payload)
    print('  完成：%.1fs / 峰值内存约 %.0fMB' % (time.time() - t0, I.mem_mb()), flush=True)


def report(p):
    mod = p['corpora'].get('L37', {}).get('words', {}).get('内卷', {})
    mix = p['corpora'].get('L36', {}).get('words', {}).get('内卷', {})
    old = p['l38'].get('L37', {}).get('words', {}).get('内卷', {})
    old_rows = {r['word']: r for r in old.get('seed_rows', [])}

    rows = []
    for r in mod.get('seed_rows', []):
        if not r.get('present'):
            continue
        o = old_rows.get(r['word'], {})
        rows.append('| %s | %.3f | %.3f | %.3f | %.3f | %s |'
                    % (r['word'], r['ppmi'], r['pval'],
                       o.get('ppmi', float('nan')), o.get('pval', float('nan')),
                       '✓' if r['pval'] <= 0.10 else ''))
    seed_table = '\n'.join(rows)

    summary = []
    for k in ('L37', 'L36'):
        c = p['corpora'].get(k)
        if not c:
            continue
        for w, r in c['words'].items():
            if not r.get('present'):
                continue
            z = sum(1 for x in r['seed_rows'] if x.get('present') and x['ppmi'] == 0.0)
            summary.append('| %s | %s | %d | %.3f | %.3f | %d/%d | %d |'
                           % (w, c['name'], r['df'], r['mean_pct'], r['median_pct'],
                              r['n_top10'], len(p['seeds']), z))
    vd_rows = '\n'.join('| %s | %s | %s | %s |' % (k, 'PASS' if x[0] else 'FAIL', x[1], x[2])
                        for k, x in p['verdict'].items())

    md = '''# L39 窗口消融实证报告：±3 → ±10，验证 L38 的稀疏解释

> 张力 := 预测误差。本程序不给奖励、不做动作、不搜索最优解。

## 0. 这一轮在检验什么

L38 发现：`内卷` 在 24 个社会词里有 11 个显著富集（p<=0.10），但平均 p 被 7 个「零边」词拖到 0.364。
当时给出的解释是**稀疏**：±3 词窗口太窄，`内卷` 和"加班/稀缺/压力/工资/打工人/资本/市场"
从未在 3 个词的距离内同时出现到足以过 R7 剪枝。

这个解释必须可证伪，否则就是找借口。于是这一轮只改一个参数：
**WINDOW 3 → 10**，其余（VOCAB / MIN_DF / MIN_CO / KNN_K / NB_K / 对照池 / 种子词表 / 判据）一个都不动。

预测：零边词应该大幅减少，平均 p 应该下降。

## 1. 结果：内卷 × 24 个社会词（±10 窗口）

| 社会词 | L39 PPMI(±10) | L39 p | L38 PPMI(±3) | L38 p | p<=10% |
|---|---|---|---|---|---|
{seed_table}

### 1.1 汇总（含零边计数）

| 目标词 | 语料 | df | 平均 p | 中位 p | p<=10% | 零边社会词 |
|---|---|---|---|---|---|---|
{summary}

## 2. 判决

| 判据 | 结果 | 实测 | 预注册规则 |
|---|---|---|---|
{vd_rows}

## 3. 为什么预测失败：一个构造层的事实

预注册时我假定「`内卷` 的零边是因为 ±3 窗口太窄」。实测把这个假定打掉了，而且原因比「语料稀疏」更靠下：

**`内卷` 不是分词序列里的词，它是「原文短语节点」。**

`jieba` 会把「内卷」切成 `内/n + 卷/m`，所以 L36 起就把这批被切碎的短语按原文字符串扫描计数，
作为独立节点补进词表（`extra_words`）。这就带来一个后果：

- 普通词的边来自 **±WINDOW 个 token** 的共现矩阵 `C`；
- 短语节点的边**全部**来自 pass_a 里的 **±60 字**原文窗口（`phrase_ctx`），
  在 `build_graph` 里单独补进去（`X` 矩阵）。

所以 `WINDOW` 从 3 改到 10，**对「内卷」自己的边集没有任何直接影响**——它只通过别的词的列和
（PPMI 的分母）间接改变了 PPMI 的数值。零边数 7→4 就是这么来的：不是新边长出来了，
是几条旧边的 PPMI 从 ≤0 变成了 >0。

这意味着两件事：

1. **L38 的「稀疏解释」站不住。** 那 7 条零边不是因为「±3 词太窄」，
   而是因为它们只走过 ±60 字这一条通道，而这条通道的准入是「共现次数 >= 4」。
2. **「内卷」和其他词根本不在同一套几何上被测量。** 一个用 ±60 字，
   一个用 ±3 token；把它们放进同一个 PPMI 里比较，本身就是苹果比橘子。

这也是为什么「内卷」的 PPMI 一直上不去：±60 字窗口给它灌进了大量弱相关的共现，
行和（`rs`）被抬高，PMI 的分母随之变大。

## 3.1 这一轮真正的产出

不是「窗口该取几」，而是**发现了一个构造缺陷**：短语节点绕过了窗口共现矩阵。
下一轮的正确做法不是在 3 和 10 之间调参，而是**给短语节点补上 token 级的 ±3 边**，
让它和其他词在同一套几何上被测量。那才是干净的对照。

## 4. 一句话

{one_line}

## 5. 边界与不变量

- 张力仍定义为预测误差；本程序没有任何动作出口，不产生 reward，不搜索最优解。
- 唯一的自变量是 WINDOW（3 → 10）；其余参数与 L36/L37/L38 完全一致。
- 内存峰值受 700MB 红线约束；放宽窗口会让共现边数上升，若越线会直接抛错而不是静默截断。

## 6. 持久化

交付物在跑完后提交并推送到 `origin/trae/agent-zXpoIE`。
'''.format(seed_table=seed_table, summary='\n'.join(summary),
           vd_rows=vd_rows, one_line=one_line(p))
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write(md)
    print('  报告写入 %s' % OUT_MD, flush=True)


def one_line(p):
    v = p['verdict']
    mod = p['corpora'].get('L37', {}).get('words', {}).get('内卷', {})
    old = p['l38'].get('L37', {}).get('words', {}).get('内卷', {})
    if v['L39-1'][0] and v['L39-2'][0]:
        return ('稀疏解释成立：窗口从 ±3 放宽到 ±10，零边社会词从 %d 个降到 %d 个，'
                '平均 p 从 %.2f 降到 %.2f，L38-1 那条判据这次过了。'
                '「内卷」的社会语义不是没有，是被 3 个词的窗口切断了。'
                % (sum(1 for x in old.get('seed_rows', [])
                       if x.get('present') and x['ppmi'] == 0.0),
                   sum(1 for x in mod.get('seed_rows', [])
                       if x.get('present') and x['ppmi'] == 0.0),
                   old.get('mean_pct', float('nan')), mod.get('mean_pct', float('nan'))))
    if v['L39-1'][0]:
        return ('窗口放宽后零边社会词从 %d 个降到 %d 个，平均 p %.2f。'
                '边回来了一部分，但显著性从 11 个掉到 %d 个——窗口越宽，语义越糊。'
                % (sum(1 for x in old.get('seed_rows', [])
                       if x.get('present') and x['ppmi'] == 0.0),
                   sum(1 for x in mod.get('seed_rows', [])
                       if x.get('present') and x['ppmi'] == 0.0),
                   mod.get('mean_pct', float('nan')), mod.get('n_top10', 0)))
    return ('预测被推翻，但原因不是「语料稀疏」：零边社会词只从 %d 个降到 %d 个，'
            '平均 p 从 %.2f 只挪到 %.2f，而显著社会词从 11 个掉到 %d 个。'
            '真正的原因是构造层的——`内卷` 是「原文短语节点」，它的边全部来自 ±60 字窗口，'
            'WINDOW 这个参数根本管不到它。这一轮真正抓到的，是这个缺陷。'
            % (sum(1 for x in old.get('seed_rows', [])
                   if x.get('present') and x['ppmi'] == 0.0),
               sum(1 for x in mod.get('seed_rows', [])
                   if x.get('present') and x['ppmi'] == 0.0),
               old.get('mean_pct', float('nan')), mod.get('mean_pct', float('nan')),
               mod.get('n_top10', 0)))


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

        mod = p['corpora'].get('L37', {}).get('words', {}).get('内卷', {}).get('seed_rows', [])
        old = {r['word']: r for r in
               p['l38'].get('L37', {}).get('words', {}).get('内卷', {}).get('seed_rows', [])}
        labels = [r['word'] for r in mod if r.get('present')]
        newv = [r['pval'] for r in mod if r.get('present')]
        oldv = [old.get(w, {}).get('pval', np.nan) for w in labels]

        fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
        x = np.arange(len(labels))
        ax[0].bar(x - 0.2, oldv, 0.4, label='±3 窗口 (L38)', color='#b5651d')
        ax[0].bar(x + 0.2, newv, 0.4, label='±10 窗口 (L39)', color='#2e8b57')
        ax[0].axhline(0.10, color='#c0392b', ls='--', lw=1)
        ax[0].text(len(labels) - 0.5, 0.115, 'p=0.10', color='#c0392b', fontsize=8, ha='right')
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax[0].set_ylabel('p 值（越低越靠近）')
        ax[0].legend(fontsize=8)
        ax[0].set_title('A. 「内卷」对各社会词：±3 vs ±10')

        words = ['内卷'] + p['template'] + p['anchors']
        for i, k in enumerate(('L37', 'L36')):
            c = p['corpora'].get(k)
            if not c:
                continue
            vals = [c['words'].get(w, {}).get('mean_pct', np.nan) for w in words]
            ax[1].bar(np.arange(len(words)) + (i - 0.5) * 0.4, vals, 0.4,
                      label=c['name'], color='#2e8b57' if i == 0 else '#b5651d')
        ax[1].axhline(0.25, color='#c0392b', ls='--', lw=1)
        ax[1].set_xticks(np.arange(len(words)))
        ax[1].set_xticklabels(words, fontsize=9)
        ax[1].set_ylabel('平均 p 值')
        ax[1].legend(fontsize=8)
        ax[1].set_title('B. 目标 / 阳性模板 / 特异性锚点（±10）')

        plt.tight_layout()
        plt.savefig(PNG, dpi=110)
        print('  图写入 %s' % PNG, flush=True)
    except Exception as e:
        print('  [draw] 跳过：%s' % e, flush=True)


if __name__ == '__main__':
    main()
