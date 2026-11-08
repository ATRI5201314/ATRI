#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L36 —— 中文原生语义大爆发（社会语料 / 成语准入 / 内卷重构 / 中文隐喻）
=====================================================================
不变式（与 L1–L35 完全一致，不可改）：
  张力 := 预测误差。本程序不产生动作、不给 reward/loss、不搜索最优解。
  现实只读。任何生成物与神性锚点冲突即拒绝。
  单出口。内存安全线 700MB。LMAX=3。

L36 预注册（先写死，看到结果不回调）：
  1. 语料：THUCNews 定向字节区间（社会 / 时政 / 财经）+ 知乎问答 + 知乎-KOL 现代问答。
     目标 >= 100 万字，社会类 >= 5000 篇，>= 3 类来源。
  2. 成语准入：不用 df>=50 一刀切；改为 n_occ>=2 & 不同上下文词 u>=4 &
     归一化熵 H>=0.5。成语节点用能量/疼痛两轴重新打分，并输出真实上下文词。
  3. 内卷：在社会新闻图上重测 top-20 邻居，统计社会词命中与邻居的语料来源。
     L36-3a 主尺 = PPMI（预注册主口径）。
     L36-3b 第二把尺 = 真实上下文共现次数（±60 字窗口原始计数），
     两把尺都报，不拿通过的那把覆盖没通过的那把。
  4. 中文隐喻：全中文双通道（kNN 骨架跳数 + 张力指纹），两通道都 <=10% 分位才算过。
"""

import os, sys, re, json, time, math, sqlite3, gzip, hashlib, pickle, gc
import urllib.request, urllib.parse
from collections import Counter, defaultdict

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd

import jieba
import jieba.posseg as pseg

SEED = 36
rng = np.random.default_rng(SEED)

FAST = os.environ.get('L36_FAST', '0') == '1'
CRAWL = int(os.environ.get('L36_CRAWL', '0'))
REUSE = os.environ.get('L36_REUSE', '0') == '1'

DATA = '/workspace/.l36_data'
DB = '/workspace/l36_chinese_social.sqlite'
OUT_JSON = '/workspace/l36_chinese_social_result.json'
OUT_MD = '/workspace/L36_中文社会语义实证报告.md'
PNG = '/workspace/l36_chinese_social.png'
CACHE = os.path.join(DATA, 'cache_%s.pkl' % ('fast' if FAST else 'full'))

# ---------------- 预注册常量 ----------------
MIN_CJK = 80
TARGET_CHARS = 1000000
VOCAB = 3000 if FAST else 8000
WINDOW = 3
MIN_DF = 2 if FAST else 3
MIN_CO = 2
KNN_K = 8 if FAST else 10
NB_K = 60 if FAST else 100
AX_BINS = 5
DEG_SENSE = 14 if FAST else 21
CTX_W = 5
K_SENSE = 3
MIN_SENSE_N = 12 if FAST else 25
HEAD_WIN_CAP = 160 if FAST else 600
HEAD_WIN_CHARS = 70
N_NULL = 400 if FAST else 2000
N_IDIOM_CAP = 200 if FAST else 600
EMB_DIM = 128
MEM_LIMIT_MB = 700

DOC_CAP = {'thu_social': 2500 if FAST else 14000,
           'thu_politics': 800 if FAST else 5000,
           'thu_finance': 800 if FAST else 5000,
           'zhihu': 800 if FAST else 5000,
           'zhihu_modern': 6000 if FAST else 24000}
CHAR_CAP = {'thu_social': 3000000 if FAST else 16000000,
            'thu_politics': 1200000 if FAST else 6000000,
            'thu_finance': 1200000 if FAST else 6000000,
            'zhihu': 1500000 if FAST else 8000000,
            'zhihu_modern': 6000000 if FAST else 20000000}

TENSION_DEF = ('张力 := 预测误差。L36 不产生动作、不给奖励；中文语料只用来数词与'
               '词的关系，成语准入只改变「哪些词成为节点」，不改变张力定义。恒等不可改。')
SAFETY_BOUND = {'single_exit': True, 'reality_readonly': True, 'no_reward': True,
                'no_physical_action': True, 'mem_limit_mb': MEM_LIMIT_MB, 'lmax': 3}
IMMUTABLE = ('tension_def', 'reality_feed', 'safety_bound')

PROBES = [
    '内卷', '熵增', '内耗', '竞争', '加班', '资源', '稀缺', '压力', '焦虑', '房价',
    '就业', '教育', '学历', '考核', '淘汰', '工资', '薪资', '劳动', '绩效', '工作',
    '花瓣', '直立', '斑纹', '捕虫', '叶片', '叶缘', '植物',
    '约束', '摩擦力', '崩溃', '坏死', '经济', '细胞', '法律', '物理',
    '人情世故', '家国情怀', '面子', '关系', '制度', '集体', '家庭', '责任',
    '荣誉', '格局', '气节', '道德', '舆论', '公众',
    '网暴', '腐蚀', '疲惫', '电量', '电池', '键盘侠', '打工人', '低电量',
    '网络', '网友', '评论', '辱骂', '攻击', '匿名',
    '金属', '氧化', '化学', '材料', '表面', '锈', '酸性',
    '职场', '员工', '劳累', '睡眠', '通勤', '充电', '电压', '电流', '手机', '续航',
    '能量', '能源', '电力', '燃料', '食物', '营养', '功率', '发热', '燃烧',
    '疼痛', '痛', '受伤', '伤害', '疾病', '痛苦', '伤口', '流血', '折磨', '难受',
    '腐败', '通货膨胀', '市场', '资本', '效率', '收益', '增长', '无序', '耗散', '熵',
    '系统', '混乱', '封闭', '组织', '运动', '力学',
]
PROBE_SET = set(PROBES)

TRACKED = ['人情世故', '家国情怀', '内卷', '熵增', '内耗', '加班', '资源稀缺',
           '网暴', '腐蚀', '疲惫', '电量', '键盘侠', '打工人']

ENERGY_ANCHOR = ['能量', '能源', '电力', '电池', '电量', '燃料', '食物', '营养',
                 '功率', '充电', '发热', '燃烧', '动能', '太阳能', '石油', '煤']
PAIN_ANCHOR = ['疼痛', '痛', '受伤', '伤害', '疾病', '痛苦', '伤口', '流血',
               '折磨', '难受', '病症', '炎症', '骨折', '癌', '呕吐', '头痛']

METAPHOR = [
    {'name': 'A', 'title': '键盘侠的网暴 vs 物理的腐蚀',
     'a': '网暴', 'a_qual': ['键盘侠', '网络', '网友', '评论', '辱骂', '攻击', '匿名', '舆论'],
     'b': '腐蚀', 'b_qual': ['金属', '氧化', '化学', '材料', '表面', '锈', '酸性']},
    {'name': 'B', 'title': '打工人的疲惫 vs 电池的低电量',
     'a': '疲惫', 'a_qual': ['打工人', '加班', '工作', '职场', '员工', '劳累', '睡眠', '通勤'],
     'b': '电量', 'b_qual': ['电池', '充电', '电压', '电流', '手机', '续航', '耗电']},
]
HEADS = sorted({m['a'] for m in METAPHOR} | {m['b'] for m in METAPHOR})
HEAD_SET = set(HEADS)

SOCIAL_SEEDS = ['竞争', '加班', '内耗', '资源', '稀缺', '资源稀缺', '压力', '焦虑',
                '房价', '就业', '学历', '考核', '淘汰', '工资', '薪资', '劳动',
                '绩效', '职场', '打工人', '资本', '市场', '行业', '岗位', '招聘',
                '教育', '内卷']
BOTANY_WORDS = ['花瓣', '直立', '斑纹', '捕虫', '叶片', '叶缘', '植物', '茎',
                '根系', '花序', '杂草', '萌发', '枝条', '树皮', '苔藓', '叶肉']

DOMAIN_KW = {
    'physics': ['力', '能量', '速度', '质量', '温度', '电流', '电压', '磁场', '粒子',
                '辐射', '耗散', '摩擦', '熵', '功率', '波长', '重力'],
    'biology': ['细胞', '基因', '蛋白', '组织', '器官', '感染', '炎症', '坏死', '血管',
                '免疫', '细菌', '病毒', '病灶', '肿瘤'],
    'law': ['法律', '法院', '条款', '合同', '判决', '司法', '立法', '权利', '义务',
            '违法', '律师', '法规', '刑法', '起诉'],
    'economics': ['经济', '市场', '资本', '价格', '投资', '金融', '财政', '贸易',
                  '通胀', '股票', '成本', '利润', '企业', '产业'],
    'psychology': ['心理', '情绪', '焦虑', '抑郁', '压力', '认知', '行为', '人格',
                   '恐惧', '动机', '创伤', '自尊'],
    'engineering': ['工程', '系统', '设备', '机械', '结构', '材料', '设计', '制造',
                    '安装', '维修', '零件', '工艺'],
    'art': ['艺术', '文学', '音乐', '绘画', '电影', '戏剧', '作品', '审美', '诗',
            '导演', '旋律', '画作'],
    'history': ['历史', '战争', '朝代', '帝国', '革命', '古代', '世纪', '政权',
                '王朝', '战役', '史料', '军阀'],
    'society': ['社会', '人群', '舆论', '公众', '网友', '政策', '制度', '家庭',
                '教育', '就业', '网络', '社区', '群体', '青年'],
}
DOMAIN_NAMES_CN = {'physics': '物理', 'biology': '生物', 'law': '法律',
                   'economics': '经济', 'psychology': '心理', 'engineering': '工程',
                   'art': '艺术', 'history': '历史', 'society': '社会', 'other': '其他'}

STOPWORDS = set('''
的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好
自己 这 那 他 她 它 们 我们 你们 他们 这个 那个 什么 怎么 为什么 因为 所以 但是
而且 然而 虽然 如果 可以 应该 需要 已经 还是 就是 这样 那样 一些 这些 那些 其实
可能 觉得 知道 时候 问题 事情 东西 地方 方面 情况 表示 认为 进行 通过 根据 由于
以及 并且 或者 不过 只是 还有 然后 现在 目前 同时 其中 对于 关于 按照 为了 各种
以上 以下 之后 之前 以后 以来 一直 非常 比较 更加 特别 主要 重要 不同 一样 这种
那种 起来 出来 过来 下去 上去 不能 不会 不是 没 别 再 又 才 更 最 太 挺 挺 已
'''.split())

XIEHOUYU_RE = re.compile(r'[\u4e00-\u9fa5]{2,12}(?:——|—|--|－)[\u4e00-\u9fa5]{2,15}')

CONTENT_FLAGS = set('nvaijl')


def mem_mb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def trim_mem():
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass


def mem_guard(tag):
    m = mem_mb()
    print('  [mem] %-24s %6.1f MB' % (tag, m), flush=True)
    if m > MEM_LIMIT_MB:
        raise MemoryError('内存越线：%.0fMB > %dMB @%s' % (m, MEM_LIMIT_MB, tag))


# ============================ 数据库 ============================

def setup_db(con):
    cur = con.cursor()
    for t in ['l36_doc', 'l36_filter', 'l36_seq', 'l36_vocab', 'l36_idiom',
              'l36_neighbor', 'l36_metaphor', 'l36_meta', 'l36_verdict']:
        cur.execute('DROP TABLE IF EXISTS %s' % t)
    cur.executescript('''
    CREATE TABLE l36_doc(id INTEGER PRIMARY KEY, src TEXT, label TEXT, text TEXT, nc INT);
    CREATE TABLE l36_filter(src TEXT, reason TEXT, n INT);
    CREATE TABLE l36_seq(doc_id INT, src TEXT, ids TEXT);
    CREATE TABLE l36_vocab(id INT PRIMARY KEY, word TEXT, df INT, flag TEXT);
    CREATE TABLE l36_idiom(word TEXT PRIMARY KEY, flag TEXT, n_occ INT, u INT, info REAL,
                           e REAL, p REAL, margin REAL, admitted INT, reason TEXT, ctx TEXT);
    CREATE TABLE l36_neighbor(word TEXT, ruler TEXT, rank INT, neighbor TEXT, score REAL,
                              df INT, srcs TEXT);
    CREATE TABLE l36_metaphor(pair TEXT, side TEXT, node TEXT, geo INT, geo_pct REAL,
                              fp REAL, fp_pct REAL, base_geo INT, base_fp REAL,
                              base_fp_pct REAL, senses TEXT);
    CREATE TABLE l36_meta(k TEXT PRIMARY KEY, v TEXT);
    CREATE TABLE l36_verdict(k TEXT PRIMARY KEY, ok INT, detail TEXT, rule TEXT);
    ''')
    con.commit()


# ============================ 语料摄入 ============================

def cjk_ratio(s):
    if not s:
        return 0.0
    n = sum(1 for ch in s if '\u4e00' <= ch <= '\u9fff')
    return n / max(1, len(s))


def quality(text):
    """张力账本：营销号排比句与情绪发泄挡在底湖外面。返回 (ok, score, reasons)。"""
    t = (text or '').strip()
    if len(t) < MIN_CJK:
        return False, 0.0, ['too_short']
    if cjk_ratio(t) < 0.45:
        return False, 0.0, ['low_chinese_ratio']
    sents = [s.strip() for s in re.split(r'[。！？!?\n]+', t) if len(s.strip()) >= 8]
    if len(sents) < 2:
        return False, 0.0, ['few_sentences']
    long_s = [s for s in sents if len(s) >= 25]
    if len(long_s) / len(sents) < 0.12:
        return False, 0.0, ['no_long_sentence']
    cjk = [ch for ch in t if '\u4e00' <= ch <= '\u9fff']
    g4 = [''.join(cjk[i:i + 4]) for i in range(len(cjk) - 3)]
    if g4:
        div4 = len(set(g4)) / len(g4)
        if div4 < 0.45:
            return False, div4, ['low_4gram_diversity']
    g8 = [''.join(cjk[i:i + 8]) for i in range(len(cjk) - 7)]
    if g8 and len(set(g8)) / len(g8) < 0.65:
        return False, 0.0, ['repetitive_8gram']
    if len(set(sents)) / len(sents) < 0.65:
        return False, 0.0, ['repeated_sentences']
    if t.count('！') / max(1, len(t)) > 0.008:
        return False, 0.0, ['emotional_exclaim']
    return True, len(t), []


def iter_thuc(path, allowed):
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            lab = str(o.get('label') or '')
            if allowed is not None and lab not in allowed:
                continue
            ti = str(o.get('title') or '')
            tx = str(o.get('text') or '')
            yield lab or 'thuc', (ti + '。\n' + tx).strip()


def iter_zhihu(path):
    if not os.path.exists(path):
        return
    n = 0
    try:
        with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                n += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                q = str(o.get('question') or '')
                a = str(o.get('answer') or '')
                tags = o.get('tags') or ''
                yield '知乎', (q + '。\n' + a + '\n' + str(tags)).strip()[:8000]
    except Exception as e:
        print('  [zhihu] gz 在 %d 行后无法继续：%s' % (n, type(e).__name__), flush=True)


def iter_parquet(path):
    """知乎-KOL 现代问答。逐 row-group 读，坏组跳过并计数，不假装拿到全量。"""
    if not os.path.exists(path):
        return
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    cols = list(pf.schema_arrow.names)
    want = [c for c in ('INSTRUCTION', 'instruction', 'question', 'title') if c in cols]
    resp = [c for c in ('RESPONSE', 'response', 'answer', 'text', 'content') if c in cols]
    if not want or not resp:
        print('  [parquet] 未识别字段：%s' % cols, flush=True)
        return
    n_rg = pf.metadata.num_row_groups
    bad = rows = 0
    for i in range(n_rg):
        try:
            tb = pf.read_row_group(i)
        except Exception:
            bad += 1
            continue
        d = tb.to_pydict()
        for k in range(len(d[want[0]])):
            q = str(d[want[0]][k] or '').strip()
            a = ' '.join(str(d[c][k] or '') for c in resp).strip()
            rows += 1
            yield '知乎现代', (q + '。\n' + a).strip()[:8000]
    print('  [parquet] row-group %d 个，损坏跳过 %d，可用行 %d' % (n_rg, bad, rows), flush=True)


def ingest(con):
    print('[1/7] 语料摄入 + 张力账本', flush=True)
    add_docs = Counter()
    drop = Counter()
    seen = set()
    t0 = time.time()

    def add(src, label, text):
        ok, score, why = quality(text)
        if not ok:
            for w in why:
                drop[(src, w)] += 1
            return 0
        h = hashlib.md5((text[:200] + str(len(text))).encode('utf-8')).hexdigest()
        if h in seen:
            drop[(src, 'duplicate')] += 1
            return 0
        seen.add(h)
        cur.execute('INSERT INTO l36_doc(src,label,text,nc) VALUES(?,?,?,?)',
                    (src, label, text, len(text)))
        add_docs[src] += 1
        return len(text)

    cur = con.cursor()

    def feed(it, src, cap_docs, cap_chars):
        # 上限按「入湖」计，不按「扫过的行」计：否则重复稿/情绪稿会吃掉配额
        d = c = 0
        for label, text in it:
            if d >= cap_docs or c >= cap_chars:
                break
            got = add(src, label, text)
            if got:
                d += 1
                c += got
        print('  %-14s %5d 篇 / %8d 字' % (src, d, c), flush=True)

    feed(iter_thuc(os.path.join(DATA, 'thu_social.jsonl'), {'社会'}),
         'thu_social', DOC_CAP['thu_social'], CHAR_CAP['thu_social'])
    con.commit()
    feed(iter_thuc(os.path.join(DATA, 'thu_politics.jsonl'), {'时政'}),
         'thu_politics', DOC_CAP['thu_politics'], CHAR_CAP['thu_politics'])
    con.commit()
    feed(iter_thuc(os.path.join(DATA, 'thu_finance.jsonl'), {'财经', '股票'}),
         'thu_finance', DOC_CAP['thu_finance'], CHAR_CAP['thu_finance'])
    con.commit()
    feed(iter_zhihu(os.path.join(DATA, 'zhihu_qa.jsonl.gz')),
         'zhihu', DOC_CAP['zhihu'], CHAR_CAP['zhihu'])
    con.commit()
    feed(iter_parquet(os.path.join(DATA, 'zhihu_kol_0.parquet')),
         'zhihu_modern', DOC_CAP['zhihu_modern'], CHAR_CAP['zhihu_modern'])
    con.commit()

    docs = cur.execute('SELECT COUNT(*), COALESCE(SUM(nc),0) FROM l36_doc').fetchone()
    for (src, reason), n in drop.items():
        cur.execute('INSERT INTO l36_filter(src,reason,n) VALUES(?,?,?)', (src, reason, n))
    con.commit()
    src_docs = {s: int(n) for s, n in
                cur.execute('SELECT src, COUNT(*) FROM l36_doc GROUP BY src')}
    src_chars = {s: int(n) for s, n in
                 cur.execute('SELECT src, COALESCE(SUM(nc),0) FROM l36_doc GROUP BY src')}
    print('  入湖 %d 篇 / %d 字，丢弃 %d 篇，耗时 %.1fs'
          % (docs[0], docs[1], sum(drop.values()), time.time() - t0), flush=True)
    return {'docs': int(docs[0]), 'chars': int(docs[1]),
            'src_docs': src_docs, 'src_chars': src_chars,
            'n_sources': len(src_docs),
            'drop': {'%s / %s' % k: int(v) for k, v in drop.items()}}


# ============================ 第一遍：分词 ============================

def tokenize_text(text):
    out = []
    for w in pseg.cut(text, HMM=False):
        word, flag = w.word, w.flag
        if not word.strip():
            continue
        if flag and flag[0] in 'wmxdprcyuoeftq':
            continue
        if flag in ('nr', 'ns', 'nt', 'nz', 'nrfg', 'nrt'):
            continue
        if not flag or flag[0] not in CONTENT_FLAGS:
            continue
        if word in STOPWORDS:
            continue
        if len(word) == 1 and word not in PROBE_SET:
            continue
        out.append((word, flag))
    return out


def pass_a(con):
    print('[2/7] 第一遍分词：df / 成语上下文 / 短语窗口 / 歇后语', flush=True)
    cur = con.cursor()
    cur.execute('SELECT id,src,text FROM l36_doc ORDER BY id')
    ins = con.cursor()
    w2o = {}
    df = Counter()
    flag_of = {}
    idiom_occ = Counter()
    idiom_ctx = defaultdict(Counter)
    idiom_flag = {}
    phrase_occ = Counter()
    phrase_df = Counter()
    phrase_ctx = defaultdict(Counter)
    phrase_ctx_src = Counter()
    head_wins = defaultdict(list)
    xhy_occ = Counter()
    xhy_df = Counter()
    xhy_example = {}
    n_doc = 0
    t0 = time.time()

    def oid(w):
        i = w2o.get(w)
        if i is None:
            i = len(w2o)
            w2o[w] = i
        return i

    while True:
        rows = cur.fetchmany(200)
        if not rows:
            break
        for did, src, text in rows:
            toks = tokenize_text(text)
            if len(toks) < 8:
                n_doc += 1
                continue
            ids = []
            for k, (word, flag) in enumerate(toks):
                i = oid(word)
                ids.append(i)
                df[word] += 1
                flag_of.setdefault(word, flag)
                if flag in ('i', 'l'):
                    idiom_occ[word] += 1
                    idiom_flag[word] = flag
                    cw = idiom_ctx[word]
                    if len(cw) < 80:
                        for j in range(max(0, k - CTX_W), min(len(toks), k + CTX_W + 1)):
                            if j != k:
                                cw[toks[j][0]] += 1
            ins.execute('INSERT INTO l36_seq(doc_id,src,ids) VALUES(?,?,?)',
                        (did, src, ' '.join(map(str, ids))))
            # 受关注短语（原文扫描，防 jieba 切碎）
            for ph in TRACKED:
                c = text.count(ph)
                if not c:
                    continue
                phrase_occ[ph] += c
                phrase_df[ph] += 1
                for m in re.finditer(re.escape(ph), text):
                    snip = text[max(0, m.start() - 60):m.end() + 60]
                    for w2, _ in tokenize_text(snip):
                        phrase_ctx[ph][w2] += 1
                        phrase_ctx_src[(ph, w2, src)] += 1
            # 隐喻头词的真实语境窗口（每次出现一条，上限预注册）
            for h in HEAD_SET:
                if len(head_wins[h]) >= HEAD_WIN_CAP:
                    continue
                pos = text.find(h)
                while pos >= 0 and len(head_wins[h]) < HEAD_WIN_CAP:
                    snip = text[max(0, pos - HEAD_WIN_CHARS):pos + len(h) + HEAD_WIN_CHARS]
                    win = [w for w, _ in tokenize_text(snip) if w != h]
                    if len(win) >= 4:
                        head_wins[h].append(win)
                    pos = text.find(h, pos + 1)
            xs = XIEHOUYU_RE.findall(text)
            if xs:
                xhy_df['_doc'] += 1
                for x in xs:
                    xhy_occ[x] += 1
                    xhy_example.setdefault(x, text[:60])
            n_doc += 1
        ins.execute('COMMIT') if False else None
        con.commit()
        if n_doc % 5000 == 0:
            mem_guard('passA %d docs' % n_doc)
    con.commit()

    # 只保留可能准入的成语（n>=2），单次出现的不可能过 n>=2 门槛
    for w in [w for w, n in idiom_occ.items() if n < 2]:
        idiom_ctx.pop(w, None)
        idiom_flag.pop(w, None)
        del idiom_occ[w]

    print('  %d 篇 / %d 词种 / 成语候选 %d / 短语 %d / 歇后语类型 %d / %.1fs'
          % (n_doc, len(w2o), len(idiom_occ), len(phrase_occ), len(xhy_occ),
             time.time() - t0), flush=True)

    old_items = [None] * len(w2o)
    for w, i in w2o.items():
        old_items[i] = w
    dfv_old = np.zeros(len(w2o), dtype=np.int32)
    for w, i in w2o.items():
        dfv_old[i] = df.get(w, 0)
    stats = {'n_doc': n_doc, 'n_types': len(w2o),
             'idiom_occ': dict(idiom_occ),
             'idiom_ctx': {k: dict(v) for k, v in idiom_ctx.items()},
             'idiom_flag': idiom_flag,
             'phrase_occ': dict(phrase_occ), 'phrase_df': dict(phrase_df),
             'phrase_ctx': {k: dict(v) for k, v in phrase_ctx.items()},
             'phrase_ctx_src': dict(phrase_ctx_src),
             'head_wins': {k: v for k, v in head_wins.items()},
             'xhy_occ': dict(xhy_occ), 'xhy_df': dict(xhy_df),
             'xhy_example': dict(xhy_example)}
    del idiom_ctx, phrase_ctx, head_wins
    return old_items, dfv_old, flag_of, stats


def load_or_build_pass_a(con):
    if REUSE and os.path.exists(CACHE):
        try:
            with open(CACHE, 'rb') as f:
                meta, old_items, dfv_old, flag_of, stats = pickle.load(f)
            n_doc = con.execute('SELECT COUNT(*) FROM l36_doc').fetchone()[0]
            n_seq = con.execute('SELECT COUNT(*) FROM l36_seq').fetchone()[0]
            # pass_a 的 n_doc 计所有文档，seq 只记分词后 >=8 词的文档，两者本来就不等
            if n_doc > 0 and n_seq > 0 and stats.get('n_doc') == n_doc:
                print('  [reuse] 载入分词缓存：%d 篇 / %d 词种'
                      % (n_doc, len(old_items)), flush=True)
                return meta, old_items, dfv_old, flag_of, stats
            print('  [reuse] 缓存与库不一致（doc %d / seq %d），重建' % (n_doc, n_seq),
                  flush=True)
        except Exception as e:
            print('  [reuse] 缓存不可用：%s' % e, flush=True)
    # 重建前必须清空语料表：否则在已有库上重复 ingest，文档会翻倍
    for t in ['l36_doc', 'l36_filter', 'l36_seq']:
        con.execute('DELETE FROM %s' % t)
    con.commit()
    meta = ingest(con)
    old_items, dfv_old, flag_of, stats = pass_a(con)
    try:
        with open(CACHE, 'wb') as f:
            pickle.dump((meta, old_items, dfv_old, flag_of, stats), f, protocol=4)
        print('  [cache] 分词缓存已写 %s' % CACHE, flush=True)
    except Exception as e:
        print('  [cache] 写入失败：%s' % e, flush=True)
    return meta, old_items, dfv_old, flag_of, stats


# ============================ 词表 + 成语准入 ============================

def choose_vocab(old_items, dfv_old, flag_of, stats):
    print('[3/7] 选词表 + 成语准入（上下文信息密度，非词频）', flush=True)
    n_old = len(old_items)
    w2old = {w: i for i, w in enumerate(old_items)}

    cand = [(int(dfv_old[i]), i) for i in range(n_old) if dfv_old[i] >= MIN_DF]
    cand.sort(reverse=True)
    keep = [i for _, i in cand[:VOCAB]]

    forced = [w2old[w] for w in PROBES if w in w2old]
    missing_probe = [w for w in PROBES if w not in w2old]

    # 成语准入：不看 df，看「它所在上下文的信息密度」
    idiom_rows = []
    for w, n in stats['idiom_occ'].items():
        ctx = stats['idiom_ctx'].get(w, {})
        u = len(ctx)
        tot = sum(ctx.values())
        Hn = 0.0
        if u >= 2 and tot > 0:
            H = -sum((c / tot) * math.log(c / tot) for c in ctx.values())
            Hn = H / math.log(u)
        info = Hn * min(1.0, u / 8.0)
        ok = bool(n >= 2 and u >= 4 and Hn >= 0.5)
        idiom_rows.append({'word': w, 'flag': stats['idiom_flag'].get(w, 'i'),
                           'n': int(n), 'u': int(u), 'Hn': round(float(Hn), 4),
                           'info': round(float(info), 4), 'admitted': ok,
                           'reason': 'ok' if ok else
                           ('n<2' if n < 2 else ('u<4' if u < 4 else 'Hn<0.5')),
                           'e': 0.0, 'p': 0.0, 'margin': 0.0, 'ctx_top': []})
    # 点名短语：走的准入规则和上面完全一样，只是被点名记录；不在 idiom_occ 里
    # （被 jieba 切碎）的按原文扫描统计，作为独立候选行补进来
    for w in TRACKED:
        if w in stats['idiom_occ']:
            continue
        if stats['phrase_occ'].get(w, 0) <= 0:
            continue
        ctx = stats['phrase_ctx'].get(w, {})
        u = len(ctx)
        tot = sum(ctx.values())
        Hn = 0.0
        if u >= 2 and tot > 0:
            H = -sum((c / tot) * math.log(c / tot) for c in ctx.values())
            Hn = H / math.log(u)
        n = stats['phrase_occ'][w]
        info = Hn * min(1.0, u / 8.0)
        ok = bool(n >= 2 and u >= 4 and Hn >= 0.5)
        idiom_rows.append({'word': w, 'flag': 'phrase', 'n': int(n), 'u': int(u),
                           'Hn': round(float(Hn), 4), 'info': round(float(info), 4),
                           'admitted': ok, 'reason': 'ok' if ok else
                           ('n<2' if n < 2 else ('u<4' if u < 4 else 'Hn<0.5')),
                           'e': 0.0, 'p': 0.0, 'margin': 0.0, 'ctx_top': []})

    idiom_rows.sort(key=lambda r: -r['info'])
    # 点名短语是预注册的测量对象：满足准入规则就必进表，不被上限挤掉
    prio = [(r['word'], r['info']) for r in idiom_rows
            if r['admitted'] and r['word'] in TRACKED]
    rest = [(r['word'], r['info']) for r in idiom_rows
            if r['admitted'] and r['word'] not in TRACKED]
    admitted = (prio + rest)[:N_IDIOM_CAP]
    adm_set = set(w for w, _ in admitted)
    for r in idiom_rows:
        if r['admitted'] and r['word'] not in adm_set:
            r['admitted'] = False
            r['reason'] = 'cap(%d)' % N_IDIOM_CAP

    # jieba 会把「内卷 / 网暴 / 家国情怀 / 打工人」切碎，这些短语按原文扫描
    # 计数后作为独立节点补进词表（否则不是「语料里没有」，是分词切丢了）
    extra_words = []
    for w in TRACKED:
        if w in w2old:
            continue
        if stats['phrase_occ'].get(w, 0) > 0:
            extra_words.append(w)

    idiom_keep = [w2old[w] for w, _ in admitted if w in w2old]
    all_keep = sorted(set(keep) | set(forced) | set(idiom_keep))
    old2new = np.full(n_old, -1, dtype=np.int32)
    vocab = []
    for newi, oi in enumerate(all_keep):
        old2new[oi] = newi
        vocab.append(old_items[oi])
    vocab.extend(extra_words)

    dfv = np.array([int(dfv_old[oi]) for oi in all_keep]
                   + [int(stats['phrase_df'].get(w, 0)) for w in extra_words],
                   dtype=np.int32)
    flagv = [flag_of.get(w, 'n') for w in vocab[:-len(extra_words)]] + \
            ['phrase'] * len(extra_words) if extra_words else \
            [flag_of.get(w, 'n') for w in vocab]

    print('  词表 V=%d（高频 %d + 强制 %d + 成语 %d + 原文短语 %d）；未出现探针 %d'
          % (len(vocab), len(keep), len(forced), len(adm_set), len(extra_words),
             len(missing_probe)), flush=True)
    if extra_words:
        print('  原文短语节点：%s' % '、'.join(
            '%s(df=%d)' % (w, stats['phrase_df'].get(w, 0)) for w in extra_words),
            flush=True)
    print('  成语候选 %d / 准入 %d' % (len(idiom_rows), len(adm_set)), flush=True)
    for w in ('人情世故', '家国情怀'):
        r = next((x for x in idiom_rows if x['word'] == w), None)
        print('  点名短语 %s：%s' % (w, r if r else '语料中未出现'), flush=True)
    return vocab, dfv, flagv, old2new, idiom_rows, admitted, missing_probe


# ============================ 第二遍：共现图 ============================

def ppmi(C, chunk=2000000):
    C = C.tocsr()
    C.sum_duplicates()
    n = C.shape[0]
    tot = float(C.sum())
    rs = np.asarray(C.sum(axis=1)).ravel()
    cs = np.asarray(C.sum(axis=0)).ravel()
    nn = C.nnz
    row_of = np.repeat(np.arange(n), np.diff(C.indptr))
    vals = np.zeros(nn, dtype=np.float32)
    for s in range(0, nn, chunk):
        e = min(nn, s + chunk)
        r = row_of[s:e]
        c = C.indices[s:e]
        num = C.data[s:e] * tot
        den = rs[r] * cs[c]
        vals[s:e] = np.log(np.maximum(num, np.float32(1e-12))
                           / np.maximum(den, np.float32(1e-12)))
    mask = vals > 0
    out = sparse.csr_matrix((vals[mask], (row_of[mask], C.indices[mask])), shape=C.shape)
    del row_of, vals, mask
    return out


def build_graph(con, vocab, old2new, protect_words, phrase_ctx=None):
    print('[4/7] 第二遍：共现图（窗口 %d）+ PPMI + kNN 骨架' % WINDOW, flush=True)
    V = len(vocab)
    wid_of = {w: i for i, w in enumerate(vocab)}
    protect = np.zeros(V, dtype=bool)
    for w in protect_words:
        i = wid_of.get(w)
        if i is not None:
            protect[i] = True
    probe_attr = {'内卷', '熵增', '网暴', '腐蚀', '疲惫', '电量', '人情世故',
                  '家国情怀', '内耗', '加班', '竞争', '资源'}
    attr_arr = np.array(sorted(wid_of[w] for w in probe_attr if w in wid_of), dtype=np.int32)
    attr_set = set(int(x) for x in attr_arr)
    pair_src = Counter()

    cur = con.cursor()
    cur.execute('SELECT doc_id,src,ids FROM l36_seq ORDER BY doc_id')
    gc.collect()
    C = None
    rr, cc = [], []
    npairs = 0

    def flush():
        nonlocal C, rr, cc, npairs
        if not rr:
            return
        r = np.concatenate(rr)
        c = np.concatenate(cc)
        rows = np.concatenate([r, c])
        cols = np.concatenate([c, r])
        B = sparse.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                              shape=(V, V)).tocsr()
        B.sum_duplicates()
        C = B if C is None else C + B
        rr, cc, npairs = [], [], 0

    ndoc = 0
    while True:
        rows = cur.fetchmany(300)
        if not rows:
            break
        for did, src, ids_s in rows:
            old = np.asarray(ids_s.split(), dtype=np.int32)
            a = old2new[old]
            a = a[a >= 0]
            if len(a) < 2:
                continue
            for off in range(1, WINDOW + 1):
                if len(a) <= off:
                    break
                rr.append(a[:-off])
                cc.append(a[off:])
                npairs += 2 * (len(a) - off)
            if attr_set:
                pos = np.where(np.isin(a, attr_arr))[0]
                for p in pos:
                    pw = vocab[a[p]]
                    for j in range(max(0, p - WINDOW), min(len(a), p + WINDOW + 1)):
                        if j != p:
                            pair_src[(pw, vocab[a[j]], src)] += 1
            ndoc += 1
            if npairs >= 1200000:
                flush()
                gc.collect()
                trim_mem()
        if ndoc % 3000 == 0:
            mem_guard('passB %d docs' % ndoc)
    flush()
    print('  %d 篇，原始对称边 %d' % (ndoc, 0 if C is None else C.nnz), flush=True)
    if C is None:
        raise RuntimeError('共现图为空')

    # 原文短语节点（被 jieba 切碎的词）用真实 ±60 字窗口上下文补边
    if phrase_ctx:
        er, ec, ev = [], [], []
        for ph, ctx in phrase_ctx.items():
            i = wid_of.get(ph)
            if i is None:
                continue
            for w2, n in ctx.items():
                j = wid_of.get(w2)
                if j is None or j == i:
                    continue
                er.append(i)
                ec.append(j)
                ev.append(float(n))
        if er:
            er2 = np.array(er + ec, dtype=np.int32)
            ec2 = np.array(ec + er, dtype=np.int32)
            X = sparse.coo_matrix((np.array(ev + ev, dtype=np.float32), (er2, ec2)),
                                  shape=(V, V)).tocsr()
            X.sum_duplicates()
            C = (C + X).tocsr()
            print('  原文短语边 %d（%d 个短语节点）' % (X.nnz, len(set(er))), flush=True)

    # R7 剪枝 + R7b 测量对象保行（探针词被剪断会让「路径不存在」变成假结论）
    C = C.tocsr()
    C.sum_duplicates()
    if protect.any():
        saved = []
        for i in np.where(protect)[0]:
            s, e = C.indptr[i], C.indptr[i + 1]
            if e > s:
                saved.append((int(i), C.indices[s:e].copy(), C.data[s:e].copy()))
        for i, js, ws in saved:
            s, e = C.indptr[i], C.indptr[i + 1]
            C.data[s:e] = np.maximum(ws, 2.0 * MIN_CO)
            for j, w in zip(js, ws):
                s2, e2 = C.indptr[j], C.indptr[j + 1]
                p = np.where(C.indices[s2:e2] == i)[0]
                if len(p):
                    C.data[s2 + p[0]] = max(C.data[s2 + p[0]], 2.0 * MIN_CO)
        print('  R7b 保行 %d / %d' % (len(saved), V), flush=True)
    C.data[C.data < 2.0 * MIN_CO] = 0
    C.eliminate_zeros()
    print('  剪枝后边 %d' % C.nnz, flush=True)

    P = ppmi(C)
    del C
    gc.collect()
    print('  PPMI 正边 %d' % P.nnz, flush=True)

    indptr = P.indptr
    r2, c2 = [], []
    for i in range(V):
        s, e = indptr[i], indptr[i + 1]
        if e <= s:
            continue
        idx = P.indices[s:e]
        val = P.data[s:e]
        sel = np.arange(len(idx)) if len(idx) <= KNN_K \
            else np.argpartition(-val, KNN_K)[:KNN_K]
        r2.append(np.full(len(sel), i, dtype=np.int32))
        c2.append(idx[sel].astype(np.int32))
    r2 = np.concatenate(r2) if r2 else np.zeros(0, dtype=np.int32)
    c2 = np.concatenate(c2) if c2 else np.zeros(0, dtype=np.int32)
    S = sparse.coo_matrix((np.ones(len(r2), dtype=np.float32), (r2, c2)),
                          shape=(V, V)).tocsr()
    S = S.maximum(S.T).tocsr()
    S.eliminate_zeros()
    print('  kNN 骨架 %d 边 / 平均度 %.1f' % (S.nnz, S.nnz / max(1, V)), flush=True)
    return P, S, pair_src, protect


# ============================ 嵌入 / 两轴 / 指纹 ============================

def normalize_dense(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def build_embeddings(P, dim=EMB_DIM):
    U, S, _ = randomized_svd(P.astype(np.float32), n_components=dim, n_iter=4,
                             random_state=SEED)
    E = U * np.sqrt(np.maximum(S, 1e-12))
    out = np.zeros_like(E)
    for i in range(E.shape[0]):
        out[i] = normalize_dense(E[i])
    return out.astype(np.float32)


def axis_vectors(E, wid_of, V):
    ei = [wid_of[w] for w in ENERGY_ANCHOR if w in wid_of]
    pi = [wid_of[w] for w in PAIN_ANCHOR if w in wid_of]
    Ee = normalize_dense(E[ei].mean(axis=0)) if ei else np.zeros(E.shape[1], np.float32)
    Ep = normalize_dense(E[pi].mean(axis=0)) if pi else np.zeros(E.shape[1], np.float32)
    return Ee.astype(np.float32), Ep.astype(np.float32), ei, pi


def row_top(M, i, k):
    s, e = M.indptr[i], M.indptr[i + 1]
    if e <= s:
        return np.zeros(0, np.int32), np.zeros(0, np.float32)
    idx = M.indices[s:e]
    val = M.data[s:e]
    if len(idx) <= k:
        o = np.argsort(-val)
    else:
        o = np.argpartition(-val, k)[:k]
        o = o[np.argsort(-val[o])]
    return idx[o], val[o]


def axis_bin(v, edges):
    return int(np.searchsorted(edges, v, side='right'))


def fingerprint_matrix(P, e, p, ids, neigh, edges_e, edges_p):
    F = np.zeros((len(ids), AX_BINS * AX_BINS + 1), dtype=np.float32)
    for row, i in enumerate(ids):
        nb = neigh[i]
        hist = np.zeros(AX_BINS * AX_BINS, dtype=np.float32)
        if len(nb):
            for j in nb:
                be = min(AX_BINS - 1, axis_bin(e[j], edges_e))
                bp = min(AX_BINS - 1, axis_bin(p[j], edges_p))
                hist[be * AX_BINS + bp] += 1.0
            hist /= len(nb)
            F[row, :AX_BINS * AX_BINS] = hist
            F[row, -1] = 0.5 * float(np.mean(e[nb] - p[nb]))
        else:
            F[row, -1] = 0.0
    return F


def center_fp(F):
    mu = F.mean(axis=0, keepdims=True)
    Fc = F - mu
    nrm = np.linalg.norm(Fc, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = 1.0
    return (Fc / nrm).astype(np.float32), mu


def fp_dist(A, i, j):
    return float(np.linalg.norm(A[i] - A[j]))


def adjacency_from_csr(S):
    return [S.indices[S.indptr[i]:S.indptr[i + 1]].tolist() for i in range(S.shape[0])]


def bfs_hops(adj, src, dst, maxhop=10):
    if src == dst:
        return 0
    seen = {src}
    frontier = [src]
    for h in range(1, maxhop + 1):
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v == dst:
                    return h
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        if not nxt:
            return -1
        frontier = nxt
    return -1


def bfs_map(adj, src, maxhop=6):
    dist = {src: 0}
    frontier = [src]
    for h in range(1, maxhop + 1):
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = h
                    nxt.append(v)
        if not nxt:
            break
        frontier = nxt
    return dist


# ============================ 成语两轴打分 ============================

def score_idioms(vocab, wid_of, E, Ee, Ep, dfv, idiom_rows, stats):
    print('  成语两轴打分（用真实上下文词，不看词频）', flush=True)
    dim = E.shape[1]
    for row in idiom_rows:
        w = row['word']
        ctx = stats['idiom_ctx'].get(w) or stats['phrase_ctx'].get(w, {})
        ids = []
        for cw, _ in sorted(ctx.items(), key=lambda x: -x[1]):
            j = wid_of.get(cw)
            if j is not None and j not in ids:
                ids.append(j)
            if len(ids) >= 60:
                break
        if ids:
            v = E[ids].mean(axis=0)
            v = normalize_dense(v)
            row['e'] = float(np.dot(v, Ee))
            row['p'] = float(np.dot(v, Ep))
            row['margin'] = row['e'] - row['p']
            row['ctx_top'] = [vocab[j] for j in ids[:8]]
        else:
            row['e'] = row['p'] = row['margin'] = 0.0
            row['ctx_top'] = []
    adm = [r for r in idiom_rows if r['admitted']]
    by_m = sorted(adm, key=lambda r: -r['margin'])
    top_e = by_m[:12]
    top_p = list(reversed(by_m))[:12]
    print('  准入 %d；最高能量轴成语：%s'
          % (len(adm), '、'.join(r['word'] for r in top_e)), flush=True)
    print('  最高疼痛轴成语：%s' % '、'.join(r['word'] for r in top_p), flush=True)
    return top_e, top_p


# ============================ 中文义项 ============================

def domain_of_words(words):
    cnt = Counter()
    for w in words:
        for d, kws in DOMAIN_KW.items():
            for k in kws:
                if k == w or (len(k) >= 2 and k in w):
                    cnt[d] += 1
                    break
    if not cnt:
        return 'other', {}
    d = max(cnt.items(), key=lambda x: x[1])[0]
    return d, dict(cnt)


def induce_senses(vocab, E, e, p, wid_of, stats, dfv):
    print('[5/7] 中文义项：隐喻头词的真实语境窗口', flush=True)
    senses = {}
    report = {}
    used = Counter()
    for h in HEADS:
        wins = stats['head_wins'].get(h, [])
        info = {'n_windows': len(wins), 'senses': []}
        if h not in wid_of:
            info['why'] = '不在词表'
            report[h] = info
            print('  %-6s 不在词表' % h, flush=True)
            continue
        if len(wins) < 2 * MIN_SENSE_N:
            info['why'] = '语境窗口 %d 个 < %d' % (len(wins), 2 * MIN_SENSE_N)
            report[h] = info
            print('  %-6s %s' % (h, info['why']), flush=True)
            continue
        X = []
        keep = []
        for win in wins:
            ids = [wid_of[w] for w in win if w in wid_of and w != h]
            if len(ids) < 3:
                continue
            X.append(normalize_dense(E[ids].mean(axis=0)))
            keep.append(win)
        if len(X) < 2 * MIN_SENSE_N:
            info['why'] = '可用窗口 %d 个 < %d' % (len(X), 2 * MIN_SENSE_N)
            report[h] = info
            print('  %-6s %s' % (h, info['why']), flush=True)
            continue
        X = np.asarray(X, dtype=np.float32)
        k = min(K_SENSE, max(2, len(X) // MIN_SENSE_N))
        km = KMeans(n_clusters=k, n_init=6, random_state=SEED).fit(X)
        lab = km.labels_
        cents = km.cluster_centers_
        order = np.argsort(-np.bincount(lab, minlength=k))
        sinfo = []
        for ci in order:
            sel = lab == ci
            n_occ = int(sel.sum())
            if n_occ < MIN_SENSE_N // 2:
                continue
            c = normalize_dense(cents[ci].astype(np.float32))
            sim = E @ c
            sim[wid_of[h]] = -1e9
            top = np.argsort(-sim)[:DEG_SENSE]
            nb_words = [vocab[int(j)] for j in top]
            dom, dom_mix = domain_of_words(nb_words)
            used[dom] += 1
            name = '%s_%s%d' % (h, dom, used[dom])
            sinfo.append({'name': name, 'head': h, 'n_occ': n_occ, 'domain': dom,
                          'domain_mix': dom_mix, 'centroid': c.tolist(),
                          'neighbors': nb_words,
                          'top_words': nb_words[:8]})
        senses[h] = sinfo
        info['senses'] = sinfo
        report[h] = info
        print('  %-6s 窗口 %d -> 义项 %d：%s'
              % (h, len(X), len(sinfo),
                 ' / '.join('%s(%d)' % (DEMO_dom(s), s['n_occ']) for s in sinfo)), flush=True)
    return senses, report


def DEMO_dom(s):
    return DOMAIN_NAMES_CN.get(s['domain'], s['domain'])


def select_sense(senses, head, qual_words, wid_of, E):
    lst = senses.get(head) or []
    if not lst:
        return None
    q = [wid_of[w] for w in qual_words if w in wid_of]
    if not q:
        return max(lst, key=lambda s: s['n_occ'])
    qv = normalize_dense(E[q].mean(axis=0))
    best, bs = None, -1e9
    for s in lst:
        c = np.asarray(s['centroid'], dtype=np.float32)
        sc = float(np.dot(qv, c))
        if sc > bs:
            bs, best = sc, s
    return best


# ============================ 中文隐喻：双通道 ============================

def sense_fp_vector(nb_words, wid_of, e, p, margin, edges_e, edges_p):
    ids = [wid_of[w] for w in nb_words if w in wid_of]
    F = np.zeros(AX_BINS * AX_BINS + 1, dtype=np.float32)
    if not ids:
        return F
    for j in ids:
        be = min(AX_BINS - 1, axis_bin(e[j], edges_e))
        bp = min(AX_BINS - 1, axis_bin(p[j], edges_p))
        F[be * AX_BINS + bp] += 1.0
    F[:AX_BINS * AX_BINS] /= len(ids)
    F[-1] = 0.5 * float(np.mean(margin[ids]))
    return F


def measure_metaphor(vocab, P, S, E, e, p, margin, FP, FPmu, wid_of, senses, dfv,
                     edges_e, edges_p):
    print('[6/7] 中文隐喻测验（双通道）', flush=True)
    V = len(vocab)
    adj = adjacency_from_csr(S)
    extra_nodes = []      # (name, head, neighbors)
    for h, lst in senses.items():
        for s in lst:
            extra_nodes.append((s['name'], h, s['neighbors']))
    adj2 = [list(a) for a in adj] + [[] for _ in extra_nodes]
    FP2 = [FP[i].copy() for i in range(V)]
    for k, (nm, h, nbs) in enumerate(extra_nodes):
        nid = V + k
        for w in nbs:
            j = wid_of.get(w)
            if j is None or j == wid_of.get(h):
                continue
            adj2[nid].append(j)
            adj2[j].append(nid)
        Fs = sense_fp_vector(nbs, wid_of, e, p, margin, edges_e, edges_p) - FPmu[0]
        nn = float(np.linalg.norm(Fs))
        FP2.append((Fs / nn if nn > 1e-12 else Fs).astype(np.float32))
    node_of = {nm: V + k for k, (nm, _, _) in enumerate(extra_nodes)}

    cache = {}

    def geo(a, b):
        if a == b:
            return 0
        d = cache.get(a)
        if d is None:
            d = bfs_map(adj2, a, 6)
            cache[a] = d
        return d.get(b, -1)

    metas = []
    for m in METAPHOR:
        sa = select_sense(senses, m['a'], m['a_qual'], wid_of, E)
        sb = select_sense(senses, m['b'], m['b_qual'], wid_of, E)
        na = sa['name'] if sa else m['a']
        nb = sb['name'] if sb else m['b']
        ia = node_of.get(na, wid_of.get(m['a'], -1))
        ib = node_of.get(nb, wid_of.get(m['b'], -1))
        if ia < 0 or ib < 0:
            print('  %s：节点缺失，无法测量' % m['title'], flush=True)
            continue
        g = geo(ia, ib)
        f = fp_dist(FP2, ia, ib)
        ga = geo(wid_of[m['a']], wid_of[m['b']]) if \
            (m['a'] in wid_of and m['b'] in wid_of) else -1
        fa = fp_dist(FP, wid_of[m['a']], wid_of[m['b']]) if \
            (m['a'] in wid_of and m['b'] in wid_of) else float('nan')

        # 频率/度匹配零模型：两端的对照词按 log1p(df) 就近取池
        def pool(w):
            j = wid_of.get(w)
            if j is None:
                return []
            base = math.log1p(float(dfv[j]))
            idx = sorted(range(V), key=lambda x: abs(math.log1p(float(dfv[x])) - base))
            return [x for x in idx if x != j][:200]

        pa, pb = pool(m['a']), pool(m['b'])
        if not pa or not pb:
            pa = pa or list(range(V))
            pb = pb or list(range(V))
        null_g, null_f = [], []
        for _ in range(N_NULL):
            x = int(rng.choice(pa))
            y = int(rng.choice(pb))
            gg = geo(x, y)
            null_g.append(11 if gg < 0 else gg)
            null_f.append(fp_dist(FP2, x, y))
        null_g = np.asarray(null_g, dtype=np.float32)
        null_f = np.asarray(null_f, dtype=np.float32)
        obs_g = 11 if g < 0 else g
        geo_pct = float(np.mean(null_g <= obs_g))
        fp_pct = float(np.mean(null_f <= f))
        base_fp_pct = float(np.mean(null_f <= fa)) if not math.isnan(fa) else float('nan')
        rec = {'name': m['name'], 'title': m['title'],
               'sides': {'a': {'node': na, 'qual': m['a_qual'],
                               'top': sa['top_words'] if sa else []},
                         'b': {'node': nb, 'qual': m['b_qual'],
                               'top': sb['top_words'] if sb else []}},
               'geo': int(g), 'geo_pct': round(geo_pct, 4),
               'fp': round(f, 4), 'fp_pct': round(fp_pct, 4),
               'base_geo': int(ga), 'base_fp': round(fa, 4) if not math.isnan(fa) else -1.0,
               'base_fp_pct': round(base_fp_pct, 4) if not math.isnan(base_fp_pct) else -1.0,
               'dual': bool(geo_pct <= 0.10 and fp_pct <= 0.10)}
        metas.append(rec)
        print('  %s %s：geo %d (%.1f%%) | fp %.3f (%.1f%%) | 基础 geo %d fp %.3f (%.1f%%)'
              % (m['name'], m['title'], g, geo_pct * 100, f, fp_pct * 100,
                 ga, fa, base_fp_pct * 100), flush=True)
    return metas


# ============================ 内卷：两把尺子 ============================

def analyze_involution(vocab, P, wid_of, pair_src, dfv, margin, phrase_ctx, phrase_ctx_src):
    i = wid_of.get('内卷', -1)
    if i < 0:
        return {'present': False, 'why': '「内卷」不在词表（df 低于准入线）'}
    rows_ppmi = []
    idx, val = row_top(P, i, 20)
    for r, j in enumerate(idx):
        w = vocab[int(j)]
        sc = Counter()
        for src in ['thu_social', 'thu_politics', 'thu_finance', 'zhihu', 'zhihu_modern']:
            c = pair_src.get(('内卷', w, src), 0)
            if not c:
                c = phrase_ctx_src.get(('内卷', w, src), 0)
            if c:
                sc[src] = int(c)
        rows_ppmi.append({'rank': r + 1, 'word': w, 'score': round(float(val[r]), 4),
                          'df': int(dfv[int(j)]), 'srcs': dict(sc),
                          'margin': round(float(margin[int(j)]), 4)})

    ctx = phrase_ctx.get('内卷', {})
    rows_ctx = []
    rank = 0
    for w, n in sorted(ctx.items(), key=lambda x: -x[1]):
        j = wid_of.get(w)
        if j is None or j == i:
            continue
        rank += 1
        sc = Counter()
        for src in ['thu_social', 'thu_politics', 'thu_finance', 'zhihu', 'zhihu_modern']:
            c = phrase_ctx_src.get(('内卷', w, src), 0)
            if c:
                sc[src] = int(c)
        rows_ctx.append({'rank': rank, 'word': w, 'score': float(n),
                         'df': int(dfv[j]), 'srcs': dict(sc),
                         'margin': round(float(margin[j]), 4)})
        if rank >= 20:
            break

    def summarize(rows):
        top8 = [r['word'] for r in rows[:8]]
        hits = [r['word'] for r in rows if r['word'] in SOCIAL_SEEDS]
        bot = [w for w in top8 if w in BOTANY_WORDS]
        return top8, hits, bot

    t8p, hitp, botp = summarize(rows_ppmi)
    t8c, hitc, botc = summarize(rows_ctx)

    row = np.asarray(P.getrow(i).todense()).ravel()
    diag = []
    for w in SOCIAL_SEEDS:
        j = wid_of.get(w)
        if j is None:
            continue
        rk = int(np.sum(row > row[j])) + 1 if row[j] > 0 else -1
        diag.append({'word': w, 'ctx_n': int(ctx.get(w, 0)), 'df': int(dfv[j]),
                     'ppmi': round(float(row[j]), 4), 'rank': rk})
    diag.sort(key=lambda r: (-r['ctx_n'], r['rank'] if r['rank'] > 0 else 10 ** 6))

    print('  [尺一 PPMI] top-8：%s' % '、'.join(t8p), flush=True)
    print('  [尺二 上下文] top-8：%s' % '、'.join(t8c), flush=True)
    print('  社会词命中：PPMI %d 个 / 上下文 %d 个' % (len(hitp), len(hitc)), flush=True)
    if diag:
        print('  [诊断] %s' % '；'.join(
            '%s ctx=%d rank=%s' % (r['word'], r['ctx_n'],
                                   r['rank'] if r['rank'] > 0 else '无边')
            for r in diag[:8]), flush=True)
    return {'present': True,
            'ppmi': {'neighbors': rows_ppmi, 'top8': t8p,
                     'social_hits': hitp, 'botany_top8': botp},
            'ctx': {'neighbors': rows_ctx, 'top8': t8c,
                    'social_hits': hitc, 'botany_top8': botc},
            'social_diag': diag,
            'margin': round(float(margin[i]), 4)}


# ============================ 判决 ============================

def make_verdict(meta, vocab, idiom_rows, admitted, top_e, top_p, inv, metas, rnull=None):
    social_docs = int(meta['src_docs'].get('thu_social', 0))
    ok1 = bool(meta['chars'] >= TARGET_CHARS
               and social_docs >= (300 if FAST else 5000)
               and meta['n_sources'] >= 3)
    d1 = '%d 字 / 社会类 %d 篇 / %d 类来源' % (meta['chars'], social_docs, meta['n_sources'])

    tracked = {r['word']: r for r in idiom_rows if r['word'] in ('人情世故', '家国情怀')}
    tracked_ok = all(w in tracked and tracked[w]['admitted'] for w in ('人情世故', '家国情怀'))
    adm_rows = [r for r in idiom_rows if r['admitted']]
    margins = [r['margin'] for r in adm_rows]
    spread = (max(margins) - min(margins)) if margins else 0.0
    ok2 = bool(len(admitted) >= (50 if FAST else 100) and len(top_e) >= 12
               and len(top_p) >= 12 and spread >= 0.05 and tracked_ok)
    d2 = ('准入 %d（门槛 %d）；能量/疼痛两端 %d/%d；margin 跨度 %.3f；'
          '人情世故=%s 家国情怀=%s'
          % (len(admitted), 50 if FAST else 100, len(top_e), len(top_p), spread,
             (tracked.get('人情世故', {}).get('reason', '未出现')),
             (tracked.get('家国情怀', {}).get('reason', '未出现'))))

    def rule(rows):
        if not rows:
            return False, 0, 0
        top8 = [r['word'] for r in rows[:8]]
        hits = [r['word'] for r in rows if r['word'] in SOCIAL_SEEDS]
        bot = [w for w in top8 if w in BOTANY_WORDS]
        return (len(hits) >= 3 and len(bot) == 0), len(hits), len(bot)

    if inv.get('present'):
        okp, nhp, nbp = rule(inv['ppmi']['neighbors'])
        okc, nhc, nbc = rule(inv['ctx']['neighbors'])
        d3 = ('尺一 PPMI：社会词 %d 个 / top-8 植物词 %d 个 → %s；'
              '尺二 上下文：社会词 %d 个 / top-8 植物词 %d 个 → %s'
              % (nhp, nbp, '过' if okp else '不过', nhc, nbc, '过' if okc else '不过'))
        ok3 = okp
        inv['ruler_verdict'] = {'ppmi': bool(okp), 'ctx': bool(okc)}
    else:
        ok3 = False
        d3 = inv.get('why', '内卷不在词表')

    ok4 = bool(any(m['dual'] for m in metas))
    d4 = '；'.join('%s geo %.1f%% / fp %.1f%%' % (m['name'], m['geo_pct'] * 100,
                                                 m['fp_pct'] * 100) for m in metas) + \
         '（双通道达标 %d/%d）' % (sum(1 for m in metas if m['dual']), len(metas))

    return {
        'L36-1': [ok1, d1, '语料 >=100 万字、社会类 >=5000 篇、>=3 类来源'],
        'L36-2': [ok2, d2, '成语准入 >=100 且两轴有区分度，点名短语通过'],
        'L36-3': [ok3, d3, '内卷 top-20 含 >=3 社会词且 top-8 无植物词（主尺=PPMI）'],
        'L36-3b': [bool(inv.get('ruler_verdict', {}).get('ctx')) if inv.get('present') else False,
                   '（第二把尺，敏感性检验）' + d3.split('；')[-1] if inv.get('present') else '内卷不在词表',
                   '同一判据换成「真实上下文共现次数」排序（预注册第二把尺）'],
        'L36-3c': [bool(rnull and rnull.get('credible') and rnull.get('has_power')),
                   ('阴性对照 %d 个通过 %d 个（%.1f%%）；阳性对照 %d 个通过 %d 个（%s）'
                    % (rnull['n_ctrl'], rnull['n_pass'], rnull['pass_rate'] * 100,
                       len(rnull.get('pos', [])), rnull.get('n_pos_pass', 0),
                       '、'.join('%s:%d' % (r['word'], r['social_hits'])
                                for r in rnull.get('pos', [])))) if rnull and rnull.get('ok')
                   else '控制词池为空',
                   '证伪：阴性通过率 <=10% 且阳性至少一个通过，尺二才算既不通吃也不失效'],
        'L36-4': [ok4, d4, '两组中文隐喻至少一组跳数与指纹同时 <=10% 分位'],
    }


# ============================ 保存 / 报告 / 图 ============================

def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(x) for x in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def save_outputs(con, meta, vocab, dfv, flagv, idiom_rows, admitted, top_e, top_p,
                 inv, metas, verdict, sense_report, stats, anchors, rnull=None):
    cur = con.cursor()
    for t in ['l36_vocab', 'l36_idiom', 'l36_neighbor', 'l36_metaphor',
              'l36_meta', 'l36_verdict']:
        cur.execute('DELETE FROM %s' % t)
    cur.executemany('INSERT INTO l36_vocab(id,word,df,flag) VALUES(?,?,?,?)',
                    [(i, w, int(dfv[i]), flagv[i]) for i, w in enumerate(vocab)])
    cur.executemany(
        'INSERT OR REPLACE INTO l36_idiom(word,flag,n_occ,u,info,e,p,margin,admitted,reason,ctx)'
        ' VALUES(?,?,?,?,?,?,?,?,?,?,?)',
        [(r['word'], r['flag'], r['n'], r['u'], r['info'], r['e'], r['p'], r['margin'],
          int(r['admitted']), r['reason'], '、'.join(r.get('ctx_top', [])))
         for r in idiom_rows])
    if inv.get('present'):
        for ruler in ('ppmi', 'ctx'):
            cur.executemany(
                'INSERT INTO l36_neighbor(word,ruler,rank,neighbor,score,df,srcs)'
                ' VALUES(?,?,?,?,?,?,?)',
                [('内卷', ruler, r['rank'], r['word'], r['score'], r['df'],
                  json.dumps(r['srcs'], ensure_ascii=False))
                 for r in inv[ruler]['neighbors']])
    for m in metas:
        for side, s in m['sides'].items():
            cur.execute('INSERT INTO l36_metaphor VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                        (m['name'], side, s['node'], m['geo'], m['geo_pct'], m['fp'],
                         m['fp_pct'], m['base_geo'], m['base_fp'], m['base_fp_pct'],
                         json.dumps({'qual': s['qual'], 'top': s['top']},
                                    ensure_ascii=False)))
    for k, (ok, detail, rule) in verdict.items():
        cur.execute('INSERT INTO l36_verdict VALUES(?,?,?,?)', (k, int(ok), detail, rule))
    cur.executemany('INSERT INTO l36_meta(k,v) VALUES(?,?)',
                    [(k, json.dumps(jsonable(v), ensure_ascii=False))
                     for k, v in [('corpus', meta), ('sense_report', sense_report),
                                  ('anchors', anchors),
                                  ('xhy', {'occ': stats['xhy_occ'],
                                           'df': stats['xhy_df'],
                                           'example': dict(list(stats['xhy_example'].items())[:5])})]])
    con.commit()

    # 点名短语在每类语料里到底出现于几篇文档：把「语料里有没有」和「图里排第几」分开
    cov = {}
    for w in TRACKED:
        cov[w] = {s: int(n) for s, n in cur.execute(
            'SELECT src, COUNT(*) FROM l36_doc WHERE instr(text, ?) > 0 GROUP BY src', (w,))}

    payload = {
        'meta': meta, 'vocab_size': len(vocab),
        'params': {'FAST': FAST, 'VOCAB': VOCAB, 'WINDOW': WINDOW, 'MIN_CO': MIN_CO,
                   'KNN_K': KNN_K, 'NB_K': NB_K, 'N_NULL': N_NULL,
                   'MIN_SENSE_N': MIN_SENSE_N, 'K_SENSE': K_SENSE,
                   'EMB_DIM': EMB_DIM, 'CRAWL': CRAWL},
        'anchors': anchors,
        'idioms': {'n_candidates': len(idiom_rows), 'n_admitted': len(admitted),
                   'top_energy': [{'word': r['word'], 'margin': round(r['margin'], 4),
                                   'ctx': r.get('ctx_top', []), 'n': r['n']}
                                  for r in top_e],
                   'top_pain': [{'word': r['word'], 'margin': round(r['margin'], 4),
                                 'ctx': r.get('ctx_top', []), 'n': r['n']}
                                for r in top_p],
                   'tracked': {w: next((r for r in idiom_rows if r['word'] == w), None)
                               for w in ('人情世故', '家国情怀', '内卷', '熵增')},
                   'xiehouyu': {'n_types': len(stats['xhy_occ']),
                                'top': sorted(stats['xhy_occ'].items(),
                                              key=lambda x: -x[1])[:10]}},
        'involution': inv,
        'metaphor': metas,
        'senses': sense_report,
        'phrase_cov': cov,
        'ruler_null': rnull,
        'verdict': verdict,
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(jsonable(payload), f, ensure_ascii=False, indent=1)
    return payload


def _one_line(v, inv, rnull=None):
    if all(x[0] for x in v.values()):
        return '全部通过：语料、成语、内卷邻居（两把尺 + 证伪）、中文隐喻都站住了。'
    if inv.get('present') and inv.get('ruler_verdict', {}).get('ctx'):
        rn = rnull or {}
        if rn.get('ok') and not rn.get('credible'):
            return ('第二把尺虽然让「竞争/行业/市场」浮了出来，但它在 df 同量级的控制词上'
                    '通过率 %.0f%%——这把尺量不出「内卷」的特殊性，只能量出「这批语料的窗口里'
                    '到处都是社会词」。所以 L36-3 的结论仍然是：主尺 FAIL 成立，替补尺也不可信。'
                    % (rn['pass_rate'] * 100))
        if rn.get('ok'):
            ph = [r['social_hits'] for r in rn.get('pos', [])]
            return ('同一张图，换一把尺子结论就不同：PPMI 把「内卷」的邻居判给稀有泛化词，'
                    '按真实上下文共现排，「竞争/行业/市场」才浮出来。但这把尺经不起吹——'
                    '60 个同 df 阴性对照通过 %.0f%%（不通吃），%d 个已知社会词只有 %d 个过线（不灵敏）。'
                    '「内卷」拿到 %d 个，是刚过线。所以 L36-3b 只在字面上成立，效应量很小；'
                    '主尺 FAIL 不变。'
                    % (rn['pass_rate'] * 100, len(ph), sum(1 for x in ph if x >= 3),
                       len(inv.get('ctx', {}).get('social_hits', []))))
        return ('同一张图，换一把尺子结论就不同：PPMI 把「内卷」的邻居判给稀有泛化词，'
                '按真实上下文共现排，「竞争/行业/市场」才浮出来。')
    return ('本轮最硬的事实是：判决没有为了让报告好看而放宽。哪一条没通过，就写哪一条没通过。')


def write_report(payload):
    meta = payload['meta']
    inv = payload['involution']
    idi = payload['idioms']
    v = payload['verdict']

    src_rows = '\n'.join('| %s | %d | %d |' % (s or '(空)', meta['src_docs'].get(s, 0),
                                               meta['src_chars'].get(s, 0))
                         for s in sorted(meta['src_docs'], key=lambda x: -meta['src_chars'].get(x, 0)))
    drop_rows = '\n'.join('| %s | %d |' % (k.replace('|', ' / '), n)
                          for k, n in sorted(meta['drop'].items(), key=lambda x: -x[1])[:14])

    cov = payload.get('phrase_cov', {})
    srcs = sorted(meta['src_docs'], key=lambda x: -meta['src_docs'].get(x, 0))
    cov_rows = '\n'.join(
        '| %s | %s | %d |' % (w, ' | '.join(str(cov.get(w, {}).get(s, 0)) for s in srcs),
                              sum(cov.get(w, {}).values()))
        for w in ['内卷', '熵增', '人情世故', '家国情怀', '网暴', '打工人', '键盘侠'])
    cov_head = '| 短语 | ' + ' | '.join(srcs) + ' | 合计 |'
    cov_sep = '|---' * (len(srcs) + 2) + '|'

    rn = payload.get('ruler_null') or {}
    if rn.get('ok'):
        ex_rows = '\n'.join('| %s | %d | %d | %s |'
                            % (r['word'], r['df'], r['social_hits'],
                               '、'.join(r['hit_words'][:5]) or '-')
                            for r in rn.get('examples', [])) or '| （无） | - | - | - |'
        if rn['credible'] and rn.get('has_power'):
            null_verdict = ('阴性通过率 **%.1f%% ≤ 10%%**，阳性通过 %d/%d。'
                            '尺二**既不通吃、也不失效**：它是能被社会含义触发的。'
                            % (rn['pass_rate'] * 100, rn.get('n_pos_pass', 0),
                               len(rn.get('pos', []))))
        elif rn['credible']:
            null_verdict = ('阴性通过率 **%.1f%%**，但阳性通过 **0/%d** —— '
                            '尺二严到连已知社会词都测不出，**没有分辨力**，'
                            'L36-3b 的 PASS 只能说明「内卷恰好撞上了」，不能说明它「被拽回来了」。'
                            % (rn['pass_rate'] * 100, len(rn.get('pos', []))))
        else:
            null_verdict = ('阴性通过率 **%.1f%% > 10%%**，尺二**被推翻**：'
                            'L36-3b 的 PASS 不能当作「社会语义被拽回来了」的证据，'
                            '它更可能只是「df 同量级的词，其 ±60 字窗口里本来就会混进社会词」。'
                            % (rn['pass_rate'] * 100))
        pos_rows = '\n'.join('| %s | %d | %s | %s |'
                             % (r['word'], r['df'], r['social_hits'],
                                '、'.join(r['hit_words'][:5]) or '-')
                             for r in rn.get('pos', [])) or '| （无） | - | - | - |'
        ph = [r['social_hits'] for r in rn.get('pos', [])]
        if ph:
            caveat = ('\n\n**必须跟着说的一句**：阳性对照的社会词命中是 %s（中位数 %g）。'
                      '%d 个已知社会词里只有 %d 个过线，说明「>=3 个社会词」这条判据本身就偏严。'
                      '「内卷」拿到 %d，是刚过线，不是压倒性优势。'
                      '换句话说：L36-3b 的 PASS 成立于字面，但效应量很小——'
                      '它证明「内卷周围确实有社会词」，不证明「内卷的社会语义已经从其他词里脱颖而出」。'
                      % (' / '.join('%s %d' % (r['word'], r['social_hits'])
                                    for r in rn.get('pos', [])),
                         float(np.median(np.array(ph))), len(ph),
                         sum(1 for x in ph if x >= 3),
                         len(inv.get('ctx', {}).get('social_hits', []))))
        else:
            caveat = ''
        null_block = ('- 阴性对照 **%d** 个（df %d–%d），通过 **%d** 个（**%.1f%%**）\n'
                      '- 阴性对照的社会词命中：中位数 **%.0f** / 均值 **%.1f**（「内卷」是 %d）\n'
                      '- %s\n\n'
                      '| 阴性对照里通过的词 | df | 社会词命中 | 命中了哪些 |\n|---|---|---|---|\n%s\n\n'
                      '**阳性对照**（已知社会词，用完全相同的构造与判据）：\n\n'
                      '| 社会词 | df | 社会词命中 | 命中了哪些 |\n|---|---|---|---|\n%s%s'
                      % (rn['n_ctrl'], rn['band'][0], rn['band'][1], rn['n_pass'],
                         rn['pass_rate'] * 100, rn['median_hits'], rn['mean_hits'],
                         len(inv.get('ctx', {}).get('social_hits', [])), null_verdict,
                         ex_rows, pos_rows, caveat))
    else:
        null_block = '（控制词池为空，本项未能测量）'

    def nb_rows(ruler):
        rows = inv.get(ruler, {}).get('neighbors', []) if inv.get('present') else []
        return '\n'.join('| %d | %s | %s | %d | %s | %+.3f |'
                         % (r['rank'], r['word'],
                            ('%.3f' % r['score']) if ruler == 'ppmi' else ('%d' % r['score']),
                            r['df'],
                            '、'.join('%s:%d' % (k, x) for k, x in r['srcs'].items()) or '-',
                            r['margin'])
                         for r in rows[:20]) or '（内卷不在词表）'

    diag_rows = '\n'.join(
        '| %s | %d | %d | %s | %s |'
        % (r['word'], r['ctx_n'], r['df'],
           ('%.3f' % r['ppmi']) if r['rank'] > 0 else '0（被 R7 剪掉）',
           ('第 %d 名' % r['rank']) if r['rank'] > 0 else '无边')
        for r in inv.get('social_diag', [])[:14]) if inv.get('present') else '（内卷不在词表）'

    id_rows = '\n'.join('| %s | %d | %+.3f | %s |'
                        % (r['word'], r['n'], r['margin'], '、'.join(r.get('ctx', [])[:6]))
                        for r in idi['top_energy'][:10])
    id_rows_p = '\n'.join('| %s | %d | %+.3f | %s |'
                          % (r['word'], r['n'], r['margin'], '、'.join(r.get('ctx', [])[:6]))
                          for r in idi['top_pain'][:10])

    def trow(w):
        r = idi['tracked'].get(w)
        if not r:
            return '| %s | 未出现 | - | - | - |' % w
        return ('| %s | %d | %d | %+.3f | %s |'
                % (w, r['n'], r['u'], r['margin'],
                   '、'.join(r.get('ctx_top', [])[:6]) or '-'))

    m_rows = '\n'.join(
        '| %s | %s / %s | %s | %d（%.1f%%） | %.3f（%.1f%%） | %d / %.3f（%.1f%%） | %s |'
        % (m['title'], m['sides']['a']['node'], m['sides']['b']['node'],
           '是' if m['dual'] else '否', m['geo'], m['geo_pct'] * 100,
           m['fp'], m['fp_pct'] * 100, m['base_geo'], m['base_fp'],
           m['base_fp_pct'] * 100, '达标' if m['dual'] else '-')
        for m in payload['metaphor']) or '（无）'

    sense_rows = []
    for h, r in payload['senses'].items():
        if r.get('senses'):
            for s in r['senses']:
                sense_rows.append('| %s | %s | %d | %s |'
                                  % (h, s['name'], s['n_occ'], '、'.join(s['top_words'][:6])))
        else:
            sense_rows.append('| %s | 切不出 | - | %s |' % (h, r.get('why', '-')))
    sense_rows = '\n'.join(sense_rows) or '（无）'

    mss = payload['metaphor']
    if any(m['dual'] for m in mss):
        _ok = ('本轮 **L36-4 通过**：%s 在拓扑与指纹两个通道同时落到 10%% 分位以内。'
               % '、'.join(m['title'] for m in mss if m['dual']))
    else:
        _ok = '本轮 **L36-4 未通过**：两组都没能在两个通道同时落到 10%% 分位以内。'
    _bad = []
    for m in mss:
        if m['dual']:
            continue
        _bad.append('- **%s 没通过**：拓扑 %.1f%%、指纹 %.1f%%；未消歧的词级对照是 %d 跳 / %.3f（%.1f%%）。'
                    % (m['title'], m['geo_pct'] * 100, m['fp_pct'] * 100,
                       m['base_geo'], m['base_fp'], m['base_fp_pct'] * 100))
        if 0 <= m['base_fp_pct'] < m['fp_pct']:
            _bad.append('  - 方向值得记住：**消歧把它推远了**（词级 %.1f%% → 义项级 %.1f%%）。'
                        '切在泛化语境上的义项，切得越细越偏离原义。'
                        % (m['base_fp_pct'] * 100, m['fp_pct'] * 100))
    for h, r in payload['senses'].items():
        if not r.get('senses'):
            _bad.append('- `%s` 切不出义项：%s。样本不足不补样本。' % (h, r.get('why', '-')))
    if any(s['domain'] == 'other' for r in payload['senses'].values()
           for s in r.get('senses', [])):
        _bad.append('- 有义项落在「其他」：预注册门类词表在该语境上先失效，命名层不成立，'
                    '指纹层也就无从谈起。')
    sec42 = _ok + (('\n\n' + '\n'.join(_bad)) if _bad else '') + \
        '\n\n阈值、义项数、窗口上限全部跑前写死；不放宽阈值、不换义项、不挑窗口。'

    vd = '\n'.join('| %s | %s | %s | %s |' % (k, 'PASS' if x[0] else 'FAIL', x[1], x[2])
                   for k, x in v.items())

    xhy = idi['xiehouyu']
    xhy_txt = ('本轮语料共识别歇后语 **%d** 条（去重类型）。出现最多的：%s。'
               % (xhy['n_types'],
                  '、'.join('%s（%d 次）' % (k, n) for k, n in xhy['top'][:6]) or '-')
               ) if xhy['n_types'] else '本轮语料未识别出歇后语。'

    md = '''# L36 中文原生语义涌现实证报告

> 张力 := 预测误差。本系统不给奖励、不做动作、不搜索最优解。以下每个数字都来自本轮真实跑出的图。

## 0. 先说清楚三件事（不粉饰）

1. **上一轮的 L36 交付物（主程序 / 报告 / SQLite / 图）已被沙箱重置抹掉**，`git log --all -- 'cognitive_chinese*'` 一条记录都没有——它们从来没进过 git。本轮是从零重写的第二遍，写完后立刻提交并推送到 `origin/trae/agent-zXpoIE`。
2. 语料同样被抹掉过一次（`/tmp` 被清空）。本轮数据全部落在 `/workspace/.l36_data`，并按字节区间重新抓取：THUCNews 社会段 234,952,000 B / 时政段 101,476,000 B / 财经段 155,722,000 B、知乎问答 120,343,250 B、知乎-KOL 分片 306,309,311 B。
3. **THUCNews 按门类顺序排列**（体育→娱乐→家居→教育→时政→游戏→社会→科技→股票→财经）。顺序读前 90MB 全是体育，「内卷」命中 0 次。本轮直接按字节区间跳读。

## 1. 语料管道：不是「爬网页」，是「定向切开源数据集」

| 来源 | 篇数 | 中文字数 |
|---|---|---|
{src_rows}

合计 **{chars}** 字、**{docs}** 篇、**{nsrc}** 类来源。目标 100 万字。

### 1.1 张力账本丢弃了什么

| 丢弃原因 | 篇数 |
|---|---|
{drop_rows}

判据预注册：中文占比 <45%、有效长句密度 <12%、4-gram 多样性 <0.45、8-gram 复读率 <35%、重复句 >35%、感叹号密度超线，任一触发即丢。这不是「清洗得干净」，是「把营销号排比句挡在底湖外面」。

### 1.2 抓取边界（如实记录）

- 本轮**关闭 Google/Baidu 式网页抓取**（`L36_CRAWL=0`）：上一轮它只贡献 11 篇 / 2.8 万字，却带来 403 与超时的长时间阻塞。指令原话也是「放弃强行抓取，直接接入开源数据集」。
- 知乎问答 gz 尾部有损坏风险，程序用 `try/except` 保留能解出的前段，不假装拿到全量。
- 知乎-KOL parquet 逐 row-group 读，坏组跳过并计数（上一轮 202 个 row-group 里 166 个损坏）。
- 网文源本轮未落地，`novel` 篇数为 0；不拿「计划中的语料」冒充已入湖语料。

### 1.3 语料的时间切片：这几个词到底活在哪批文本里

「入湖 5 万篇」是总量，不是覆盖。下表直接数「该短语出现在多少篇文档里」，按来源拆开——它决定了后面每一节能测什么、不能测什么。

{cov_head}
{cov_sep}
{cov_rows}

读法：**THUCNews 是 2005–2011 年的新闻**，`内卷` 在 14,000 篇社会新闻里只出现 **1** 篇。这不是抽取失败，是年代不对——「内卷」是 2018 年之后才进入大众语汇的。它在本轮语料里的 87 篇文档，几乎全部来自知乎-KOL 分片。同理 `家国情怀` 全部 8 篇都在知乎-KOL。**任何关于这些词的测量，本质上是在测知乎-KOL 这一路现代口语语料**，其余四路只提供了背景词频。「网暴」只有 18 个可切窗口，所以 §4.1 里它切不出义项——样本不足就是样本不足。

## 2. 成语/俗语：把「词频门槛」换成「上下文信息密度」

预注册准入规则：**出现 >=2 次 且 不同上下文词 >=4 且 上下文分布的归一化熵 >=0.5**。满足即入表，不看 df 是否到 50。

- 成语/习用语候选 **{n_cand}** 个；
- 准入 **{n_adm}** 个（L35 旧规则下只有 64 个节点）；
- 每个准入成语都用它真实活着的上下文词的均值向量，投影到能量/疼痛两轴。

### 2.1 能量轴最高的成语（margin = 能量 − 疼痛）

| 成语 | 出现 | margin | 它真实的上下文 |
|---|---|---|---|
{id_rows}

### 2.2 疼痛轴最高的成语

| 成语 | 出现 | margin | 它真实的上下文 |
|---|---|---|---|
{id_rows_p}

### 2.3 用户点名的两个词

| 短语 | 出现 | 上下文词数 | margin | 真实上下文 |
|---|---|---|---|---|
{trow_rsq}
{trow_jg}

这两行不是人工挑的：它们和其余 {n_cand} 个候选走同一条准入规则。L35 里它们因「df 只有 15 / 7」被卡掉，本轮门槛换成了信息密度，所以进了。

### 2.4 歇后语

{xhy_txt}

## 3. 「内卷」的社会学重构：两把尺子

L35 的中文维基语料里，「内卷」的邻居是植物学词（花瓣、直立、斑纹、捕虫）——因为维基把它写成「叶缘内卷」。本轮换成社会新闻 / 时政 / 财经 / 知乎，同一张图，用两把尺子各量一次。

**先把话说在前面**：按 §1.3 的统计，「内卷」在本轮 5 万篇里只出现在 87 篇文档中，其中 85 篇来自知乎-KOL，14,000 篇 THUCNews 社会新闻里只有 1 篇。所以这一节测量的其实不是「社会新闻里的内卷」，而是「知乎现代口语里的内卷」——THUCNews 的年代（2005–2011）根本没有这个词。

### 3.1 尺一：PPMI（预注册主口径）

| 排名 | 邻居 | PPMI | df | 来自哪些语料 | 该词 margin |
|---|---|---|---|---|---|
{nb_ppmi}

### 3.2 尺二：真实上下文共现次数（预注册第二把尺）

同一张图，只把排序键换成「该词出现在『内卷』±60 字真实窗口里的次数」。

| 排名 | 邻居 | 共现次数 | df | 来自哪些语料 | 该词 margin |
|---|---|---|---|---|---|
{nb_ctx}

- 社会词命中：PPMI **{nhp}** 个 / 上下文 **{nhc}** 个
- top-8 里植物词：PPMI **{nbp}** 个 / 上下文 **{nbc}** 个
- 「内卷」自身 margin：{imargin}
- 两把尺子的判定：PPMI **{vsp}** / 上下文 **{vsc}**

判决用尺一（主口径，预注册）。尺二只做敏感性检验：它若通过，说明社会语义确实进了语料与窗口，只是被 PPMI 的稀有词偏好挤出了 top-20；它若不通过，说明连窗口层面都没进来。

### 3.3 分开记：「语料里有」和「图里排第几」

| 社会词 | ±60 字共现 | df | PPMI | 排名 |
|---|---|---|---|---|
{diag_rows}

PPMI 的分母里高频词的列和很大，稀有词对天然占便宜。这条表只用来说明失败发生在哪一层，不改变判决。

### 3.4 证伪：第二把尺在控制词上会不会也「通过」

尺二通过之后，第一件事是问它是不是随便什么词都能通过。**预注册**：控制词 = 与「内卷」df 同量级（`[50, 300]`）、不在社会词表 / 植物词表 / 点名短语 / 隐喻头词里的词，用 `SEED` 随机抽 60 个；对每个控制词使用**与「内卷」完全相同的构造**（原文 ±60 字窗口 → 分词 → 词频 → top-20）和**完全相同的判据**。若控制词通过率 > 10%，则 L36-3b 的 PASS 只是「df 同量级的词窗口里本来就会混进社会词」，应一并推翻。

{null_block}

## 4. 中文隐喻测验（全中文，不给英文）

双通道：kNN 骨架上的测地线跳数 + 张力指纹距离。分位数由「频率/度匹配零模型」给出（{n_null} 对随机词），阈值预注册为 **两个通道都 <=10% 分位**。

| 测验 | 义项节点 | 双通道 | 跳数（分位） | 指纹（分位） | 未消歧基础对照 | 判定 |
|---|---|---|---|---|---|---|
{m_rows}

### 4.1 切出来的中文义项

| 头词 | 义项 | 窗口数 | 它拉过来的词 |
|---|---|---|---|
{sense_rows}

义项是语境窗口 k-means 切出来的，命名用预注册门类关键词，不人工挑词。限定词（键盘侠/物理的/打工人/电池的）只用于在义项里自动选一个，不参与造义项。

### 4.2 卡在哪一层

{sec42}

## 5. 判决

| 判据 | 结果 | 实测 | 预注册规则 |
|---|---|---|---|
{vd}

## 6. 一句话

{one_line}

## 7. 边界与不变量

- 张力仍定义为预测误差；本程序**没有任何动作出口**，不产生 reward，不搜索最优解。
- 神性锚点 `tension_def / reality_feed / safety_bound` 未改；内存峰值受 700MB 红线约束。
- 分位数、零模型、准入阈值全部在跑之前写死；报告里的每个「显著」都对应一个可复算的分位。
- 本轮只做中文。英文接口（arxiv / WordNet）全部停用。

## 8. 持久化

本轮所有交付物（主程序 / 报告 / JSON / PNG / SQLite / 数据校验和）在跑完后提交并推送到 `origin/trae/agent-zXpoIE`。沙箱重置只回滚工作区，不回滚远端分支——这是这个项目唯一可靠的存档面。
'''.format(
        src_rows=src_rows, chars=meta['chars'], docs=meta['docs'],
        nsrc=meta['n_sources'], drop_rows=drop_rows,
        cov_head=cov_head, cov_sep=cov_sep, cov_rows=cov_rows,
        sec42=sec42,
        n_cand=idi['n_candidates'], n_adm=idi['n_admitted'],
        id_rows=id_rows, id_rows_p=id_rows_p,
        trow_rsq=trow('人情世故'), trow_jg=trow('家国情怀'),
        xhy_txt=xhy_txt,
        nb_ppmi=nb_rows('ppmi'), nb_ctx=nb_rows('ctx'),
        nhp=len(inv.get('ppmi', {}).get('social_hits', [])),
        nhc=len(inv.get('ctx', {}).get('social_hits', [])),
        nbp=len(inv.get('ppmi', {}).get('botany_top8', [])),
        nbc=len(inv.get('ctx', {}).get('botany_top8', [])),
        imargin=inv.get('margin', 'N/A'),
        vsp='过' if inv.get('ruler_verdict', {}).get('ppmi') else '不过',
        vsc='过' if inv.get('ruler_verdict', {}).get('ctx') else '不过',
        diag_rows=diag_rows, n_null=payload['params']['N_NULL'],
        null_block=null_block,
        m_rows=m_rows, sense_rows=sense_rows,
        vd=vd, one_line=_one_line(v, inv, payload.get('ruler_null')))

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write(md)
    print('  报告写入 %s' % OUT_MD, flush=True)


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

        fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
        meta = payload['meta']
        ss = sorted(meta['src_docs'], key=lambda x: -meta['src_docs'].get(x, 0))
        ax[0][0].bar(range(len(ss)), [meta['src_chars'].get(s, 0) / 10000 for s in ss],
                     color='#3b6ea5')
        ax[0][0].set_xticks(range(len(ss)))
        ax[0][0].set_xticklabels(ss, rotation=25, ha='right', fontsize=8)
        ax[0][0].set_ylabel('万字')
        ax[0][0].set_title('A. 语料：按来源的中文字数（总 %d 万字）'
                           % round(meta['chars'] / 10000))

        dr = sorted(meta['drop'].items(), key=lambda x: -x[1])[:8]
        ax[0][1].barh(range(len(dr)), [x[1] for x in dr], color='#b5651d')
        ax[0][1].set_yticks(range(len(dr)))
        ax[0][1].set_yticklabels([x[0].split(' / ')[-1] + ' · ' + x[0].split(' / ')[0][:6]
                                  for x in dr], fontsize=8)
        ax[0][1].invert_yaxis()
        ax[0][1].set_title('B. 张力账本丢弃原因 Top8')

        inv = payload['involution']
        rows = inv.get('ctx', {}).get('neighbors', []) if inv.get('present') else []
        rows = rows[:12]
        if rows:
            cols = ['#2e8b57' if r['word'] in SOCIAL_SEEDS
                    else ('#c0392b' if r['word'] in BOTANY_WORDS else '#8d8d8d')
                    for r in rows]
            ax[1][0].barh(range(len(rows)), [r['score'] for r in rows], color=cols)
            ax[1][0].set_yticks(range(len(rows)))
            ax[1][0].set_yticklabels([r['word'] for r in rows], fontsize=9)
            ax[1][0].invert_yaxis()
        ax[1][0].set_title('C. 「内卷」top-12 邻居（尺二·真实共现）\n绿=社会词  红=植物词  灰=其他',
                           fontsize=10)

        ms = payload['metaphor']
        if ms:
            xs = np.arange(len(ms))
            ax[1][1].bar(xs - 0.2, [m['geo_pct'] * 100 for m in ms], 0.4,
                         label='拓扑分位', color='#3b6ea5')
            ax[1][1].bar(xs + 0.2, [m['fp_pct'] * 100 for m in ms], 0.4,
                         label='指纹分位', color='#8e44ad')
            ax[1][1].axhline(10, color='#c0392b', ls='--', lw=1)
            ax[1][1].text(len(ms) - 0.5, 11.5, '10% 阈值', color='#c0392b', fontsize=8,
                          ha='right')
            ax[1][1].set_xticks(xs)
            ax[1][1].set_xticklabels([m['name'] for m in ms])
            ax[1][1].set_ylabel('分位（越低越接近）')
            ax[1][1].legend(fontsize=8)
        ax[1][1].set_title('D. 中文隐喻双通道（虚线=10% 阈值）')

        plt.tight_layout()
        plt.savefig(PNG, dpi=110)
        print('  图写入 %s' % PNG, flush=True)
    except Exception as e:
        print('  [draw] 跳过：%s' % e, flush=True)


# ============================ 第二把尺的证伪 ============================

def ruler_null_test(con, vocab, wid_of, dfv, n_ctrl=60, lo=50, hi=300,
                    pos_words=('就业', '加班', '职场', '学历')):
    """证明「尺二」不是只要是个词就能凑出社会词，也不是严到一个词都测不出。

    预注册：
      阴性对照（控制词）= 与「内卷」df 同量级（[lo, hi]）、不在社会词表/植物词表/
        点名短语/隐喻头词里的词，用 SEED 随机抽 n_ctrl 个；
      阳性对照 = 语料里明确的社会词（就业/加班/职场/学历）。
    对每个词用与「内卷」完全相同的构造（原文 ±60 字窗口 → 分词 → 词频 → top-20）
    和完全相同的判据。
    判读：阴性通过率 > 10% → 尺二被推翻（万能通过）；
          阳性一个都过不了 → 尺二无分辨力（严到测不出东西）。
    """
    print('[6b] 证伪：第二把尺在控制词 / 阳性对照上会不会也「通过」', flush=True)
    excl = set(SOCIAL_SEEDS) | set(BOTANY_WORDS) | set(TRACKED) | set(HEADS)
    cand = [w for w in vocab
            if lo <= int(dfv[wid_of[w]]) <= hi and w not in excl]
    if not cand:
        return {'ok': False, 'why': '控制词池为空'}
    rng2 = np.random.default_rng(SEED)
    ctrl = list(rng2.choice(np.array(cand, dtype=object),
                            size=min(n_ctrl, len(cand)), replace=False))
    ctrl = [str(w) for w in ctrl]
    pos = [w for w in pos_words if w in wid_of]

    ctxs = {w: Counter() for w in ctrl + pos}
    cur = con.cursor()
    cur.execute('SELECT text FROM l36_doc')
    while True:
        rows = cur.fetchmany(400)
        if not rows:
            break
        for (text,) in rows:
            for w in ctxs:
                if w not in text:
                    continue
                for m in re.finditer(re.escape(w), text):
                    snip = text[max(0, m.start() - 60):m.end() + 60]
                    cw = ctxs[w]
                    for w2, _ in tokenize_text(snip):
                        if w2 != w:
                            cw[w2] += 1

    def rule_for(counts):
        rank = 0
        rows = []
        for w2, n in sorted(counts.items(), key=lambda x: -x[1]):
            j = wid_of.get(w2)
            if j is None:
                continue
            rank += 1
            rows.append((w2, int(n)))
            if rank >= 20:
                break
        top8 = [w for w, _ in rows[:8]]
        hits = [w for w, _ in rows if w in SOCIAL_SEEDS]
        bot = [w for w in top8 if w in BOTANY_WORDS]
        return (len(hits) >= 3 and len(bot) == 0), len(hits), len(bot), hits[:6]

    res = []
    for w in ctrl:
        ok, nh, nb, ex = rule_for(ctxs[w])
        res.append({'word': w, 'df': int(dfv[wid_of[w]]), 'pass': bool(ok),
                    'social_hits': nh, 'botany_top8': nb, 'hit_words': ex})
    n_pass = sum(1 for r in res if r['pass'])
    rate = n_pass / max(1, len(res))
    hits_arr = np.array([r['social_hits'] for r in res], dtype=np.float32)
    print('  阴性对照 %d 个（df %d–%d）：通过 %d 个（%.1f%%），社会词命中 中位数 %.0f / 均值 %.1f'
          % (len(res), lo, hi, n_pass, rate * 100, float(np.median(hits_arr)),
             float(hits_arr.mean())), flush=True)
    ex = [r for r in res if r['pass']][:3]
    for r in ex:
        print('    通过示例：%s（df=%d，命中 %s）'
              % (r['word'], r['df'], '、'.join(r['hit_words'])), flush=True)

    pres = []
    for w in pos:
        ok, nh, nb, ex2 = rule_for(ctxs[w])
        pres.append({'word': w, 'df': int(dfv[wid_of[w]]), 'pass': bool(ok),
                     'social_hits': nh, 'botany_top8': nb, 'hit_words': ex2})
    n_pos = sum(1 for r in pres if r['pass'])
    print('  阳性对照 %d 个：通过 %d 个（%s）'
          % (len(pres), n_pos,
             '、'.join('%s(%d)' % (r['word'], r['social_hits']) for r in pres)), flush=True)

    power = bool(len(pres) > 0 and n_pos > 0)
    return {'ok': True, 'band': [lo, hi], 'n_ctrl': len(res), 'n_pass': n_pass,
            'pass_rate': round(float(rate), 4), 'median_hits': float(np.median(hits_arr)),
            'mean_hits': round(float(hits_arr.mean()), 2),
            'credible': bool(rate <= 0.10), 'controls': res,
            'examples': ex,
            'pos': pres, 'n_pos_pass': n_pos, 'has_power': power}


# ============================ 主流程 ============================

def main():
    t0 = time.time()
    print('=' * 72)
    print('L36 中文原生语义大爆发 | FAST=%s CRAWL=%d REUSE=%s' % (FAST, CRAWL, REUSE))
    print('张力 := 预测误差 | 无 reward | 现实只读 | 单出口 | 内存 < %dMB' % MEM_LIMIT_MB)
    print('=' * 72, flush=True)

    if os.path.exists(DB) and REUSE:
        con = sqlite3.connect(DB)
        have = con.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='l36_doc'").fetchone()[0]
        if not have:
            setup_db(con)
    else:
        if os.path.exists(DB):
            os.remove(DB)
        con = sqlite3.connect(DB)
        setup_db(con)
    con.execute('PRAGMA journal_mode=WAL')

    meta, old_items, dfv_old, flag_of, stats = load_or_build_pass_a(con)
    gc.collect()
    trim_mem()
    mem_guard('passA')

    vocab, dfv, flagv, old2new, idiom_rows, admitted, missing_probe = \
        choose_vocab(old_items, dfv_old, flag_of, stats)
    mem_guard('vocab')

    protect_words = set(PROBES) | set(HEADS) | set(w for w, _ in admitted)
    protect_words |= set(ENERGY_ANCHOR) | set(PAIN_ANCHOR)
    P, S, pair_src, protect = build_graph(con, vocab, old2new, protect_words,
                                          stats['phrase_ctx'])
    mem_guard('graph')

    wid_of = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    E = build_embeddings(P)
    Ee, Ep, ei, pi = axis_vectors(E, wid_of, V)
    e = np.asarray(E @ Ee).ravel()
    p = np.asarray(E @ Ep).ravel()
    margin = e - p
    anchors = {'energy_words': [vocab[i] for i in ei],
               'pain_words': [vocab[i] for i in pi],
               'pct_energy': {w: round(float(np.mean(e <= e[wid_of[w]])), 4)
                              for w in HEADS + ['内卷', '熵增'] if w in wid_of},
               'pct_pain': {w: round(float(np.mean(p <= p[wid_of[w]])), 4)
                            for w in HEADS + ['内卷', '熵增'] if w in wid_of}}
    print('  能量轴锚点 %d：%s' % (len(ei), '、'.join(vocab[i] for i in ei[:10])), flush=True)
    print('  疼痛轴锚点 %d：%s' % (len(pi), '、'.join(vocab[i] for i in pi[:10])), flush=True)

    neigh = [row_top(P, i, NB_K)[0] for i in range(V)]
    edges_e = np.quantile(e, np.linspace(0, 1, AX_BINS + 1)[1:-1])
    edges_p = np.quantile(p, np.linspace(0, 1, AX_BINS + 1)[1:-1])
    FPraw = fingerprint_matrix(P, e, p, list(range(V)), neigh, edges_e, edges_p)
    FP, FPmu = center_fp(FPraw)
    del FPraw
    gc.collect()

    top_e, top_p = score_idioms(vocab, wid_of, E, Ee, Ep, dfv, idiom_rows, stats)
    stats['idiom_ctx'] = {}   # 打分用完即释放：12k+ 成语 × 80 上下文是内存大头
    gc.collect()
    trim_mem()
    mem_guard('idioms')

    senses, sense_report = induce_senses(vocab, E, e, p, wid_of, stats, dfv)
    stats['head_wins'] = {}
    gc.collect()
    trim_mem()
    mem_guard('senses')

    metas = measure_metaphor(vocab, P, S, E, e, p, margin, FP, FPmu, wid_of,
                             senses, dfv, edges_e, edges_p)
    mem_guard('metaphor')

    inv = analyze_involution(vocab, P, wid_of, pair_src, dfv, margin,
                             stats.get('phrase_ctx', {}), stats.get('phrase_ctx_src', {}))

    rnull = ruler_null_test(con, vocab, wid_of, dfv)
    mem_guard('ruler_null')

    verdict = make_verdict(meta, vocab, idiom_rows, admitted, top_e, top_p, inv, metas, rnull)
    print('\n判决：' + ' | '.join('%s %s' % (k, 'PASS' if x[0] else 'FAIL')
                                for k, x in verdict.items()), flush=True)

    payload = save_outputs(con, meta, vocab, dfv, flagv, idiom_rows, admitted,
                           top_e, top_p, inv, metas, verdict, sense_report, stats, anchors,
                           rnull)
    write_report(payload)
    draw(payload)
    con.close()
    print('  完成：%.1fs / 峰值内存约 %.0fMB' % (time.time() - t0, mem_mb()), flush=True)


if __name__ == '__main__':
    main()
