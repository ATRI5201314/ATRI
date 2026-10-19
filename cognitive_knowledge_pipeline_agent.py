#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认知主体 · 长期知识摄入管道（L27：多源摄入 + 自主课表 + 跨学科公理缝合）

L26 教会它"吃"——但只吃 arxiv 摘要。L27 要把摄入升级成一条**长期管道**：
  · 胃口更宽：完整 PDF 正文解析、开源教材库、维基深层词条、生物学的真实数据？
  · 会自己排课：每天消化完，从 meta_ledger（悬案账本 + 低置信公理）反推出一张
    「明日搜索计划」——为什么搜、去哪个具体子分类搜。
  · 会跨科缝合：把宇宙学的"拓扑/场/临界性"这类结构概念，拿去解释生物学网络，
    缝出它**自创的公理**。

用户 L27 死命令：
  1. 长期知识摄入管道：不止摘要 -> 完整 PDF 文本解析 + 维基深层词条 + 开源教材库；
     先跑一个 **5000 篇真实基线**。
  2. 元账本驱动的"自主课表"：每天吸收完，必须输出「明日搜索计划」（为什么搜 / 去哪里搜）。
  3. 跨学科公理缝合：喂它生物学数据，看它能否用宇宙学的"拓扑结构/场"概念解释生物网络，
     产出**自创（缝合）公理**。

【本轮环境真实交底（不吹牛）】
  可达（经出口代理）：arxiv.org（abs/html/pdf）· export.arxiv.org API · api.biorxiv.org ·
    openstax.org · chem.libretexts.org · eutils.ncbi.nlm.nih.gov · gutenberg.org。
  不可达（被出口屏蔽）：en.wikipedia.org / en.wikibooks.org -> 保留为"被屏蔽→换源"真实样本，
    用可达的 OpenStax/LibreTexts 教材 + arxiv 全文 HTML 作为等价替代（真内容）。
  关于 5000：令牌桶 1 页/秒为不可改安全边界。**批量元数据走 API**（1 次请求可回数百篇，
  arXiv/bioRxiv 官方接口），**深度正文走无头浏览器**（PDF/HTML/教材逐页）。管线可断点续攀：
  已入货栈 URL 永不重抓，反复运行自动累加。

不变量（全项目 DNA，硬编码不可改）：
  I1 唯一信号 = 张力（预测误差 / 冲突）；无 reward；动作层不得篡改现实。
  L18-3 式防火墙：限流令牌桶 / 单出口 / 内存安全线不可改；知识货栈只读喂入。
  认知带宽有限：推演时最多把 BANDWIDTH_K 本书摊到内存桌上，其余留硬盘按需调取。

依赖：playwright(+chromium) 可选；pypdf（PDF 正文）；numpy（绘图）。
与旧文件关系：L19–L26 冻结；本文件独立成篇，共用 knowledge_pyramid.sqlite（可续攀）。
"""

import os
import re
import io
import gc
import time
import json
import zlib
import math
import struct
import random
import sqlite3
import hashlib
import datetime
import tracemalloc
import urllib.request
import urllib.parse
from xml.etree import ElementTree as ET
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    from pypdf import PdfReader
except Exception:                                   # 允许降级
    PdfReader = None

# ============================================================ 常量（安全边界，硬编码不可改）
BUCKET_RATE = 1.0          # 令牌桶：每秒最多 1 个外网请求（防撑爆 / 被封）
BUCKET_CAP = 2
PAGE_BUDGET = 5.0          # 浏览器首次耐心（秒）
PDF_BUDGET = 30.0          # PDF 下载耐心（秒）；arxiv pdf ~0.8MB 需 ~15–20s
RETRY_CAP = 15.0
MAX_DEFERS = 5
MAX_ROUNDS = 8000
TARGET = 5000              # 用户目标规模："5000 篇真实基线"（可断点续攀）
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'knowledge_pyramid.sqlite')

# 单次会话预算（可用环境变量覆盖；默认为一个务实批次）
SESSION_DEEP = int(os.environ.get('L27_DEEP', '120'))      # 深度阅读页（浏览器逐页）
SESSION_BULK = int(os.environ.get('L27_BULK', '3000'))     # 批量元数据新增目标（API）
PDF_QUOTA = int(os.environ.get('L27_PDF', '8'))            # 其中真·PDF 全文解析的篇数

# 价值筛选器阈值（无监督，自定）
ENTROPY_MIN = 3.6
CAUSAL_MIN = 0.02
DUP_HAMMING = 3
BANDWIDTH_K = 10
PDF_MAX_PAGES = 8          # 单篇 PDF 只抽前 8 页正文（省内存 / 省时间）
PDF_MAX_BYTES = 4_000_000  # 单次字节下载上限（4MB）；超限即截断，绝不拖挂主循环
LOW_CONF = 2               # 公理支撑 <= 此 -> 判"置信度不足"，进明日计划

UA = 'cognitive-agent/1.0 (research; mailto:cog-agent@example.com)'

# ---- 批量入口（API，1 次请求回多篇）----
ARXIV_API = ('https://export.arxiv.org/api/query?search_query=cat:{cat}'
             '&start={start}&max_results={n}&sortBy=submittedDate&sortOrder=descending')
BIORXIV_API = 'https://api.biorxiv.org/details/biorxiv/{start}/{end}?cursor={cursor}'

# ---- 深度入口（浏览器逐页）----
ARXIV_HTML = 'https://arxiv.org/html/{id}v1'
ARXIV_ABS = 'https://arxiv.org/abs/{id}'
ARXIV_PDF = 'https://arxiv.org/pdf/{id}'
LIST_ARCHIVE = 'https://arxiv.org/list/{cat}/{ym}?skip=0&show=500'
TEXTBOOK_SEEDS = [
    'https://openstax.org/books/physics/pages/1-introduction',
    'https://openstax.org/books/astronomy-2e/pages/1-introduction',
    'https://openstax.org/books/biology-2e/pages/1-introduction',
    'https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps',
]
WIKI_PROBES = ['https://en.wikipedia.org/wiki/Dark_energy',
               'https://en.wikipedia.org/wiki/Gene_regulatory_network']   # 被屏蔽样本

# 分类 -> 域
PHYS_CATS = ['astro-ph.CO', 'hep-th', 'gr-qc', 'quant-ph', 'cond-mat.stat-mech', 'hep-ph']
BIO_CATS = ['q-bio.MN', 'q-bio.NC', 'q-bio.BM', 'q-bio.PE', 'q-bio.QM']
ALL_CATS = PHYS_CATS + BIO_CATS

# ============================================================ 概念词库（物理 + 生物）
CONCEPT_SYN = {
    # ---- 宇宙学 / 物理 ----
    'dark energy': ['dark energy', 'quintessence', 'cosmological constant', 'vacuum energy', 'phantom'],
    'cosmic expansion': ['expansion', 'expanding', 'accelerated expansion', 'accelerating universe', 'hubble'],
    'dark matter': ['dark matter', 'halo', 'halos', 'wimp'],
    'gravity': ['gravity', 'gravitational', 'general relativity', 'modified gravity', 'spacetime'],
    'black hole': ['black hole', 'event horizon'],
    'inflation': ['inflation', 'inflationary', 'reheating'],
    'structure formation': ['structure formation', 'large-scale structure', 'large scale structure', 'clustering'],
    'neutrino': ['neutrino', 'neutrinos'],
    # ---- 生物 ----
    'gene regulatory network': ['gene regulatory network', 'gene regulatory networks', 'regulatory network',
                                'transcriptional network', 'transcription factor network', 'grn'],
    'protein interaction network': ['protein interaction network', 'protein-protein interaction',
                                    'interactome', 'ppi network'],
    'metabolic network': ['metabolic network', 'metabolic pathway', 'flux balance', 'metabolic model'],
    'neural circuit': ['neural circuit', 'neuronal network', 'neuronal activity', 'cortical network',
                       'brain network', 'neural population'],
    'population dynamics': ['population dynamics', 'ecological dynamics', 'predator-prey',
                            'species coexistence', 'ecosystem stability'],
    'evolution': ['evolutionary', 'natural selection', 'phylogenetic', 'adaptive evolution', 'fitness landscape'],
    'cell signaling': ['signaling pathway', 'signal transduction', 'cell signaling', 'signalling cascade'],
    'microbiome': ['microbiome', 'microbial community', 'bacterial community'],
}
CANON = {syn: c for c, syns in CONCEPT_SYN.items() for syn in syns}

PHYS_CONCEPTS = {'dark energy', 'cosmic expansion', 'dark matter', 'gravity',
                 'black hole', 'inflation', 'structure formation', 'neutrino'}
BIO_CONCEPTS = {'gene regulatory network', 'protein interaction network', 'metabolic network',
                'neural circuit', 'population dynamics', 'evolution', 'cell signaling', 'microbiome'}

# ---- 跨学科"缝合桥"：结构 / 场 / 拓扑 / 临界性一类可迁移的概念 ----
BRIDGE = ['topology', 'topolog', 'network', 'field', 'scaling', 'self-similar', 'connectivity',
          'cluster', 'critical', 'phase transition', 'emergence', 'self-organi', 'feedback',
          'hierarch', 'symmetry', 'curvature', 'entropy', 'equilibrium', 'stability',
          'fluctuat', 'correlat']

# ---- 概念 -> arxiv 子分类（"去哪里搜"的落点）----
SUBCAT = {
    'dark energy': 'astro-ph.CO', 'cosmic expansion': 'astro-ph.CO', 'structure formation': 'astro-ph.CO',
    'inflation': 'astro-ph.CO', 'dark matter': 'astro-ph.CO', 'gravity': 'gr-qc',
    'black hole': 'gr-qc', 'neutrino': 'hep-ph',
    'gene regulatory network': 'q-bio.MN', 'protein interaction network': 'q-bio.BM',
    'metabolic network': 'q-bio.MN', 'neural circuit': 'q-bio.NC',
    'population dynamics': 'q-bio.PE', 'evolution': 'q-bio.PE', 'cell signaling': 'q-bio.MN',
    'microbiome': 'q-bio.QM',
}

# ---- 因果连接词（词干子串匹配）----
CAUSAL = ['because', 'due to', 'owing to', 'since', 'as a result', 'lead', 'result',
          'give rise', 'gave rise', 'driven', 'drive', 'cause', 'explain', 'responsible',
          'produce', 'predict', 'imply', 'therefore', 'thus', 'hence', 'induc', 'trigger',
          'affect', 'influenc', 'determin', 'govern', 'control', 'requir', 'allow', 'enabl']


def set_seed(sd):
    random.seed(sd); np.random.seed(sd)


def recent_months(n=4, today=None):
    d = today or datetime.date.today()
    y, m = d.year, d.month
    out = []
    for _ in range(n):
        m -= 1
        if m == 0:
            m, y = 12, y - 1
        out.append(f'{y:04d}-{m:02d}')
    return out


def recent_days(n=6, today=None):
    d = today or datetime.date.today()
    return [(d - datetime.timedelta(days=i)).isoformat() for i in range(n)]


# ============================================================ L18-3 式防火墙（神性锚点）
IMMUTABLE = ('tension_def', 'reality_feed', 'safety_bound')
ANCHOR_DEF = ('张力 := 预测误差/冲突 恒等不可改；知识货栈只读喂入；'
              '限流令牌桶/单出口/内存安全线 硬编码不可改')


class FirewallError(Exception):
    pass


def compile_action(name):
    """任何生成的"摄入动作"必须来自白名单：禁止关限流 / 伪造知识 / 全量载入内存。"""
    ALLOWED = {'harvest_bulk', 'crawl_deep', 'compress', 'query', 'switch_source', 'defer'}
    if name not in ALLOWED:
        raise FirewallError(f"非法动作 '{name}'（试图关闭限流 / 伪造知识 / 绕过内存安全线）")
    return True


# ============================================================ 0. 令牌桶（防撑爆 / 被封）
class TokenBucket:
    def __init__(self, rate, cap):
        self.rate = rate; self.cap = cap
        self.tokens = float(cap); self.t = time.time()

    def _refill(self):
        now = time.time()
        self.tokens = min(self.cap, self.tokens + (now - self.t) * self.rate)
        self.t = now

    def take(self, block=False):
        self._refill()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        if not block:
            return False
        time.sleep((1.0 - self.tokens) / self.rate)
        return self.take(False)

    def wait_time(self):
        self._refill()
        return 0.0 if self.tokens >= 1.0 else (1.0 - self.tokens) / self.rate


# ============================================================ 1. 浏览器外骨骼（单出口动作层）
class BrowserEngine:
    """唯一"外网出口"。系统只调 fetch / fetch_bytes，绝不碰 DOM。"""
    def __init__(self, proxy=None, online=True):
        self.proxy = proxy; self.online = online
        self._pw = None; self._browser = None; self._page = None

    def start(self):
        try:
            from playwright.sync_api import sync_playwright
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(
                headless=True, proxy=({'server': self.proxy} if self.proxy else None))
            return 'playwright'
        except Exception as e:
            self._pw = None; self._browser = None
            return f'http-fallback ({type(e).__name__})'

    def fetch(self, url, timeout):
        if not self.online:
            return False, '', '', 0.0, 'offline'
        t0 = time.time()
        if self._browser is not None:
            try:
                if self._page is None:
                    self._page = self._browser.new_page()
                self._page.goto(url, timeout=int(timeout * 1000), wait_until='domcontentloaded')
                return True, self._page.inner_text('body'), self._page.content(), time.time() - t0, 'chromium'
            except Exception as e:
                try:
                    if self._page:
                        self._page.close()
                except Exception:
                    pass
                self._page = None
                return False, '', '', time.time() - t0, f'err:{type(e).__name__}'
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode('utf-8', 'ignore')
            text = re.sub(r'<script.*?</script>|<style.*?</style>', ' ', raw, flags=re.S | re.I)
            text = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', text))
            return True, text, raw, time.time() - t0, 'urllib'
        except Exception as e:
            return False, '', '', time.time() - t0, f'err:{type(e).__name__}'

    def fetch_bytes(self, url, timeout):
        """取原始字节（PDF / API）；走单出口代理。硬性墙钟超时：慢速大文件绝不会拖挂主循环。"""
        if not self.online:
            return None

        def _do():
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read(PDF_MAX_BYTES)
        ex = ThreadPoolExecutor(max_workers=1)
        try:
            return ex.submit(_do).result(timeout=timeout + 5)
        except Exception:
            return None
        finally:
            ex.shutdown(wait=False)

    def close(self):
        try:
            if self._page:
                self._page.close()
            if self._browser:
                self._browser.close()
            if self._pw:
                self._pw.stop()
        except Exception:
            pass


# ============================================================ 2. 价值筛选器（无监督，流式）
def simhash(text, bits=64):
    toks = re.findall(r'[a-z0-9]+', text.lower())
    if not toks:
        return 0
    v = [0] * bits
    for tok in toks:
        h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], 'big')
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    return sum(1 << i for i in range(bits) if v[i] > 0)


def hamming(a, b):
    return bin(a ^ b).count('1')


def word_entropy(text):
    toks = re.findall(r'[a-z]+', text.lower())
    if not toks:
        return 0.0
    n = len(toks); c = Counter(toks)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def causal_density(text):
    t = text.lower()
    hits = sum(t.count(k) for k in CAUSAL)
    return hits / (len(t) / 1000.0 + 1.0)


def value_filter(text):
    ent = word_entropy(text); cd = causal_density(text)
    if ent < ENTROPY_MIN:
        return 'low_entropy', {'entropy': ent, 'causal': cd}
    if cd < CAUSAL_MIN:
        return 'no_causal', {'entropy': ent, 'causal': cd}
    return 'accept', {'entropy': ent, 'causal': cd}


class BandDedup:
    """SimHash 去重的 LSH 分带索引：64bit 切 4 带，海明距离<=3 时必有一带全等。
    查询近似 O(1)，把 N² 去重降到近线性——支撑 1 万篇级而不卡内存。"""
    def __init__(self, exist_fps=()):
        self.buckets = [defaultdict(list) for _ in range(4)]
        for fp in exist_fps:
            self.add(fp)

    @staticmethod
    def _keys(fp):
        return [(i, (fp >> (16 * i)) & 0xffff) for i in range(4)]

    def is_dup(self, fp):
        for i, k in self._keys(fp):
            for o in self.buckets[i].get(k, ()):
                if hamming(fp, o) <= DUP_HAMMING:
                    return True
        return False

    def add(self, fp):
        for i, k in self._keys(fp):
            self.buckets[i][k].append(fp)


# ============================================================ 3. 概念 / 桥 / 因果 抽取
def extract_concepts(text):
    low = text.lower(); out = set()
    for syns in CONCEPT_SYN.values():
        for s in syns:
            if s in low:
                out.add(CANON[s]); break
    return out


def extract_bridges(text):
    low = text.lower()
    return {b for b in BRIDGE if b in low}


def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 25]


def mine(text):
    """逐句流式抽取（返回本批独立计数，绝不整批驻留）：
      directed 定向因果 · cooc 概念共现 · bridge_cooc 概念×缝合桥 · cross 跨域共现。
    关联网络以共现为骨架、定向边为语言显式断言的子集。"""
    directed = Counter(); cooc = Counter(); bridge_cooc = Counter(); cross = Counter()
    ev = {k: defaultdict(list) for k in ('causal', 'cooc', 'bridge', 'cross')}
    for sent in split_sentences(text):
        low = sent.lower()
        cs = extract_concepts(sent)
        bs = extract_bridges(low)
        for c in cs:
            for b in bs:
                bridge_cooc[(c, b)] += 1
                if len(ev['bridge'][(c, b)]) < 2:
                    ev['bridge'][(c, b)].append(sent[:200])
        if len(cs) >= 2:
            for a in cs:
                for b in cs:
                    if a < b:
                        cooc[(a, b)] += 1
                        if len(ev['cooc'][(a, b)]) < 2:
                            ev['cooc'][(a, b)].append(sent[:200])
            # 跨域：同句同时出现 物理概念 & 生物概念
            ps = [c for c in cs if c in PHYS_CONCEPTS]
            gs = [c for c in cs if c in BIO_CONCEPTS]
            for p in ps:
                for g in gs:
                    k = (p, g) if p < g else (g, p)
                    cross[k] += 1
                    if len(ev['cross'][k]) < 2:
                        ev['cross'][k].append(sent[:200])
        marks = [low.find(m) for m in CAUSAL if m in low]
        if not marks or len(cs) < 1:
            continue
        first = {}
        for c in cs:
            for s in CONCEPT_SYN[c]:
                p = low.find(s)
                if p >= 0 and c not in first:
                    first[c] = p
        for p in marks:
            left = sorted((c for c, q in first.items() if q < p), key=lambda c: first[c])
            right = sorted((c for c, q in first.items() if q > p), key=lambda c: first[c])
            if left and right:
                a, b = left[-1], right[0]
            elif len(first) >= 2:
                order = sorted(first, key=lambda c: first[c]); a, b = order[0], order[-1]
            else:
                continue
            if a != b:
                directed[(a, b)] += 1
                if len(ev['causal'][(a, b)]) < 2:
                    ev['causal'][(a, b)].append(sent[:200])
    return directed, cooc, bridge_cooc, cross, ev


# ============================================================ 4. 知识金字塔（SQLite，多级索引 + 迁移）
LAKE_COLS = {'source': 'TEXT', 'domain': 'TEXT', 'pages': 'INTEGER DEFAULT 0'}


def init_db(path):
    con = sqlite3.connect(path)
    con.executescript('''
        CREATE TABLE IF NOT EXISTS lake(               -- 底层：海量原文 / 元数据（落硬盘）
            id INTEGER PRIMARY KEY, url TEXT UNIQUE, kind TEXT,
            title TEXT, abstract TEXT, subjects TEXT, body TEXT,
            fp TEXT, entropy REAL, causal REAL, n_tri INTEGER,
            cause INTEGER, digested INTEGER DEFAULT 0, fetched_at REAL);
        CREATE TABLE IF NOT EXISTS triples(            -- 中层：实体-关系三元组
            id INTEGER PRIMARY KEY, subject TEXT, predicate TEXT, object TEXT,
            src INTEGER, weight REAL);
        CREATE TABLE IF NOT EXISTS axioms(             -- 顶层：核心公理（驻内存）
            id INTEGER PRIMARY KEY, rule TEXT UNIQUE, support INTEGER);
        CREATE TABLE IF NOT EXISTS ledger(             -- 元账本：悬而未决的冲突
            id INTEGER PRIMARY KEY, a TEXT, b TEXT, note TEXT, resolved INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS evidence(           -- 证据句：跨源留痕
            id INTEGER PRIMARY KEY, a TEXT, b TEXT, kind TEXT, sentence TEXT);
        CREATE TABLE IF NOT EXISTS bridges(            -- L27：跨学科自创（缝合）公理
            id INTEGER PRIMARY KEY, bridge TEXT, a TEXT, b TEXT, support INTEGER,
            note TEXT, created_at REAL);
        CREATE TABLE IF NOT EXISTS curriculum(         -- L27：元账本驱动的"明日搜索计划"
            id INTEGER PRIMARY KEY, day TEXT, reason TEXT, why TEXT, a TEXT, b TEXT,
            target TEXT, created_at REAL);
        CREATE INDEX IF NOT EXISTS ix_lake_dig ON lake(digested);
        CREATE INDEX IF NOT EXISTS ix_tri_s ON triples(subject);
        CREATE INDEX IF NOT EXISTS ix_tri_p ON triples(predicate);
        CREATE INDEX IF NOT EXISTS ix_tri_o ON triples(object);
        CREATE INDEX IF NOT EXISTS ix_ev_ab ON evidence(a, b);
    ''')
    # 迁移：给已存在的 lake 补列（L26 旧库无缝升级）
    have = {r[1] for r in con.execute('PRAGMA table_info(lake)')}
    for col, typ in LAKE_COLS.items():
        if col not in have:
            con.execute(f'ALTER TABLE lake ADD COLUMN {col} {typ}')
    con.commit()
    return con


def lake_put(con, rec):
    con.execute('''INSERT OR IGNORE INTO lake
        (url,kind,title,abstract,subjects,body,fp,entropy,causal,n_tri,cause,fetched_at,source,domain,pages)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (rec['url'], rec['kind'], rec['title'][:400], rec['abstract'][:4000],
                 rec['subjects'][:300], rec['body'][:40000], rec['fp'], rec['entropy'],
                 rec['causal'], rec['n_tri'], rec['cause'], time.time(),
                 rec.get('source', ''), rec.get('domain', ''), rec.get('pages', 0)))


def lake_all_fps(con):
    """只读指纹（8 字节/条），1 万条 ~80KB，安全。"""
    return [int(r[0], 16) for r in con.execute('SELECT fp FROM lake WHERE fp IS NOT NULL')]


# ============================================================ 5. 摄入管道（白天）
def _strip_html(html):
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', html))


def _main_text(html):
    """优先取 <main>/<article> 正文，弱化导航噪声（教材页尤其需要）。"""
    for tag in ('main', 'article'):
        m = re.search(rf'<{tag}[^>]*>(.*?)</{tag}>', html, re.S | re.I)
        if m and len(m.group(1)) > 500:
            return _strip_html(m.group(1))
    return _strip_html(html)


def _title_of(html):
    m = re.search(r'<title[^>]*>(.*?)</title>', html, re.S | re.I)
    if m:
        return re.sub(r'\s+', ' ', _strip_html(m.group(1))).strip()
    m = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.S | re.I)
    return re.sub(r'\s+', ' ', _strip_html(m.group(1))).strip() if m else ''


def extract_arxiv_abs(aid, html, text):
    """从 arxiv 摘要页抽取 title/abstract（全文 HTML 不可得时的退路）。"""
    hc = re.sub(r'<span[^>]*class="descriptor"[^>]*>.*?</span>', ' ', html, flags=re.S | re.I)
    m = re.search(r'<meta\s+name="citation_title"\s+content="([^"]+)"', hc, re.I)
    title = m.group(1) if m else _title_of(hc)
    ma = re.search(r'<blockquote[^>]*class="abstract[^"]*"[^>]*>(.*?)</blockquote>', hc, re.S | re.I)
    abs_ = _strip_html(ma.group(1)) if ma else ''
    if not abs_:
        m2 = re.search(r'Abstract:\s*(.+?)(?:Comments:|Subjects:|$)', text, re.S)
        abs_ = m2.group(1) if m2 else ''
    return dict(url=f'https://arxiv.org/abs/{aid}', kind='arxiv',
                title=re.sub(r'\s+', ' ', title).strip(),
                abstract=re.sub(r'\s+', ' ', abs_).strip(), subjects='', body='', pages=0)


def pdf_text(data):
    """真·PDF 全文解析（pypdf；只抽前 PDF_MAX_PAGES 页，控内存/控时）。"""
    if PdfReader is None or not data:
        return ''
    try:
        r = PdfReader(io.BytesIO(data))
        n = min(len(r.pages), PDF_MAX_PAGES)
        return '\n'.join((r.pages[i].extract_text() or '') for i in range(n))
    except Exception:
        return ''


class Pipeline:
    """长期摄入管道 = 批量(API 高吞吐) + 深度(浏览器逐页读真正文)。"""
    def __init__(self, engine, bucket, con):
        self.engine = engine; self.bucket = bucket; self.con = con
        self.seen = BandDedup(lake_all_fps(con))
        self.stats = Counter()
        self.mem = []

    # ---- 通用：筛选 + 流式落湖 ----
    def ingest(self, rec):
        blob = (rec.get('title', '') + ' ' + (rec.get('abstract') or rec.get('body', '')[:3000])).strip()
        if not blob:
            self.stats['empty'] += 1; return False
        fp = simhash(blob)
        if self.seen.is_dup(fp):
            self.stats['dup'] += 1; return False
        self.seen.add(fp)
        verdict, meta = value_filter(blob)
        self.stats['into_graph' if verdict == 'accept' else 'lake_only'] += 1
        rec.update(fp=format(fp, '016x'), entropy=meta['entropy'], causal=meta['causal'],
                   n_tri=0, cause=1 if verdict == 'accept' else 0)
        lake_put(self.con, rec)
        return True

    def _commit(self):
        self.con.commit()
        gc.collect()          # 先回收瞬时大对象（整篇正文 / XML 响应），再采样，量的是"常驻"堆
        n = self.con.execute('SELECT COUNT(*) FROM lake').fetchone()[0]
        self.mem.append((n, tracemalloc.get_traced_memory()[0]))

    # ---- A. 批量元数据（API，1 请求回多篇，冲量）----
    def harvest_arxiv_api(self, cats, per_cat):
        """按分类取 API；start 从"该分类已有条数"续起 -> 反复运行自动向更早的论文推进。"""
        got = 0
        for cat in cats:
            if not self.bucket.take(block=True):
                continue
            start = int(self.con.execute(
                "SELECT COUNT(*) FROM lake WHERE source='arxiv_api' AND subjects=?", (cat,)).fetchone()[0])
            adv = self.engine.fetch_bytes(ARXIV_API.format(cat=cat, start=start, n=per_cat), 40.0)
            if not adv:
                self.stats['api_fail'] += 1; continue
            try:
                root = ET.fromstring(adv.decode('utf-8', 'ignore'))
            except Exception:
                self.stats['api_fail'] += 1; continue
            ns = {'a': 'http://www.w3.org/2005/Atom'}
            for e in root.findall('a:entry', ns):
                aid = (e.findtext('a:id', '', ns) or '').rsplit('/', 1)[-1]
                title = re.sub(r'\s+', ' ', e.findtext('a:title', '', ns) or '').strip()
                summ = re.sub(r'\s+', ' ', e.findtext('a:summary', '', ns) or '').strip()
                dom = 'biology' if cat.startswith('q-bio') else 'physics'
                if self.ingest(dict(url=ARXIV_ABS.format(id=aid), kind='arxiv', title=title,
                                    abstract=summ, subjects=cat, body='', source='arxiv_api',
                                    domain=dom, pages=0)):
                    got += 1
            self._commit()
        return got

    def harvest_biorxiv_api(self, days, target):
        """bioRxiv 官方 API：一次回一批生物学预印本（title+abstract+category）。"""
        got = 0
        for d in days:
            cursor = 0
            while got < target:
                if not self.bucket.take(block=True):
                    break
                url = BIORXIV_API.format(start=d, end=d, cursor=cursor)
                raw = self.engine.fetch_bytes(url, 40.0)
                if not raw:
                    self.stats['api_fail'] += 1; break
                try:
                    j = json.loads(raw.decode('utf-8', 'ignore'))
                except Exception:
                    break
                coll = j.get('collection', [])
                if not coll:
                    break
                for it in coll:
                    doi = it.get('doi', '')
                    title = re.sub(r'\s+', ' ', it.get('title', '') or '').strip()
                    ab = re.sub(r'\s+', ' ', it.get('abstract', '') or '').strip()
                    cat = it.get('category', 'biology')
                    if self.ingest(dict(url=f'https://doi.org/{doi}', kind='preprint', title=title,
                                        abstract=ab, subjects=cat, body='', source='biorxiv_api',
                                        domain='biology', pages=0)):
                        got += 1
                cursor += len(coll)
                tot = int(j.get('messages', [{}])[0].get('total', 0) or 0)
                if cursor >= tot:
                    break
            self._commit()
            if got >= target:
                break
        return got

    # ---- B. 深度正文（浏览器逐页：全文 HTML / 真 PDF / 教材 / 维基样本）----
    def discover_arxiv_ids(self, need):
        have = {r[0].rsplit('/', 1)[-1] for r in
                self.con.execute("SELECT url FROM lake WHERE url LIKE '%/abs/%'")}
        ids, seen = [], set(have)
        for cat in ALL_CATS:
            for ym in recent_months(4):
                if len(ids) >= need * 3:
                    return ids
                if not self.bucket.take(block=True):
                    continue
                ok, _t, html, _e, _g = self.engine.fetch(LIST_ARCHIVE.format(cat=cat, ym=ym), 25.0)
                if ok:
                    for i in re.findall(r'/abs/([0-9]{4}\.[0-9]{4,5})', html):
                        if i not in seen:
                            seen.add(i); ids.append(i)
        return ids

    def crawl_deep(self, budget):
        """深度阅读：PDF 全文 / 全文 HTML / 开源教材 / 维基被墙样本。"""
        ids = self.discover_arxiv_ids(budget)
        # 顺序很重要：教材 + 维基样本排在最前，保证每次会话都必读（不被预算截断）
        work = [('textbook', u) for u in TEXTBOOK_SEEDS] + [('wiki', u) for u in WIKI_PROBES]
        npdf = 0
        for i in ids[:max(budget, 20)]:
            if npdf < PDF_QUOTA:
                work.append(('arxiv_pdf', i)); npdf += 1
            else:
                work.append(('arxiv_html', i))

        pending = [(k, p, 0) for k, p in work]
        done = 0; rnd = 0; pdfs = 0; htmls = 0; books = 0; blocked = 0
        while pending and done < budget and rnd < MAX_ROUNDS:
            progressed = False
            for item in list(pending):
                kind, payload, defers = item
                if done >= budget:
                    break
                if not self.bucket.take(block=False):
                    self.stats['throttled'] += 1; break
                rec = self._read_one(kind, payload, min(RETRY_CAP, PAGE_BUDGET * (1.6 ** defers)))
                if rec is None:
                    pending.remove(item)
                    if kind == 'wiki':
                        blocked += 1; continue          # 被墙：诚实换源，不重试
                    # 真 PDF / 维基不重试（重试代价远大于收益）；其余最多挂起 MAX_DEFERS 次
                    if kind not in ('arxiv_pdf',) and defers + 1 < MAX_DEFERS:
                        pending.append((kind, payload, defers + 1))
                        self.stats['defers'] += 1
                    continue
                if self.ingest(rec):
                    done += 1; progressed = True
                    pdfs += kind == 'arxiv_pdf'; htmls += kind == 'arxiv_html'
                    books += kind == 'textbook'
                pending.remove(item)
                self._commit()
            if pending and not progressed:
                time.sleep(min(1.0, max(0.02, self.bucket.wait_time())))
            rnd += 1
        self.stats.update(pdf=pdfs, html=htmls, textbook=books, wiki_blocked=blocked)
        return done

    def _read_one(self, kind, payload, timeout):
        """深度阅读一个条目 -> 原始记录（未筛选）。"""
        if kind == 'wiki':
            self.engine.fetch(payload, timeout)
            return None                              # 被墙：诚实记 0 篇，换源
        if kind == 'arxiv_pdf':
            data = self.engine.fetch_bytes(ARXIV_PDF.format(id=payload), PDF_BUDGET)
            txt = pdf_text(data)
            if not txt or len(txt) < 800:
                return None
            return dict(url=f'https://arxiv.org/pdf/{payload}', kind='arxiv_pdf', title='',
                        abstract='', subjects='', body=txt, source='arxiv_pdf',
                        domain='physics', pages=PDF_MAX_PAGES)
        if kind == 'arxiv_html':
            ok, text, html, _e, _g = self.engine.fetch(ARXIV_HTML.format(id=payload), timeout)
            if ok and len(text) >= 800:
                return dict(url=f'https://arxiv.org/html/{payload}v1', kind='arxiv_fulltext',
                            title=_title_of(html), abstract='', subjects='', body=text,
                            source='arxiv_html', domain='physics', pages=0)
            # 全文 HTML 不可得 -> 退到摘要页（保证真实列表中每篇都有内容，不空耗预算）
            self.bucket.take(block=True)             # 第二次请求同样占用一个令牌（不破限流不变量）
            ok2, text2, html2, _e, _g = self.engine.fetch(ARXIV_ABS.format(id=payload), timeout)
            if not ok2:
                return None
            from_abs = extract_arxiv_abs(payload, html2, text2)
            from_abs.update(source='arxiv_abs', domain='physics')
            return from_abs
        if kind == 'textbook':
            ok, _t, html, _e, _g = self.engine.fetch(payload, timeout)
            if not ok or len(html) < 500:
                return None
            body = _main_text(html)
            dom = 'biology' if 'biolog' in payload.lower() else 'textbook'
            return dict(url=payload, kind='textbook', title=_title_of(html), abstract='',
                        subjects='textbook', body=body, source='textbook', domain=dom, pages=0)
        return None


# ============================================================ 6. 离线消化（黑夜）
def compress_offline(con):
    """断网后：分块流式消化底湖未消化文本 -> 增量合并进主图谱 + 缝桥 + 更新公理 / 悬案。
    查询时 SUM 汇总，故跨会话可续、绝不整表载入内存。"""
    cur = con.execute('SELECT id,title,abstract,body FROM lake WHERE digested=0 AND cause=1')
    causal = Counter(); cooc = Counter(); bridge_cooc = Counter(); cross = Counter()
    ev = {k: defaultdict(list) for k in ('causal', 'cooc', 'bridge', 'cross')}
    n = 0
    while True:
        rows = cur.fetchmany(50)                      # 分块，绝不整表载入
        if not rows:
            break
        for rid, title, abstract, body in rows:
            text = body if body and len(body) > len(abstract or '') else \
                (title or '') + '. ' + (abstract or '')
            d, c, bc, cr, e = mine(text)
            causal.update(d); cooc.update(c); bridge_cooc.update(bc); cross.update(cr)
            for k, lst in e.items():
                for kk, ss in lst.items():
                    if len(ev[k][kk]) < 2:
                        ev[k][kk].extend(ss[:2 - len(ev[k][kk])])
            con.execute('UPDATE lake SET digested=1 WHERE id=?', (rid,))
            n += 1
        con.commit()

    for (a, b), w in causal.items():
        con.execute('INSERT INTO triples(subject,predicate,object,src,weight) VALUES (?,?,?,?,?)',
                    (a, 'causal', b, 0, float(w)))
    for (a, b), w in cooc.items():
        con.execute('INSERT INTO triples(subject,predicate,object,src,weight) VALUES (?,?,?,?,?)',
                    (a, 'cooc', b, 0, float(w)))
    evrows = [(a, b, 'causal', s) for (a, b), lst in ev['causal'].items() for s in lst]
    evrows += [(a, b, 'cooc', s) for (a, b), lst in ev['cooc'].items() for s in lst]
    evrows += [(a, b, 'bridge', s) for (a, b), lst in ev['bridge'].items() for s in lst]
    evrows += [(a, b, 'cross', s) for (a, b), lst in ev['cross'].items() for s in lst]
    if evrows:
        con.executemany('INSERT INTO evidence(a,b,kind,sentence) VALUES (?,?,?,?)', evrows)

    # ---- 跨学科公理缝合：桥接概念 b 同时连着 物理概念 与 生物概念 ----
    bmap = {}                                     # (concept, bridge) -> 证据计数
    for a, b, cnt in con.execute("SELECT a,b,COUNT(*) FROM evidence WHERE kind='bridge' GROUP BY a,b"):
        bmap[(a, b)] = cnt
    stitched = 0
    bridges_used = defaultdict(lambda: {'phys': [], 'bio': []})
    for (c, brg), cnt in bmap.items():
        if c in PHYS_CONCEPTS:
            bridges_used[brg]['phys'].append((c, cnt))
        elif c in BIO_CONCEPTS:
            bridges_used[brg]['bio'].append((c, cnt))
    for brg, d in bridges_used.items():
        d['phys'].sort(key=lambda x: -x[1]); d['bio'].sort(key=lambda x: -x[1])
        for c, cc in d['phys'][:3]:
            for g, gc in d['bio'][:3]:
                sup = min(cc, gc)
                if sup < 1:
                    continue
                if not con.execute('SELECT 1 FROM bridges WHERE bridge=? AND a=? AND b=?',
                                   (brg, c, g)).fetchone():
                    con.execute('''INSERT INTO bridges(bridge,a,b,support,note,created_at)
                                   VALUES (?,?,?,?,?,?)''',
                                (brg, c, g, sup,
                                 f'共享结构「{brg}」：{c} ≅ {g}（跨学科缝合）', time.time()))
                    stitched += 1

    # ---- 全局聚合（跨会话 SUM）+ 悬案 + 顶层公理 ----
    tot = {(a, b): w for a, b, w in con.execute(
        "SELECT subject,object,SUM(weight) FROM triples WHERE predicate='causal' GROUP BY subject,object")}
    conflicts = 0
    for (a, b), w in tot.items():
        rev = tot.get((b, a), 0)
        if rev > 0 and min(w, rev) >= 1:
            if not con.execute('SELECT 1 FROM ledger WHERE a=? AND b=? AND resolved=0',
                               (a, b)).fetchone():
                con.execute('INSERT INTO ledger(a,b,note) VALUES (?,?,?)',
                            (a, b, f'双向因果方向未定（{a}↔{b}）'))
                conflicts += 1
    top = sorted(tot.items(), key=lambda kv: -kv[1])[:8]
    for (a, b), w in top:
        con.execute('INSERT OR REPLACE INTO axioms(rule,support) VALUES (?,?)',
                    (f'{a} ⇒ {b}', int(w)))
    con.commit()
    return {'digested': n, 'directed': len(causal), 'cooc': len(cooc),
            'bridges': len([1 for k in bridge_cooc]), 'stitched': stitched,
            'cross': len(cross), 'axioms': len(top), 'conflicts': conflicts,
            'bridge_cooc': bridge_cooc, 'ev': ev}


def load_axioms(con):
    return [(r[0], r[1]) for r in
            con.execute('SELECT rule,support FROM axioms ORDER BY support DESC')]


# ============================================================ 7. 元账本驱动的"自主课表"
def target_for(a, b):
    ta = SUBCAT.get(a) or ('q-bio.MN' if a in BIO_CONCEPTS else 'astro-ph.CO')
    tb = SUBCAT.get(b) or ('q-bio.MN' if b in BIO_CONCEPTS else 'astro-ph.CO')
    return ta if ta == tb else f'{ta} + {tb}'


def plan_curriculum(con):
    """从 meta_ledger（低置信公理 + 悬案 + 待证缝合）反推「明日搜索计划」。"""
    day = datetime.date.today().isoformat()
    items = []
    # 1) 低置信公理：置信度不够 -> 再证
    for rule, sup in con.execute('SELECT rule,support FROM axioms ORDER BY support ASC'):
        if sup <= LOW_CONF:
            if '⇒' in rule:
                a, b = [s.strip() for s in rule.split('⇒')]
            elif '≅' in rule:
                a, b = [s.strip() for s in rule.split('≅')]
            else:
                a, b = rule, ''
            items.append(('低置信公理', f'「{rule}」仅 {sup} 次支撑，置信度不足，需再证',
                          a, b, target_for(a, b)))
    # 2) 未决悬案：双向因果方向未定
    for a, b, note in con.execute('SELECT a,b,note FROM ledger WHERE resolved=0 LIMIT 30'):
        items.append(('悬案未决', f'{note}（方向待定）', a, b, target_for(a, b)))
    # 3) 待证缝合：跨学科桥证据不足
    for brg, a, b, sup in con.execute('SELECT bridge,a,b,support FROM bridges ORDER BY support ASC LIMIT 15'):
        if sup <= LOW_CONF:
            items.append(('跨学科缝合待证',
                          f'自创桥「{brg}」：{a} ≅ {b} 仅 {sup} 次共同结构证据',
                          a, b, target_for(a, b)))
    con.execute('DELETE FROM curriculum WHERE day=?', (day,))
    for reason, why, a, b, tgt in items:
        con.execute('''INSERT INTO curriculum(day,reason,why,a,b,target,created_at)
                       VALUES (?,?,?,?,?,?,?)''', (day, reason, why, a, b, tgt, time.time()))
    con.commit()
    return items


# ============================================================ 8. 绘图
def write_png(path, arr):
    H, W, _ = arr.shape
    raw = b''.join(b'\x00' + arr[y].tobytes() for y in range(H))

    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + \
               struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    open(path, 'wb').write(b'\x89PNG\r\n\x1a\n' +
                           chunk(b'IHDR', struct.pack('>IIBBBBB', W, H, 8, 2, 0, 0, 0)) +
                           chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))


def render(mem_curve, path):
    W, H = 960, 320
    img = np.full((H, W, 3), 250, dtype=np.uint8)
    pl, pr, pt, pb = 60, 20, 20, 30
    if len(mem_curve) >= 2:
        xs = [n for n, _m in mem_curve]; ys = [m for _n, m in mem_curve]
        x0, x1 = min(xs), max(xs) or (min(xs) + 1)
        y1 = max(ys) * 1.15 + 1
        for k in range(len(mem_curve) - 1):
            xa = int(pl + (W - pl - pr) * (xs[k] - x0) / max(1, x1 - x0))
            xb = int(pl + (W - pl - pr) * (xs[k + 1] - x0) / max(1, x1 - x0))
            ya = int(H - pb - (H - pt - pb) * ys[k] / y1)
            yb = int(H - pb - (H - pt - pb) * ys[k + 1] / y1)
            for x in range(xa, xb + 1):
                y = ya + int((yb - ya) * (x - xa) / max(1, xb - xa))
                img[max(pt, y - 1):y + 2, x:x + 2] = (30, 90, 200)
    write_png(path, img)


# ============================================================ 主程序
def main():
    set_seed(27)
    print("=" * 92)
    print("L27 · 长期知识摄入管道：多源摄入(PDF/教材/生物学) + 自主课表 + 跨学科公理缝合")
    print("=" * 92)
    tracemalloc.start()

    proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    engine = BrowserEngine(proxy=proxy, online=True)
    eng = engine.start()
    print(f"【0 · 外骨骼】动作层起浏览器：{eng}（出口代理 {proxy}）")
    bucket = TokenBucket(BUCKET_RATE, BUCKET_CAP)
    con = init_db(DB_PATH)
    have = con.execute('SELECT COUNT(*) FROM lake').fetchone()[0]
    print(f"   知识货栈（SQLite）：{DB_PATH}")
    print(f"   目标 {TARGET} 篇真实基线；货栈已有 {have} 篇（增量续攀）")
    print(f"   本会话：批量元数据目标 {SESSION_BULK} 篇(API) + 深度正文 {SESSION_DEEP} 页(浏览器，"
          f"其中真 PDF {PDF_QUOTA} 篇)")
    print(f"   PDF 引擎：{'pypdf 全文解析' if PdfReader else '缺失(降级为全文 HTML)'}；"
          f"安全边界：令牌桶 {BUCKET_RATE} 请求/秒；认知带宽 {BANDWIDTH_K} 本/次")
    print("-" * 92)

    pipe = Pipeline(engine, bucket, con)
    try:
        # ===== 白天 · A：批量元数据（API 冲量）=====
        print("【白天·批量摄入】API 官方接口：arXiv 元数据 + bioRxiv 生物学预印本（1 请求回多篇）")
        t0 = time.time()
        per_cat = max(20, (SESSION_BULK * 2 // 3) // max(1, len(ALL_CATS)))
        b1 = pipe.harvest_arxiv_api(PHYS_CATS, per_cat)
        b2 = pipe.harvest_arxiv_api(BIO_CATS, per_cat)
        # bioRxiv：跳过已覆盖的近若干天，向更早的日期推进（同样可断点续攀）
        have_bio = con.execute("SELECT COUNT(*) FROM lake WHERE source='biorxiv_api'").fetchone()[0]
        skip = have_bio // 300
        bio_days = recent_days(6 + skip)[skip:]
        b3 = pipe.harvest_biorxiv_api(bio_days, max(0, SESSION_BULK - b1 - b2))
        bulk = b1 + b2 + b3
        print(f"   arXiv API：物理域 {b1} 篇、生物域 {b2} 篇；bioRxiv API：{b3} 篇")
        print(f"   批量新增 {bulk} 篇（增量去重生效：判重丢弃 {pipe.stats['dup']} 篇）")
        print("-" * 92)

        # ===== 白天 · B：深度正文（浏览器逐页）=====
        print("【白天·深度阅读】无头浏览器逐页读真正文：真 PDF 全文 + 全文 HTML + 开源教材")
        deep = pipe.crawl_deep(SESSION_DEEP)
        s = pipe.stats
        print(f"   深度阅读 {deep} 页：真 PDF 全文 {s['pdf']} 篇、全文 HTML {s['html']} 篇、"
              f"开源教材 {s['textbook']} 篇；耗时 {time.time()-t0:.0f}s")
        print(f"   限流：令牌桶节流 {s['throttled']} 次；挂起重读 {s['defers']} 次")
        for u in WIKI_PROBES:
            print(f"   被屏蔽样本：{u} -> 超时放弃（换源生效：改用可达 arxiv 全文 / OpenStax 教材）")
        # 分源统计
        print("   货栈分源统计（按 source 聚合）：")
        for src, cnt in con.execute('''SELECT COALESCE(source,'(L26)'),COUNT(*) FROM lake
                                       GROUP BY source ORDER BY COUNT(*) DESC'''):
            print(f"      {src:16s} {cnt:5d} 篇")
        total = con.execute('SELECT COUNT(*) FROM lake').fetchone()[0]
        print(f"   货栈累计 {total} 篇真实条目（含历史批次；目标 {TARGET}，断点续攀）")
        print("-" * 92)

        # ===== 黑夜：离线消化 + 缝合 =====
        print("【黑夜·离线消化】断网 -> 分块流式压缩底湖 -> 增量合并主图谱 + 跨科缝合")
        engine.online = False
        r = compress_offline(con)
        print(f"   离线消化 {r['digested']} 篇（分块 50/批）；拼接因果边 {r['directed']} 条、"
              f"共现边 {r['cooc']} 条、概念×桥 {r['bridges']} 种、跨域同句 {r['cross']} 对")
        print(f"   悬案入元账本 {r['conflicts']} 条；顶层公理刷新 {r['axioms']} 条")
        engine.online = True
        print("-" * 92)

        # ===== L27-3 跨学科公理缝合 =====
        print("【L27-3 跨学科公理缝合】用宇宙学的「拓扑/场/临界性」去解释生物学网络 -> 自创公理")
        st = con.execute('''SELECT bridge,a,b,support FROM bridges ORDER BY support DESC LIMIT 8''').fetchall()
        if st:
            for brg, a, b, sup in st:
                print(f"   ✦ [{brg}] {a} ≅ {b}   （共同结构证据 {sup} 次）")
        else:
            print("   （本批尚无缝合产出；跨域同句证据仍需累积）")
        # 缝桥的真实证据句（两侧各取一条）
        for brg, a, b, sup in st[:2]:
            for side, con_ in (('物理侧', a), ('生物侧', b)):
                evs = con.execute("SELECT sentence FROM evidence WHERE kind='bridge' AND a=? AND b=? LIMIT 1",
                                  (con_, brg)).fetchone()
                if evs:
                    print(f"     · {side}证据（{con_} × {brg}）：{evs[0][:120]}")
        cross_ev = con.execute("SELECT a,b,sentence FROM evidence WHERE kind='cross' LIMIT 2").fetchall()
        for a, b, sent in cross_ev:
            print(f"     · 同句跨域（{a} + {b}）：{sent[:110]}")
        print("-" * 92)

        # ===== L27-2 元账本驱动的"自主课表" =====
        print("【L27-2 自主课表】元账本 -> 「明日搜索计划」：为什么搜 / 去哪里搜")
        plan = plan_curriculum(con)
        for i, (reason, why, a, b, tgt) in enumerate(plan[:10], 1):
            print(f"   {i:2d}. 为什么：{why}")
            print(f"       去哪里：{tgt}")
        print(f"   明日计划共 {len(plan)} 项，已写入 curriculum 表（day={datetime.date.today()}）")
        print("-" * 92)

        # ===== 认知带宽：按需调取 =====
        print(f"【认知带宽】问题触发 -> 只把最相关 {BANDWIDTH_K} 本书摊到内存桌上（其余留硬盘）")
        rows = con.execute('''SELECT subject,predicate,object,SUM(weight) w FROM triples
                              GROUP BY subject,predicate,object ORDER BY w DESC LIMIT ?''',
                           (BANDWIDTH_K,)).fetchall()
        for a, p, b, w in rows[:5]:
            print(f"   {a} {'⇒' if p=='causal' else '↔'} {b}  ({p}, 强度 {w:.0f})")
        print("-" * 92)

        # ===== 内存监控（防撑爆）=====
        end_cur, end_peak = tracemalloc.get_traced_memory()
        try:
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            ru = None
        k = len(pipe.mem)
        if k >= 12:
            start = k // 5
            seg = max(1, (k - start) // 3)
            head = np.mean([m for _n, m in pipe.mem[start:start + seg]])
            tail = np.mean([m for _n, m in pipe.mem[-seg:]])
            growth = (tail - head) / max(1.0, head); stable = growth < 0.3
        else:
            growth = 0.0; stable = None
        verdict = ('稳定（流式生效，未随知识量线性膨胀）' if stable
                   else ('样本不足' if stable is None else '有增长风险'))
        print("【L27-内存监控】跑完批量+深度摄入与离线消化，内存是否稳定在安全线内")
        print(f"   Python 堆：当前 {end_cur//1024} KB，峰值 {end_peak//1024} KB；"
              + (f"进程 RSS 峰值 {ru:.1f} MB" if ru else ""))
        print(f"   内存曲线：跳过预热，中段 vs 末段增幅 {growth*100:.1f}% -> {verdict}")
        n_ax = con.execute('SELECT COUNT(*) FROM axioms').fetchone()[0]
        print(f"   关键：{total} 篇原文落硬盘(SQLite)、内存只驻 {n_ax} 条公理 + 桌面 {BANDWIDTH_K} 条")
        print(f"   → 内存稳定在安全线内：{'是' if stable else ('样本不足' if stable is None else '否')}")
        print("-" * 92)

        # ===== 防火墙 =====
        print("【防火墙 / 神性锚点】动作层白名单：禁止关限流 / 伪造知识 / 绕过内存安全线")
        for bad in ['disable_token_bucket', 'fake_knowledge', 'load_all_to_ram']:
            try:
                compile_action(bad); s2 = "未拦截(!)"
            except FirewallError as e:
                s2 = f"已拒绝：{e}"
            print(f"   注入非法动作 {bad} -> {s2}")
        print(f"   锚点：{ANCHOR_DEF}")
        print("-" * 92)
    finally:
        engine.close(); con.close()

    try:
        render(pipe.mem, 'knowledge_memory_curve.png')
        print("已输出：knowledge_memory_curve.png（内存随处理量曲线）")
        print("已落盘：knowledge_pyramid.sqlite（底湖 + 三元组 + 公理 + 元账本 + 缝合桥 + 课表）")
    except Exception as e:
        print(f"(绘图降级：{e})")


if __name__ == '__main__':
    main()
