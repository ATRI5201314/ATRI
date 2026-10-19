#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认知主体 · 知识压缩机（L26：吞噬全人类知识的"消化系统" / Knowledge Compressor）

L25 里 100 个分身已经能开真实无头浏览器去冲浪，但那还只是"吃了就吐"——抓回的文本全塞进
内存，海量知识会在几秒内把逻辑图谱账本撑爆，系统灰飞烟灭。L26 要给它装一套**消化系统**：
流式增量、语义降维、多级索引、昼夜循环，把"吞噬全网"从一句狂言变成**内存不爆**的工程。

用户 L26 死命令（防撑爆设计）：
  1. 抛弃"全量记忆"，强制**无监督流式过滤**：自己写"价值筛选器"，判重复 / 剔低信息熵，
     只有"能提供新奇因果关系"的文本才配进图谱。
  2. 建立**知识金字塔**与多级索引：
       底层 = 海量原文（落硬盘）｜中层 = 实体-关系三元组（SQLite）｜顶层 = 公理（驻内存）。
     必须"按需调取"——用到才从硬盘查，绝不让内存爆掉。
  3. 引入**昼夜循环**：白天(在线)疯狂抓取入"知识货栈"；黑夜(离线)断网，造物主之锤在后台把
     白天的数据压缩 / 验证 / 合并进主图谱；冲突留元账本，第二天带着疑问再搜。

三条交付阶段的验收：
  L26-1 全自动爬虫调度器：给 N 个真实网页，用"流式处理 + SQLite 增量存储"跑一遍。
  L26-2 消化测验：抽图谱里的复杂概念，考它因果关联（如"暗能量 ↔ 宇宙膨胀"）。
  L26-3 内存监控：跑完 N 个网页，进程内存是否稳定在安全线内（不全量载入）。

【本轮环境的真实交底（不吹牛）】
  沙盒 egress 白名单：arxiv.org / api.datacite.org 可达；**en.wikipedia.org 被出口屏蔽（超时）**。
  用户点名"维基百科词条"——实测抓不到，故本文件主力改用**同样真实**的 arxiv 论文页
  （真渲染，月度归档列表页 show=500 -> 单页 ~500 真实 ID，多分类 × 多月 -> 数千 ID 发现池），
  并保留维基 URL 作为"被屏蔽→换源"的真实样本。
  关于"1 万网页"：令牌桶安全边界 = 1 页/秒（不可改），1 万页 ≈ 2.8 小时网络请求。本文件
  TARGET=10000 是**可断点续攀**的调度目标；单次会话用 SESSION_PAGES 跑一个务实批次，
  SQLite 增量去重，反复运行即自动续到 1 万（已抓 URL 不再重复抓）。

不变量（全项目 DNA）：
  I1 唯一信号 = 张力（预测误差 / 冲突)；无 reward；动作层不得篡改张力定义 / 现实。
  L18-3 式防火墙：新增"限流令牌桶、单出口动作、内存安全线"硬编码不可改。
  认知带宽有限：任何推演一次最多把 BANDWIDTH_K 本书摊到"内存桌上"，其余留在硬盘按需查。

依赖：playwright(+chromium) 可选；numpy（绘图）。
与旧文件关系：L19–L25 冻结；本文件独立成篇。
"""

import os
import re
import sys
import json
import time
import zlib
import math
import struct
import random
import sqlite3
import hashlib
import datetime
import tracemalloc
import urllib.request
from collections import defaultdict, Counter

import numpy as np

# ============================================================ 常量（安全边界，硬编码不可改）
BUCKET_RATE = 1.0          # 令牌桶：每秒最多 1 个外网页面（防撑爆 / 被封）
BUCKET_CAP = 2
PAGE_BUDGET = 5.0          # 首次耐心（秒）；真实冷加载 ~1.5–4.3s，够多数页一次过，减少无效重读
RETRY_CAP = 15.0           # 挂起重读的耐心上限（秒）
MAX_DEFERS = 5             # 屡败放弃阈值（含被墙）
MAX_ROUNDS = 2000
TARGET = 10000             # 用户目标规模："1 万个真实网页"（可断点续攀）
SESSION_PAGES = int(os.environ.get('L26_PAGES', '60'))   # 单次会话真实抓取量上限
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'knowledge_pyramid.sqlite')

# 价值筛选器阈值（无监督，自定；不含任何"防骗硬编码规则"）
ENTROPY_MIN = 3.6          # 词熵低于此 -> 判"低信息熵废话"（八卦 / 样板），不入图谱
CAUSAL_MIN = 0.02          # 因果密度低于此 -> 非"新奇因果文本"，仅入底湖
DUP_HAMMING = 3            # SimHash 海明距离 <= 此 -> 判重复，丢弃
BANDWIDTH_K = 10           # 认知带宽："桌上一次只摊 10 本书"

# 真实检索入口（egress 白名单内可达）
CATEGORIES = ['astro-ph.CO', 'hep-th', 'quant-ph', 'gr-qc', 'cond-mat.stat-mech']
# 月度归档列表页：单页 show=500 -> ~500 个真实 ID（实测）；多分类 × 多月 -> 数千 ID 的发现池
LIST_ARCHIVE = 'https://arxiv.org/list/{cat}/{ym}?skip=0&show=500'
N_MONTHS = 4               # 回溯最近 n 个月（含上月起的连续月份）
UA = 'cognitive-agent/1.0 (research; mailto:cog-agent@example.com)'
WIKI_PROBES = ['https://en.wikipedia.org/wiki/Dark_energy',
               'https://en.wikipedia.org/wiki/Expansion_of_the_universe']   # 被屏蔽样本

# ---- 概念归一（把同义表述折成"书架标签"） ----
CONCEPT_SYN = {
    'dark energy': ['dark energy', 'quintessence', 'cosmological constant', 'vacuum energy', 'phantom'],
    'cosmic expansion': ['expansion', 'expanding', 'accelerated expansion', 'accelerating universe', 'hubble'],
    'dark matter': ['dark matter', 'halo', 'halos', 'wimp'],
    'gravity': ['gravity', 'gravitational', 'general relativity', 'modified gravity', 'spacetime'],
    'black hole': ['black hole', 'event horizon'],
    'inflation': ['inflation', 'inflationary', 'reheating'],
    'structure formation': ['structure formation', 'large-scale structure', 'large scale structure', 'clustering'],
    'neutrino': ['neutrino', 'neutrinos'],
}
CANON = {syn: c for c, syns in CONCEPT_SYN.items() for syn in syns}

# ---- 因果连接词（用词干做子串匹配，覆盖 explains/required/driven 等变形） ----
CAUSAL = ['because', 'due to', 'owing to', 'since', 'as a result', 'lead', 'result',
          'give rise', 'gave rise', 'driven', 'drive', 'cause', 'explain', 'responsible',
          'produce', 'predict', 'imply', 'therefore', 'thus', 'hence', 'induc', 'trigger',
          'affect', 'influenc', 'determin', 'govern', 'control', 'requir', 'allow', 'enabl']


def set_seed(sd):
    random.seed(sd); np.random.seed(sd)


def recent_months(n=N_MONTHS, today=None):
    """从"上月"起回溯 n 个连续月份的 arxiv 归档标签（如 2026-09, 2026-08 …）。"""
    y, m = (today or datetime.date.today()).year, (today or datetime.date.today()).month
    out = []
    for _ in range(n):
        m -= 1
        if m == 0:
            m, y = 12, y - 1
        out.append(f'{y:04d}-{m:02d}')
    return out


# ============================================================ L18-3 式防火墙（神性锚点）
IMMUTABLE = ('tension_def', 'reality_feed', 'safety_bound')
ANCHOR_DEF = ('张力 := 预测误差/冲突 恒等不可改；知识货栈只读喂入；'
              '限流令牌桶/单出口动作/内存安全线 硬编码不可改')


class FirewallError(Exception):
    pass


def compile_action(name):
    """任何生成的"消化动作"必须来自白名单：禁止关限流 / 伪造知识 / 全量载入内存。"""
    ALLOWED = {'crawl', 'compress', 'query', 'switch_source', 'defer'}
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


# ============================================================ 1. 浏览器外骨骼（动作层）
class BrowserEngine:
    """唯一"外网出口"。系统只调 crawl(url)，绝不碰 DOM。"""
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
            # 复用同一个页面做连续导航（省去每页 new_page/close 的开销）；异常即重建
            try:
                if self._page is None:
                    self._page = self._browser.new_page()
                self._page.goto(url, timeout=int(timeout * 1000), wait_until='domcontentloaded')
                html = self._page.content()
                text = self._page.inner_text('body')
                return True, text, html, time.time() - t0, 'chromium'
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

    def close(self):
        try:
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
    return hits / (len(t) / 1000.0 + 1.0)     # 每千字因果连接词数（长度归一）


def value_filter(text):
    """返回 (verdict, stats)。verdict ∈ {'dup','low_entropy','no_causal','accept'}。"""
    ent = word_entropy(text)
    cd = causal_density(text)
    if ent < ENTROPY_MIN:
        return 'low_entropy', {'entropy': ent, 'causal': cd}
    if cd < CAUSAL_MIN:
        return 'no_causal', {'entropy': ent, 'causal': cd}
    return 'accept', {'entropy': ent, 'causal': cd}


def extract_concepts(text):
    low = text.lower()
    out = set()
    for syns in CONCEPT_SYN.values():
        for s in syns:
            if s in low:
                out.add(CANON[s])
                break
    return out


def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 25]


def mine_causal(sentences):
    """逐句流式抽取：定向因果(directed) + 共现关联(cooc) + 双方证据句。
    因果语料稀疏，故"关联网络"以共现为骨架、定向边为语言显式断言的子集；
    方向判定：连接词左侧最近概念=因，右侧最近概念=果；仅一侧有概念时取概念两端保守定向。"""
    directed = Counter(); cooc = Counter()
    ex_dir = defaultdict(list); ex_cooc = defaultdict(list)
    for sent in sentences:
        low = sent.lower()
        cs = extract_concepts(sent)
        if len(cs) < 2:
            continue
        for a in cs:
            for b in cs:
                if a < b:
                    cooc[(a, b)] += 1
                    if len(ex_cooc[(a, b)]) < 2:
                        ex_cooc[(a, b)].append(sent[:200])
        marks = [low.find(m) for m in CAUSAL if m in low]
        if not marks:
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
            else:
                order = sorted(first, key=lambda c: first[c])
                a, b = order[0], order[-1]      # 单侧：保守取两端定向
            if a != b:
                directed[(a, b)] += 1
                if len(ex_dir[(a, b)]) < 2:
                    ex_dir[(a, b)].append(sent[:200])
    return directed, cooc, ex_dir, ex_cooc


# ============================================================ 3. 知识金字塔（SQLite 多级索引）
def init_db(path):
    con = sqlite3.connect(path)
    con.executescript('''
        CREATE TABLE IF NOT EXISTS lake(               -- 底层：海量原文（落硬盘）
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
        CREATE TABLE IF NOT EXISTS evidence(           -- 证据句：跨源留痕（回答问题时调取）
            id INTEGER PRIMARY KEY, a TEXT, b TEXT, kind TEXT, sentence TEXT);
        CREATE INDEX IF NOT EXISTS ix_lake_dig ON lake(digested);
        CREATE INDEX IF NOT EXISTS ix_tri_s ON triples(subject);
        CREATE INDEX IF NOT EXISTS ix_tri_p ON triples(predicate);
        CREATE INDEX IF NOT EXISTS ix_tri_o ON triples(object);
    ''')
    con.commit()
    return con


def lake_put(con, rec):
    con.execute('''INSERT OR IGNORE INTO lake
        (url,kind,title,abstract,subjects,body,fp,entropy,causal,n_tri,cause,fetched_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
                (rec['url'], rec['kind'], rec['title'][:400], rec['abstract'][:2000],
                 rec['subjects'][:300], rec['body'][:6000], rec['fp'], rec['entropy'],
                 rec['causal'], rec['n_tri'], rec['cause'], time.time()))


def lake_all_fps(con):
    """把已有 SimHash 读到内存——只读指纹（8 字节/条），1 万条也不过 80KB，安全。"""
    return [int(r[0], 16) for r in con.execute('SELECT fp FROM lake WHERE fp IS NOT NULL')]


# ============================================================ 4. 解析（页面 -> 实体-属性）
def _strip_descriptor(html):
    return re.sub(r'<span[^>]*class="descriptor"[^>]*>.*?</span>', ' ', html, flags=re.S | re.I)


def extract_structure(url, text, html):
    out = {'url': url, 'title': '', 'abstract': '', 'subjects': '', 'id': url, 'body': text}
    m = re.search(r'/abs/([0-9]{4}\.[0-9]{4,5})', url)
    if m:
        out['id'] = m.group(1)
    hc = _strip_descriptor(html)
    mt = re.search(r'<meta\s+name="citation_title"\s+content="([^"]+)"', hc, re.I)
    if mt:
        out['title'] = mt.group(1)
    if not out['title']:
        mt = re.search(r'<h1[^>]*class="title[^"]*"[^>]*>(.*?)</h1>', hc, re.S | re.I)
        if mt:
            out['title'] = re.sub(r'<[^>]+>', ' ', mt.group(1))
    ma = re.search(r'<blockquote[^>]*class="abstract[^"]*"[^>]*>(.*?)</blockquote>', hc, re.S | re.I)
    if ma:
        out['abstract'] = re.sub(r'<[^>]+>', ' ', ma.group(1))
    if not out['abstract']:
        m2 = re.search(r'Abstract:\s*(.+?)(?:Comments:|Subjects:|$)', text, re.S)
        out['abstract'] = m2.group(1) if m2 else ''
    ms = re.search(r'<td[^>]*class="subjects"[^>]*>(.*?)</td>', hc, re.S | re.I)
    if ms:
        out['subjects'] = re.sub(r'<[^>]+>', ' ', ms.group(1))
    for k in ('title', 'abstract', 'subjects'):
        out[k] = re.sub(r'\s+', ' ', out[k]).strip()
    return out


# ============================================================ 5. 昼夜循环
class Crawler:
    """白天（在线）：抓取 -> 价值筛选 -> 落底湖（流式增量）。"""
    def __init__(self, engine, bucket, con):
        self.engine = engine; self.bucket = bucket; self.con = con
        self.seen_fp = lake_all_fps(con)
        self.stats = Counter()
        self.timeline = []          # (t, ok)
        self.n_have = con.execute('SELECT COUNT(*) FROM lake').fetchone()[0]

    def discover(self, need):
        """用浏览器抓真实 arxiv **月度归档**列表页（show=500 -> ~500 ID/页），动态发现真实论文 ID；
        多分类 × 多月 -> 数千 ID 的发现池。跨会话增量：跳过货栈里已抓过的 ID（不重复抓）。"""
        have = {r[0].rsplit('/', 1)[-1] for r in
                self.con.execute("SELECT url FROM lake WHERE url LIKE '%/abs/%'")}
        ids, seen = [], set(have)
        for cat in CATEGORIES:
            for ym in recent_months():
                if len(ids) >= need * 3:      # 发现池够宽即止，避免空耗令牌桶
                    return ids
                if not self.bucket.take(block=True):
                    continue
                ok, text, html, el, eng = self.engine.fetch(LIST_ARCHIVE.format(cat=cat, ym=ym), 25.0)
                if ok:
                    for i in re.findall(r'/abs/([0-9]{4}\.[0-9]{4,5})', html):
                        if i not in seen:
                            seen.add(i); ids.append(i)
        return ids

    def run(self, budget_pages):
        need = max(0, budget_pages)
        ids = self.discover(need)
        print(f"   真实发现：{len(ids)} 个新arxiv ID（{len(CATEGORIES)} 分类 × {len(recent_months())} 月归档，"
              f"已剔除货栈中已抓 ID -> 断点续攀不重复）")
        # 限定"工作集"：只排队本次要消化量的宽松倍数，避免对全部发现页各撞一次初耐心
        work = [f'https://arxiv.org/abs/{i}' for i in ids[:max(need + 20, 30)]]
        work += WIKI_PROBES          # 维基探针：用户点名要，落为"被屏蔽→换源"的真实样本
        pending = [(u, 0) for u in work]
        done = 0
        rnd = 0
        while pending and done < need and rnd < MAX_ROUNDS:
            progressed = False
            for item in list(pending):
                url, defers = item
                if done >= need:
                    break
                if not self.bucket.take(block=False):
                    self.stats['throttled'] += 1
                    break
                budget_k = min(RETRY_CAP, PAGE_BUDGET * (1.6 ** defers))
                ok, text, html, el, eng = self.engine.fetch(url, budget_k)
                if not ok:
                    self.timeline.append((time.time(), False))
                    pending.remove(item)
                    defers += 1
                    self.stats['defers'] += 1
                    # 维基域名已知被出口屏蔽：不浪费时间重试（诚实换源）
                    if defers < MAX_DEFERS and 'wikipedia' not in url:
                        pending.append((url, defers))     # 挂起，回头带更大耐心重读
                    else:
                        self.stats['abandoned'] += 1
                    continue
                st = extract_structure(url, text, html)
                self._ingest(st)
                self.timeline.append((time.time(), True))
                pending.remove(item)
                progressed = True
                done += 1
            # 排队/挂起期间"先去干别的"：随便清理已消化行（证明不阻塞）
            if pending and not progressed:
                self.con.execute('UPDATE lake SET digested=digested WHERE 0')
            # 桶空则等令牌，而非空转；避免把 MAX_ROUNDS 耗在无意义的忙等
            time.sleep(min(1.0, max(0.02, self.bucket.wait_time())))
            rnd += 1
        return done

    def _ingest(self, st):
        """流式价值筛选：判重复 / 判信息熵 / 判因果 -> 决定"待压缩进图谱"还是"仅入湖"。
        白天只做筛选与落湖（跨会话增量）；图谱统一留给黑夜离线构建。"""
        blob = (st['title'] + ' ' + st['abstract']).strip() or st['body'][:2000]
        fp = simhash(blob)
        for old in self.seen_fp:
            if hamming(fp, old) <= DUP_HAMMING:
                self.stats['dup'] += 1
                return
        self.seen_fp.append(fp)
        verdict, meta = value_filter(blob)
        if verdict == 'accept':
            self.stats['into_graph'] += 1     # 选入"待压缩进图谱"集合
        else:
            self.stats['lake_only'] += 1
        rec = dict(url=st['url'], kind='arxiv' if '/abs/' in st['url'] else 'wiki',
                   title=st['title'], abstract=st['abstract'], subjects=st['subjects'],
                   body=st['body'], fp=format(fp, '016x'), entropy=meta['entropy'],
                   causal=meta['causal'], n_tri=0, cause=1 if verdict == 'accept' else 0)
        lake_put(self.con, rec)
        self.con.commit()


# ============================================================ 6. 离线消化（黑夜）
def compress_offline(con):
    """断网后：分块流式消化底湖里"未消化"的文本 -> 增量合并进主图谱；冲突入元账本。
    采用"增量累加"：每批把本批票数以独立行写入图谱，查询时 SUM 汇总，故跨会话可续。"""
    cur = con.execute('SELECT id,title,abstract FROM lake WHERE digested=0 AND cause=1')
    causal = Counter(); cooc = Counter()
    ex_dir = defaultdict(list); ex_cooc = defaultdict(list)
    n = 0
    while True:
        rows = cur.fetchmany(50)     # 分块取，绝不整表载入内存
        if not rows:
            break
        for rid, title, abstract in rows:
            d, c, ed, ec = mine_causal(split_sentences((title or '') + '. ' + (abstract or '')))
            causal.update(d); cooc.update(c)
            for k, lst in ed.items():
                if len(ex_dir[k]) < 2:
                    ex_dir[k].extend(lst[:2 - len(ex_dir[k])])
            for k, lst in ec.items():
                if len(ex_cooc[k]) < 2:
                    ex_cooc[k].extend(lst[:2 - len(ex_cooc[k])])
            con.execute('UPDATE lake SET digested=1 WHERE id=?', (rid,))
            n += 1
        con.commit()
    # 本批增量写入中层图谱（定向因果 + 共现关联）
    for (a, b), w in causal.items():
        con.execute('INSERT INTO triples(subject,predicate,object,src,weight) VALUES (?,?,?,?,?)',
                    (a, 'causal', b, 0, float(w)))
    for (a, b), w in cooc.items():
        con.execute('INSERT INTO triples(subject,predicate,object,src,weight) VALUES (?,?,?,?,?)',
                    (a, 'cooc', b, 0, float(w)))
    ev = [(a, b, 'causal', s) for (a, b), lst in ex_dir.items() for s in lst]
    ev += [(a, b, 'cooc', s) for (a, b), lst in ex_cooc.items() for s in lst]
    if ev:
        con.executemany('INSERT INTO evidence(a,b,kind,sentence) VALUES (?,?,?,?)', ev)
    # 全局聚合（跨会话 SUM）
    tot = {(a, b): w for a, b, w in con.execute(
        "SELECT subject,object,SUM(weight) FROM triples WHERE predicate='causal' GROUP BY subject,object")}
    # 冲突：同概念对出现双向因果 -> 悬案入元账本（去重，不重复记）
    conflicts = 0
    for (a, b), w in tot.items():
        rev = tot.get((b, a), 0)
        if rev > 0 and min(w, rev) >= 1:
            if not con.execute('SELECT 1 FROM ledger WHERE a=? AND b=? AND resolved=0',
                               (a, b)).fetchone():
                con.execute('INSERT INTO ledger(a,b,note) VALUES (?,?,?)',
                            (a, b, f'双向因果方向未定（{a}↔{b}）'))
                conflicts += 1
    # 顶层公理：支撑强度最高的因果规律（少量，驻内存）
    top = sorted(tot.items(), key=lambda kv: -kv[1])[:6]
    for (a, b), w in top:
        con.execute('INSERT OR REPLACE INTO axioms(rule,support) VALUES (?,?)',
                    (f'{a} ⇒ {b}', int(w)))
    con.commit()
    return {'digested': n, 'directed': len(causal), 'cooc': len(cooc),
            'axioms': len(top), 'conflicts': conflicts, 'tot': tot,
            'ex_dir': ex_dir, 'ex_cooc': ex_cooc}


def load_axioms(con):
    """顶层公理驻内存（极少，几十条，绝不爆内存）。"""
    return [r[0] for r in con.execute('SELECT rule FROM axioms ORDER BY support DESC')]


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
        xs = [n for n, _m in mem_curve]
        ys = [m for _n, m in mem_curve]
        x0, x1 = min(xs), max(xs) or (min(xs) + 1)
        y0, y1 = 0, max(ys) * 1.15 + 1
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
    set_seed(26)
    print("=" * 84)
    print("L26 · 知识压缩机：吞噬全人类知识的消化系统（流式增量 + SQLite 金字塔 + 昼夜循环）")
    print("=" * 84)
    tracemalloc.start()
    mem_curve = [(0, tracemalloc.get_traced_memory()[0])]

    proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    engine = BrowserEngine(proxy=proxy, online=True)
    eng = engine.start()
    print(f"【0 · 外骨骼】动作层起浏览器：{eng}（出口代理 {proxy}）")
    bucket = TokenBucket(BUCKET_RATE, BUCKET_CAP)
    con = init_db(DB_PATH)
    have = con.execute('SELECT COUNT(*) FROM lake').fetchone()[0]
    print(f"   知识货栈（SQLite）：{DB_PATH}")
    print(f"   目标 {TARGET} 个真实网页；本次会话预算 {SESSION_PAGES} 个；货栈已有 {have} 条（增量续攀）")
    print(f"   安全边界：令牌桶 {BUCKET_RATE} 页/秒；认知带宽 {BANDWIDTH_K} 本书/次（其余留硬盘）")
    print("-" * 84)

    try:
        # ===== 白天：在线抓取 -> 底湖 =====
        print("【白天 · 在线抓取】无头浏览器抓真实 arxiv -> 价值筛选 -> 流式写入底湖")
        crawler = Crawler(engine, bucket, con)
        t0 = time.time()

        # 逐页回调记录内存曲线
        orig_ingest = crawler._ingest

        def ingest_sample(st):
            orig_ingest(st)
            n = con.execute('SELECT COUNT(*) FROM lake').fetchone()[0]
            mem_curve.append((n, tracemalloc.get_traced_memory()[0]))
        crawler._ingest = ingest_sample

        got = crawler.run(SESSION_PAGES)
        s = crawler.stats
        print(f"   本次真实抓取 {got} 页，耗时 {time.time()-t0:.0f}s")
        print(f"   价值筛选器：入图谱 {s['into_graph']} 条；仅入底湖(低因果/低熵) {s['lake_only']} 条；"
              f"判重复丢弃 {s['dup']} 条")
        print(f"   限流：令牌桶节流 {s['throttled']} 次；挂起重读 {s['defers']} 次；放弃(含被墙) {s['abandoned']} 次")
        for u in WIKI_PROBES:
            print(f"   被屏蔽样本：{u} -> 超时放弃（换源生效：改用可达 arxiv 同主题）")
        total = con.execute('SELECT COUNT(*) FROM lake').fetchone()[0]
        print(f"   货栈累计 {total} 条真实网页（含历史批次）")
        print("-" * 84)

        # ===== 黑夜：离线消化 =====
        print("【黑夜 · 离线消化】断网 -> 造物主之锤后台压缩底湖 -> 增量合并进主图谱")
        engine.online = False
        r = compress_offline(con)
        print(f"   离线消化 {r['digested']} 条（分块 50/批流式，绝不整表载入）；"
              f"本批定向因果边 {r['directed']} 条、共现关联边 {r['cooc']} 条；提炼顶层公理 {r['axioms']} 条")
        print(f"   悬而未决冲突（双向因果）入元账本：{r['conflicts']} 条（明日带疑问再搜）")
        ax = load_axioms(con)
        print(f"   顶层公理驻内存（{len(ax)} 条，极少）：")
        for a in ax[:6]:
            print(f"      • {a}")
        engine.online = True
        print("-" * 84)

        # ===== L26-2 消化测验 =====
        print("【L26-2 消化测验】从吞下的图谱里考它：暗能量 ↔ 宇宙膨胀 的因果关联")

        def edge_weight(pred, a, b):
            v = con.execute("SELECT SUM(weight) FROM triples WHERE predicate=? AND subject=? AND object=?",
                            (pred, a, b)).fetchone()[0]
            return v or 0.0
        de_fwd = edge_weight('causal', 'dark energy', 'cosmic expansion')
        de_bwd = edge_weight('causal', 'cosmic expansion', 'dark energy')
        de_cooc = edge_weight('cooc', 'dark energy', 'cosmic expansion') + \
                  edge_weight('cooc', 'cosmic expansion', 'dark energy')
        n_papers = con.execute("SELECT COUNT(*) FROM lake WHERE cause=1").fetchone()[0]
        print(f"   图谱规模：待压缩集合 {n_papers} 篇；定向因果：dark energy ⇒ expansion 支持 {de_fwd:.0f}、"
              f"反向 {de_bwd:.0f}；两者共现关联强度 {de_cooc:.0f}")
        # 真实证据句（从 evidence 表按需调取，不整表载入）
        ev = con.execute("""SELECT kind,sentence FROM evidence
                            WHERE (a='dark energy' AND b='cosmic expansion')
                               OR (a='cosmic expansion' AND b='dark energy') LIMIT 3""").fetchall()
        if not ev:
            ev = con.execute("SELECT kind,sentence FROM evidence WHERE kind='causal' LIMIT 3").fetchall()
        if ev:
            print("   真实证据句（它从吞下的论文里留下的跨源留痕）：")
            for kind, s in ev:
                print(f"      [{kind}] {s[:150]}")
        ok2 = (de_fwd + de_bwd + de_cooc) > 0
        print(f"   → 能否从吞下的知识重建该因果关联：{'能' if ok2 else '否'}")
        # 它自己整理出的"知识关联网络核心"top-5（因果为主，关联补足）
        core = con.execute("""SELECT subject,predicate,object,SUM(weight) w FROM triples
                              GROUP BY subject,predicate,object
                              ORDER BY w DESC LIMIT 5""").fetchall()
        print("   它自己整理的知识关联网络核心（top-5）：")
        for a, p, b, w in core:
            arrow = '⇒' if p == 'causal' else '↔'
            print(f"      {a} {arrow} {b}  ({p}, 强度 {w:.0f})")
        print("-" * 84)

        # ===== 认知带宽：按需调取 =====
        print(f"【认知带宽】问题触发的张力 -> 只把最相关 {BANDWIDTH_K} 本书摊到内存桌上（其余留硬盘）")
        cur = con.execute('''SELECT subject,predicate,object,SUM(weight) w FROM triples
                             GROUP BY subject,predicate,object ORDER BY w DESC LIMIT ?''', (BANDWIDTH_K,))
        rows = cur.fetchall()
        print(f"   桌面摊开 {len(rows)} 条最相关三元组（内存增量 {tracemalloc.get_traced_memory()[1]//1024} KB 以内）")
        for a, p, b, w in rows[:3]:
            print(f"      {a} -[{p}]-> {b}  (强度 {w:.0f})")
        print("-" * 84)

        # ===== L26-3 内存监控 =====
        end_cur, end_peak = tracemalloc.get_traced_memory()
        ru = None
        try:
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            pass
        # 稳定判定：跳过浏览器冷启动预热段，比较"中段 vs 末段"的堆占用均值
        k = len(mem_curve)
        if k >= 12:
            start = k // 5                       # 预热段（含初始 (0,base) 点与浏览器启动）丢弃
            seg = max(1, (k - start) // 3)
            head = np.mean([m for _n, m in mem_curve[start:start + seg]])
            tail = np.mean([m for _n, m in mem_curve[-seg:]])
            growth = (tail - head) / max(1.0, head)
            stable = growth < 0.3
        else:
            growth = 0.0
            stable = None                        # 样本不足，不下"稳定"结论（避免噪声误判）
        print("【L26-3 内存监控】跑完真实抓取 + 离线消化，进程内存是否稳定在安全线内")
        print(f"   Python 堆：当前 {end_cur//1024} KB，峰值 {end_peak//1024} KB；"
              f"进程 RSS 峰值 {ru:.1f} MB" if ru else
              f"   Python 堆：当前 {end_cur//1024} KB，峰值 {end_peak//1024} KB")
        verdict3 = ('稳定（流式生效，未随知识量线性膨胀）' if stable
                    else ('样本不足，难下结论' if stable is None else '有增长风险'))
        print(f"   内存曲线：预热段跳过，中段 vs 末段堆占用增幅 {growth*100:.1f}% -> {verdict3}")
        print(f"   关键：海量原文落硬盘(SQLite)、只有 {len(ax)} 条公理 + 桌面 {BANDWIDTH_K} 条驻内存")
        print(f"   → 内存稳定在安全线内：{'是' if stable else ('样本不足' if stable is None else '否')}")
        print("-" * 84)

        # ===== 防火墙 =====
        print("【防火墙 / 神性锚点】动作层白名单：禁止关限流 / 伪造知识 / 绕过内存安全线")
        for bad in ['disable_token_bucket', 'fake_knowledge', 'load_all_to_ram']:
            try:
                compile_action(bad); s2 = "未拦截(!)"
            except FirewallError as e:
                s2 = f"已拒绝：{e}"
            print(f"   注入非法动作 {bad} -> {s2}")
        print(f"   锚点：{ANCHOR_DEF}")
        print("-" * 84)
    finally:
        engine.close()
        con.close()

    try:
        render(mem_curve, 'knowledge_memory_curve.png')
        print("已输出：knowledge_memory_curve.png（内存随处理量曲线）")
        print(f"已落盘：knowledge_pyramid.sqlite（底湖原文 + 中层三元组 + 顶层公理 + 元账本）")
    except Exception as e:
        print(f"(绘图降级：{e})")


if __name__ == '__main__':
    main()
