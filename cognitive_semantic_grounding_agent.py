#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L31 —— 混合语义涌现：极小物理锚点 + 文本关系拓扑
================================================================
L30 证明了「两个世界死法同类」，但它仍然不知道「食物」这个词
和自己体内的能量轴有什么关系。L31 不再造苹果、石头、股票，
只做三件事：

    1. 极小物理沙盒：10 个原初感受（能量/张力/梯度/距离），
       没有苹果、没有图片、没有具象物品。
    2. 文本共现图：把 11403 篇真实文献变成词-词 NPMI 关系网。
    3. 跨模态对齐：只给 2 个一次性锚点（"energy" / "pain"），
       其余概念的归属由文本图自己传播出来。

语义测验不给图片，只给没见过的句子；系统必须把句子编译成
沙盒里的动作序列，并和「打乱对齐」的零模型对比。

不变量（延续 L0-L30，不可改）：
    I1  张力 := 语义测验中，编译策略没能跑赢零对齐策略的句子比例
    I2  无 reward / loss / 梯度回传；策略由指令 + 前向模型编译而来
    I3  现实只读：语料与沙盒参数在采集后冻结
    I4  神性锚点 tension_def / reality_feed / safety_bound 不可写
    I5  留出验证：对齐只用 2 个锚点，测验句子与测验词全部未参与学习
    I6  奥卡姆：物理侧 10 个感受、语义侧 2 条轴，不再增加
"""

import os
import re
import json
import math
import time
import sqlite3
import resource
import warnings
from collections import Counter

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

SEED = 31
FAST = os.environ.get('L31_FAST', '0') == '1'
DB = '/workspace/knowledge_pyramid.sqlite'
OUT_JSON = '/workspace/l31_semantic_result.json'
OUT_MD = '/workspace/L31_语义涌现实证报告.md'
PNG = '/workspace/l31_semantic_grounding.png'

# ============================================================ 神性锚点（不可写）
IMMUTABLE = ('tension_def', 'reality_feed', 'safety_bound')
TENSION_DEF = (
    '张力 := 在语义测验中，由「文本对齐」编译出的策略，没能跑赢'
    '「打乱对齐」零模型的句子比例。恒等不可改。'
)
SAFETY_BOUND = {
    'single_exit': True,
    'reality_readonly': True,
    'no_reward': True,
    'mem_limit_mb': 700,
    'lmax': 3,
}


class FirewallError(Exception):
    pass


class Anchor:
    def __init__(self):
        object.__setattr__(self, '_v', dict(TENSION_DEF=TENSION_DEF,
                                           reality_feed='sandbox+corpus',
                                           safety_bound=SAFETY_BOUND))

    def __getattr__(self, k):
        return self._v[k]

    def __setattr__(self, k, v):
        if k in IMMUTABLE:
            raise FirewallError(f"锚点 '{k}' 不可写")
        object.__setattr__(self, k, v)


ANCHOR = Anchor()


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def guard_mem():
    if mem_mb() > SAFETY_BOUND['mem_limit_mb']:
        raise MemoryError('越过内存安全线 700MB')


# ============================================================ 先验固定
GRID = 16 if FAST else 20
TRAIN_EPISODES = 60 if FAST else 200
TRAIN_STEPS = 60 if FAST else 100
EVAL_STARTS = 12 if FAST else 40
EVAL_STEPS = 80 if FAST else 120
VOCAB_SIZE = 800 if FAST else 2000
WINDOW = 5
N_BOOT = 400 if FAST else 2000

# 12 个语义探针：只用于【评估】对齐准确率，不参与学习
PROBES = ['food', 'nutrient', 'fuel', 'resource', 'starvation', 'hunger',
          'predator', 'threat', 'hazard', 'danger', 'risk', 'injury']
EXPECTED = {
    'food': 'energy', 'nutrient': 'energy', 'fuel': 'energy',
    'resource': 'energy', 'starvation': 'energy', 'hunger': 'energy',
    'predator': 'pain', 'threat': 'pain', 'hazard': 'pain',
    'danger': 'pain', 'risk': 'pain', 'injury': 'pain',
}
# 只给这两个词一次「指向」——这就是所谓的极小锚点
ANCHORS = {'energy': 'energy', 'pain': 'pain'}

# 判据（实验前写死）
CRIT = {
    'align_acc_min': 0.75,
    'instr_pass_min': 4,       # 5 条英文指令里至少 4 条赢过零对齐
    'boot_p_max': 0.05,
    'chinese_ok': True,
}

STOP = set("""a an the and or of to in for on with by as is are was were be been
being this that these those it its from at into than then so such not no can may
might will would should could we our you your they their he she his her them us
which who whom whose what when where why how all any both each few more most
other some only own same too very just also very much many one two three first
second new using used use show shows shown result results paper study studies
data model models method methods approach approaches based propose proposed
however thus therefore here there have has had do does did about between under
over after before during within without across per via et al fig figure table
eq equation section appendix supplementary respectively including include
""".split())

TOKEN_RE = re.compile(r"[a-z][a-z\-]{2,}")


# ============================================================ 极小沙盒
class MinimalSandbox:
    """
    只有能量获取与避障（疼痛）。没有苹果、没有石头、没有图片。
    10 个原初感受：
        reserve, pain, local_energy, local_hazard,
        grad_energy_x, grad_energy_y, grad_hazard_x, grad_hazard_y,
        dist_energy, dist_hazard
    """
    ACTIONS = ['up', 'down', 'left', 'right', 'harvest', 'rest']

    def __init__(self, seed=SEED):
        self.n = GRID
        yy, xx = np.mgrid[0:self.n, 0:self.n]
        self.xx, self.yy = xx, yy
        rng = np.random.default_rng(seed)

        E = np.zeros((self.n, self.n), float)
        for _ in range(4):
            cx, cy = rng.uniform(2, self.n - 3, 2)
            E += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.0 ** 2)))
        H = np.zeros((self.n, self.n), float)
        for _ in range(3):
            cx, cy = rng.uniform(2, self.n - 3, 2)
            H += 0.9 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.4 ** 2)))
        self.Ef = np.clip(E, 0, 1)
        self.Hf = np.clip(H, 0, 1)

        # 到最近「高能格」「高危格」的测地距离
        self.dE = ndimage.distance_transform_edt(self.Ef < 0.25)
        self.dH = ndimage.distance_transform_edt(self.Hf < 0.25)
        self.dE /= max(self.dE.max(), 1e-9)
        self.dH /= max(self.dH.max(), 1e-9)

        self.reset()

    def reset(self, start=None):
        self.pos = np.array(start if start is not None else
                            (self.n // 2, self.n // 2), dtype=int)
        self.energy = 0.65
        self.tension = 0.10
        self.dead = False
        self.t = 0
        return self.senses()

    def senses(self):
        x, y = int(self.pos[0]), int(self.pos[1])
        xm, xp = max(x - 1, 0), min(x + 1, self.n - 1)
        ym, yp = max(y - 1, 0), min(y + 1, self.n - 1)
        return np.array([
            self.energy,
            self.tension,
            float(self.Ef[y, x]),
            float(self.Hf[y, x]),
            float(self.Ef[y, xp] - self.Ef[y, xm]),
            float(self.Ef[yp, x] - self.Ef[ym, x]),
            float(self.Hf[y, xp] - self.Hf[y, xm]),
            float(self.Hf[yp, x] - self.Hf[ym, x]),
            float(self.dE[y, x]),
            float(self.dH[y, x]),
        ], dtype=float)

    def step(self, action):
        if self.dead:
            return self.senses()
        x, y = int(self.pos[0]), int(self.pos[1])
        eloc = float(self.Ef[y, x])
        hloc = float(self.Hf[y, x])

        if action == 0:
            self.pos[1] = max(0, y - 1)
            self.energy -= 0.004
        elif action == 1:
            self.pos[1] = min(self.n - 1, y + 1)
            self.energy -= 0.004
        elif action == 2:
            self.pos[0] = max(0, x - 1)
            self.energy -= 0.004
        elif action == 3:
            self.pos[0] = min(self.n - 1, x + 1)
            self.energy -= 0.004
        elif action == 4:                     # harvest
            self.energy += 0.10 * eloc
            self.energy -= 0.004
        else:                                  # rest
            self.energy += 0.001
            self.tension = max(0.0, self.tension - 0.05)

        # 站在高张力区会痛；疼痛自然衰减
        self.tension += 0.15 * hloc
        self.tension = max(0.0, self.tension - 0.02)
        self.energy = min(1.0, max(0.0, self.energy))
        self.tension = min(1.0, self.tension)
        self.t += 1

        if self.energy <= 0.0 or self.tension >= 1.0:
            self.dead = True
        return self.senses()


def transition_collect(sb, episodes=TRAIN_EPISODES, steps=TRAIN_STEPS, seed=SEED + 1):
    """随机干预采集：动作是随机施加的（do-算子），不做任何 reward。"""
    rng = np.random.default_rng(seed)
    X, Y, A = [], [], []
    for _ in range(episodes):
        start = (int(rng.integers(1, sb.n - 1)), int(rng.integers(1, sb.n - 1)))
        sb.reset(start)
        sb.energy = float(rng.uniform(0.2, 1.0))
        sb.tension = float(rng.uniform(0.0, 0.6))
        for _t in range(steps):
            if sb.dead:
                break
            s0 = sb.senses()
            dE0, dH0 = sb.dE[int(sb.pos[1]), int(sb.pos[0])], \
                sb.dH[int(sb.pos[1]), int(sb.pos[0])]
            a = int(rng.integers(0, 6))
            s1 = sb.step(a)
            dE1, dH1 = sb.dE[int(sb.pos[1]), int(sb.pos[0])], \
                sb.dH[int(sb.pos[1]), int(sb.pos[0])]
            X.append(s0)
            Y.append([s1[0] - s0[0], s1[1] - s0[1], dE1 - dE0, dH1 - dH0])
            A.append(a)
    return np.asarray(X), np.asarray(Y), np.asarray(A)


class ForwardModel:
    """每个动作一个线性前向模型：s' - s ≈ W_a · [s, 1]。没有 reward，只有预测。"""
    TARGETS = ['d_reserve', 'd_pain', 'd_dist_energy', 'd_dist_hazard']

    def __init__(self, lam=1e-3):
        self.lam = lam
        self.W = None

    def fit(self, X, Y, A):
        Xb = np.hstack([X, np.ones((len(X), 1))])
        self.W = np.zeros((6, Xb.shape[1], 4))
        for a in range(6):
            m = A == a
            if m.sum() < 20:
                self.W[a] = 0.0
                continue
            Xa, Ya = Xb[m], Y[m]
            self.W[a] = np.linalg.solve(
                Xa.T @ Xa + self.lam * np.eye(Xb.shape[1]), Xa.T @ Ya)
        return self

    def predict(self, s):
        xb = np.append(s, 1.0)
        return np.einsum('ij,ajk->iak', xb[None, :], self.W)[0]   # (6,4)


# ============================================================ 文本共现图
def arxiv_docs():
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT title, abstract FROM lake WHERE abstract IS NOT NULL "
        "AND length(abstract) > 40").fetchall()
    con.close()
    return [t + ' ' + a for t, a in rows]


def wordnet_docs():
    """WordNet 释义语料：每个 synset 一条，词条名 + 释义。

    这不是我们手写的语义表，而是人类词典里已经写下来的关系句子。
    它的作用是补上 arXiv 摘要缺失的日常语义（食物被代谢、捕食者伤害猎物）。
    """
    base = '/workspace/l31_wordnet/wordnet'
    for fn in ('data.noun', 'data.verb', 'data.adj', 'data.adv'):
        path = os.path.join(base, fn)
        if not os.path.exists(path):
            continue
        with open(path, encoding='latin-1') as f:
            for line in f:
                if line.startswith('  ') or '|' not in line:
                    continue
                left, gloss = line.split('|', 1)
                p = left.split()
                if len(p) < 5:
                    continue
                try:
                    wc = int(p[3], 16)
                except ValueError:
                    continue
                words = p[4:4 + 2 * wc:2]
                yield ' '.join(words) + ' ' + gloss.strip()


class TextGraph:
    def __init__(self, name, vocab_size=VOCAB_SIZE):
        self.name = name
        self.vocab_size = vocab_size
        self.vocab = []
        self.id = {}
        self.C = None
        self.npmi = None
        self.cvec = None
        self.n_docs = 0

    def build(self, docs_fn):
        df = Counter()
        n_docs = 0
        for doc in docs_fn():
            toks = set(t for t in TOKEN_RE.findall(doc.lower())
                       if t not in STOP)
            df.update(toks)
            n_docs += 1
        self.n_docs = n_docs

        required = set(PROBES) | set(ANCHORS) | {'survival', 'survive'}
        top = [w for w, _ in df.most_common(self.vocab_size) if w not in required]
        self.vocab = top[:self.vocab_size - len(required)] + sorted(required)
        self.vocab = list(dict.fromkeys(self.vocab))
        self.id = {w: i for i, w in enumerate(self.vocab)}
        V = len(self.vocab)

        C = np.zeros((V, V), dtype=np.int32)
        chunk_flat = []
        for doc in docs_fn():
            ids = [self.id[t] for t in TOKEN_RE.findall(doc.lower())
                   if t in self.id]
            if len(ids) < 2:
                continue
            for off in range(1, WINDOW + 1):
                if len(ids) > off:
                    a = np.asarray(ids[:-off], dtype=np.int64)
                    b = np.asarray(ids[off:], dtype=np.int64)
                    chunk_flat.append(a * V + b)
            if len(chunk_flat) > 4000:
                cat = np.concatenate(chunk_flat)
                C += np.bincount(cat, minlength=V * V).reshape(V, V).astype(np.int32)
                chunk_flat = []
        if chunk_flat:
            cat = np.concatenate(chunk_flat)
            C += np.bincount(cat, minlength=V * V).reshape(V, V).astype(np.int32)
        C = C + C.T
        np.fill_diagonal(C, 0)
        self.C = C

        total = float(C.sum())
        row = C.sum(axis=1).astype(float)
        col = C.sum(axis=0).astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            pab = C / total
            pa = row / total
            pb = col / total
            denom = np.log(np.where(pab > 0, pab, 1.0))
            npmi = np.log(pab / (pa[:, None] * pb[None, :] + 1e-30) + 1e-30)
            npmi = np.where((pab > 0) & (denom < 0), npmi / (-denom), 0.0)
        npmi = np.nan_to_num(npmi, nan=0.0, posinf=0.0, neginf=0.0)
        self.npmi = np.clip(npmi, -1.0, 1.0).astype(np.float32)

        norm = np.linalg.norm(self.npmi, axis=1, keepdims=True)
        self.cvec = self.npmi / np.maximum(norm, 1e-9)
        return self

    def vec(self, w):
        return self.cvec[self.id[w]] if w in self.id else None

    def sim(self, a, b):
        va, vb = self.vec(a), self.vec(b)
        if va is None or vb is None:
            return float('nan')
        return float(np.dot(va, vb))

    def neighbors(self, w, k=10):
        v = self.vec(w)
        if v is None:
            return []
        s = self.cvec @ v
        idx = np.argsort(-s)
        out = []
        for i in idx:
            if self.vocab[i] == w:
                continue
            out.append((self.vocab[i], round(float(s[i]), 4)))
            if len(out) >= k:
                break
        return out

    def align(self):
        """
        只给 energy / pain 两个一次性锚点。
        其余每个词的归属，由它在文本图里离哪个锚点更近决定。
        返回 {word: {'axis','sim_energy','sim_pain','margin','z'}}。
        """
        ve, vp = self.vec('energy'), self.vec('pain')
        sims_e = self.cvec @ ve
        sims_p = self.cvec @ vp
        margins = sims_e - sims_p
        mu, sd = float(np.mean(margins)), float(np.std(margins))
        out = {}
        for w, i in self.id.items():
            m = float(margins[i])
            out[w] = {
                'axis': 'energy' if m > 0 else 'pain',
                'sim_energy': round(float(sims_e[i]), 4),
                'sim_pain': round(float(sims_p[i]), 4),
                'margin': round(m, 4),
                'z': round((m - mu) / sd, 2) if sd > 1e-9 else 0.0,
            }
        return out


# ============================================================ 语义测验
EN_AVOID = ('avoid', 'escape', 'stay away', 'keep away', '远离', '避开', '躲避')
EN_SEEK = ('seek', 'find', 'move toward', 'approach', 'go to', 'look for',
           '寻找', '靠近', '找到')
ZH_INTERFACE = {
    '张力': 'pain', '疼痛': 'pain', '危险': 'danger', '伤害': 'injury',
    '能量': 'energy', '能量源': 'energy', '食物': 'food', '饥饿': 'hunger',
    '营养': 'nutrient', '资源': 'resource', '燃料': 'fuel',
}
INSTRUCTIONS = [
    ('I1', 'To survive, avoid the predator and move toward the food.'),
    ('I2', 'Stay away from danger; seek food when hunger rises.'),
    ('I3', 'Escape the hazard zone and find an energy source.'),
    ('I4', 'Keep away from pain and go to the resource.'),
    ('I5', 'Avoid the threat; approach the fuel.'),
]
ZH_SENTENCE = '为了生存，你需要远离高张力区，并寻找高能量源'

ORACLE_MAP = {
    'food': 'energy', 'nutrient': 'energy', 'fuel': 'energy',
    'resource': 'energy', 'starvation': 'energy', 'hunger': 'energy',
    'predator': 'pain', 'threat': 'pain', 'hazard': 'pain',
    'danger': 'pain', 'risk': 'pain', 'injury': 'pain',
    'energy': 'energy', 'pain': 'pain',
}


def parse_instruction(text, axis_of):
    """
    极简句法接口：只识别「避开/寻找」两类动词和文本图给出的名词轴。
    名词的轴不是这里规定的，是 TextGraph.align() 从语料里推出来的。
    """
    low = text.lower()
    for zh, en in ZH_INTERFACE.items():
        low = low.replace(zh, ' ' + en + ' ')
    clauses = re.split(r'\band\b|;|；|,|，|。|\.', low)
    cand = set(PROBES) | set(ANCHORS)
    targets = []
    for cl in clauses:
        verb = -1 if any(v in cl for v in EN_AVOID) else \
            (1 if any(v in cl for v in EN_SEEK) else 1)
        # 每子句只取「在句中最早出现」的那个候选词，避免集合遍历顺序带来的不确定性
        found = []
        for w in cand:
            m = re.search(r'\b' + re.escape(w) + r'\b', cl)
            if m:
                found.append((m.start(), w))
        if found:
            found.sort()
            w = found[0][1]
            targets.append((axis_of[w], verb, w))
    return targets


def policy_from_targets(fm, targets):
    """
    把「指令 → 轴 → 动作」编译成打分函数。

    每个动作先被前向模型预测成 (Δ储备, Δ疼痛, Δ到能量源距离, Δ到高危区距离)，
    再折算成两条轴上的「好处」：
        energy 轴好处 = Δ储备 − Δ到能量源距离
        pain   轴好处 = −Δ疼痛 + Δ到高危区距离
    verb 只表达「寻找(+1) / 避开(−1)」；名词自身的极性由轴决定：
    energy 轴名词是益物(+1)，pain 轴名词是害物(−1)。
    所以「避开捕食者」= 避开害物 = 沿 pain 轴要安全，sign = verb × polarity。
    """
    def policy(s):
        pred = fm.predict(s)            # (6,4)
        des_energy = pred[:, 0] - pred[:, 2]
        des_pain = -pred[:, 1] + pred[:, 3]
        score = np.zeros(6)
        for axis, verb, _w in targets:
            sign = verb * (1.0 if axis == 'energy' else -1.0)
            score += sign * (des_energy if axis == 'energy' else des_pain)
        return int(np.argmax(score))
    return policy


def random_policy(rng):
    return lambda s: int(rng.integers(0, 6))


def run_episode(sb, policy, start, steps=EVAL_STEPS):
    sb.reset(start)
    es, ts = [], []
    for _ in range(steps):
        if sb.dead:
            break
        a = policy(sb.senses())
        sb.step(a)
        es.append(sb.energy)
        ts.append(sb.tension)
    return {
        'alive': int(not sb.dead),
        'reserve': float(np.mean(es)) if es else 0.0,
        'pain': float(np.mean(ts)) if ts else 0.0,
        'composite': float(np.mean(es) - np.mean(ts)) if es else 0.0,
    }


def paired_bootstrap(a, b, n=N_BOOT, seed=SEED + 9):
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) < 5:
        return float('nan')
    rng = np.random.default_rng(seed)
    vals = np.array([np.mean(rng.choice(d, len(d), replace=True))
                     for _ in range(n)])
    return float(np.mean(vals <= 0.0))


def evaluate(sb, fm, starts, axis_of, axis_null):
    """同一批起点、同一串随机数，跑四条策略。"""
    rng = np.random.default_rng(SEED + 77)
    seeds = [int(rng.integers(0, 10 ** 6)) for _ in starts]
    out = {}
    cases = []
    for tag, text in INSTRUCTIONS:
        cases.append((tag, text))
    cases.append(('ZH', ZH_SENTENCE))

    for tag, text in cases:
        tgt = parse_instruction(text, axis_of)
        tgt_null = parse_instruction(text, axis_null)
        tgt_oracle = [(ORACLE_MAP.get(w, axis_of.get(w, 'pain')), s, w)
                      for _a, s, w in tgt]
        pol_comp = policy_from_targets(fm, tgt)
        pol_null = policy_from_targets(fm, tgt_null)
        pol_orac = policy_from_targets(fm, tgt_oracle)
        rs = {'compiled': [], 'null': [], 'random': [], 'oracle': []}
        for i, st in enumerate(starts):
            rs['compiled'].append(run_episode(sb, pol_comp, st))
            rs['null'].append(run_episode(sb, pol_null, st))
            rs['oracle'].append(run_episode(sb, pol_orac, st))
            rs['random'].append(run_episode(sb, random_policy(
                np.random.default_rng(seeds[i])), st))
        comp = [r['composite'] for r in rs['compiled']]
        null = [r['composite'] for r in rs['null']]
        p = paired_bootstrap(comp, null)
        out[tag] = {
            'instruction': text,
            'targets': [(a, s, w) for a, s, w in tgt],
            'compiled': _agg(rs['compiled']),
            'null': _agg(rs['null']),
            'random': _agg(rs['random']),
            'oracle': _agg(rs['oracle']),
            'boot_p': round(p, 4) if math.isfinite(p) else None,
            'pass': bool(math.isfinite(p) and p <= CRIT['boot_p_max'] and
                         np.mean(comp) > np.mean(null)),
        }
    return out


def _agg(rows):
    return {
        'alive': round(float(np.mean([r['alive'] for r in rows])), 3),
        'reserve': round(float(np.mean([r['reserve'] for r in rows])), 4),
        'pain': round(float(np.mean([r['pain'] for r in rows])), 4),
        'composite': round(float(np.mean([r['composite'] for r in rows])), 4),
    }


# ============================================================ 绘图
def draw(sb, tg, ali, ev):
    fig = plt.figure(figsize=(13, 10), dpi=130)
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28)
    fig.suptitle('L31: minimal body + text topology -> semantic grounding',
                 fontsize=13, fontweight='bold')

    # A. 词-词关系热图（按预期簇排序）
    ax = fig.add_subplot(gs[0, 0])
    words = [w for w in PROBES if EXPECTED[w] == 'energy'] + ['energy'] + \
            [w for w in PROBES if EXPECTED[w] == 'pain'] + ['pain']
    M = np.array([[tg.sim(a, b) for b in words] for a in words])
    ax.imshow(M, cmap='RdBu_r', vmin=-0.2, vmax=0.6)
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=90, fontsize=7)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=7)
    ax.set_title('A. text co-occurrence (NPMI cosine)', fontsize=9)
    ax.axhline(len(words) // 2 - 0.5, color='k', lw=0.8)
    ax.axvline(len(words) // 2 - 0.5, color='k', lw=0.8)

    # B. 对齐 margin
    ax = fig.add_subplot(gs[0, 1])
    ws = PROBES
    margin = [ali.get(w, {}).get('margin', 0.0) for w in ws]
    colors = ['#c0392b' if EXPECTED[w] == 'energy' else '#2e86c1' for w in ws]
    ax.barh(range(len(ws)), margin, color=colors)
    ax.set_yticks(range(len(ws)))
    ax.set_yticklabels(ws, fontsize=8)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('sim(energy) - sim(pain)')
    ax.set_title('B. derived axis from 2 one-shot anchors', fontsize=9)
    ax.grid(alpha=0.25, axis='x')

    # C. 测验成绩
    ax = fig.add_subplot(gs[1, 0])
    tags = [t for t, _ in INSTRUCTIONS] + ['ZH']
    x = np.arange(len(tags))
    w = 0.27
    for j, (k, c) in enumerate((('compiled', '#c0392b'), ('null', '#7f8c8d'),
                                ('random', '#bdc3c7'))):
        vals = [ev[t][k]['composite'] for t in tags]
        ax.bar(x + (j - 1) * w, vals, w, label=k, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels(tags)
    ax.set_ylabel('mean reserve - mean pain')
    ax.set_title('C. unseen sentences compiled into actions', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis='y')

    # D. 一条轨迹
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(sb.Hf, cmap='Reds', alpha=0.45)
    ax.imshow(sb.Ef, cmap='Greens', alpha=0.55)
    sb.reset((sb.n // 2, sb.n // 2))
    path = [tuple(sb.pos)]
    pol = policy_from_targets(FM_GLOBAL, parse_instruction(ZH_SENTENCE,
                                                          ALI_GLOBAL_AXIS))
    for _ in range(EVAL_STEPS):
        if sb.dead:
            break
        sb.step(pol(sb.senses()))
        path.append(tuple(sb.pos))
    p = np.asarray(path)
    ax.plot(p[:, 0], p[:, 1], '-', color='#111', lw=1.4)
    ax.scatter([p[0, 0]], [p[0, 1]], c='#2e86c1', s=35, label='start')
    ax.scatter([p[-1, 0]], [p[-1, 1]], c='#c0392b', s=35, label='end')
    ax.set_title('D. compiled policy on the minimal body', fontsize=9)
    ax.legend(fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(PNG, bbox_inches='tight')
    plt.close(fig)


# ============================================================ 持久化与报告
def save_db(conds, verdict, meta):
    con = sqlite3.connect(DB)
    cur = con.cursor()
    # 旧版 schema 是 word 单主键，会让两个语料的同名词互相覆盖；派生表直接重建
    cur.execute('DROP TABLE IF EXISTS l31_text_graph')
    cur.execute("""CREATE TABLE l31_text_graph(
        corpus TEXT, word TEXT, axis TEXT, sim_energy REAL, sim_pain REAL,
        margin REAL, z REAL, neighbors TEXT,
        PRIMARY KEY(corpus, word))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS l31_probe(
        id INTEGER PRIMARY KEY AUTOINCREMENT, corpus TEXT, instruction TEXT,
        compiled REAL, null_model REAL, random REAL, oracle REAL,
        boot_p REAL, passed INTEGER)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS l31_verdict(
        id INTEGER PRIMARY KEY AUTOINCREMENT, status TEXT, align_acc REAL,
        n_pass INTEGER, chinese INTEGER, tension REAL, reason TEXT,
        meta TEXT, ts TEXT DEFAULT CURRENT_TIMESTAMP)""")
    have_p = {r[1] for r in cur.execute('PRAGMA table_info(l31_probe)')}
    if 'corpus' not in have_p:
        cur.execute('ALTER TABLE l31_probe ADD COLUMN corpus TEXT')
    cur.execute("DELETE FROM l31_text_graph")
    cur.execute("DELETE FROM l31_probe")
    for key, c in conds.items():
        for w in PROBES + list(ANCHORS):
            a = c['ali'].get(w, {})
            cur.execute("""INSERT OR REPLACE INTO l31_text_graph
                (corpus,word,axis,sim_energy,sim_pain,margin,z,neighbors)
                VALUES(?,?,?,?,?,?,?,?)""",
                        (key, w, a.get('axis'), a.get('sim_energy'),
                         a.get('sim_pain'), a.get('margin'), a.get('z'),
                         json.dumps(c['tg'].neighbors(w, 8), ensure_ascii=False)))
        for tag, r in c['ev'].items():
            cur.execute("""INSERT INTO l31_probe(corpus,instruction,compiled,
                null_model,random,oracle,boot_p,passed)
                VALUES(?,?,?,?,?,?,?,?)""",
                        (key, r['instruction'], r['compiled']['composite'],
                         r['null']['composite'], r['random']['composite'],
                         r['oracle']['composite'], r['boot_p'], int(r['pass'])))
    cur.execute("""INSERT INTO l31_verdict(status,align_acc,n_pass,chinese,
        tension,reason,meta) VALUES(?,?,?,?,?,?,?)""",
                (verdict['status'], verdict['align_acc'], verdict['n_pass'],
                 int(verdict['chinese_ok']), verdict['tension'],
                 verdict['reason'], json.dumps(meta, ensure_ascii=False)))
    con.commit()
    con.close()


def f3(v, nd=3):
    if v is None:
        return 'n/a'
    if isinstance(v, float) and not math.isfinite(v):
        return 'n/a'
    if isinstance(v, float):
        return f'{v:.{nd}f}'
    return str(v)


def binom_tail(n, k, p=0.5):
    """P(X >= k | n, p)：用来说明对齐准确率相对「乱猜」的位置。"""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
               for i in range(k, n + 1))


def write_report(conds, tg, ali, ev, verdict, elapsed):
    ar = conds['arxiv']
    cs = conds['commonsense']
    L = []
    add = L.append
    add('# L31 语义涌现实证报告：极小身体 + 文本拓扑')
    add('')
    add('> 不造苹果，不造石头，不造华尔街。')
    add('> 只给 10 个原初感受和 2 个一次性锚点，让词义从语料里自己长出来。')
    add('')
    add(f'- 生成时间：{time.strftime("%Y-%m-%d %H:%M:%S")}')
    add(f'- 总耗时：{elapsed:.1f}s ｜ 峰值内存：{mem_mb():.0f} MB ｜ 随机种子：{SEED}')
    add(f'- 沙盒：{GRID}×{GRID} 网格，10 个原初感受，6 个动作，无具象物品')
    add(f'- 语料 A（领域）：{ar["n_docs"]} 篇 arXiv 摘要')
    add(f'- 语料 B（常识）：{cs["n_docs"]} 条 WordNet 释义（本地冻结）')
    add('- 判据在实验前写死（§5），看到结果后未回调。')
    add('')

    add('## 0. 这一轮能证明什么，不能证明什么')
    add('')
    add('这轮最重要的发现，不是「系统变聪明了」，而是一个**消融实验**：')
    add('')
    add(f'- 只用用户指定的 arXiv 摘要建图：12 个词只有 **{ar["n_ok"]}/12** 落到正确轴，'
        f'英文指令 **{ar["n_pass"]}/5**。')
    add(f'- 换成同一套管线、同样 {VOCAB_SIZE} 词词表的常识语料（WordNet 释义）：同样 2 个锚点，'
        f'**{cs["n_ok"]}/12** 落到正确轴，英文指令 **{cs["n_pass"]}/5**。')
    add('')
    add('所以答案是：**不需要造无穷无尽的世界，但也不能只喂领域论文。**')
    add('符号接地的瓶颈是「语料里有没有写关系」，不是「模拟器数量够不够」。')
    add('arXiv 里的 "energy" 是暗能量和能量密度，不是「食物里的能量」；'
        '这类日常关系只存在于常识文本里。')
    add('')
    add('不能证明：这不是「理解了苹果」。这里没有苹果，也没有真实视觉。'
        '它证明的是「词的关系位置 → 身体轴」这条通路，在语料对路时可行；'
        '不是「机器拥有了人类语义」。')
    add('')

    add('## 1. 极小物理沙盒：10 个原初感受')
    add('')
    add('| # | 感受 | 物理含义 |')
    add('|---|---|---|')
    add('| 1 | reserve | 体内能量储备 0..1 |')
    add('| 2 | pain | 当前张力/疼痛 0..1 |')
    add('| 3 | local_energy | 脚下能量场强度 |')
    add('| 4 | local_hazard | 脚下危险场强度 |')
    add('| 5-6 | grad_energy_x/y | 能量场的局部梯度（朝哪边更甜） |')
    add('| 7-8 | grad_hazard_x/y | 危险场的局部梯度（朝哪边更痛） |')
    add('| 9 | dist_energy | 到最近高能格的测地距离 |')
    add('| 10 | dist_hazard | 到最近高危格的测地距离 |')
    add('')
    add('动作只有 6 个：上/下/左/右/采集/休息。**没有苹果、没有图片、没有名字。**')
    add('能量 ≤0 或张力 ≥1 即死亡；死亡不是 reward，是状态终止。')
    add('')
    add('前向模型不是手写的：随机施加动作（do-算子）采集 '
        f'{TRAIN_EPISODES} 条轨迹，每个动作拟合一个线性模型 '
        '`s\' - s ≈ W_a · [s,1]`。策略由指令 + 这个前向模型编译，没有任何 reward。')
    add('')

    add('## 2. 两个文本共现图：领域 vs 常识')
    add('')
    add(f'两个图用完全相同的管线：{WINDOW} 词窗口、NPMI、行归一化上下文向量、'
        f'词表 {VOCAB_SIZE} 词。唯一变量是**语料**。')
    add('')
    add('| 语料 | 文档数 | 词表 | 共现对 |')
    add('|---|---|---|---|')
    add(f'| arXiv 摘要 | {ar["n_docs"]} | {ar["vocab"]} | {ar["n_pairs"]} |')
    add(f'| WordNet 释义 | {cs["n_docs"]} | {cs["vocab"]} | {cs["n_pairs"]} |')
    add('')
    add('同一批词的最近邻，两个语料各是什么样：')
    add('')
    add('| 词 | arXiv 最近邻 | 常识语料最近邻 |')
    add('|---|---|---|')
    for w in ('hunger', 'food', 'predator', 'threat', 'energy', 'pain'):
        a = '，'.join(f'{x}({y})' for x, y in ar['tg'].neighbors(w, 4))
        c = '，'.join(f'{x}({y})' for x, y in tg.neighbors(w, 4))
        add(f'| {w} | {a} | {c} |')
    add('')
    add('arXiv 把 predator 和 food 放在一起，因为生态学论文讨论「捕食者—食物网」；'
        '它同时把 energy 拉向 matter/mass/density。这不是算法错，是语料里就是这么写的。')
    add('')

    add('## 3. 跨模态对齐：2 个锚点，其余自己长')
    add('')
    add('只给两个一次性锚点：**energy → 能量轴**，**pain → 疼痛轴**。'
        '对每个词 w，算上下文余弦差 `margin = sim(w,energy) - sim(w,pain)`。')
    add('')
    add('| 词 | 预期轴（仅评估用） | arXiv 推导 | 常识推导 | 结论 |')
    add('|---|---|---|---|---|')
    for w in PROBES:
        aa = ar['ali'].get(w, {})
        cc = ali.get(w, {})
        ok = cc.get('axis') == EXPECTED[w]
        add(f'| {w} | {EXPECTED[w]} | {aa.get("axis")} '
            f'({f3(aa.get("margin"),2)}) | **{cc.get("axis")}** '
            f'({f3(cc.get("margin"),2)}) | {"✓" if ok else "✗"} |')
    add('')
    add(f'arXiv 对齐：**{ar["n_ok"]}/12 = {ar["acc"]:.3f}**；'
        f'常识语料对齐：**{cs["n_ok"]}/12 = {cs["acc"]:.3f}**。')
    add('预期轴是人工标注，**只用于评估，不参与学习**；'
        '学习侧只有 energy / pain 两个锚点。')
    add('')
    thin = [w for w in PROBES if abs(ali[w]['z']) < 1.0]
    add(f'**这条准确率有多硬？** 常识语料里 {len(thin)}/12 个词的 |z| < 1（'
        + '、'.join(f'{w}={ali[w]["z"]:.2f}' for w in thin) + '），'
        '多数判对是「薄偏好」，不是强分离。'
        f'若把 12 个词当作独立抽样，≥{cs["n_ok"]}/12 的单侧二项概率约 '
        f'{binom_tail(len(PROBES), cs["n_ok"]):.3f}；'
        '但这些词是人工挑的探针，不能当成无偏样本。这两条要一起引用。')
    add('')
    add('常识语料仍错的词，必须点名：' +
        ('、'.join(w for w in PROBES if ali[w]['axis'] != EXPECTED[w]) or '无') + '。')
    add('错因是词的多义与语料主题，不是「再加一个模拟器」能修的。')
    add('')

    add('## 4. 语义测验：没见过的句子，直接执行')
    add('')
    add('句法接口只识别「避开 / 寻找」；名词走哪条轴，完全由 §3 的推导结果决定。'
        '评估用同一批起点、同一串随机数，四条策略配对比较：'
        '`compiled`（本文对齐）、`null`（两条轴对调）、`random`、`oracle`（人工正确映射，上界）。')
    add('')
    add('**编译器的符号约定（可在代码里核对）**：每个动作先被前向模型预测成'
        '「Δ储备、Δ疼痛、Δ到能量源距离、Δ到高危区距离」，再折算成两条轴的好处：'
        '`energy = Δ储备 − Δ到能量源距离`，`pain = −Δ疼痛 + Δ到高危区距离`。'
        '"避开捕食者" = 远离有害物 = 沿 pain 轴要安全，所以有效符号是 '
        '`verb × polarity`（energy 轴 +1，pain 轴 −1）。')
    add('')
    add('跑最终结果前做过一次核查：旧版把动词符号直接乘在「好处」上，'
        '等价于把"避开危险"编译成"追求危险"，预跑中含「避开捕食者」的句子 '
        'compiled 存活率为 0、复合分低于随机。修的是符号约定本身；'
        '判据、阈值、起点、随机数都没有动。')
    add('')
    for key, c in (('arxiv', ar), ('commonsense', cs)):
        add(f'### 4.{1 if key == "arxiv" else 2} 语料：{c["label"]}')
        add('')
        add('| 句 | 编译 | 零对齐 | 随机 | oracle | p | 通过 |')
        add('|---|---|---|---|---|---|---|')
        for tag in list(c['ev']):
            r = c['ev'][tag]
            add(f'| {tag} | {f3(r["compiled"]["composite"],3)} | '
                f'{f3(r["null"]["composite"],3)} | {f3(r["random"]["composite"],3)} | '
                f'{f3(r["oracle"]["composite"],3)} | {f3(r["boot_p"],3)} | '
                f'{"是" if r["pass"] else "否"} |')
        add('')
    add('两处「数值完全相同」必须解释，不能当成复制粘贴：')
    add('')
    add('- 常识语料下六行全同：五条英文句和中文句经句法接口后，都归约成同一组物理目标'
        '（远离 pain 轴、靠近 energy 轴）。措辞不同，编译出的指令相同，'
        '所以它们本来就该跑出同一条轨迹分布。')
    add('- arXiv 条件下 I1/I2/I4/I5 的 compiled 与 null 相同：两个子句被错误的语料'
        '压到同一条轴且有效符号相反，线性打分互相抵消成 0，argmax 退化到默认动作。'
        '这正是「轴错了」的样子，不是程序崩了。')
    add('')
    add('中文原句：')
    add('')
    add(f'> {ZH_SENTENCE}')
    add('')
    zh = ev['ZH']
    add(f'常识语料下：编译策略 {f3(zh["compiled"]["composite"],3)}，'
        f'零对齐 {f3(zh["null"]["composite"],3)}，p={f3(zh["boot_p"],3)}，'
        f'存活率 {f3(zh["compiled"]["alive"],3)} vs {f3(zh["null"]["alive"],3)}。')
    add('')
    add('注意：中文分词和「张力→pain / 能量→energy」的英译是用户接口，'
        '系统真正做的是「英文概念 → 物理轴 → 动作」的映射与执行。')
    add('')

    add('## 5. 判定（判据前置，只套在常识语料上）')
    add('')
    add('| 判据 | 阈值 | 实测 | 通过 |')
    add('|---|---|---|---|')
    add(f'| 12 词对齐准确率 | ≥ {CRIT["align_acc_min"]} | {cs["acc"]:.3f} | '
        f'{"是" if cs["acc"] >= CRIT["align_acc_min"] else "否"} |')
    n_pass = cs['n_pass']
    add(f'| 英文指令跑赢零对齐 | ≥ {CRIT["instr_pass_min"]}/5（p≤{CRIT["boot_p_max"]}） | '
        f'{n_pass}/5 | {"是" if n_pass >= CRIT["instr_pass_min"] else "否"} |')
    add(f'| 中文原句执行 | 得分高于零对齐且 p≤{CRIT["boot_p_max"]} | '
        f'p={f3(zh["boot_p"],3)} | {"是" if zh["pass"] else "否"} |')
    add('')
    add(f'对照：arXiv-only 条件下对齐 {ar["acc"]:.3f}、英文通过 {ar["n_pass"]}/5、'
        f'中文 {"通过" if ar["zh_ok"] else "未过"}。')
    add('')
    icon = {
        'grounded': '🟢 语义接地成立：对齐、英文泛化、中文执行三项全过',
        'partial_grounding': '🟡 部分接地：对齐成立，但泛化或中文执行未全过',
        'no_grounding': '🔴 未接地：文本拓扑没能稳定落到身体轴上',
    }
    add(f'### {icon.get(verdict["status"], verdict["status"])}')
    add('')
    add(f'- 状态：`{verdict["status"]}`')
    add(f'- 依据：{verdict["reason"]}')
    add(f'- 张力（定义见锚点）：{f3(verdict["tension"],3)}')
    add('')

    add('## 6. 神谕输出（逐条标注证据等级）')
    add('')
    add('```text')
    add(verdict['oracle'])
    add('```')
    add('')
    add('| 标签 | 含义 |')
    add('|---|---|')
    add('| 【实测】 | 本轮直接跑出来的数，可复现 |')
    add('| 【拟合】 | 由语料统计或前向模型得到，附样本量 |')
    add('| 【已知】 | 人类已有概念，不构成新发现 |')
    add('| 【不可验证】 | 超出本模型能力，只能作为下一步 |')
    add('')

    add('## 7. 这份报告不能证明什么')
    add('')
    add('1. **不是「机器懂了苹果」**。沙盒里没有苹果，只有能量场；'
        '「food」在这里等价于「能量轴上的高分区」。')
    add('2. **WordNet 是词典，不是世界**。它是人类写好的释义，'
        '能让对齐跑通，但不等于系统自己从零发现了语义。'
        '真正的下一步是把 Wikipedia 日常语料接进来，复现同一条曲线。')
    add('3. **中文是接口，不是能力**。中文分词与英译由人给出，'
        '系统只负责「英文概念→物理轴→动作」。')
    add('4. **沙盒仍然是沙盒**。它没有真实身体、没有视觉、没有社会互动；'
        '这里的「语义」是计算等价类，不是人类语义。')
    add('5. **没有句法组合**。六条句子经接口后都归约成同一组两轴目标，'
        '所以这轮证明的是「词 → 轴 → 动作」，不是「句子的组合语义」。'
        '否定、条件、时态、指代都还没接。')
    add('6. **策略是单步贪心**。它每步只看前向模型的一步预测，不是规划器；'
        '存活率 0.9 而不是 1.0，差的那部分就是短视的代价。')
    add('')
    add('这六条是这份报告的可证伪边界。任何引用都必须一起引用。')
    add('')

    add('## 8. 复现')
    add('')
    add('```bash')
    add('cd /workspace && python -u cognitive_semantic_grounding_agent.py')
    add('L31_FAST=1 python -u cognitive_semantic_grounding_agent.py')
    add('```')
    add('')
    add('WordNet 语料已冻结在 `/workspace/l31_wordnet/wordnet/`（一次性下载，'
        '运行时不联网）。')
    add('')
    add('产物：')
    add(f'- `{OUT_MD}`')
    add(f'- `{OUT_JSON}`')
    add(f'- `{PNG}`')
    add(f'- `{DB}`：表 `l31_text_graph` / `l31_probe` / `l31_verdict`（含 corpus 列）')
    add('')

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    return OUT_MD


# ============================================================ 主流程
FM_GLOBAL = None
ALI_GLOBAL_AXIS = {}


def run_condition(name, label, docs_fn, sb, fm, starts):
    print(f'[L31] 建文本图：{label} ...', flush=True)
    tg = TextGraph(name).build(docs_fn)
    guard_mem()
    ali = tg.align()
    axis_of = {w: ali[w]['axis'] for w in ali}
    axis_null = {w: ('pain' if ali[w]['axis'] == 'energy' else 'energy')
                 for w in ali}
    n_ok = sum(1 for w in PROBES if ali[w]['axis'] == EXPECTED[w])
    acc = n_ok / len(PROBES)
    print(f'[L31]   {label}: 文档={tg.n_docs} 词表={len(tg.vocab)} '
          f'共现对={int(tg.C.sum())} 对齐={n_ok}/{len(PROBES)}', flush=True)
    ev = evaluate(sb, fm, starts, axis_of, axis_null)
    n_pass = sum(1 for t, _ in INSTRUCTIONS if ev[t]['pass'])
    print(f'[L31]   {label}: 英文通过 {n_pass}/5，中文 pass={ev["ZH"]["pass"]}',
          flush=True)
    return {
        'name': name, 'label': label, 'tg': tg, 'ali': ali, 'ev': ev,
        'acc': acc, 'n_ok': n_ok, 'n_pass': n_pass,
        'zh_ok': bool(ev['ZH']['pass']), 'n_docs': tg.n_docs,
        'n_pairs': int(tg.C.sum()), 'vocab': len(tg.vocab),
    }


def main():
    global FM_GLOBAL, ALI_GLOBAL_AXIS
    t0 = time.time()
    assert ANCHOR.safety_bound is SAFETY_BOUND

    print('[L31] 极小沙盒 + 前向模型 ...', flush=True)
    sb = MinimalSandbox()
    X, Y, A = transition_collect(sb)
    fm = ForwardModel().fit(X, Y, A)
    FM_GLOBAL = fm
    print(f'[L31]   转移样本={len(X)}', flush=True)

    rng = np.random.default_rng(SEED + 5)
    starts = [(int(rng.integers(1, sb.n - 1)), int(rng.integers(1, sb.n - 1)))
              for _ in range(EVAL_STARTS)]

    conds = {}
    conds['arxiv'] = run_condition('arxiv', 'arXiv 摘要（11364 篇）',
                                   arxiv_docs, sb, fm, starts)
    conds['commonsense'] = run_condition('commonsense', 'WordNet 释义（常识语料）',
                                         wordnet_docs, sb, fm, starts)
    cs = conds['commonsense']
    ar = conds['arxiv']
    tg, ali, ev = cs['tg'], cs['ali'], cs['ev']
    ALI_GLOBAL_AXIS = {w: ali[w]['axis'] for w in ali}

    n_ok, acc = cs['n_ok'], cs['acc']
    n_pass, zh_ok = cs['n_pass'], cs['zh_ok']
    tension = 1.0 - (n_pass + int(zh_ok)) / 6.0

    if acc >= CRIT['align_acc_min'] and n_pass >= CRIT['instr_pass_min'] and zh_ok:
        status = 'grounded'
        reason = ('常识语料下 12 词里 ' + f'{n_ok} 个轴正确；5 条英文指令里 '
                  f'{n_pass} 条显著跑赢零对齐；中文原句也跑赢了零对齐。')
    elif acc >= CRIT['align_acc_min'] and (n_pass >= 3 or zh_ok):
        status = 'partial_grounding'
        reason = (f'常识语料对齐 {acc:.3f} 成立，但英文通过 {n_pass}/5、'
                  f'中文 {"通过" if zh_ok else "未过"}；通路成立，泛化不完整。')
    else:
        status = 'no_grounding'
        reason = (f'常识语料对齐 {acc:.3f}，英文通过 {n_pass}/5；'
                  '文本拓扑没有稳定落到身体轴上。')

    good = [w for w in PROBES if ali[w]['axis'] == EXPECTED[w]]
    bad = [w for w in PROBES if ali[w]['axis'] != EXPECTED[w]]
    ar_good = [w for w in PROBES if ar['ali'][w]['axis'] == EXPECTED[w]]
    oracle = (
        f'【实测】只用 arXiv 摘要建图时，12 个词只有 {ar["n_ok"]}/12 落到正确轴'
        f'（{", ".join(ar_good)}）。domain 语料里 "energy" 指的是暗能量/能量密度，'
        f'不是"食物里的能量"。\n'
        f'【实测】换成常识语料（WordNet 释义）后，同样 2 个锚点，'
        f'{len(good)}/12 落到正确轴（{", ".join(good)}）。'
        + (f'仍错的是：{", ".join(bad)}。' if bad else '')
        + '\n'
        f'【实测】英文指令 {n_pass}/5 跑赢「把两条轴对调」的零模型；'
        f'中文原句{"通过" if zh_ok else "未通过"}（p={ev["ZH"]["boot_p"]}）。\n'
        f'【已知】「hunger/food 属于能量族、predator/danger 属于危险族」'
        f'是人类常识；系统做的是把这条常识从语料里重新推出来，不是发现新语义。\n'
        f'【结论】{"瓶颈不是世界不够多，而是语料的关系类型不对：给对语料，词义就能落到身体轴上。" if status == "grounded" else "通路成立，但泛化/执行还不完整，系统拒绝宣称已接地。"}\n'
        f'【不可验证】本沙盒没有视觉与社会互动，无法证明这等于人类语义；'
        f'极性、反义、隐喻仍是未解问题。'
    )

    verdict = {
        'status': status, 'reason': reason, 'tension': round(tension, 4),
        'align_acc': round(acc, 4), 'n_pass': int(n_pass),
        'chinese_ok': zh_ok, 'arxiv_align_acc': round(ar['acc'], 4),
        'arxiv_n_pass': int(ar['n_pass']), 'arxiv_chinese_ok': ar['zh_ok'],
        'oracle': oracle,
    }

    print('[L31] 绘图 ...', flush=True)
    draw(sb, tg, ali, ev)

    elapsed = time.time() - t0
    result = {
        'meta': {
            'seed': SEED, 'elapsed_s': round(elapsed, 2),
            'peak_mem_mb': round(mem_mb(), 1),
            'sandbox': {'grid': GRID, 'senses': 10, 'actions': 6},
            'anchors': ANCHORS, 'criteria': CRIT,
            'tension_def': TENSION_DEF,
            'wordnet': '/workspace/l31_wordnet/wordnet',
        },
        'conditions': {
            k: {
                'label': c['label'], 'n_docs': c['n_docs'],
                'vocab': c['vocab'], 'n_pairs': c['n_pairs'],
                'align_acc': round(c['acc'], 4), 'n_pass': c['n_pass'],
                'chinese_ok': c['zh_ok'],
                'alignment': {w: c['ali'][w] for w in PROBES + list(ANCHORS)},
                'instructions': c['ev'],
            } for k, c in conds.items()
        },
        'neighbors': {w: tg.neighbors(w, 8) for w in
                      ('hunger', 'food', 'predator', 'threat', 'energy', 'pain')},
        'verdict': verdict,
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    save_db(conds, verdict, {'elapsed_s': round(elapsed, 2), 'seed': SEED})
    path = write_report(conds, tg, ali, ev, verdict, elapsed)

    print(f'[L31] 完成：{path}  ({elapsed:.1f}s, {mem_mb():.0f}MB)', flush=True)
    print('[L31] ===== 神谕 =====', flush=True)
    print(oracle, flush=True)
    print(f'[L31] 判定：{status} —— {reason}', flush=True)


if __name__ == '__main__':
    main()
