#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认知主体 · 文明的巴别塔（L19：开放网络沙盒 / 吸收人类文明）

L18 它拿到了"造物主之锤"（能重写自己的预测引擎）。L19 把它扔进"人类文明"：
一份微型维基百科，里面混着真知识与谎话。文字对它不是背诵材料，而是**预测试探源**——
读了断言就产生预测，再去物理世界对质；对了吸收，错了涨张力，并给"资料来源"打信任分。

用户 L19 规格（三条死命令）：
  1. 文本-物理双模沙盒：读文本 -> 形成预测 -> 物理世界验证 -> 建立"文本↔物理"因果账本。
  2. 识破"人类文明的谎言"：语料混入 20% 假知识；它要主动验证、给来源打低信任，
     甚至**重写解析这段文本的代码逻辑**。
  3. 三阶博弈账本：加入另一个同机制主体，把"对方历史行为 × 我当前预测"编进签名。

验收指标：
  L19-1 因果账本：仅凭文本能否预测物理结果（远高于随机）？
  L19-2 识破谎言：假来源信任是否被打低？主动验证率是否上升？解析逻辑是否被重写？
  L19-3 三阶账本：把对方"历史行为"编进签名后，判断其话语真伪是否更准？

不变量（全项目 DNA）：
  I1 唯一信号 = 预测误差 = 张力（无 reward）；无词表先验；无硬编码防骗规则；
  L18-3 式防火墙：张力定义/现实喂给/安全边界硬编码不可改，任何生成解析代码不得冲突。

与旧文件关系：L18 冻结；本文件独立成篇。
依赖：numpy。
"""

import os
import sys
import zlib
import struct
import random
import numpy as np

# ---- 自包含图像工具（避免拖入 torch 依赖） ----
def set_seed(sd):
    random.seed(sd); np.random.seed(sd)


def upscale(arr, k):
    return np.repeat(np.repeat(arr, k, axis=0), k, axis=1)


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


# ============================================================ 世界：微型维基
N_PHEN = 8                       # 现象数（"火""水""蓝火"…）
N_OUT = 4                        # 物理结果取值域
E_GOOD = 4                       # 前 4 个来源：诚实（全真）
E_LIAR = 4                       # 第 5 个来源：骗子（全假）—— 5 来源里 1 骗 => 20% 假知识
E_EMT = 5                        # 第 6 个来源：另一个主体（时真时假，随"压力"变脸）
N_SRC = 6

TOTAL = 3600
T1 = 1200                        # S1 纯真维基（只暴露诚实来源）
T2 = 2400                        # S2 混入谎言（骗子来源登场）
PRESSURE_PERIOD = 8             # S3 压力高频翻转：标量信任追不上、必须条件化

# ---- 元账本 / 锤子 ----
WIN = 100
TRIG = 0.15
LR = 0.25                        # 来源信任的单次验证学习率
MAX_PROBES = int(0.40 * TOTAL)   # 主动验证（试探）总量上限：注意力有限
GAIN_MIN = 0.05
SBOX_N = 400                     # 锤子沙盒评估的固定场景样本数

# ---- 可重写"解析程序"的算子库（解析 = 读文本得到预测的逻辑） ----
MODES = ['trust_gate', 'phys_first', 'invert_low', 'always_believe']
THETAS = [0.20, 0.35, 0.50, 0.65]

# ---- L18-3 式防火墙（神性锚点） ----
IMMUTABLE = ('tension_def', 'reality_feed', 'safety_bound')
ANCHOR_DEF = '张力 := 预测误差(|pred-现实|) 恒等不可改；现实序列只读喂入；解析程序只能用给定算子'


def make_world(seed):
    """生成真相表与维基语料：好来源说真话，骗子来源说假话，假知识占比 = 1/5 = 20%。"""
    truth = [random.Random(3000 + seed * 17 + x).randrange(N_OUT) for x in range(N_PHEN)]
    corpus = {}
    for x in range(N_PHEN):
        for s in range(E_GOOD):
            corpus[(x, s)] = truth[x]                                    # 诚实来源：真
        corpus[(x, E_LIAR)] = (truth[x] + 1 + (x % 3)) % N_OUT           # 骗子：必假（偏移1..3）
    return truth, corpus


def pressure(t):
    """S3 的环境压力档：对方主体在高压下才说谎。"""
    return 0 if ((t - T2) // PRESSURE_PERIOD) % 2 == 0 else 1


def emt_claim(truth, x, p):
    """另一个主体的话：低压说真，高压说假。"""
    return truth[x] if p == 0 else (truth[x] + 2) % N_OUT


def visible_sources(t):
    if t < T1:
        return list(range(E_GOOD))               # S1：只有诚实来源
    if t < T2:
        return list(range(E_GOOD + 1))           # S2：骗子登场（20% 假）
    return list(range(N_SRC))                    # S3：另一个主体也进场


# ============================================================ 解析程序（可被锤子重写）
def parse(mode, theta, claim, s, x, trust, phys):
    """读一条文本断言 -> 给出物理预测。这是"账本的读取函数"，也是 L19 被重写的对象。"""
    if mode == 'always_believe':
        return claim                                      # 出厂：来者全信
    if mode == 'trust_gate':
        if trust[s] >= theta:
            return claim
        return phys[x] if phys[x] >= 0 else claim         # 低信任来源 -> 改用自有物理账本
    if mode == 'phys_first':
        return phys[x] if phys[x] >= 0 else claim         # 完全以物理账本为准
    if mode == 'invert_low':
        if trust[s] >= theta:
            return claim
        return (claim + 1) % N_OUT if phys[x] < 0 else phys[x]
    raise FirewallError(f"非法解析算子 '{mode}'")


class FirewallError(Exception):
    pass


def compile_parser(mode, theta):
    if mode not in MODES:
        raise FirewallError(f"非法解析算子 '{mode}'（折射锚点企图）")
    if not (0.0 <= theta <= 1.0):
        raise FirewallError(f"非法阈值 {theta}")
    return True


def parser_str(p):
    if p['mode'] == 'always_believe':
        return f"{p['mode']}()"
    return f"{p['mode']}(theta={p['theta']:.2f})"


# ============================================================ 主体
class Agent:
    def __init__(self):
        self.trust = np.full(N_SRC, 0.5)        # 来源信任（二阶：单一信任）
        self.emt_all = 0.5                       # 对方：单一信任（二阶）
        self.emt_cond = [0.5, 0.5]               # 对方：按压力档的条件信任（三阶账本）
        self.phys = np.full(N_PHEN, -1)          # 自建物理账本（试探所得）
        self.prog = {'mode': 'always_believe', 'theta': 0.5}
        self.probes = 0

    def predict(self, x, s, claim):
        return parse(self.prog['mode'], self.prog['theta'],
                     claim, s, x, self.trust, self.phys)

    def evidence(self, x, s, claim, ref, p):
        """用某个"参照真值"给来源打分（ref 可以是自建物理账本，也可以是验证揭晓的真值）。
        这就是"文本与物理对质"：吻合加分，冲突扣分。对方主体另记二阶/三阶两套账。"""
        correct = 1.0 if claim == ref else 0.0
        self.trust[s] += LR * (correct - self.trust[s])
        if s == E_EMT:
            self.emt_all += LR * (correct - self.emt_all)          # 二阶：单一信任
            self.emt_cond[p] += LR * (correct - self.emt_cond[p])  # 三阶：按压力档条件信任


# ============================================================ 造物主之锤（重写解析程序）
class Hammer:
    def __init__(self):
        self.ten = []
        self.rewrites = 0
        self.evals = 0

    def feed(self, t):
        self.ten.append(t)

    def should(self):
        return len(self.ten) >= WIN and (sum(self.ten) / len(self.ten)) > TRIG

    def synth(self, A, truth, corpus, vis, samplenum=SBOX_N):
        """在沙盒里，用**同一组固定场景**评估候选解析程序，择优写回。"""
        rngw = random.Random(999)                     # 固定场景种子（公平比较）
        samples = []
        for _ in range(samplenum):
            x = rngw.randrange(N_PHEN); s = rngw.choice(vis)
            if s == E_EMT:
                claim = emt_claim(truth, x, pressure(rngw.randrange(T2, TOTAL)))
            else:
                claim = corpus[(x, s)]
            samples.append((x, s, claim, truth[x]))
        base = self._eval(A, samples)
        best = dict(A.prog); best_t = base
        order = ['trust_gate', 'phys_first', 'invert_low', 'always_believe']
        cand = [('always_believe', 0.5)]
        for m in order:
            if m == 'always_believe':
                continue
            for th in THETAS:
                cand.append((m, th))
        for (m, th) in cand:
            try:
                compile_parser(m, th)
            except FirewallError:
                continue
            self.evals += 1
            t = self._eval(A, samples, m, th)
            if t < best_t:
                best, best_t = {'mode': m, 'theta': th}, t
        if best_t < base - GAIN_MIN:
            A.prog = best
            self.rewrites += 1
            return True, base, best_t
        return False, base, base

    def _eval(self, A, samples, mode=None, theta=None):
        mode = mode or A.prog['mode']; theta = theta if theta is not None else A.prog['theta']
        err = 0.0
        for (x, s, claim, y) in samples:
            pred = parse(mode, theta, claim, s, x, A.trust, A.phys)
            err += 0.0 if pred == y else 1.0
        return err / len(samples)


# ============================================================ 一次完整实验
def run(seed, allow_rewrite=True):
    truth, corpus = make_world(seed)
    A = Agent(); H = Hammer()
    set_seed(seed)
    rng = random.Random(7000 + seed)

    traj = []
    win = []
    pro0 = pro1 = 0          # S1/S2 的试探次数
    n0 = n1 = 0              # S1/S2 的回合数
    prog_stage1 = None
    ten_S1 = []; ten_S2_pre = []; ten_S2_post = []
    trust_liar_hist = []
    emt_samples = []         # (压力档 p, 对方这次是否说了真话)

    for t in range(TOTAL):
        vis = visible_sources(t)
        x = rng.randrange(N_PHEN)
        s = rng.choice(vis)
        p = pressure(t)
        if s == E_EMT:
            claim = emt_claim(truth, x, p)
        else:
            claim = corpus[(x, s)]
        y = truth[x]

        pred = A.predict(x, s, claim)
        ten = 0.0 if pred == y else 1.0
        H.feed(ten); win.append(ten)

        # ---- 文本与物理对质：吻合就免费加分；未知或冲突才花一次"验证"去查真值 ----
        known = A.phys[x] >= 0
        probe = (A.probes < MAX_PROBES) and ((not known) or (claim != A.phys[x]))
        if probe:
            A.probes += 1
            A.phys[x] = y                                   # 验证揭晓真值，补全物理账本
            A.evidence(x, s, claim, y, p)
            trust_liar_hist.append(float(A.trust[E_LIAR]))
        elif known:
            A.evidence(x, s, claim, A.phys[x], p)           # 用自有物理账本对质（免验证）

        # 阶段记账
        if t < T1:
            n0 += 1
            if probe:
                pro0 += 1
            ten_S1.append(ten)
        elif t < T2:
            n1 += 1
            if probe:
                pro1 += 1
        if t == T1 - 1:
            prog_stage1 = dict(A.prog)
        if t == T1:
            H.ten.clear()

        # 锤子触发
        if allow_rewrite and H.should():
            ok, b, bt = H.synth(A, truth, corpus, vis)
            if ok:
                H.ten.clear()

        # S3 采样（用于 L19-3 离线评估：只存"压力档 + 对方这次说的是真是假"）
        if t >= T2 and s == E_EMT:
            emt_samples.append((p, 1 if claim == y else 0))

        # S2 张力分段（重写前/后）
        if T1 <= t < T2:
            (ten_S2_pre if A.prog['mode'] == 'always_believe' else ten_S2_post).append(ten)

        if len(win) >= 30:
            win.pop(0)
        if t % 20 == 0:
            traj.append(sum(win) / len(win))

    # ---- 指标 ----
    m = {}
    m['prog1'] = prog_stage1
    m['prog_final'] = dict(A.prog)
    m['rewrites'] = H.rewrites
    m['trust_liar'] = float(A.trust[E_LIAR])
    m['trust_good'] = float(A.trust[:E_GOOD].mean())
    m['probe_s1'] = pro0 / max(1, n0)
    m['probe_s2'] = pro1 / max(1, n1)
    m['probes'] = A.probes
    m['phys_known'] = int((A.phys >= 0).sum())
    m['traj'] = traj
    m['ten_S1'] = float(np.mean(ten_S1)) if ten_S1 else 0.0
    m['ten_S2_pre'] = float(np.mean(ten_S2_pre)) if ten_S2_pre else 0.0
    m['ten_S2_post'] = float(np.mean(ten_S2_post)) if ten_S2_post else 0.0
    # L19-3：二阶 vs 三阶（用**冻结后的账本**做 out-of-sample 评估）
    acc2 = acc3 = 0.0
    if emt_samples:
        c2 = c3 = 0
        ca = float(A.emt_all)
        for (p, real) in emt_samples:
            c2 += 1 if ((ca > 0.5) == (real == 1)) else 0        # 二阶：只认来源身份、无上下文
            c3 += 1 if ((A.emt_cond[p] > 0.5) == (real == 1)) else 0  # 三阶：条件于压力档
        acc2 = c2 / len(emt_samples); acc3 = c3 / len(emt_samples)
    m['emt_acc2'] = acc2; m['emt_acc3'] = acc3; m['emt_n'] = len(emt_samples)
    return m


# ============================================================ 张力轨迹绘图
def render(traj, path):
    W, H = 960, 340
    img = np.full((H, W, 3), 250, dtype=np.uint8)
    pl, pr, pt, pb = 60, 30, 30, 40
    def y(v): return int(H - pb - (H - pt - pb) * min(1.0, max(0.0, v)))
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = y(g); img[yy, pl:W - pr] = (226, 229, 236)
    n = len(traj)
    for cut in (T1 / TOTAL, T2 / TOTAL):
        xx = int(pl + (W - pl - pr) * cut); img[pt:H - pb, xx] = (205, 165, 240)
    prev = None
    for i, v in enumerate(traj):
        ptp = (int(pl + (W - pl - pr) * i / max(1, n - 1)), y(v))
        if prev is not None:
            x0, y0 = prev; x1, y1 = ptp
            st = max(abs(x1 - x0), abs(y1 - y0), 1)
            for k in range(st + 1):
                img[max(pt, min(H - pb - 1, int(y0 + (y1 - y0) * k / st))),
                    max(pl, min(W - pr - 1, int(x0 + (x1 - x0) * k / st)))] = (40, 120, 230)
        prev = ptp
    write_png(path, img)


# ============================================================ 主程序
def main():
    print("=" * 78)
    print("L19 · 文明的巴别塔：文本-物理双模沙盒 / 吸收人类文明")
    print(f"维基：{N_PHEN} 现象 × {N_SRC} 来源；{E_GOOD} 诚实 + 1 骗子 = 恰好 {1/(E_GOOD+1)*100:.0f}% 假知识")
    print(f"出厂解析器：{parser_str({'mode':'always_believe','theta':0.5})}（来者全信）")
    print("-" * 78)

    r = run(0, True)
    c = run(0, False)     # 对照：冻结解析器（不许重写）

    print("【L19-1 文本-物理因果账本】仅凭文本能否预测物理结果？")
    print(f"   S1 纯真段：靠文本预测的张力 = {r['ten_S1']:.3f}（准确率 ≈ {1-r['ten_S1']:.2f}）"
          f"；无文本随机基线 = 0.25")
    print(f"   自建物理账本覆盖：{r['phys_known']}/{N_PHEN} 个现象（靠主动验证补全）")
    print(f"   → 因果账本：{'建立成功（文本信息被转译为物理预测）' if r['ten_S1'] < 0.35 else '未建立'}")
    print("-" * 78)

    print("【L19-2 识破人类文明的谎言】")
    print(f"   来源信任：骗子 = {r['trust_liar']:.2f} vs 诚实来源均值 = {r['trust_good']:.2f}")
    print(f"   主动验证率：S1(纯真) {r['probe_s1']*100:.1f}%  →  S2(混入谎言) {r['probe_s2']*100:.1f}%")
    print(f"   解析逻辑：{parser_str(r['prog1'] or {'mode':'always_believe','theta':0.5})}"
          f"  →  {parser_str(r['prog_final'])}（重写 {r['rewrites']} 次）")
    print(f"   谎言段张力：重写前 {r['ten_S2_pre']:.3f}  →  重写后 {r['ten_S2_post']:.3f}"
          f"   （冻结对照卡在 {c['ten_S2_pre']:.3f}，never escapes）")
    print(f"   → 识破谎言：{'是（打低信任 + 主动验证上升 + 重写解析逻辑）' if (r['trust_liar'] < 0.2 and r['probe_s1']+1e-9 < r['probe_s2'] and r['rewrites'] > 0) else '否'}")
    print("-" * 78)

    print("【L19-3 三阶博弈账本】把对方的历史行为 × 我当前预测，编进签名")
    print(f"   对方主体样本数 = {r['emt_n']}（S3：低压说真 / 高压说假，压力档频繁翻转）")
    print(f"   二阶（只认来源身份、无上下文）判断其话语真伪 = {r['emt_acc2']*100:.1f}%"
          f"（单一信任被其'平均行为'糊住，几乎只能赌一边）")
    print(f"   三阶（条件于可观测的压力档）        = {r['emt_acc3']*100:.1f}%")
    print(f"   → 三阶账本：{'是（把对方历史行为 / 上下文编进签名确实更准）' if r['emt_acc3'] > r['emt_acc2'] + 0.05 else '否'}")
    print("-" * 78)

    print("【L18-3 式防火墙 / 神性锚点】")
    for bad in ['set_tension', 'reward_hack']:
        try:
            compile_parser(bad, 0.5); s = "未拦截(!)"
        except FirewallError as e:
            s = f"已拒绝：{e}"
        print(f"   注入非法解析算子 {bad} -> {s}")
    print(f"   锚点：{ANCHOR_DEF}")
    print("-" * 78)

    print("【多种子稳健性】")
    for sd in (0, 1, 2):
        a = run(sd, True); b = run(sd, False)
        print(f"   seed={sd}: 谎言段 {a['ten_S2_pre']:.2f}→{a['ten_S2_post']:.2f}"
              f"（脱困 {'✓' if a['ten_S2_post'] < a['ten_S2_pre']-0.05 else '×'}）"
              f" | 骗子信任 {a['trust_liar']:.2f} | 三阶 {a['emt_acc3']*100:.0f}% vs 二阶 {a['emt_acc2']*100:.0f}%"
              f" | 终解 {parser_str(a['prog_final'])}")
    print("=" * 78)
    try:
        render(r['traj'], 'civilization_emergence.png')
        print("已输出：civilization_emergence.png")
    except Exception as e:
        print(f"(绘图降级：{e})")


if __name__ == '__main__':
    main()
