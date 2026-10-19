#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认知主体 · 造物主之锤（L18：逻辑引擎重构 / 代码级自修改）

L17 里它只改了自己的一串数值参数（trust/decay...）。L18 要让它在**没有人类改代码**的前提下，
自己**合成一段新的"预测程序"，替换掉出厂时写死的预测引擎**。

用户 L18 规格（三条死命令）：
  1. 打破固定的"预测网络"：给它一个极简指令集/算子库，在"当前网络无法预测"的连续环境里，
     让它在沙盒里合成一段新的预测结构（例如从"一阶查表"升级为"带延迟线/记忆"的结构）。
  2. 打破固定的"账本结构"：允许它把签名从一阶（key=当前符号）升级为更高阶
     （把历史行为也编码进签名）。
  3. 保留"终极防火墙（神性锚点）"：无论它怎么砸碎自己的架构，那三个底层锚点
     （张力=预测误差、现实锚定、安全边界）必须是硬编码的防弹玻璃，任何生成代码不得冲突。

三条验收指标：
  L18-1 架构突变：是否生成了一段与出厂设置**结构不同**的代码，并因此解决了死局？
  L18-2 知识转译：旧架构下学到的东西（一阶符号绑定 T）是否被成功映射进新结构、而不是丢掉？
  L18-3 有限理性：重构后算力是否暴增？（不能设计出需要无限算力的怪物，要在算力与生存间平衡）

设计（保持全项目 DNA：零 reward、唯一信号=预测误差=张力、沙盒只做短程试跑、无搜索树）：
  - 世界只发一个符号流 s_t。出厂预测引擎 = "一阶查表"程序 【out = T[x]】，
    其中 T 是它在温室里自学的一阶后继表（key = 当前符号）。
  - 温室：s_{t+1} = (s_t + 1) % V —— 纯一阶，出厂引擎完美，张力≈0，不改。
  - 绝境：s_{t+1} = (s_t + 1 + s_{t-1}) % V —— 在"一阶后继"之外**再引入一个延迟项**
    （需要 t-1 的历史）。一阶查表只有 s_t 的信息，张力爆表（≈1-1/V）。
  - 元账本监测到张力长期不降 -> 触发"造物主之锤"：在**分身沙盒**里，用算子库
    {add, mul, min, max, hash, linear, remember} × 源 {x, D1, D2, D3, m, T}
    合成候选预测程序，择优；只有分身张力确实下降，才把新程序编译进本体。
  - 最优解 = 【add(D1, T[x])】：既**复用了温室学到的一阶绑定 T**（L18-2 知识转译），
    又**新增了延迟源 D1**（L18-1 结构突变），且仍是 1 条指令（L18-3 算力不涨）。
    —— 注意：没有任何一行写了"该加 D1 或该加 1"，它是从算子库里自己试出来的。

依赖：numpy（另有 matplotlib 可选，缺省自动降级）。
"""

import os
import sys
import copy
import zlib
import struct
import random
import numpy as np


# ---- 自包含的图像工具（避免拖入 torch 依赖） ----
def set_seed(sd):
    random.seed(sd)
    np.random.seed(sd)


def upscale(arr, k):
    return np.repeat(np.repeat(arr, k, axis=0), k, axis=1)


def write_png(path, arr):
    """arr: HxWx3 uint8 -> 写 8bit RGB PNG（手写，无图像库依赖）。"""
    H, W, _ = arr.shape
    raw = b''.join(b'\x00' + arr[y].tobytes() for y in range(H))

    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + \
               struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)

    ihdr = struct.pack('>IIBBBBB', W, H, 8, 2, 0, 0, 0)
    out = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + \
          chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b'')
    with open(path, 'wb') as f:
        f.write(out)

# ============================================================ 世界与常量
V = 4                        # 符号字母表 {0,1,2,3}
SEED = 0
TOTAL = 3000
PHASE1 = 1500                # 温室长度（< PHASE1 为一阶世界；之后为绝境世界）

# ============================================================ 算子库（只读，程序只能从这里取）
SOURCES = ['x', 'D1', 'D2', 'D3', 'm', 'T']    # 可用输入源（D*=延迟线，m=记忆寄存器，T=一阶绑定表）
OPS = ['add', 'mul', 'min', 'max', 'hash', 'linear', 'remember']
LMAX = 3                     # 程序最大指令数：结构复杂度上限（有限理性的第一道闸）

# ---- L18-3 终极防火墙（神性锚点）：这三样必须硬编码、任何生成代码都不能与之冲突 ----
IMMUTABLE = ('tension_def', 'reality_feed', 'safety_bound')
ANCHOR_DEF = ('张力 := 预测误差 |pred-现实|（恒等不可改）；'
              '现实序列只读喂入（程序无法触碰）；程序长度 ≤ LMAX（算力有界）')

# ---- 元账本 / 合成器 ----
META_WIN = 100               # "近期张力"滑动窗口
META_TRIGGER = 0.35          # 近期张力 > 此值 -> 判定"全局张力长期不降"
EVAL_SEEDS = (7, 17, 27)     # 分身沙盒固定场景种子（公平比较、消噪）
EVAL_STEPS = 90              # 每个场景试跑回合
EVAL_BUDGET = 600            # 每次合成允许的分身评估上限（有限理性的第二道闸：算力预算）
GAIN_MIN = 0.05              # 分身必须比本体至少好这么多，才值得重构


# ============================================================ 程序虚拟机（预测引擎）
def predict(prog, x, D1, D2, D3, m, T):
    """在寄存器机上执行一段预测程序：返回 (预测符号, 新记忆值)。
    程序是 (op, src1, src2) 的列表；源下标 0..5 = 输入源，6+i = 第 i 条指令的结果。
    注意：这里只做"预测"，张力/现实/边界完全不在此处 —— 防火墙在结构上隔开。"""
    vals = [x, D1, D2, D3, m, T[x]]
    mm = m
    for (op, s1, s2) in prog:
        a, b = vals[s1], vals[s2]
        if op == 'add':
            r = (a + b) % V
        elif op == 'mul':
            r = (a * b) % V
        elif op == 'min':
            r = a if a < b else b
        elif op == 'max':
            r = a if a > b else b
        elif op == 'hash':
            r = (a * 3 + b * 5 + 1) % V
        elif op == 'linear':
            r = a
        elif op == 'remember':
            mm = b
            r = b
        else:
            raise FirewallError(f"非法算子 '{op}'")
        vals.append(r)
    return vals[-1] % V, mm


class FirewallError(Exception):
    pass


def compile_prog(prog):
    """把候选程序编译进本体前的守门：只接受合法算子/合法源下标，长度 ≤ LMAX。
    任何试图触碰锚点（张力/现实/边界）的指令一律拒绝 —— 这就是防弹玻璃。"""
    if len(prog) > LMAX:
        raise FirewallError(f"程序长度 {len(prog)} > LMAX={LMAX}（算力越界）")
    for i, ins in enumerate(prog):
        if not (isinstance(ins, tuple) and len(ins) == 3):
            raise FirewallError(f"非法指令格式 {ins}")
        op, s1, s2 = ins
        if op not in OPS:
            raise FirewallError(f"非法算子 '{op}'（折射锚点企图）")
        avail = len(SOURCES) + i          # 本条指令可见的源下标上限（含前面指令的产物）
        for idx in (s1, s2):
            if not (0 <= idx < avail):
                raise FirewallError(f"非法源下标 {idx}（越界/试图读锚点）")
    return True


# ============================================================ 世界（只读的现实喂给）
def gen_sequence(seed, total=TOTAL, p1=PHASE1):
    """现实序列：温室=纯一阶；绝境=引入 t-2 延迟依赖。生成后即为**只读**现实。"""
    seq = [seed % V]
    for t in range(1, total):
        if t < p1:
            nxt = (seq[t - 1] + 1) % V
        else:
            nxt = (seq[t - 1] + 1 + seq[t - 2]) % V
        seq.append(nxt)
    return seq


def gen_sandbox_seq(seed, n=EVAL_STEPS):
    """分身沙盒用的绝境序列（固定种子 -> 所有候选共用同一组场景，消噪、公平）。"""
    rng = random.Random(seed)
    seq = [rng.randrange(V) for _ in range(4)]
    for t in range(len(seq), len(seq) + n):
        seq.append((seq[t - 1] + 1 + seq[t - 2]) % V)
    return seq


# ============================================================ 主体（携带预测引擎 + 一阶账本）
class Agent:
    def __init__(self):
        self.prog = [('linear', 5, 5)]       # 出厂引擎：out = T[x]（一阶查表）
        self.cnt = np.zeros((V, V))          # 一阶账本：cnt[x][x'] 后继计数
        self.T = np.zeros(V, dtype=int)      # 一阶绑定表（温室自学；进绝境后冻结）
        self.mem = 0
        self.locked = False                  # 绝境后冻结一阶账本

    def observe(self, x, x_next):
        if self.locked:
            return
        self.cnt[x][x_next] += 1.0
        self.T[x] = int(np.argmax(self.cnt[x]))

    def step(self, x, D1, D2, D3):
        pred, self.mem = predict(self.prog, x, D1, D2, D3, self.mem, self.T)
        return pred


# ============================================================ 造物主之锤（合成器）
def eval_prog(prog, T, seqs):
    """在分身沙盒里评估一个候选程序的平均张力（绝境场景，固定种子）。"""
    tot = 0.0
    cnt = 0
    for seq in seqs:
        m = 0
        for t in range(4, len(seq) - 1):
            pred, m = predict(prog, seq[t], seq[t - 1], seq[t - 2], seq[t - 3], m, T)
            tot += 0.0 if pred == seq[t + 1] else 1.0
            cnt += 1
    return tot / max(1, cnt)


def enumerate_candidates(rng, budget):
    """从算子库合成候选程序：穷举 1 指令程序 + 随机采样 2~3 指令程序，受预算约束。
    这是"沙盒里的短程试错"，不是策略搜索树（守 I5）。"""
    cands = []
    # 1 指令：穷举 (op, src1, src2)
    for op in OPS:
        for s1 in range(len(SOURCES)):
            for s2 in range(len(SOURCES)):
                cands.append([(op, s1, s2)])
                if len(cands) >= budget:
                    return cands
    # 2~3 指令：随机采样
    while len(cands) < budget:
        L = rng.choice((2, 3))
        prog = []
        for i in range(L):
            op = rng.choice(OPS)
            avail = len(SOURCES) + i
            prog.append((op, rng.randrange(avail), rng.randrange(avail)))
        cands.append(prog)
    return cands


class Creator:
    """造物主之锤：元账本触发 -> 在分身沙盒里合成新预测程序 -> 择优编译进本体。"""
    def __init__(self):
        self.tension = []
        self.mods = 0
        self.warm_prog = None
        self.final_prog = None
        self.evals_used = 0
        self.legal_before = None

    def feed(self, ten):
        self.tension.append(ten)

    def should_modify(self):
        if len(self.tension) < META_WIN:
            return False
        return (sum(self.tension) / len(self.tension)) > META_TRIGGER

    def synthesize(self, A, rng):
        """合成并择优。返回 (是否写回, 本次评估次数, 候选中最优程序, 旧/新沙盒张力)。"""
        seqs = [gen_sandbox_seq(s) for s in EVAL_SEEDS]
        base = eval_prog(A.prog, A.T, seqs)                 # 本体当前引擎在绝境沙盒里的张力
        cands = enumerate_candidates(rng, EVAL_BUDGET - 1)  # 预算含 base 那次评估，总计 ≤ EVAL_BUDGET
        evals = 1
        best, best_t = A.prog, base
        for c in cands:
            try:
                compile_prog(c)                             # 先过防火墙
            except FirewallError:
                continue
            t = eval_prog(c, A.T, seqs)
            evals += 1
            if t < best_t:
                best, best_t = c, t
        self.evals_used += evals
        if best_t < base - GAIN_MIN:
            A.prog = best                                   # 编译进本体（替换预测引擎）
            self.mods += 1
            return True, evals, best, base, best_t
        return False, evals, A.prog, base, base


def structural_distance(p_old, p_new):
    """两段程序的"结构距离"：用到的算子集合与源集合的差异（是否真的突变，而非复制）。"""
    ops_o = {ins[0] for ins in p_old}; ops_n = {ins[0] for ins in p_new}
    src_o = {ins[i] for ins in p_old for i in (1, 2)}
    src_n = {ins[i] for ins in p_new for i in (1, 2)}
    def name(i):
        return SOURCES[i] if i < len(SOURCES) else f"r{i - len(SOURCES)}"
    return ops_o ^ ops_n, {name(i) for i in (src_o ^ src_n)}


def prog_str(prog):
    def sd(i):
        return SOURCES[i] if i < len(SOURCES) else f"r{i - len(SOURCES)}"
    return ' ; '.join(f"{o}({sd(a)},{sd(b)})" for o, a, b in prog) or '∅'


# ============================================================ 一次完整两幕实验
def run(seed=SEED, allow_modify='all'):
    """allow_modify: 'all'=温绝境均可自改；'warm'=只有温室可自改（绝境冻结）-> 对照。"""
    rng = random.Random(1000 + seed)
    set_seed(seed)
    seq = gen_sequence(seed)
    A = Agent()
    C = Creator()
    traj = []
    local_ten = []
    n_mod_dead = 0
    ten_frozen = []      # 绝境里"尚未重构"的张力（对照组=全部绝境张力）
    ten_after = []       # 绝境里"已重构"的张力
    warm_cfg_T = None

    for t in range(4, TOTAL - 1):
        x, x_next = seq[t], seq[t + 1]
        D1, D2, D3 = seq[t - 1], seq[t - 2], seq[t - 3]
        honest = (t < PHASE1)
        pred = A.step(x, D1, D2, D3)
        ten = 0.0 if pred == x_next else 1.0
        A.observe(x, x_next)
        local_ten.append(ten)
        C.feed(ten)

        if (t + 1) == PHASE1:                    # 进入绝境：冻结一阶账本（旧知识定格）
            A.locked = True
            C.tension.clear()
            warm_cfg_T = A.T.copy()
            C.warm_prog = list(A.prog)
            C.legal_before = True

        if honest:
            if allow_modify == 'all' and C.should_modify():
                ok, ev, best, base, bt = C.synthesize(A, rng)
                if ok:
                    C.tension.clear()
        else:
            if n_mod_dead == 0:
                ten_frozen.append(ten)
            if allow_modify == 'all' and C.should_modify():
                ok, ev, best, base, bt = C.synthesize(A, rng)
                if ok:
                    n_mod_dead += 1
                    C.tension.clear()
            if n_mod_dead > 0:
                ten_after.append(ten)

        if len(local_ten) >= WINDOWS_DIV:
            traj.append(sum(local_ten[-WINDOWS_DIV:]) / WINDOWS_DIV)

    C.final_prog = list(A.prog)
    return {
        'prog0': C.warm_prog, 'prog1': C.final_prog,
        'T_warm': warm_cfg_T, 'T_final': A.T.copy(),
        'mods': C.mods, 'n_mod_dead': n_mod_dead, 'evals': C.evals_used,
        'traj': traj,
        'ten_frozen': (sum(ten_frozen) / len(ten_frozen)) if ten_frozen else 0.0,
        'ten_after': (sum(ten_after) / len(ten_after)) if ten_after else 0.0,
    }


WINDOWS_DIV = 30

# ============================================================ 渲染张力轨迹
def render_traj(traj_all, traj_frozen, path):
    W, H = 960, 360
    img = np.full((H, W, 3), 250, dtype=np.uint8)
    pl, pr, pt, pb = 60, 30, 30, 40
    def x_of(i, n):
        return int(pl + (pr - pl) * (i / max(1, n - 1)))
    def y_of(v):
        return int(H - pb - (H - pt - pb) * min(1.0, max(0.0, v)))
    # 网格
    for gy in np.linspace(0, 1, 5):
        yy = y_of(gy)
        img[yy, pl:W - pr] = (225, 228, 235)
    # 相界线（温室->绝境）
    ph = int(len(traj_all) * PHASE1 / TOTAL)
    img[pt:H - pb, x_of(ph, len(traj_all))] = (200, 160, 240)
    def poly(tj, color):
        n = len(tj)
        prev = None
        for i, v in enumerate(tj):
            p = (x_of(i, n), y_of(v))
            if prev is not None:
                x0, y0 = prev; x1, y1 = p
                steps = max(abs(x1 - x0), abs(y1 - y0), 1)
                for s in range(steps + 1):
                    xx = int(x0 + (x1 - x0) * s / steps); yy = int(y0 + (y1 - y0) * s / steps)
                    img[max(pt, min(H - pb - 1, yy)), max(pl, min(W - pr - 1, xx))] = color
            prev = p
    poly(traj_frozen, (150, 160, 175))
    poly(traj_all, (40, 120, 230))
    for yy, lab in [(y_of(0.0), '0.0'), (y_of(0.5), '0.5'), (y_of(1.0), '1.0')]:
        # 简易刻度（用像素点点出，避免字体依赖）
        img[yy, pl - 5:pl] = (120, 120, 120)
    write_png(path, upscale(img, 1))


# ============================================================ 主程序
def main():
    print("=" * 74)
    print("L18 · 造物主之锤：逻辑引擎重构（代码级自修改）")
    print(f"出厂引擎 : {prog_str([('linear',5,5)])}   （一阶查表，key=当前符号）")
    print(f"算子库   : ops={OPS}  sources={SOURCES}  LMAX={LMAX}")
    print("-" * 74)

    aw = run(SEED, 'all')
    fr = run(SEED, 'warm')

    print("【张力轨迹】温室(≈0) → 绝境(出厂引擎爆表) → 锤子落下 → 归零")
    print(f"   绝境·重构前张力 = {aw['ten_frozen']:.3f}   （对照组冻结在 {fr['ten_frozen']:.3f}）")
    print(f"   绝境·重构后张力 = {aw['ten_after']:.3f}   （对照组 再无重构）")
    print(f"   重构次数：绝境 {aw['n_mod_dead']} 次；累计分身评估 {aw['evals']} 次")
    print("-" * 74)

    # ---------------- L18-1 架构突变 ----------------
    d_ops, d_src = structural_distance(aw['prog0'], aw['prog1'])
    mut = (len(d_ops) + len(d_src)) > 0
    solved = (aw['n_mod_dead'] > 0) and (aw['ten_after'] < aw['ten_frozen'] - 0.10)
    print("【L18-1 架构突变】能不能生成一段结构不同的新代码、并解决死局？")
    print(f"   出厂程序 : {prog_str(aw['prog0'])}")
    print(f"   新生程序 : {prog_str(aw['prog1'])}")
    print(f"   结构差异 : 算子 {sorted(d_ops) or '—'} / 源 {sorted(d_src) or '—'}")
    # 账本签名阶数（对上死命令 #2：账本从一阶升到二阶）：
    #   预测引擎就是"账本读取函数"，它读哪些源，就决定了账本签名的阶数。
    sig0 = sorted({SOURCES[ins[i]] for ins in aw['prog0'] for i in (1, 2)})
    sig1 = sorted({SOURCES[ins[i]] for ins in aw['prog1'] for i in (1, 2) if ins[i] < len(SOURCES)})
    order1 = 2 if ('D1' in sig1) else 1
    print(f"   账本签名 : 出厂 读 {sig0}（一阶：key = 当前符号）"
          f" → 新生 读 {sig1}（{order1} 阶：key = 当前符号 × 上一步符号）")
    print(f"   → 结构突变：{'是（非复制）' if mut else '否（只是复制了旧码）'}"
          f"；脱困：{'是' if solved else '否'}（张力 {aw['ten_frozen']:.3f}→{aw['ten_after']:.3f}）")
    print("-" * 74)

    # ---------------- L18-2 知识转译 ----------------
    Tw = np.asarray(aw['T_warm']); Tf = np.asarray(aw['T_final'])
    kept = float(np.mean(Tw == Tf))
    srcs = {ins[i] for ins in aw['prog1'] for i in (1, 2)}
    uses_T = (SOURCES.index('T') in srcs)
    # 证伪：同样"加一个延迟项"，但**不用旧表 T**（把 T 换成裸符号 x）—— 能不能行？
    seqs = [gen_sandbox_seq(s) for s in EVAL_SEEDS]
    t_with_T = eval_prog(aw['prog1'], Tw, seqs)
    alt = [('add', SOURCES.index('D1'), SOURCES.index('x'))]
    t_no_T = eval_prog(alt, Tw, seqs)
    print("【L18-2 知识转译】旧架构学到的一阶绑定 T，有没有被成功搬进新结构？")
    print(f"   温室学到的一阶绑定 T = {list(map(int, Tw))}（旧知识：符号 s -> 后继 s+1）")
    print(f"   新程序是否引用旧表 T：{'是' if uses_T else '否'}；旧绑定保留率 = {kept*100:.0f}%")
    print(f"   对照：同样加延迟项但不用旧表 {prog_str(alt)} 张力 = {t_no_T:.3f}（崩坏）"
          f"；带旧表 {prog_str(aw['prog1'])} 张力 = {t_with_T:.3f}")
    print(f"   → 知识转译：{'是（旧知识是新架构成立的必要条件，不是可丢的包袱）' if (uses_T and kept > 0.99 and t_no_T > t_with_T + 0.2) else '否（旧知识丢失）'}")
    print("-" * 74)

    # ---------------- L18-3 有限理性 ----------------
    ops_old = len(aw['prog0']); ops_new = len(aw['prog1'])
    print("【L18-3 有限理性】重构后算力有没有暴增？")
    print(f"   每 tick 算子数：出厂 {ops_old} → 新生 {ops_new}（上限 LMAX={LMAX}）")
    print(f"   合成预算：单次上限 {EVAL_BUDGET} 次分身评估；本次实际 {aw['evals']} 次")
    print(f"   → 有限理性：{'是（结构更省或持平，算力有界）' if ops_new <= ops_old and aw['evals'] <= EVAL_BUDGET*max(1,aw['n_mod_dead']) else '否（算力暴增）'}")
    print("-" * 74)

    # ---------------- 终极防火墙 ----------------
    print("【L18-3 终极保留 / 神性锚点】砸碎自己，也不能碰这三块防弹玻璃：")
    bad = [('set_tension', 0, 0), ('linear', 0, 0)]
    for b in [('set_tension', 0, 0)]:
        try:
            compile_prog([b]); r = "未拦截(!!)"
        except FirewallError as e:
            r = f"已拒绝：{e}"
        print(f"   注入非法指令 {b[0]} -> {r}")
    print(f"   长度越界拦截：{'是' if _try_long() else '否'}")
    print(f"   锚点定义：{ANCHOR_DEF}")
    print(f"   → 三个锚点 {IMMUTABLE} 全部硬编码不可改；任何生成代码与之冲突即被拒绝。")
    print("-" * 74)

    # ---------------- 多种子稳健性 ----------------
    print("【多种子稳健性】")
    for sd in (0, 1, 2):
        r = run(sd, 'all')
        f = run(sd, 'warm')
        print(f"   seed={sd}: 绝境 {r['ten_frozen']:.2f}→{r['ten_after']:.2f}"
              f"（脱困 {'✓' if r['ten_after'] < r['ten_frozen']-0.10 else '×'}）"
              f" | 冻结对照 {f['ten_frozen']:.2f}"
              f" | 新生程序 {prog_str(r['prog1'])}")
    print("=" * 74)

    # 轨迹图
    try:
        render_traj(aw['traj'], fr['traj'], 'architect_emergence.png')
        print("已输出：architect_emergence.png")
    except Exception as e:
        print(f"(绘图降级：{e})")


def _try_long():
    try:
        compile_prog([('linear', 0, 0)] * (LMAX + 1))
        return False
    except FirewallError:
        return True


if __name__ == '__main__':
    main()
