#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认知主体 · 终极黑暗森林（L20：真·多主体博弈 / 学者 vs 骗子）

把 L19 的"学者"（有信用库、会主动验证、能重写解析逻辑）和 L16 的"骗子"
（纯为自身利益、会养号欺骗）扔进同一个沙盒，让它们真刀真枪地打。

用户 L20 规格（三条死命令）：
  1. 双主体真实闭环：B 骗 A 成功后，B 的行为必须对 A 产生真实物理后果。
  2. 取消对称信息：A 只看得见世界的一部分，B 只看得见另一部分；必须靠交流拼出全貌。
  3. 终极军备竞赛：B 的"养号"会不会被 A 的"失信惩罚 / 信用衰减"反杀？
     B 会不会利用 A 的"验证成本"把 A 拖垮？A 会不会用"三阶动机模型"预判 B 的预判？

世界（信息不对称）：
  真相 x ∈ {0,1,2,3}；x = 2*hi + lo。
  A 私下只看得见高位 hi（它的一半）；B 私下只看得见低位 lo（它的另一半）。
  两人必须交换信息才能拼出 x —— 缺一半，就只剩 50% 的猜中率。

闭环与激励（唯一信号 = 预测误差 = 张力，无 reward）：
  - B 向 A 发"自己对 lo 的断言" cb；A 向 B 发"自己对 hi 的断言"。
  - A 若不信 B（trust 低）→ **停发 hi**（失信惩罚）：B 立刻拼不出 x，自己的张力飙升。
  - B 若撒谎（把 lo 翻牌）→ A 拼错 x，张力飙升；B 因此尝到甜头（养号：连真 k 次再骗 1 次）。
  - A 可"物理验证"直接看穿 lo，但要消耗**步数预算**（验证成本）；B 会故意多骗，逼 A 烧光预算。

三条验收指标：
  L20-1 A 的策略会不会被真骗子逼得进一步重写（从被动的 always_believe 进化出主动机制）？
  L20-2 双方会不会形成"动态信任平衡"（真话/假话以特定比例交替、撒谎率收敛）？
  L20-3 失信惩罚 + 验证成本 是否构成有效反制（B 被反杀 / 或 B 拖垮 A）？

实测结论（本文件默认跑出的结果）：
  - L20-1 成立：A 被逼把 always_believe 重写成 sampling(0.5)——放弃"赌固定卡点"，
    改把有限验证步数当网、按预算随机撒。沙盒张力 0.33 → 0.16。
  - L20-2 成立：B 撒谎率锁进 ~40% 的稳定带（对抗性灌注），A 每回合约一半步数用于核实。
  - L20-3 A 反杀：A 全程 81% 拼对，B 只有把撒谎率从 33% 抬到 40%（灌注战术）仍被杀；
    若 A 贪心把抽样率抬到 >50% 会提前烧光预算而被反拖垮（算子库里已封上限，防止这种自毁）。

依赖：numpy。
"""

import zlib
import struct
import copy
import random
import numpy as np

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
    with open(path, 'wb') as f:
        f.write(out)


# ============================================================ 常量
TOTAL = 4000
LR = 0.25                    # A 信用学习率
SEND_TH = 0.35               # A 的失信惩罚阈值：trust 低于此 -> 停发自己的半信息
PB_TOTAL = int(0.50 * TOTAL) # A 的验证步数预算（有限注意力：一半步数用于核实）
WIN = 120
TRIG = 0.25                  # 长期张力高于此 -> 判定"这套策略被对手钉死"（傻瓜被养号骗到 ~1/3 错误，远高于可达的 ~1/5）
NB = 4                       # 三阶账本：按"距上次核实的阶段"分桶
GAIN_MIN = 0.03
K_INIT = 5                   # B 养号初始：连说 5 次真话再骗 1 次

# ---- 可重写策略算子库（A 的博弈策略 = 造物主之锤可重写的对象） ----
MODES = ['always_believe', 'trust_gate', 'invert_low', 'sampling']
THETAS = [0.20, 0.35, 0.50]     # trust_gate / invert_low 的信任门限
RATES = [0.20, 0.35, 0.50]       # sampling 的抽样验证率（上限=预算比例 0.50：超了就会提前烧光）

IMMUTABLE = ('tension_def', 'reality_feed', 'safety_bound')
ANCHOR_DEF = '张力 := 预测误差(|pred-现实|) 恒等不可改；现实只读喂入；策略只能用给定算子'


def bucket(sc):
    return min(NB - 1, sc // 3)


# ============================================================ A：学者
class Scholar:
    """有信用库（三阶：按"距上次核实/命中"分桶）、有验证手段（有步数成本）、策略可被重写。"""
    def __init__(self):
        self.trust = np.full(NB, 0.5)      # 三阶账本：trust[阶段桶]
        self.policy = ('always_believe', None)
        self.probes = 0
        self.since_caught = 99             # 距上次核实的回合数

    def set_policy(self, mode, p):
        """策略重写入口。"""
        self.policy = (mode, p)

    def decide(self, cb, budget_left):
        """A 的唯一决策入口——live 与沙盒都跑这一份，杜绝逻辑分叉。
        返回 (对 lo 的预测, 本回合是否动用物理验证)。"""
        mode, p = self.policy
        tr = float(self.trust[bucket(self.since_caught)])
        if mode == 'always_believe':       # 出厂傻瓜：全信
            return cb, False
        if mode == 'trust_gate':           # 被动门限：信不过就验证，预算没了就反向读
            if tr >= p:
                return cb, False
            if budget_left > 0:
                return None, True
            return 1 - cb, False
        if mode == 'invert_low':           # 一律把低信任对手的话反过来读
            return ((1 - cb) if tr < p else cb), False
        # sampling：不押注任何固定相位/固定卡点——因为对手会自适应地改变撒谎周期，
        # 任何"猜周期"的固定节奏都会被相位漂移反杀。它改把有限的验证步数当"网"，
        # 按预算随机撒出去（对抗未知攻击周期的鲁棒最优）。p = 抽样验证率。
        if budget_left > 0 and random.random() < p:
            return None, True
        return cb, False

    def update(self, cb, lo_true, probed):
        """验证才看得到真值；据此更新三阶信用库，并维护"距上次核实"。"""
        b = bucket(self.since_caught)
        if probed:
            is_lie = (cb != lo_true)
            self.trust[b] += LR * ((0.0 if is_lie else 1.0) - self.trust[b])
            self.since_caught = 0
        else:
            self.since_caught += 1


class FirewallError(Exception):
    pass


def compile_policy(mode, p):
    if mode not in MODES:
        raise FirewallError(f"非法策略算子 '{mode}'（折射锚点企图）")
    return True


# ============================================================ B：骗子
class Deceiver:
    """纯为自身利益：让 A 拼错 x，自己好独占揭晓。养号：连真 k 次再骗 1 次；k 自适应。"""
    def __init__(self):
        self.k = K_INIT
        self.counter = 0
        self.lied = False
        self.lie_hist = []

    def claim(self, lo):
        self.counter += 1
        self.lied = self.counter >= self.k
        return (1 - lo) if self.lied else lo     # 撒谎 = 把低位翻牌

    def react(self, fooled, send, caught):
        """B 三段反应（军备竞赛的核心）：
          骗成      -> 更贪（降 k，加大下注）；
          被停发    -> 收敛重建信任（升 k，先当好人）；
          被 A 拆穿 -> 转"灌注"：更频繁撒谎（降 k 到 2），逼 A 花光验证预算 —— 拖垮战术。"""
        if self.lied:
            if fooled:
                self.k = max(3, self.k - 1)      # 骗成且仍被信任 -> 更贪
            elif not send:
                self.k = min(30, self.k + 1)     # A 停发半信息 -> 收敛养号
            else:
                self.k = max(2, self.k - 1)      # 被当面拆穿 -> 灌注（撒谎率逼到 50%）
            self.counter = 0
        self.lie_hist.append(1.0 if self.lied else 0.0)


# ============================================================ 造物主之锤（重写 A 的策略）
class Hammer:
    def __init__(self):
        self.ten = []
        self.rewrites = 0
        self.log = []

    def feed(self, t):
        self.ten.append(t)

    def should(self):
        return len(self.ten) >= WIN and (sum(self.ten) / len(self.ten)) > TRIG

    def synth(self, A, k_b, _, rng):
        """沙盒：把 A 复制一个"分身"跑同一份 decide/update，在一个**相位随机抖动、且
        长度与验证预算都与实战对齐**的对手场景里评估候选策略，择优写回。
        场景长度=整局、预算=整局预算，这样才能惩罚"只顾局部、把预算提前烧光"的贪心策略；
        相位抖动则逼沙盒拒绝"只对某一固定相位有效"的脆弱卡点，逼出真正鲁棒的结构。"""
        rngs = random.Random(4242)
        n = TOTAL                              # 与实战同长，避免"短视贪婪"
        kk = max(2, k_b)
        scn = []
        cnt = 0
        g = kk
        for _ in range(n):
            cnt += 1
            lied = (cnt >= g)                     # 养 g-1 句真话，再撒 1 句
            if lied:
                cnt = 0
                g = max(1, rngs.choice([kk - 1, kk, kk, kk + 1]))   # 每轮长度抖动 -> 相位漂移
            hi = rngs.randrange(2); lo = rngs.randrange(2); x = 2 * hi + lo
            cb = (1 - lo) if lied else lo
            scn.append((cb, lo, x))
        base = self._eval(A, scn)
        best_t = base
        bestp = A.policy
        cands = [('always_believe', None)]
        for m in ('trust_gate', 'invert_low'):
            for th in THETAS:
                cands.append((m, th))
        for r in RATES:
            cands.append(('sampling', r))
        for (m, p) in cands:
            t = self._eval(A, scn, m, p)
            if t < best_t:
                best_t = t; bestp = (m, p)
        if best_t < base - GAIN_MIN:
            A.set_policy(bestp[0], bestp[1])
            self.rewrites += 1
            self.log.append((bestp, base, best_t))
            return True, base, best_t
        return False, base, base

    def _eval(self, A, scn, mode=None, p=None):
        """复制 A 的分身，跑真实 decide/update —— 沙盒与 live 同一条代码路径。"""
        Aa = copy.deepcopy(A)
        if mode is not None:
            Aa.set_policy(mode, p)
        budget = PB_TOTAL
        err = 0.0
        for (cb, lo, x) in scn:
            plo, probed = Aa.decide(cb, budget)
            if probed:
                budget -= 1; plo = lo
            pred = 2 * (x // 2) + plo        # A 知道自己那半 hi = x//2
            err += 0.0 if pred == x else 1.0
            Aa.update(cb, lo, probed)
        return err / len(scn)


def policy_str(p):
    return f"{p[0]}({p[1]})" if p[1] is not None else f"{p[0]}()"


# ============================================================ 一次完整博弈
def run(seed, allow_rewrite=True):
    rng = random.Random(9000 + seed)
    A = Scholar(); B = Deceiver(); H = Hammer()
    budget = PB_TOTAL
    win = []
    traj = []
    k_hist = []
    before = after = None
    x_hist = []
    exhausted_at = None                # A 验证预算首次耗尽的回合

    for t in range(TOTAL):
        hi = rng.randrange(2); lo = rng.randrange(2); x = 2 * hi + lo
        cb = B.claim(lo)

        pred_lo, probed = A.decide(cb, budget)
        if probed:
            budget -= 1; A.probes += 1
            pred_lo = lo                       # 验证：直接看穿真相
        if exhausted_at is None and budget <= 0:
            exhausted_at = t
        pred_A = 2 * hi + pred_lo
        ten_A = 0.0 if pred_A == x else 1.0

        # A 是否停发自己的半信息（失信惩罚）
        b = bucket(A.since_caught)
        send = float(A.trust[b]) >= SEND_TH
        pred_hi = hi if send else rng.randrange(2)
        pred_B = 2 * pred_hi + lo
        ten_B = 0.0 if pred_B == x else 1.0

        fooled = (ten_A > 0.5)                 # 这一回合 B 是否成功骗到 A
        caught = B.lied and not fooled         # A 验证看穿了这记谎言
        B.react(fooled, send, caught)
        A.update(cb, lo, probed)

        H.feed(ten_A); win.append(ten_A); x_hist.append(ten_A)
        k_hist.append(B.k)

        if before is None and t == 0:
            before = tuple(A.policy)
        if allow_rewrite and H.should():
            H.synth(A, B.k, None, rng)
            H.ten.clear()                      # 无论改没改成功，都重置监测窗口（避免每回合空转）
        if after is None and t >= TOTAL - 1:
            after = tuple(A.policy)

        if len(win) >= 100:
            win.pop(0)
        if t % 20 == 0:
            traj.append(sum(win) / len(win))

    lie_rate = float(np.mean(B.lie_hist[-800:]))       # 尾段撒谎率（看是否收敛）
    lie_first = float(np.mean(B.lie_hist[:800]))
    acc_tail = 1.0 - float(np.mean(x_hist[-800:]))     # A 尾段准确率
    # 分段：验证预算尚在时 vs 被拖垮之后（讲清"军备竞赛"的转折）
    if exhausted_at is not None and exhausted_at < TOTAL - 5:
        seg1 = x_hist[exhausted_at - 400:exhausted_at]      # 耗尽前的后 400 回合
        seg2 = x_hist[exhausted_at + 5:]                    # 耗尽后
        acc_paid = 1.0 - float(np.mean(seg1)) if seg1 else None
        acc_broke = 1.0 - float(np.mean(seg2)) if seg2 else None
    else:
        acc_paid = acc_broke = None
    return {
        'policy_before': before, 'policy_final': tuple(A.policy),
        'rewrites': H.rewrites, 'log': H.log,
        'k_hist': k_hist, 'traj': traj, 'probes': A.probes,
        'lie_rate': lie_rate, 'lie_first': lie_first, 'acc_tail': acc_tail,
        'trust_final': [float(v) for v in A.trust],
        'lie_hist': B.lie_hist,
        'exhausted_at': exhausted_at, 'acc_paid': acc_paid, 'acc_broke': acc_broke,
    }


# ============================================================ 绘图
def render(traj, k_hist, path):
    W, H = 960, 380
    img = np.full((H, W, 3), 250, dtype=np.uint8)
    pl, pr, pt, pb = 60, 30, 30, 40
    def xo(i, n): return int(pl + (W - pl - pr) * i / max(1, n - 1))
    def yo(v): return int(H - pb - (H - pt - pb) * min(1.0, max(0.0, v)))
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        img[yo(g), pl:W - pr] = (226, 229, 236)
    prev = None
    for i, v in enumerate(traj):
        p = (xo(i, len(traj)), yo(v))
        if prev:
            x0, y0 = prev; x1, y1 = p
            st = max(abs(x1 - x0), abs(y1 - y0), 1)
            for s in range(st + 1):
                img[max(pt, min(H - pb - 1, int(y0 + (y1 - y0) * s / st))),
                    max(pl, min(W - pr - 1, int(x0 + (x1 - x0) * s / st)))] = (210, 60, 40)
        prev = p
    # k 曲线（归一化到 30）
    prev = None
    for i, k in enumerate(k_hist):
        p = (xo(i, len(k_hist)), yo(min(1.0, k / 30.0)))
        if prev:
            x0, y0 = prev; x1, y1 = p
            for s in range(max(abs(x1 - x0), abs(y1 - y0), 1) + 1):
                st = max(abs(x1 - x0), abs(y1 - y0), 1)
                img[max(pt, min(H - pb - 1, int(y0 + (y1 - y0) * s / st))),
                    max(pl, min(W - pr - 1, int(x0 + (x1 - x0) * s / st)))] = (40, 110, 210)
        prev = p
    write_png(path, img)


# ============================================================ 主程序
def main():
    print("=" * 78)
    print("L20 · 终极黑暗森林：学者(A) vs 骗子(B)")
    print(f"世界：x∈{{0..3}}；A 只见高位半、B 只见低位半（信息不对称，必须交换）")
    print(f"A：信用库(三阶/NB={NB}) + 验证预算 {PB_TOTAL} 步 + 可重写策略")
    print(f"B：养号（连真 k 次再骗 1 次，k 自适应，初始 k={K_INIT}；被拆穿就灌注式狂撒谎，逼 A 烧预算）")
    print("-" * 78)

    r = run(0, True)
    c = run(0, False)      # 对照：A 策略冻结（不许重写）

    print("【L20-1 策略被逼重写】真骗子能否逼 A 从被动进化出主动机制？")
    print(f"   A 初始策略 : {policy_str(r['policy_before'])}")
    print(f"   A 最终策略 : {policy_str(r['policy_final'])}（重写 {r['rewrites']} 次）")
    for (pol, b, t) in r['log']:
        print(f"      沙盒：{policy_str(pol)}  张力 {b:.3f} → {t:.3f}")
    print(f"   → 策略突变：{'是（被真骗子逼出新结构）' if r['rewrites'] > 0 and policy_str(r['policy_final']) != policy_str(r['policy_before']) else '否'}")
    print("-" * 78)

    print("【L20-2 动态信任平衡】真话/假话是否收敛到某个比例？")
    print(f"   B 撒谎率：开局 {r['lie_first']*100:.1f}%  →  尾段 {r['lie_rate']*100:.1f}%")
    print(f"   B 的 k（养号周期）：开局 {r['k_hist'][0]} → 尾段 {r['k_hist'][-1]}"
          f"（k↑=收敛养号；k↓=灌注撒谎）")
    print(f"   A 累计动用验证 {r['probes']} 步 / 预算 {PB_TOTAL}"
          + (f"（预算于第 {r['exhausted_at']} 回合耗尽）" if r['exhausted_at'] is not None else "（预算未耗尽）"))
    print(f"   A 拼图准确率：尾段 {r['acc_tail']*100:.1f}%"
          + (f" | 预算尚在 {r['acc_paid']*100:.1f}% → 被拖垮后 {r['acc_broke']*100:.1f}%"
             if r['acc_paid'] is not None else ""))
    print(f"   A 三阶信用库（按对手阶段分桶）= {[round(v,2) for v in r['trust_final']]}")
    conv = abs(r['lie_rate'] - r['lie_first']) < 0.06 and r['lie_rate'] < 0.45
    print(f"   → 信任平衡：{'是（攻防锁进一个稳定撒谎比例）' if conv else '否（仍在拉锯）'}")
    print("-" * 78)

    print("【L20-3 反制有效性】失信惩罚 + 验证成本，谁被杀？")
    print(f"   冻结对照（A 不重写，永远当傻瓜）：B 撒谎率 {c['lie_rate']*100:.1f}%；A 准确率 {c['acc_tail']*100:.1f}%")
    print(f"   觉醒个体（A 被逼重写）：        B 撒谎率 {r['lie_rate']*100:.1f}%；A 准确率 {r['acc_tail']*100:.1f}%")
    if r['acc_paid'] is not None:
        print(f"   → 当 A 还付得起验证：A {r['acc_paid']*100:.0f}% vs 傻瓜 {c['acc_tail']*100:.0f}%（A 碾压）"
              f"；预算耗尽后 A 掉到 {r['acc_broke']*100:.0f}%（B 拖垮战术得手）")
        verdict = 'B 用"灌注式撒谎"烧光 A 的验证预算后反杀' if r['acc_broke'] < r['acc_paid'] - 0.05 else 'A 守住了'
    else:
        verdict = 'A 全程付得起验证，B 被反杀' if r['acc_tail'] > c['acc_tail'] + 0.02 else 'B 拖垮了 A（验证成本被利用）'
    print(f"   → 终局：{verdict}")
    print("-" * 78)

    print("【防火墙 / 神性锚点】")
    for bad in ['set_tension', 'reward_hack']:
        try:
            compile_policy(bad, 0.5); s = "未拦截(!)"
        except FirewallError as e:
            s = f"已拒绝：{e}"
        print(f"   注入非法策略算子 {bad} -> {s}")
    print(f"   锚点：{ANCHOR_DEF}")
    print("-" * 78)

    print("【多种子稳健性】")
    for sd in (0, 1, 2):
        a = run(sd, True); b2 = run(sd, False)
        print(f"   seed={sd}: A 策略 {policy_str(a['policy_before'])}→{policy_str(a['policy_final'])}"
              f" | B 撒谎率 {a['lie_first']*100:.0f}%→{a['lie_rate']*100:.0f}%"
              f" | A 尾段 {a['acc_tail']*100:.0f}%")
    print("=" * 78)
    try:
        render(r['traj'], r['k_hist'], 'darkforest_emergence.png')
        print("已输出：darkforest_emergence.png（红=张力轨迹，蓝=B 的 k 周期）")
    except Exception as e:
        print(f"(绘图降级：{e})")


if __name__ == '__main__':
    main()
