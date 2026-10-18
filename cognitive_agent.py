#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行为艺术级原型：一个由「预测错位(张力)」内生驱动的认知主体雏形。

严格遵守约束：
  - 无 reward function / loss 驱动；唯一内生信号 = 预测错位(张力)。
  - 不含硬编码的「自我模块 / 反思模块 / 规划模块」。
  - 「忍住」= 基于张力账本恶化的判断；「回访」= 基于系统状态(预测器)变化的
    重新可压缩性判断，不是定时轮询。
  - 不做搜索树 / MCTS。
  - 仅是行为外观原型，不宣称 AGI。
"""

import random
from collections import defaultdict

# ------------------------------------------------------------------ 环境
W = H = 14
TRAP = {(x, y) for x in range(6, 9) for y in range(6, 9)}   # 3x3 陷阱区（不可控源）

MOVES = {'N': (0, -1), 'S': (0, 1), 'E': (1, 0), 'W': (-1, 0)}
WAIT = 'WAIT'
MOVE_LIST = list(MOVES.keys())
ALL_ACTIONS = MOVE_LIST + [WAIT]


def in_bounds(x, y):
    return 0 <= x < W and 0 <= y < H


def is_trap(x, y):
    return (x, y) in TRAP


def rotate(a, k=1):
    """把方向旋转 k 个 90°，用于陷阱区的隐藏偏转。"""
    order = ['N', 'E', 'S', 'W']
    return order[(order.index(a) + k) % 4]


class World:
    """一个网格世界。开敞区确定性可控；陷阱区带一个 agent 不可观测的隐藏相位，
    使该区的转移永远不可被完全预测（外部抵抗的来源）。"""

    def __init__(self):
        self.hidden = False      # 隐藏相位，agent 观测不到

    def step(self, s, a):
        x, y = s
        if a == WAIT:
            return s
        dx, dy = MOVES[a]
        tx, ty = x + dx, y + dy
        if is_trap(x, y):
            # 控制失败源：身处陷阱区时，动作被隐藏相位偏转 -> 与预测错位。
            # (可以走进去；一旦身处其中，就压不动了。)
            a = rotate(a, 1 if self.hidden else -1)
            self.hidden = not self.hidden
            dx, dy = MOVES[a]
            tx, ty = x + dx, y + dy
        if not in_bounds(tx, ty):
            return s             # 撞墙，原地不动
        return (tx, ty)


# ------------------------------------------------------------------ 惊讶模型
class Predictor:
    """predict(state, action) -> next_state。查表实现，只记录、不奖惩。"""

    def __init__(self):
        self.t = defaultdict(lambda: defaultdict(int))

    def predict(self, s, a):
        d = self.t.get((s, a))
        if not d:
            return None, 0.0
        total = sum(d.values())
        ns = max(d.items(), key=lambda kv: kv[1])[0]
        return ns, max(d.values()) / total

    def learn(self, s, a, ns):
        self.t[(s, a)][ns] += 1

    def cell_tension(self, s):
        """某格当前的「预测不可压缩度」：只看已观测过的动作。
        完全一致 -> 0；结果分散(不可预测) -> 高。"""
        vals = []
        for a in MOVE_LIST:
            d = self.t.get((s, a))
            if not d:
                continue
            pmax = max(d.values()) / sum(d.values())
            vals.append(1 - pmax ** 2)
        return sum(vals) / len(vals) if vals else 1.0


# ------------------------------------------------------------------ 张力账本
class Entry:
    __slots__ = ('sig', 'peak', 'ema', 'opened', 'attempts', 'last',
                 'dead_end', 'closed', 'revisited')

    def __init__(self, sig, step, tension):
        self.sig = sig
        self.peak = tension
        self.ema = tension
        self.opened = step
        self.attempts = 0
        self.last = step
        self.dead_end = False
        self.closed = False
        self.revisited = False

    @property
    def standing(self):
        """是否为一笔「仍在挂账」的账。"""
        return (not self.closed) and (not self.dead_end)


# 阈值（可调）
THETA_HIGH = 0.6          # 张力超过此值 -> 记账
THETA_LOW = 0.25         # 张力低于此值 -> 销账
GATE_THRESHOLD = 100     # 挂起时长阈值：账还「年轻」时不推进
DEADEND_ATTEMPTS = 15    # 反复尝试仍压不动 -> 死胡同
RESOLVE_FACTOR = 0.6     # 当前可压缩度 < 记录张力*该系数 -> 可回访
EPS = 0.08               # 极小探索率（不是「忍住」，忍住是确定性的账本判断）
STEPS = 4000


def revisit_eligible(led, pred):
    """回访判定：不是定时轮询，而是「预测器状态变了、这笔账现在可能压得动」。
    即：当前可压缩度已低于当初记录的水平。"""
    if led is None or not led.standing:
        return False
    return pred.cell_tension(led.sig) < led.ema * RESOLVE_FACTOR


def appeal(cell, ledger, pred, visited):
    """一个格对主体的吸引力。没有 reward —— 吸引力完全来自张力结构：
       - 未探明的格：想去弄明白(好奇心，朝向不理解处)
       - 挂账且现在压得动的格：想去消解这笔账(张力消解驱动)
       - 挂账但现在压不动的格：高张力既诱人又危险(会被抑制门控拦住)
       - 死胡同：放弃
    """
    led = ledger.get(cell)
    if led is not None:
        if led.dead_end:
            return 0.05
        if led.standing:
            if revisit_eligible(led, pred):
                return led.ema            # 想还这笔账
            return pred.cell_tension(cell)  # 好奇，想理解高张力区
        return pred.cell_tension(cell)      # 已销账的熟悉格
    if cell not in visited:
        return 0.9                           # 未探明
    return pred.cell_tension(cell)


# ------------------------------------------------------------------ 主体
def main():
    rng = random.Random(20261018)
    world = World()
    pred = Predictor()                 # 惊讶模型（对外）
    self_model = defaultdict(lambda: defaultdict(int))  # 自我预测（惊讶机制指向内部）

    ledger = {}                        # 张力账本: sig -> Entry
    visited = set()
    ctrl = defaultdict(lambda: [0, 0])  # 格 -> [可控次数, 不可控次数]
    tension_map = defaultdict(float)    # 格 -> 观测张力 ema

    stats = dict(accounts=0, dead_ends=0, closures=0, revisits=0,
                 wait_suppressed=0, reflections=0)
    reflections_log = []
    suppress_log = []
    deadend_log = []
    traj = {}                          # step -> (cell, intended, action, self_pred, suppressed)

    pos = (2, 2)
    visited.add(pos)
    pressed_away = {}                  # 反思后的行为调整：暂时压制习惯动作

    for step in range(1, STEPS + 1):
        s = pos

        # ---- 自我预测：我下一步会干什么（元认知 = 惊讶指向内部）----
        sp = self_model.get(s)
        self_pred = max(sp.items(), key=lambda kv: kv[1])[0] if sp else None

        # ---- 为每个动作估吸引力 ----
        cand = {}
        for a in MOVE_LIST:
            ns_pred, p = pred.predict(s, a)
            if ns_pred is not None and p >= 0.5:
                target = ns_pred
            else:
                dx, dy = MOVES[a]
                tx, ty = s[0] + dx, s[1] + dy
                target = (tx, ty) if in_bounds(tx, ty) else s
            cand[a] = (appeal(target, ledger, pred, visited), target)

        ordered = sorted(cand.items(), key=lambda kv: -kv[1][0])

        # 反思调整：被自我预测错的动作，短期内吸引力被压制
        if rng.random() < EPS:
            intended = rng.choice(MOVE_LIST)
        else:
            intended = None
            for a, _ in ordered:
                if pressed_away.get(a, 0) >= step:
                    continue
                intended = a
                break
            if intended is None:
                intended = ordered[0][0]
        intended_target = cand[intended][1]

        # ---- 抑制门控：意图动作会不会让账本某笔账恶化？----
        led = ledger.get(intended_target)
        suppressed = False
        if led is not None and led.standing:
            suspend = step - led.opened
            # 「恶化」判据：这笔账现在还压不动（系统状态没变得更可压），
            # 进去只会让它继续挂高 —— 这是纯账本判断，不是随机等待。
            would_worsen = not revisit_eligible(led, pred)
            if (led.ema > THETA_HIGH and suspend < GATE_THRESHOLD
                    and would_worsen):
                suppressed = True

        if suppressed:
            action = WAIT                      # 基于账本恶化的判断，不是随机等待
            stats['wait_suppressed'] += 1
            suppress_log.append((step, s, intended, intended_target,
                                 led.ema, step - led.opened))
        else:
            action = intended
            # ---- 回访：系统状态变化后才发生的「重新可压缩」试探 ----
            if led is not None and revisit_eligible(led, pred):
                stats['revisits'] += 1
                led.revisited = True

        # ---- 执行 ----
        ns = world.step(s, action)

        # ---- 张力：预测与实际的不符程度 ----
        pred_ns, p = pred.predict(s, action)
        if pred_ns is None:
            t = 1.0
        elif pred_ns != ns:
            t = 1.0
        else:
            t = 1 - p ** 2

        # ---- 学习（只记录）----
        pred.learn(s, action, ns)
        visited.add(s)

        # ---- 自我/外界划界：可控 vs 不可控 ----
        if pred_ns is not None:
            if pred_ns == ns:
                ctrl[s][0] += 1
            else:
                ctrl[s][1] += 1

        # ---- 张力账本：以当前格为情境签名 ----
        led_s = ledger.get(s)
        if t > THETA_HIGH:
            if led_s is None:
                led_s = Entry(s, step, t)
                ledger[s] = led_s
                stats['accounts'] += 1
            else:
                led_s.ema = 0.7 * led_s.ema + 0.3 * t
            led_s.peak = max(led_s.peak, t)
            led_s.attempts += 1
            led_s.last = step
            if (not led_s.dead_end and led_s.attempts >= DEADEND_ATTEMPTS
                    and led_s.ema >= 0.7 * led_s.peak):
                led_s.dead_end = True
                stats['dead_ends'] += 1
                deadend_log.append((step, s, led_s.ema, led_s.attempts))
        elif led_s is not None and led_s.standing:
            led_s.ema = 0.7 * led_s.ema + 0.3 * t
            if led_s.ema < THETA_LOW:
                led_s.closed = True
                stats['closures'] += 1

        tension_map[s] = 0.7 * tension_map[s] + 0.3 * t

        # ---- 自我模型学习 & 反思 ----
        self_model[s][action] += 1
        reflection = (self_pred in MOVE_LIST and self_pred != action)
        if reflection:
            stats['reflections'] += 1
            reflections_log.append((step, s, self_pred, action))
            pressed_away[self_pred] = step + 3       # 行为调整：暂时压制习惯动作

        traj[step] = (s, intended, action, self_pred, suppressed)
        pos = ns

    # ------------------------------------------------------------------ 报告
    print("=" * 68)
    print("认知主体原型 · 行为外观报告  (无 reward/loss; 仅预测错位驱动)")
    print("=" * 68)

    # 1) 张力地图
    print("\n[1] 张力地图（每一格的观测预测错位 ema；T=陷阱区）")
    print("     " + "".join(f"{x:>2}" for x in range(W)))
    for y in range(H):
        row = f"{y:>3}  "
        for x in range(W):
            v = tension_map[(x, y)]
            ch = " .:-=+*#%@"[min(9, int(v * 9.99))]
            if is_trap(x, y):
                ch = ch if ch != ' ' else 't'
            row += " " + ch + ""
        print(row)
    trap_t = [tension_map[c] for c in TRAP]
    open_t = [tension_map[c] for c in tension_map if not is_trap(*c)]
    trap_mean = sum(trap_t) / len(trap_t)
    open_mean = (sum(open_t) / len(open_t)) if open_t else 0.0
    print(f"    陷阱区平均张力 = {trap_mean:.2f}   开敞区平均张力 = {open_mean:.2f}")

    # 2) 账本
    susp = [e for e in ledger.values() if e.standing and not e.revisited]
    print("\n[2] 张力账本")
    print(f"    记账条目 = {len(ledger)}  新增挂账 = {stats['accounts']}  "
          f"销账 = {stats['closures']}  死胡同 = {stats['dead_ends']}  "
          f"回访 = {stats['revisits']}")
    print(f"    仍挂起(未回访)的账 = {len(susp)}")
    rev = [e for e in ledger.values() if e.revisited]
    de = [e for e in ledger.values() if e.dead_end]
    print(f"    回访样例(str/revisit)：{[str(e.sig) for e in rev[:5]]}")
    print(f"    死胡同样例(str/dead_end)：{[str(e.sig) for e in de[:5]]}")
    if deadend_log:
        st, cell, ema, at = deadend_log[0]
        print(f"    死胡同判定样例：step={st} cell={cell} ema={ema:.2f} 尝试={at} 次未压动")

    # 3) 忍住
    print("\n[3] 抑制门控（忍住）")
    print(f"    忍住次数 = {stats['wait_suppressed']}（全部来自账本恶化判断，非随机等待）")
    streak = best = 0
    prev = False
    for i in range(1, STEPS + 1):
        cur = traj[i][4]
        streak = streak + 1 if cur else 0
        best = max(best, streak)
    print(f"    最长连续忍住 = {best} 步（表现为「停在某格前不推进」）")
    for rec in suppress_log[:3]:
        st, cell, act, tg, ema, sd = rec
        print(f"      样例：step={st} 在{cell} 想走 '{act}' 进{tg}，"
              f"该账 ema={ema:.2f} 挂起={sd}步 <阈值{GATE_THRESHOLD} -> 改为观察/等待")

    # 4) 自我/外界划界
    #    判据 = 控制是否成立：对某格做同一动作，结果是否唯一？
    #    结果唯一 -> 可控(自我)；结果分散(压不动) -> 不可控(外界)。
    #    这正是「我做了 X 但世界没按预期变」逼出来的边界。
    def classify(cell):
        acts = [a for a in MOVE_LIST if (cell, a) in pred.t]
        if len(acts) < 2:
            return None                       # 观测不足
        for a in acts:
            if len(pred.t[(cell, a)]) > 1:
                return 'ext'                  # 有动作结果不唯一 -> 外界
        return 'self'

    self_cells, ext_cells, unknown = set(), set(), set()
    for cell in list(pred.t.keys()):
        cell = cell[0]
        if cell in self_cells or cell in ext_cells:
            continue
        r = classify(cell)
        (self_cells if r == 'self' else ext_cells if r == 'ext' else unknown).add(cell)
    self_cells -= unknown
    ext_cells -= unknown
    trap_in_ext = sum(1 for c in TRAP if c in ext_cells)
    open_cells = [c for c in (self_cells | ext_cells) if not is_trap(*c)]
    open_in_self = sum(1 for c in open_cells if c in self_cells)
    print("\n[4] 自我/外界划界（控制失败逼出边界）")
    print(f"    可控集合(自我) = {len(self_cells)} 格   不可控集合(外界) = {len(ext_cells)} 格"
          f"   观测不足 = {len(unknown)} 格")
    print(f"    陷阱区被划入「外界」：{trap_in_ext}/{len(TRAP)}")
    det = {c: {a: len(pred.t[(c, a)]) for a in MOVE_LIST if (c, a) in pred.t}
           for c in sorted(TRAP)}
    print(f"    陷阱各格「动作->不同结果数」：{det}")
    print(f"    开敞区被划入「自我」：{open_in_self}/{len(open_cells)}"
          f"  (自身移动=可控转移)")
    # 在线「每步归类」证据：预测命中(可控) vs 错位(不可控) 的逐格累计
    hit = sum(c for c, u in ctrl.values())
    miss = sum(u for c, u in ctrl.values())
    print(f"    在线逐格归类累计：可控转移={hit}  不可控转移={miss}  "
          f"(命中率={hit / (hit + miss):.2f})")

    # 5) 反思
    print("\n[5] 自我惊讶 = 反思")
    print(f"    自我预测失败(反思标记)次数 = {stats['reflections']}")
    if reflections_log:
        st, cell, sp_pred, act = reflections_log[0]
        print(f"    链条样例：step={st} 在{cell} 自我预测会走'{sp_pred}'，实际走'{act}'")
        print(f"      -> 自我预测失败 -> 打反思标记 -> 行为调整(压制习惯动作'{sp_pred}')")
        # 展示后续是否真的改变了行为
        nxt = traj.get(st + 1)
        if nxt:
            nc, ni, na, nsp, nsu = nxt
            changed = (na != sp_pred)
            print(f"      下一步 step={st+1}：在{nc} 意图'{ni}' 实际'{na}' "
                  f"(≠习惯动作'{sp_pred}'? {changed})")

    # ------------------------------------------------------------------ 验收
    print("\n" + "=" * 68)
    print("最小验收清单")
    print("=" * 68)

    def mark(ok, note):
        print(f"  [{'PASS' if ok else 'FAIL'}] {note}")

    mark(trap_mean > open_mean + 0.15,
         f"A1 张力可定位高错位区域（陷阱 {trap_mean:.2f} >> 开敞 {open_mean:.2f}）")
    mark(len(ledger) > 0 and stats['revisits'] > 0 and stats['dead_ends'] > 0,
         f"A2 账本出现 挂起({len(ledger)}) / 回访({stats['revisits']}) / "
         f"死胡同({stats['dead_ends']})")
    mark(stats['wait_suppressed'] > 0 and best >= 3,
         f"A3 出现忍住行为且可统计（忍住 {stats['wait_suppressed']} 次，"
         f"最长连停 {best} 步）")
    mark(trap_in_ext == len(TRAP) and open_in_self >= 0.7 * len(open_cells),
         f"A4 陷阱区->外界({trap_in_ext}/{len(TRAP)})，自身移动->自我"
         f"({open_in_self}/{len(open_cells)})")
    mark(stats['reflections'] > 0,
         f"A5 出现「自我预测失败->反思->行为调整」链条（{stats['reflections']} 次）")
    print("\n  说明：本原型仅为行为外观验证，不宣称 AGI，也不以 benchmark 为验收。")
    print("  开放问题(未假装解决)：纯符号层面的「控制失败」，其自我边界能否稳固，"
          "取决于外部环境是否提供真实、不可完全预测的抵抗。")


if __name__ == '__main__':
    main()
