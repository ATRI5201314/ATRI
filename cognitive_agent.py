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

运行：
  python3 cognitive_agent.py         # 默认跑法 + 五条验收
  python3 cognitive_agent.py diag    # 四组诊断对照实验
"""

import random
import sys
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
    使该区的转移永远不可被完全预测（外部抵抗的来源）。
    jitter>0 时给开敞区加小概率随机抖动，用于「判据鲁棒性」退化测试。"""

    def __init__(self, jitter=0.0, rng=None, trap=None):
        self.hidden = False      # 隐藏相位，agent 观测不到
        self.jitter = jitter
        self.rng = rng or random.Random(0)
        self.trap = set(trap) if trap is not None else set(TRAP)

    def step(self, s, a):
        x, y = s
        if a == WAIT:
            return s
        dx, dy = MOVES[a]
        tx, ty = x + dx, y + dy
        if (x, y) in self.trap:
            # 控制失败源：身处陷阱区时，动作被隐藏相位偏转 -> 与预测错位。
            a = rotate(a, 1 if self.hidden else -1)
            self.hidden = not self.hidden
            dx, dy = MOVES[a]
            tx, ty = x + dx, y + dy
        elif self.jitter > 0 and self.rng.random() < self.jitter:
            # 开敞区随机抖动：动作偶尔被横向滑一格（环境本身带随机性）
            a = rotate(a, 1 if self.rng.random() < 0.5 else -1)
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
GATE_THRESHOLD = 100     # 挂起时长阈值
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


# ------------------------------------------------------------------ 主体主循环
def run(steps=STEPS, gate_rule='young', deadend_cap=True, jitter=0.0,
        seed=20261018, eps=EPS, trap=None):
    """gate_rule: 'young' 挂起时长<阈值才忍住(基线) / 'old' 挂起时长>=阈值才忍住
                  / 'both' 都忍住。
       deadend_cap: 是否给「死胡同」设尝试上限（False -> 账可以长期挂着）。
       trap: 陷阱(不可控)格集合，默认 TRAP；可传入自定义布局做病态环境压测。"""
    rng = random.Random(seed)
    trap_set = set(trap) if trap is not None else set(TRAP)
    world = World(jitter=jitter, rng=random.Random(seed * 7 + 1), trap=trap_set)
    pred = Predictor()                 # 惊讶模型（对外）
    self_model = defaultdict(lambda: defaultdict(int))  # 自我预测（惊讶指向内部）

    ledger = {}
    visited = set()
    ctrl = defaultdict(lambda: [0, 0])
    tension_map = defaultdict(float)

    stats = dict(accounts=0, dead_ends=0, closures=0, revisits=0,
                 wait_suppressed=0, reflections=0)
    reflections_log = []
    ref_info = []                      # 与 reflections_log 平行：机会/对准/生效跟踪
    active_refs = []                   # 仍在压制窗口内的反思标记下标
    suppress_log = []
    deadend_log = []
    traj = {}
    moved_count = 0

    pos = (2, 2)
    visited.add(pos)
    pressed_away = {}

    for step in range(1, steps + 1):
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
        base_choice = ordered[0][0]      # 反事实基线：不经过反思压制时的选择

        # 反思时效性统计：哪些标记的「机会」真的来了
        #   returned = 压制窗口内又回到了同一个格
        #   aimed    = 那一刻的最大吸引力动作正好就是被压制的那个动作
        for idx in list(active_refs):
            ri = ref_info[idx]
            if ri['window_end'] < step:
                active_refs.remove(idx)
                continue
            if ri['cell'] == s:
                ri['returned'] = True
                if base_choice == ri['act']:
                    ri['aimed'] = True

        # 反思调整：被自我预测错的动作，短期内吸引力被压制
        if rng.random() < eps:
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
            # 反事实：压制是否真的把原首选换掉了（归因到对应标记）
            if base_choice != intended and pressed_away.get(base_choice, 0) >= step:
                for idx in active_refs:
                    ri = ref_info[idx]
                    if ri['cell'] == s and ri['act'] == base_choice:
                        ri['changed'] = True
        intended_target = cand[intended][1]

        # ---- 抑制门控：意图动作会不会让账本某笔账恶化？----
        led = ledger.get(intended_target)
        suppressed = False
        if led is not None and led.standing:
            suspend = step - led.opened
            would_worsen = not revisit_eligible(led, pred)
            if gate_rule == 'young':
                gate_on = suspend < GATE_THRESHOLD
            elif gate_rule == 'old':
                gate_on = suspend >= GATE_THRESHOLD
            else:                        # 'both'
                gate_on = True
            if led.ema > THETA_HIGH and would_worsen and gate_on:
                suppressed = True

        if suppressed:
            action = WAIT                      # 基于账本恶化的判断，不是随机等待
            stats['wait_suppressed'] += 1
            suppress_log.append((step, s, intended, intended_target,
                                 led.ema, step - led.opened))
        else:
            action = intended
            moved_count += 1
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
            if (deadend_cap and not led_s.dead_end
                    and led_s.attempts >= DEADEND_ATTEMPTS
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
        if self_pred in MOVE_LIST and self_pred != action:
            stats['reflections'] += 1
            reflections_log.append((step, s, self_pred, action))
            ref_info.append(dict(cell=s, act=self_pred, window_end=step + 3,
                                 returned=False, aimed=False, changed=False))
            active_refs.append(len(ref_info) - 1)
            pressed_away[self_pred] = step + 3       # 行为调整：暂时压制习惯动作

        traj[step] = (s, intended, action, self_pred, suppressed)
        pos = ns

    # ---------------- 派生指标 ----------------
    trap_t = [tension_map[c] for c in trap_set]
    open_t = [tension_map[c] for c in tension_map if c not in trap_set]
    trap_mean = sum(trap_t) / len(trap_t)
    open_mean = (sum(open_t) / len(open_t)) if open_t else 0.0

    def classify(cell):
        acts = [a for a in MOVE_LIST if (cell, a) in pred.t]
        if len(acts) < 2:
            return None
        for a in acts:
            if len(pred.t[(cell, a)]) > 1:
                return 'ext'
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
    trap_in_ext = sum(1 for c in trap_set if c in ext_cells)
    open_cells = [c for c in (self_cells | ext_cells) if c not in trap_set]
    open_in_self = sum(1 for c in open_cells if c in self_cells)

    best = streak = 0
    for i in range(1, steps + 1):
        streak = streak + 1 if traj[i][4] else 0
        best = max(best, streak)
    tail_wait = 0
    for i in range(steps, 0, -1):
        if traj[i][4]:
            tail_wait += 1
        else:
            break

    ref_changed = {i for i, ri in enumerate(ref_info) if ri['changed']}
    ref_opportunity = sum(1 for ri in ref_info if not ri['returned'])            # 没机会
    ref_misaimed = sum(1 for ri in ref_info if ri['returned'] and not ri['aimed'])  # 没对准

    standing = [e for e in ledger.values() if e.standing]
    longest_suspend = max((steps - e.opened for e in standing), default=0)

    return dict(
        steps=steps, gate_rule=gate_rule, deadend_cap=deadend_cap, jitter=jitter,
        stats=stats, ledger=ledger, tension_map=tension_map, visited=visited,
        pred=pred, ctrl=ctrl, reflections_log=reflections_log, ref_changed=ref_changed,
        ref_info=ref_info, ref_opportunity=ref_opportunity, ref_misaimed=ref_misaimed,
        suppress_log=suppress_log, deadend_log=deadend_log, traj=traj,
        moved_count=moved_count, best_streak=best, tail_wait=tail_wait,
        standing=standing, longest_suspend=longest_suspend,
        trap_mean=trap_mean, open_mean=open_mean, trap_set=trap_set,
        self_cells=self_cells, ext_cells=ext_cells, unknown=unknown,
        trap_in_ext=trap_in_ext, open_in_self=open_in_self, open_total=len(open_cells),
    )


# ------------------------------------------------------------------ 验收报告
def report(R):
    steps = R['steps']
    stats = R['stats']
    ledger = R['ledger']
    tension_map = R['tension_map']
    pred = R['pred']
    ctrl = R['ctrl']
    traj = R['traj']

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
            if is_trap(x, y) and ch == ' ':
                ch = 't'
            row += " " + ch
        print(row)
    print(f"    陷阱区平均张力 = {R['trap_mean']:.2f}   开敞区平均张力 = {R['open_mean']:.2f}")

    # 2) 账本
    print("\n[2] 张力账本")
    print(f"    记账条目 = {len(ledger)}  新增挂账 = {stats['accounts']}  "
          f"销账 = {stats['closures']}  死胡同 = {stats['dead_ends']}  "
          f"回访 = {stats['revisits']}")
    print(f"    仍挂起(未被判死胡同)的账 = {len(R['standing'])}  "
          f"最长挂起时长 = {R['longest_suspend']} 步")
    rev = [e for e in ledger.values() if e.revisited]
    de = [e for e in ledger.values() if e.dead_end]
    print(f"    回访样例：{[str(e.sig) for e in rev[:5]]}")
    print(f"    死胡同样例：{[str(e.sig) for e in de[:5]]}")

    # 3) 忍住
    print("\n[3] 抑制门控（忍住）")
    print(f"    忍住次数 = {stats['wait_suppressed']}  最长连续忍住 = {R['best_streak']} 步")
    for rec in R['suppress_log'][:3]:
        st, cell, act, tg, ema, sd = rec
        print(f"      样例：step={st} 在{cell} 想走 '{act}' 进{tg}，"
              f"该账 ema={ema:.2f} 挂起={sd}步 -> 改为观察/等待")

    # 4) 自我/外界划界
    print("\n[4] 自我/外界划界（控制失败逼出边界）")
    print(f"    可控集合(自我) = {len(R['self_cells'])} 格   "
          f"不可控集合(外界) = {len(R['ext_cells'])} 格   观测不足 = {len(R['unknown'])} 格")
    print(f"    陷阱区被划入「外界」：{R['trap_in_ext']}/{len(TRAP)}")
    print(f"    开敞区被划入「自我」：{R['open_in_self']}/{R['open_total']}")
    hit = sum(c for c, u in ctrl.values())
    miss = sum(u for c, u in ctrl.values())
    print(f"    在线逐格归类累计：可控={hit} 不可控={miss} (命中率={hit/(hit+miss):.2f})")

    # 5) 反思 + 转化率
    n_ref = len(R['reflections_log'])
    n_chg = len(R['ref_changed'])
    print("\n[5] 自我惊讶 = 反思")
    print(f"    自我预测失败(反思标记)次数 = {n_ref}")
    if n_ref:
        print(f"    真正改变后续动作(反事实) = {n_chg}  -> 转化率 = {n_chg/n_ref:.2f}")
        print(f"    未生效拆解：没机会(窗口内未回到该格) = {R['ref_opportunity']}   "
              f"没对准(当时首选≠被压制动作) = {R['ref_misaimed']}")
    else:
        print("    无反思事件")
    if R['reflections_log']:
        st, cell, sp_pred, act = R['reflections_log'][0]
        print(f"    链条样例：step={st} 在{cell} 自我预测'{sp_pred}'，实际'{act}'"
              f" -> 反思标记 -> 压制习惯动作'{sp_pred}'")

    # 验收
    print("\n" + "=" * 68)
    print("最小验收清单")
    print("=" * 68)

    def mark(ok, note):
        print(f"  [{'PASS' if ok else 'FAIL'}] {note}")

    mark(R['trap_mean'] > R['open_mean'] + 0.15,
         f"A1 张力可定位高错位区域（陷阱 {R['trap_mean']:.2f} >> 开敞 {R['open_mean']:.2f}）")
    mark(len(ledger) > 0 and stats['revisits'] > 0 and stats['dead_ends'] > 0,
         f"A2 账本 挂起({len(ledger)}) / 回访({stats['revisits']}) / 死胡同({stats['dead_ends']})")
    mark(stats['wait_suppressed'] > 0 and R['best_streak'] >= 3,
         f"A3 忍住行为可统计（忍住 {stats['wait_suppressed']} 次，最长连停 {R['best_streak']} 步）")
    mark(R['trap_in_ext'] == len(TRAP) and R['open_in_self'] >= 0.7 * max(1, R['open_total']),
         f"A4 陷阱区->外界({R['trap_in_ext']}/{len(TRAP)})，自身移动->自我"
         f"({R['open_in_self']}/{R['open_total']})")
    mark(stats['reflections'] > 0,
         f"A5 自我预测失败->反思->行为调整 链条（{stats['reflections']} 次）")
    print("\n  说明：本原型仅为行为外观验证，不宣称 AGI，也不以 benchmark 为验收。")
    print("  开放问题(未假装解决)：纯符号层面的「控制失败」，其自我边界能否稳固，"
          "取决于外部环境是否提供真实、不可完全预测的抵抗。")


# ------------------------------------------------------------------ 诊断对照
def _summary(R):
    stats = R['stats']
    n_ref = len(R['reflections_log'])
    return dict(
        wait=stats['wait_suppressed'], streak=R['best_streak'],
        moved=R['moved_count'], coverage=len(R['visited']),
        accounts=len(R['ledger']), closures=stats['closures'],
        dead_ends=stats['dead_ends'], revisits=stats['revisits'],
        standing=len(R['standing']), longest=R['longest_suspend'],
        trap_ext=R['trap_in_ext'], open_self=R['open_in_self'], open_tot=R['open_total'],
        refs=n_ref, ref_chg=len(R['ref_changed']),
        trap_mean=R['trap_mean'], open_mean=R['open_mean'],
    )


def _line(tag, s, steps):
    rate = f"{s['ref_chg']/s['refs']:.2f}" if s['refs'] else "n/a"
    print(f"  {tag:<26} 忍住={s['wait']:<5} 连停={s['streak']:<4} "
          f"移动步={s['moved']:<5} 覆盖={s['coverage']:<4}")
    print(f"  {'':<26} 账={s['accounts']:<4} 销={s['closures']:<4} 死胡同={s['dead_ends']:<4} "
          f"挂起中={s['standing']:<3} 最长挂起={s['longest']:<5} 回访={s['revisits']}")
    print(f"  {'':<26} 陷阱->外界={s['trap_ext']}/{len(TRAP)} 开敞->自我={s['open_self']}/{s['open_tot']}"
          f"  反思={s['refs']} 转化率={rate}")


def diagnostics():
    print("#" * 68)
    print("# 诊断对照实验（不新增功能，只做对照与压测）")
    print("#" * 68)

    # --- Exp0: 基线 ---
    base = run()
    print("\n【Exp0 基线】young 门控 + 死胡同上限 + 无抖动, 4000 步")
    _line("baseline", _summary(base), 4000)

    # --- Exp1: A3 对照 —— 把门控改成「老账也忍」 ---
    print("\n【Exp1 A3 对照】门控规则改动，是否更会忍 / 还是会瘫痪")
    no_cap_young = run(gate_rule='young', deadend_cap=False)
    _line("young + 无死胡同上限", _summary(no_cap_young), 4000)
    old_nocap = run(gate_rule='old', deadend_cap=False)
    _line("old + 无死胡同上限", _summary(old_nocap), 4000)
    old_cap = run(gate_rule='old', deadend_cap=True)
    _line("old + 死胡同上限", _summary(old_cap), 4000)
    both_nocap = run(gate_rule='both', deadend_cap=False)
    _line("both + 无死胡同上限", _summary(both_nocap), 4000)

    # --- Exp2: 挂账压测 —— 长程看「省略号」能否长期存在 ---
    print("\n【Exp2 挂账压测】去掉死胡同上限，跑 12000 步")
    long_nocap = run(steps=12000, gate_rule='young', deadend_cap=False)
    _line("young + 无上限 + 长程", _summary(long_nocap), 12000)
    if long_nocap['standing']:
        led = max(long_nocap['standing'], key=lambda e: 12000 - e.opened)
        print(f"    最老的一笔仍挂账：{led.sig}  ema={led.ema:.2f}  "
              f"挂起={12000 - led.opened} 步  尝试={led.attempts} 次")

    # --- Exp3: 反思转化率（基线） ---
    print("\n【Exp3 反思转化率】反思标记后行为实际改变的比率")
    s = _summary(base)
    print(f"    反思事件 = {s['refs']}  真正改变行为 = {s['ref_chg']}  "
          f"转化率 = {s['ref_chg']/s['refs']:.2f}" if s['refs'] else "    无反思事件")

    # --- Exp4: 随机环境退化 —— 判据鲁棒性 ---
    print("\n【Exp4 随机退化】给开敞区加抖动，看 A4 划界还干不干净")
    for j in (0.0, 0.03, 0.08, 0.15):
        R = run(jitter=j)
        ss = _summary(R)
        correct = ss['trap_ext'] + ss['open_self']
        total = len(TRAP) + ss['open_tot']
        print(f"    jitter={j:<5} 陷阱->外界={ss['trap_ext']}/{len(TRAP)}  "
              f"开敞->自我={ss['open_self']}/{ss['open_tot']}  "
              f"边界正确率={correct}/{total} = {correct/total:.2f}")

    # --- Exp5: 病态环境压测 old 规则 ---
    print("\n【Exp5 病态环境压测】整个可达区都是压不动的老账 -> old 规则会不会真死锁")
    all_trap = {(x, y) for x in range(W) for y in range(H)}
    for rule in ('young', 'old'):
        R = run(trap=all_trap, gate_rule=rule, deadend_cap=False)
        verdict = ("真死锁：尾部全程不再移动" if R['tail_wait'] > 500
                   else "未死锁：仍在移动")
        print(f"    {rule:<6} 忍住={R['stats']['wait_suppressed']:<5} "
              f"最长连停={R['best_streak']:<5} 移动步={R['moved_count']:<5} "
              f"尾部连停={R['tail_wait']:<5} 覆盖={len(R['visited'])}")
        print(f"           -> {verdict}")

    # --- Exp6: 反思时效性拆分 ---
    print("\n【Exp6 反思时效性拆分】那 52% 是没机会，还是没对准")
    n = len(base['reflections_log'])
    if n:
        print(f"    总反思={n}  生效={len(base['ref_changed'])}({len(base['ref_changed'])/n:.2f})  "
              f"没机会(窗口内没回到该格)={base['ref_opportunity']}({base['ref_opportunity']/n:.2f})  "
              f"没对准(当时首选≠被压制动作)={base['ref_misaimed']}({base['ref_misaimed']/n:.2f})")

    print("\n结论请对照各组指标自行判读；本实验只给数据，不给结论性包装。")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'diag':
        diagnostics()
    else:
        report(run())
