#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认知主体 · 像素版（第一步：从离散格子 -> 连续高维世界）

目标（对应用户「第一步」指令）：
  - 环境从 GridWorld 换成 PixelWorld：64x64x3 RGB 连续画面。
  - 预测器从查表换成「小型卷积自编码器 + 潜空间动作预测器」（极简世界模型）。
  - 没有任何 reward；唯一内生信号 = 预测误差（latent 空间距离）= 张力。
  - 撞墙时，能仅凭像素 + 动作后果，把「墙」的视觉特征自动划入「不可控/边界」。

守住的六条不变量（与格子版/L1-L12 一致）：
  I1 唯一信号=预测误差；I2 无 reward/loss 目标（只有自监督重构/预测误差）；
  I3 无硬编码自我/反思/规划模块（均为行为外观）；I4 忍住是判断不是定时；
  I5 无搜索树；I6 语言先验须可证伪（本文件暂不涉及语言，只做感知底座）。

与旧文件关系：cognitive_agent.py / cognitive_language_agent.py 一律冻结，本文件独立。
依赖：numpy, torch(CPU)。
"""

import os
import sys
import math
import zlib
import struct
import random
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn

# ============================================================ 常量
SEED = 0

IMG = 64                      # 画面尺寸 64x64
DIM = 3                       # RGB
Z_DIM = 32                    # latent 维度
FEAT = IMG // 8               # 编码器 3 次 stride-2 之后的特征图边长

DOM = 1.0                     # 连续世界 [0,1]^2
AGENT_R = 0.09                # 智能体半径（色块须足够大，否则自编码器塌缩、latent 不编码位置）
STEP_LEN = 0.07               # 单步位移（连续）；须大于「原型半径」，否则一步预测总落在同一原型里
MOVES = ['N', 'E', 'S', 'W']  # 上 右 下 左
DELTA = {'N': (0.0, -1.0), 'E': (1.0, 0.0), 'S': (0.0, 1.0), 'W': (-1.0, 0.0)}
ACTIONS = MOVES + ['WAIT']    # WAIT = 忍住不动

# 内部墙（矩形 x0,y0,x1,y1）——外边界另算，等价于四面墙
# 贴边放置：使墙的「阻塞带宽」与外边界带宽合并，留出一片足够大的中央开敞区
WALLS = [
    (0.00, 0.30, 0.16, 0.70),   # 左侧凸出的竖墙
]
# 混沌区（盲区）：动作被随机篡改，等价格子版 TRAP；放在开敞区内部
CHAOS = (0.52, 0.48, 0.74, 0.66)
SPAWN = [0.45, 0.30]          # 出生在开敞区
P_CHAOS = 0.75                # 身处混沌区时，动作被篡改的概率

# 颜色（uint8）
C_FLOOR = (232, 234, 238)
C_WALL = (58, 62, 72)
C_CHAOS = (255, 214, 170)     # 混沌区地表：可被"看见"（对应 RISK_MARK）
C_AGENT = (32, 120, 255)
C_AGENT_CORE = (10, 70, 200)

# 认知层阈值
CTRL_FLOOR = 0.35             # 稳定可控度阈值：低于 -> 外界
UNCTRL = ('BOUNDARY', 'CHAOS', 'EXTERNAL')   # 「不可控」判定集合
CHAOS_ETA = 0.18              # eta^2 低于此且方差高 -> 混沌不可控
BLOCK_EPS = 0.06              # 某方向平均 latent 增量模长低于此 -> 撞墙(阻塞)
NEAR_WALL = 0.16              # 判定"贴近墙"的几何距离：撞墙发生的带宽 ≈ AGENT_R+STEP_LEN（仅评测 ground-truth）
MIN_OBS = 3                   # 单动作最少观测
# 张力阈值不写死：run() 在预热后按该感知尺度下的预测误差分位数自动校准（见 t_high）
NOVELTY = 1.6
CURIOSITY = 0.9
RECENT_PENALTY = 3.0          # 近期刚访问过的情境 -> 惩罚（打破 A<->B 局部震荡，促其游走）
RECENT_LEN = 6
EPS = 0.40
DEADEND_ATTEMPTS = 10
DEADLOCK_BREAK = 15

# 训练
WARMUP = 1800                 # 自监督预训练步数（随机游走采集）
TRAIN_PER_STEP = 3            # 每个预热步做几次梯度更新
STEPS = 3600                  # 主循环步数
BATCH = 64
BUF_CAP = 4000
LR = 3e-4
ONLINE_TRAIN = False          # 主循环是否继续训练世界模型
#   说明：感知编码器一旦在线继续训练，权重漂移会让「同一画面的 latent」随步变化，
#   使 latent 预测误差被人为放大（不是真实的"世界不可预测"，而是"自我漂移"）。
#   第一步只求「感知底座稳定」：预热定型编码器，主循环靠原型记忆+账本在线适应。
PROTO_K = 256
PROTO_LR = 0.02
PROTO_NEW_DIST = 12.0         # 新建原型的默认阈值（平方距离）；run() 会按单步 latent 位移自动重新校准


def set_seed(sd):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)


# ============================================================ 像素世界
class PixelWorld:
    """连续 2D 世界，渲染为 64x64x3 RGB。
    - 内部矩形墙 + 外边界墙：撞墙 -> 位置不变（阻塞）。
    - 混沌区：身处其中时动作被随机篡改（控制失败源）。
    渲染确定性（给定状态 -> 相同像素）。"""

    def __init__(self, rng=None):
        self.rng = rng or random.Random(0)
        self.pos = list(SPAWN)

    # ---------- 几何
    def _inflated(self):
        r = AGENT_R
        return [(x0 - r, y0 - r, x1 + r, y1 + r) for (x0, y0, x1, y1) in WALLS]

    def _blocked(self, x, y):
        if x < AGENT_R or x > DOM - AGENT_R or y < AGENT_R or y > DOM - AGENT_R:
            return True                       # 外边界墙
        for (x0, y0, x1, y1) in self._inflated():
            if x0 <= x <= x1 and y0 <= y <= y1:
                return True                   # 内部墙
        return False

    def in_chaos(self, x, y):
        x0, y0, x1, y1 = CHAOS
        return x0 <= x <= x1 and y0 <= y <= y1

    def step(self, a):
        """返回 (frame, blocked)；blocked=撞墙（意图是移动但位置没变）。"""
        x, y = self.pos
        blocked = False
        if a != 'WAIT':
            if self.in_chaos(x, y) and self.rng.random() < P_CHAOS:
                a = self.rng.choice(MOVES)     # 混沌区：动作被不可观测地篡改
            dx, dy = DELTA[a]
            nx, ny = x + dx * STEP_LEN, y + dy * STEP_LEN
            if self._blocked(nx, ny):
                blocked = True                 # 撞墙：位置不变
            else:
                self.pos = [nx, ny]
        return self.render(), blocked

    # ---------- 渲染
    def render(self):
        img = np.empty((IMG, IMG, DIM), dtype=np.uint8)
        img[:, :] = C_FLOOR
        # 混沌区地表
        x0, y0, x1, y1 = CHAOS
        self._fill(img, (x0, y0, x1, y1), C_CHAOS)
        # 墙
        for w in WALLS:
            self._fill(img, w, C_WALL)
        # 外边界墙（窄边条）
        b = max(1, int(round(AGENT_R * IMG / DOM)))
        img[:b, :] = C_WALL; img[-b:, :] = C_WALL
        img[:, :b] = C_WALL; img[:, -b:] = C_WALL
        # 智能体
        self._disc(img, self.pos[0], self.pos[1], AGENT_R, C_AGENT, C_AGENT_CORE)
        return img

    @staticmethod
    def _fill(img, rect, color):
        x0, y0, x1, y1 = rect
        c0 = int(round(x0 * IMG)); c1 = int(round(x1 * IMG))
        r0 = int(round(y0 * IMG)); r1 = int(round(y1 * IMG))
        c0, c1 = max(0, c0), min(IMG, c1)
        r0, r1 = max(0, r0), min(IMG, r1)
        img[r0:r1, c0:c1] = color

    @staticmethod
    def _disc(img, cx, cy, r, color, core):
        px, py = cx * IMG, cy * IMG
        pr = max(2.0, r * IMG)
        y0 = max(0, int(py - pr - 1)); y1 = min(IMG, int(py + pr + 2))
        x0 = max(0, int(px - pr - 1)); x1 = min(IMG, int(px + pr + 2))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
        img[y0:y1, x0:x1][d <= pr] = color
        img[y0:y1, x0:x1][d <= pr * 0.55] = core

    # ---------- 评测专用 ground-truth（智能体看不到）
    def gt_region(self):
        x, y = self.pos
        if self.in_chaos(x, y):
            return 'CHAOS'
        if self._dist_to_wall(x, y) <= NEAR_WALL:
            return 'NEAR_WALL'
        return 'OPEN'

    @staticmethod
    def _dist_to_wall(x, y):
        best = min(x, y, DOM - x, DOM - y)     # 到外边界
        for (x0, y0, x1, y1) in WALLS:
            dx = max(x0 - x, 0.0, x - x1)
            dy = max(y0 - y, 0.0, y - y1)
            best = min(best, math.hypot(dx, dy))
        return best


# ============================================================ 世界模型
class WorldModel(nn.Module):
    """极简世界模型：卷积自编码器 + 潜空间动作预测器。
    编码 64x64x3 -> z(32)；解码 z -> 画面；预测 (z, a) -> z'。
    训练完全是自监督（重构误差 + 预测误差），无 reward。"""

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
        )
        self.to_z = nn.Linear(64 * FEAT * FEAT, Z_DIM)
        # latent 归一化：否则 to_z 会产生巨大常量偏置（norm~95）而方差极小，
        # 使「距离/预测误差/原型划分」全部失去分辨力。BN 让各维 ~单位方差。
        self.bn = nn.BatchNorm1d(Z_DIM)
        self.from_z = nn.Linear(Z_DIM, 64 * FEAT * FEAT)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid(),
        )
        self.pred = nn.Sequential(
            nn.Linear(Z_DIM + len(ACTIONS), 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, Z_DIM),
        )
        self.opt = torch.optim.Adam(self.parameters(), lr=LR)

    def _zx(self, x):
        return self.bn(self.to_z(self.enc(x)))

    def encode(self, frame_u8):
        """单帧编码（eval 模式，用 BN running stats；no_grad）。"""
        self.eval()
        x = torch.from_numpy(frame_u8).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            return self._zx(x)

    def encode_batch(self, imgs):
        self.eval()
        x = torch.from_numpy(np.stack(imgs)).float().permute(0, 3, 1, 2) / 255.0
        with torch.no_grad():
            return self._zx(x)

    def predict(self, z, a_idx):
        onehot = torch.zeros(1, len(ACTIONS)); onehot[0, a_idx] = 1.0
        with torch.no_grad():
            return self.pred(torch.cat([z, onehot], dim=1))

    def train_batch(self, imgs, z_prev, aidx, z_next):
        """imgs: 当前帧(N,H,W,3)；z_prev 对应前一步 latent；z_next 当前 latent。
        重构当前帧 + 预测 前一步->当前 的 latent。"""
        self.train()
        x = torch.from_numpy(np.stack(imgs)).float().permute(0, 3, 1, 2) / 255.0
        z = self._zx(x)
        rec = self.dec(self.from_z(z).view(-1, 64, FEAT, FEAT))
        onehot = torch.zeros(len(aidx), len(ACTIONS))
        for i, a in enumerate(aidx):
            onehot[i, a] = 1.0
        zhat = self.pred(torch.cat([z_prev, onehot], dim=1))
        loss_rec = torch.mean((rec - x) ** 2)
        loss_pred = torch.mean((zhat - z.detach()) ** 2)
        loss = loss_rec + loss_pred
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.opt.step()
        return float(loss_rec.detach()), float(loss_pred.detach())


# ============================================================ 情景记忆
class ProtoMem:
    """在线原型记忆（k-means 式）：把连续 latent 离散成「情境签名」。
    每个情境按动作记录 latent 增量的在线均值/方差（Welford）。
    -> 稳定可控度 = 动作对增量分布的解释力 (eta^2)。"""

    def __init__(self, k=PROTO_K):
        self.k = k
        self.new_dist = PROTO_NEW_DIST    # 新建原型的阈值（平方距离）；run() 里按单步 latent 位移自动校准
        self.C = []                       # list of np.array(Z_DIM,)
        self.n = []                       # 每原型访问数
        self.blk = defaultdict(int)       # 原型 -> 真实撞墙(blocked)次数（评测用）
        self.gt = defaultdict(lambda: defaultdict(int))   # 原型 -> 真实region计数（仅评测）
        self.st = {}                      # (proto, aidx) -> [n, mean, M2] 向量式Welford

    def assign(self, z, create=True):
        """create=True 仅用于「真实观测」；create=False 用于「预测」——预测不得新建原型，
        否则会用想象出来的 latent 污染情境记忆。"""
        z = z.detach().cpu().numpy().ravel()
        if not self.C:
            if not create:
                return -1
            self.C.append(z.copy()); self.n.append(0)
            return 0
        d = np.array([np.sum((z - c) ** 2) for c in self.C])
        i = int(np.argmin(d))
        if d[i] > self.new_dist:
            if create and len(self.C) < self.k:
                self.C.append(z.copy()); self.n.append(0)
                return len(self.C) - 1
            return i                          # 预测/已满：就近归类
        if create:                            # 只有真实观测才更新质心（预测不得污染记忆）
            self.C[i] += PROTO_LR * (z - self.C[i])
        return i

    def nearest_dist(self, z):
        """到最近原型的平方距离（用于连续新颖度；无原型返回一个大值）。"""
        if not self.C:
            return 1e9
        z = z.detach().cpu().numpy().ravel()
        return float(min(np.sum((z - c) ** 2) for c in self.C))

    def observe(self, proto, aidx, delta):
        self.n[proto] += 1
        key = (proto, aidx)
        if key not in self.st:
            self.st[key] = [0, np.zeros_like(delta), np.zeros_like(delta)]
        s = self.st[key]
        s[0] += 1
        d = delta - s[1]
        s[1] += d / s[0]
        s[2] += d * (delta - s[1])

    def stats(self, proto):
        """返回 (可控度 eta^2, 被阻塞的移动方向数, 总增量模长, 是否混沌)。"""
        acts = [ai for ai in range(len(ACTIONS)) if (proto, ai) in self.st and self.st[(proto, ai)][0] >= 2]
        if len(acts) < 2:
            return None
        means, ws, within, tot_n = [], [], 0.0, 0
        for ai in acts:
            n, m, M2 = self.st[(proto, ai)]
            means.append(m); ws.append(n)
            within += float(np.sum(M2))
            tot_n += n
        W = np.array(ws, float); W = W / W.sum()
        M = np.stack(means)                                   # (A, Z)
        mu = (M * W[:, None]).sum(0)                          # 总均值
        ss_between = float(np.sum(W[:, None] * (M - mu) ** 2))
        ss_within = float(within)
        eta2 = ss_between / (ss_between + ss_within + 1e-9)
        # 阻塞方向：某「移动」动作平均增量模长 ~0
        blocked = 0
        for ai, m in zip(acts, means):
            if ACTIONS[ai] in MOVES and float(np.linalg.norm(m)) < BLOCK_EPS:
                blocked += 1
        total_norm = float(np.linalg.norm(mu))
        return eta2, blocked, total_norm, ss_within

    def classify(self, proto):
        s = self.stats(proto)
        if s is None:
            return 'UNKNOWN'
        eta2, blocked, total_norm, ss_within = s
        # 顺序很重要（对应用户要求：撞墙 -> 墙的视觉特征划入不可控/边界）
        if eta2 < CHAOS_ETA:
            return 'CHAOS'          # 动作解释不了结果（随机被篡改）-> 混沌/外界
        if blocked >= 1:
            return 'BOUNDARY'       # 有方向被阻塞（撞墙）-> 边界（不可控）
        if eta2 < CTRL_FLOOR:
            return 'EXTERNAL'
        return 'SELF'               # 可控（自我）


# ============================================================ PNG
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


def upscale(arr, k):
    return np.repeat(np.repeat(arr, k, axis=0), k, axis=1)


# ============================================================ 主实验
def run(steps=STEPS, warmup=WARMUP, seed=SEED, outdir='/workspace'):
    set_seed(seed)
    rng = random.Random(seed)
    world = PixelWorld(rng)
    model = WorldModel()
    proto = ProtoMem()

    # ---------- 缓冲区
    buf = deque(maxlen=BUF_CAP)

    # ---------- Phase A：自监督预热（随机游走，采经验 + 训练世界模型）
    losses = []
    z_prev = model.encode(world.render())
    for _ in range(warmup):
        a = rng.choice(ACTIONS)
        frame, blocked = world.step(a)
        z = model.encode(frame)
        buf.append((frame, z_prev.detach(), ACTIONS.index(a), z.detach()))
        z_prev = z
        for _ in range(TRAIN_PER_STEP):           # 每步多次训练，加速收敛
            if len(buf) >= BATCH:
                idx = np.random.randint(0, len(buf), BATCH)
                lr_, lp_ = model.train_batch(
                    [buf[i][0] for i in idx],
                    torch.cat([buf[i][1] for i in idx], 0),
                    [buf[i][2] for i in idx],
                    torch.cat([buf[i][3] for i in idx], 0))
                losses.append((lr_, lp_))
    if losses:
        tail = losses[-200:]
        print(f"[预热] 重构损失={np.mean([l[0] for l in tail]):.4f}  "
              f"潜在预测损失={np.mean([l[1] for l in tail]):.4f}  (训练 {len(losses)} 次)")

    # ---------- 校准：把认知层阈值钉在「感知尺度」上（换分辨率/步长不用重调魔法数）
    d2s, tens = [], []
    for i in range(1, len(buf)):
        fr, zp, ai, zn = buf[i]
        d2 = float(torch.sum((zn - zp) ** 2))
        if d2 > 1.0:                              # 排除撞墙(位移≈0)的样本
            d2s.append(d2)
            tens.append(float(torch.norm(zn - model.predict(zp, ai))))
    d2s = np.array(d2s); tens = np.array(tens)
    proto.new_dist = float(0.6 * np.median(d2s))  # 原型粒度 ≈ 一个单步位移（半步之内归为同一情境签名）
    t_high = float(np.percentile(tens, 85))       # 高张力 = 该尺度下的高分位
    print(f"[校准] 单步 latent 位移 中位²={np.median(d2s):.2f} -> PROTO_NEW_DIST={proto.new_dist:.2f} | "
          f"预测误差 中位={np.median(tens):.2f} p85={t_high:.2f} -> TENSION_HIGH={t_high:.2f}")

    # ---------- Phase B：认知主循环
    # 账本：情境 -> 统计
    led = defaultdict(lambda: {'tension': 0.0, 'n': 0, 'standing': 0,
                               'dead_end': False, 'revisit': 0, 'seen': 0})
    reflected = {}                    # 情境 -> 触发时的动作（用于转化率）
    stats = dict(waits=0, deadlock_break=0, reflect_mark=0, reflect_trig=0,
                 reflect_convert=0, blocked_events=0, tension_sum=0.0, n=0)
    log_tension = []                  # (prototype, gt_region, tension, blocked)
    z_prev = model.encode(world.render())
    sit_prev = proto.assign(z_prev)
    # 主循环从开敞区出发：预热随机游走的终点很可能贴在边界墙上，
    # 若不在开敞区起始，会被「边界=不可控->回避」直接锁死。
    world.pos = list(SPAWN)
    buf.clear()
    z_prev = model.encode(world.render())
    last_z = None                     # 上一步的 latent（预测当前的基准）
    last_action_idx = None            # 上一步的动作
    prev_blocked = False              # 上一步是否撞墙
    consec_wait = 0
    recent = deque(maxlen=RECENT_LEN) # 近期访问过的情境（用于反折返）

    for step in range(steps):
        z = z_prev
        sit = proto.assign(z)
        # 张力 = 用「上一步 latent + 上一步动作」预测「当前 latent」的误差
        if last_action_idx is not None:
            zhat = model.predict(last_z, last_action_idx)
            tension = float(torch.norm(z - zhat).item())
        else:
            tension = 0.0
        e = led[sit]; e['n'] += 1; e['tension'] += tension; e['seen'] += 1
        stats['tension_sum'] += tension; stats['n'] += 1
        gt = world.gt_region()
        log_tension.append((sit, gt, tension, False))

        # ---------------- 意图池（无 reward：唯一依据=张力账本/新颖度）
        # 每个移动都由「好奇心(该情境平均张力) + 新颖度」打分；
        # 唯一拦路的是下面的「抑制门控」（目标账仍挂起且不可控 -> 忍住）。
        intents = []
        for ai, a in enumerate(ACTIONS):
            if a == 'WAIT':
                intents.append((a, ai, 0.0))
                continue
            zp = model.predict(z, ai)
            tgt_sit = proto.assign(zp, create=False)      # 预测：不得新建原型
            dmin = proto.nearest_dist(zp)                 # 连续新颖度（可区分方向）
            novelty = NOVELTY * min(1.0, dmin / proto.new_dist)
            avg_t = 0.0
            if tgt_sit >= 0 and led[tgt_sit]['n'] > 0:
                avg_t = led[tgt_sit]['tension'] / led[tgt_sit]['n']
            score = CURIOSITY * avg_t + novelty
            if tgt_sit in recent:
                score -= RECENT_PENALTY                   # 刚访问过 -> 别马上折返
            intents.append((a, ai, score))

        # 选择：撞墙后先转向脱困；抑制门控；否则在移动里贪心（随机破平）
        move_best = max([s for (a, ai, s) in intents if a != 'WAIT'] + [-1e9])
        cur_bad = (sit >= 0 and proto.n[sit] >= MIN_OBS * 2
                   and proto.classify(sit) in UNCTRL)
        tops = [it for it in intents if it[0] != 'WAIT' and it[2] >= move_best - 1e-9]
        intended = rng.choice(tops)
        i_ai = intended[1]
        i_tgt = proto.assign(model.predict(z, i_ai), create=False)
        i_cls = (proto.classify(i_tgt)
                 if (i_tgt >= 0 and proto.n[i_tgt] >= MIN_OBS * 2) else 'UNKNOWN')
        lt = led[i_tgt] if i_tgt >= 0 else None
        lt_hot = (lt is not None and (lt['standing'] >= 3
                  or (lt['n'] > 0 and lt['tension'] / lt['n'] >= t_high + 0.8)))
        # 抑制门控（世界侧「忍住」）：意图动作会踏进「仍挂起(高张力)且不可控」的账 -> 先不动
        suppress = ((not cur_bad) and i_cls in UNCTRL and lt_hot)
        br = ''
        if prev_blocked and rng.random() < 0.7:
            kind = rng.choice(MOVES); chosen_idx = ACTIONS.index(kind); br = 'escape'
        elif suppress:
            kind = 'WAIT'; chosen_idx = ACTIONS.index('WAIT'); br = 'suppress'
        elif rng.random() < EPS:
            kind = rng.choice(MOVES); chosen_idx = ACTIONS.index(kind); br = 'eps'
        else:
            kind = intended[0]; chosen_idx = i_ai
            br = 'escape-move' if cur_bad else 'greedy'

        if kind == 'WAIT':
            consec_wait += 1; stats['waits'] += 1
        else:
            consec_wait = 0
        stats['br_' + (br or '?')] = stats.get('br_' + (br or '?'), 0) + 1

        # 防死锁兜底：连续忍住超限 -> 强制试探一次（对应格子版 DEADLOCK_BREAK）
        if consec_wait >= DEADLOCK_BREAK:
            kind = rng.choice(MOVES); chosen_idx = ACTIONS.index(kind)
            consec_wait = 0; stats['deadlock_break'] += 1

        if os.environ.get('PXDEBUG') and step < 40:
            print(f"[dbg {step:3d}] gt={world.gt_region():<9} sit={sit} "
                  f"cls={proto.classify(sit):<8} protos={len(proto.C)} "
                  f"intents={[(a, round(s, 2)) for (a, ai, s) in intents]} move_best={move_best:.2f} "
                  f"[{br}] -> {kind} (prev_blocked={prev_blocked})")

        # ---------------- 执行 + 观测
        frame, blocked = world.step(kind)
        z_new = model.encode(frame)
        if blocked:
            stats['blocked_events'] += 1
        # 情境增量统计（在 sit 上用某动作产生的 latent 增量）
        delta = (z_new - z).detach().cpu().numpy().ravel()
        proto.observe(sit, chosen_idx, delta)
        proto.gt[sit][world.gt_region()] += 1
        if blocked:
            proto.blk[sit] += 1
        recent.append(sit)

        # ---------------- 反思/归因纪律
        e = led[sit]
        # 自我预测失败：张力骤高，且该情境尚属「可控」（外界的不怪自己）
        cls = proto.classify(sit)
        if tension >= t_high and cls not in ('BOUNDARY', 'CHAOS', 'EXTERNAL'):
            if sit not in reflected:
                reflected[sit] = chosen_idx
                stats['reflect_mark'] += 1
            else:
                stats['reflect_trig'] += 1
                if reflected[sit] != chosen_idx:
                    stats['reflect_convert'] += 1
                    reflected.pop(sit, None)
        # 挂账/死胡同/回访
        e['standing'] = e['standing'] + 1 if tension >= t_high else 0
        if e['seen'] > DEADEND_ATTEMPTS and e['standing'] > e['seen'] * 0.5:
            e['dead_end'] = True
        if e['seen'] > 1:
            e['revisit'] += 1

        # 在线续训（慢速；默认关闭，见 ONLINE_TRAIN 说明）
        buf.append((frame, z.detach(), chosen_idx, z_new.detach()))
        if ONLINE_TRAIN and len(buf) >= BATCH:
            idx = np.random.randint(0, len(buf), BATCH)
            model.train_batch([buf[i][0] for i in idx],
                              torch.cat([buf[i][1] for i in idx], 0),
                              [buf[i][2] for i in idx],
                              torch.cat([buf[i][3] for i in idx], 0))

        z_prev = z_new
        last_z = z                     # 本步动作前的 latent（供下一步预测）
        last_action_idx = chosen_idx
        prev_blocked = blocked

    return dict(world=world, model=model, proto=proto, led=led, stats=stats,
                log_tension=log_tension, outdir=outdir)


# ============================================================ 评测
def region_tension(log):
    agg = defaultdict(lambda: [0.0, 0])
    for sit, gt, t, _ in log:
        agg[gt][0] += t; agg[gt][1] += 1
    return {k: (v[0] / v[1], v[1]) for k, v in agg.items()}


def report(res):
    world, model, proto = res['world'], res['model'], res['proto']
    led, stats, log = res['led'], res['stats'], res['log_tension']
    outdir = res['outdir']
    print("=" * 74)
    print("认知主体 · 像素版 —— 第一步：连续高维世界的感知底座")
    print("=" * 74)

    rt = region_tension(log)
    print("\n[P1] 感知张力定位（latent 预测误差；只看像素+动作后果）")
    for k in ('CHAOS', 'NEAR_WALL', 'OPEN'):
        if k in rt:
            print(f"    {k:<10} 平均张力 = {rt[k][0]:.3f}   (n={rt[k][1]})")
    tn = stats['tension_sum'] / max(1, stats['n'])
    print(f"    全局平均张力 = {tn:.3f}")

    print("\n[P2] 账本")
    act = [v for v in led.values() if v['n'] >= MIN_OBS]
    standing = sum(1 for v in act if v['standing'] > 0)
    deadens = sum(1 for v in act if v['dead_end'])
    revisit = sum(v['revisit'] for v in led.values())
    print(f"    情境数(≥{MIN_OBS}观测)={len(act)}  挂起={standing}  死胡同={deadens}  回访={revisit}")
    print(f"    原型总数={len(proto.C)}  latent均访问={np.mean(proto.n):.1f}")

    print("\n[P3] 忍住不说/不动（WAIT）")
    print(f"    WAIT={stats['waits']}   防死锁强制试探={stats['deadlock_break']}   撞墙事件={stats['blocked_events']}")
    print(f"    决策分支计数: " + "  ".join(f"{k[3:]}={v}" for k, v in sorted(stats.items()) if k.startswith('br_')))

    # [P4] 划界：可控(自我) vs 不可控(外界/边界)
    print("\n[P4] 划界：动作对结果的可控度 (eta^2) 分类 vs 几何真值")
    conf = defaultdict(int)
    proto_cls = {}
    # 「可判定」= 观测足够且动作至少试过 2 种（classify 不返回 UNKNOWN）。
    # 观测不足的原型不计入准确率分母，但单独报告「覆盖率」，避免用"没数据"冒充"没判错"。
    n_elig = 0
    for p in range(len(proto.C)):
        if proto.n[p] < MIN_OBS:
            continue
        n_elig += 1
        cls = proto.classify(p)
        if cls == 'UNKNOWN':
            continue
        proto_cls[p] = cls
        gtd = proto.gt[p]
        if not gtd:
            continue
        gt = max(gtd.items(), key=lambda kv: kv[1])[0]
        conf[(gt, cls)] += 1
    def rate(gt, okmasks):
        tot = sum(v for (g, c), v in conf.items() if g == gt)
        hit = sum(v for (g, c), v in conf.items() if g == gt and c in okmasks)
        return (hit, tot)
    hc, tc = rate('CHAOS', {'CHAOS', 'EXTERNAL', 'BOUNDARY'})
    hw, tw = rate('NEAR_WALL', {'BOUNDARY', 'EXTERNAL', 'CHAOS'})
    ho, to = rate('OPEN', {'SELF'})
    n_dec = sum(conf.values())
    cov = n_dec / max(1, n_elig)
    print(f"    可判定覆盖率 = {n_dec}/{n_elig} = {cov:.2f}  （观测/动作种类不足 -> 不下结论）")
    print(f"    混沌区 -> 判为外界/边界: {hc}/{tc} = {hc/max(1,tc):.2f}")
    print(f"    墙附近 -> 判为边界/外界: {hw}/{tw} = {hw/max(1,tw):.2f}")
    print(f"    开敞区 -> 判为可控(自我): {ho}/{to} = {ho/max(1,to):.2f}")
    # 明细
    print("    真值 x 判定 混淆：")
    for gt in ('CHAOS', 'NEAR_WALL', 'OPEN'):
        row = {c: v for (g, c), v in conf.items() if g == gt}
        print(f"      {gt:<10} {row}")

    print("\n[P5] 反思链（归因纪律：外界不怪自己）")
    print(f"    标记={stats['reflect_mark']}  触发={stats['reflect_trig']}  "
          f"转化={stats['reflect_convert']}  转化率={stats['reflect_convert']/max(1,stats['reflect_trig']):.2f}")

    # [P6] 感知编码可分性：原始 latent 对真值区域的「最近质心分类」平衡准确率
    print("\n[P6] 感知编码可分性（latent 对真值区域的分类能力，非方差比）")
    G6 = 24
    probe6 = PixelWorld(random.Random(1))
    Z6, GT6 = [], []
    for gy in range(G6):
        for gx in range(G6):
            probe6.pos = [(gx + 0.5) / G6, (gy + 0.5) / G6]
            Z6.append(model.encode(probe6.render()).cpu().numpy().ravel())
            GT6.append(probe6.gt_region())
    Z6 = np.array(Z6); GT6 = np.array(GT6)
    regions = sorted(set(GT6))
    cen = {r: Z6[GT6 == r].mean(0) for r in regions}
    pred6 = np.array([min(regions, key=lambda r: np.sum((z - cen[r]) ** 2)) for z in Z6])
    bal6 = float(np.mean([(pred6[GT6 == r] == r).mean() for r in regions]))
    gm = Z6.mean(0)
    between = sum((GT6 == r).sum() * np.sum((cen[r] - gm) ** 2) for r in regions)
    within = sum(np.sum((Z6[GT6 == r] - cen[r]) ** 2) for r in regions)
    eta = between / (between + within + 1e-9)
    for r in regions:
        print(f"    {r:<10} 分类准确率={(pred6[GT6 == r] == r).mean():.2f}  n={(GT6 == r).sum()}")
    print(f"    平衡准确率 = {bal6:.3f}   (eta^2 参考值 = {eta:.3f})")

    # ---------- 产物：帧 + 分类地图
    frame = world.render()
    write_png(os.path.join(outdir, 'pixelworld_frame.png'), frame)
    # 分类地图：对格点位置渲染->编码->找最近原型->取分类
    G = 32
    mp = np.zeros((G, G, 3), dtype=np.uint8)
    mp[:, :] = (210, 210, 210)
    COL = {'SELF': (40, 200, 80), 'CHAOS': (230, 40, 40),
           'BOUNDARY': (40, 90, 230), 'EXTERNAL': (150, 40, 200), 'UNKNOWN': (240, 240, 240)}
    probe = PixelWorld(random.Random(0))
    for gy in range(G):
        for gx in range(G):
            probe.pos = [(gx + 0.5) / G, (gy + 0.5) / G]
            zz = model.encode(probe.render())
            p = proto.assign(zz, create=False)
            cls = proto.classify(p) if (p >= 0 and proto.n[p] >= MIN_OBS) else 'UNKNOWN'
            mp[gy, gx] = COL.get(cls, (240, 240, 240))
    write_png(os.path.join(outdir, 'pixel_controllability_map.png'), upscale(mp, 8))
    print(f"\n[产物] pixelworld_frame.png / pixel_controllability_map.png -> {outdir}")
    print("   地图色: 绿=可控(自我) 红=混沌 蓝=边界(撞墙) 紫=外界 灰=未访问")

    # ---------- 判定
    print("\n[判定]")
    ok1 = ('CHAOS' in rt and 'OPEN' in rt and rt['CHAOS'][0] > rt['OPEN'][0] + 0.10)
    ok2 = (len(act) > 0 and (standing > 0 or revisit > 0))
    ok3 = stats['waits'] > 0
    ok4 = (cov >= 0.6 and tc > 0 and hc / tc >= 0.6
           and tw > 0 and hw / tw >= 0.6 and to > 0 and ho / to >= 0.6)
    ok5 = stats['reflect_convert'] > 0
    ok6 = bal6 >= 0.6
    print(f"    P1 张力定位={'✓' if ok1 else '✗'}  P2 账本={'✓' if ok2 else '✗'}  "
          f"P3 忍住={'✓' if ok3 else '✗'}")
    print(f"    P4 划界={'✓' if ok4 else '✗'}  P5 反思={'✓' if ok5 else '✗'}  P6 编码可分={'✓' if ok6 else '✗'}")
    if all([ok1, ok2, ok3, ok4, ok5, ok6]):
        print("    结论：连续像素世界里，张力/账本/划界全部成立——脱离图论玩具。")
    else:
        print("    结论：部分成立，需继续调参/加强感知（见各 ✗ 项）。")
    return dict(ok=[ok1, ok2, ok3, ok4, ok5, ok6])


if __name__ == '__main__':
    sd = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else SEED
    report(run(seed=sd))
