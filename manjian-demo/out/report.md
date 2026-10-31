# 漫剪成片报告 · 破晓

- 音乐 BPM：**128**　节拍数：64　时长：30s
- 节拍检测对照真值：BPM 误差 0，相位误差 18ms，命中 63/64
- 素材：15 个镜头（library: 15 shots in /workspace/manjian-demo/assets/frames）
- 剪辑点：37 段　自动审片：**87.9** / 100（ship，迭代 5 次）

## 审片指标

| 维度 | 数值 | 目标 | 得分 | 权重 | 备注 |
|---|---|---|---|---|---|
| 切点踩拍率 | 1 | ≥ 0.95 | 100 | 0.3 | 36 个内部切点中 36 个落在节拍 ±50ms 内 |
| 段落节奏 | 0.819 | 每段贴近目标镜头长度 | 82 | 0.2 | intro/build/drop/outro 的镜头长度分布 |
| 素材变化度 | 0.432 | 无连续重复、素材复用有间隔 | 43 | 0.15 | 7 个不同镜头 / 37 段，连续重复 0 次 |
| 情绪曲线 | 1 | intro→build→drop 镜头持续加速 | 100 | 0.12 | intro 1.87s / build 0.75s / drop 0.51s |
| 重音处理 | 1 | 每个重音都有闪白/故障/冲击 | 100 | 0.1 | 3/3 个重音被强化 |
| 明暗对比 | 1 | 相邻镜头亮度有起伏 | 100 | 0.08 | 相邻亮度差均值 59.9% |
| 时长准确度 | 30 | 30s | 100 | 0.05 | 成片 30.00s / 目标 30s |
| 镜头离散度 | 0.494 | 有长短变化 | 50 | 0 | 信息项，不计分 |

## 迭代轨迹

- 第 0 轮：87.9 分，0 个问题
- 第 1 轮：87.9 分，0 个问题
- 第 2 轮：87.9 分，0 个问题
- 第 3 轮：87.9 分，0 个问题
- 第 4 轮：87.9 分，0 个问题
- 第 5 轮：87.9 分，0 个问题

## 段落结构

- **intro** 0–7.481s：4 刀，平均镜头 1.87s
- **build** 7.481–11.232s：5 刀，平均镜头 0.75s
- **drop** 11.232–22.483s：22 刀，平均镜头 0.51s
- **outro** 22.483–30s：6 刀，平均镜头 1.25s

## 剪辑决策表

| # | 段落 | 素材 | 时间 | 运镜 | 转场 | 特效 |
|---|---|---|---|---|---|---|
| e001 | intro | f12.jpg | 0–2.325 | zoom-in | fade | grain |
| e002 | intro | f14.jpg | 2.325–4.2 | pan-right | cut | grain |
| e003 | intro | f09.jpg | 4.2–5.606 | zoom-out | cut | grain |
| e004 | intro | f08.jpg | 5.606–7.481 | pan-left | cut | grain |
| e005 | build | f05.jpg | 7.481–8.419 | zoom-in | cut | grain |
| e006 | build | f03.jpg | 8.419–9.357 | pan-right | cut | grain |
| e007 | build | f12.jpg | 9.357–9.826 | zoom-out | cut | grain |
| e008 | build | f14.jpg | 9.826–10.294 | pan-left | cut | grain,shake |
| e009 | build | f09.jpg | 10.294–11.232 | zoom-in | cut | grain |
| e010 | drop | f06.jpg | 11.232–11.701 | shake | glitch | grain,flash,rgb-split,bloom |
| e011 | drop | f05.jpg | 11.701–12.17 | shake | cut | grain |
| e012 | drop | f08.jpg | 12.17–12.638 | zoom-out | cut | grain,glitch |
| e013 | drop | f12.jpg | 12.638–13.107 | zoom-in | cut | grain,scanline |
| e014 | drop | f14.jpg | 13.107–13.576 | punch-in | glitch | grain,rgb-split |
| e015 | drop | f09.jpg | 13.576–14.045 | shake | cut | grain,bloom |
| e016 | drop | f06.jpg | 14.045–14.514 | zoom-out | cut | grain |
| e017 | drop | f05.jpg | 14.514–14.982 | zoom-in | cut | grain |
| e018 | drop | f08.jpg | 14.982–15.92 | punch-in | cut | grain,rgb-split |
| e019 | drop | f12.jpg | 15.92–16.389 | shake | cut | grain,scanline |
| e020 | drop | f14.jpg | 16.389–16.858 | zoom-out | cut | grain,glitch,bloom |
| e021 | drop | f09.jpg | 16.858–17.326 | punch-in | flash | grain,flash |
| e022 | drop | f06.jpg | 17.326–17.795 | punch-in | cut | grain,rgb-split |
| e023 | drop | f05.jpg | 17.795–18.264 | shake | glitch | grain |
| e024 | drop | f08.jpg | 18.264–18.733 | zoom-out | cut | grain |
| e025 | drop | f12.jpg | 18.733–19.202 | zoom-in | cut | grain,scanline,bloom |
| e026 | drop | f14.jpg | 19.202–19.671 | punch-in | cut | grain,rgb-split |
| e027 | drop | f09.jpg | 19.671–20.608 | shake | cut | grain |
| e028 | drop | f06.jpg | 20.608–21.077 | zoom-out | cut | grain,glitch |
| e029 | drop | f05.jpg | 21.077–21.546 | zoom-in | cut | grain |
| e030 | drop | f08.jpg | 21.546–22.015 | punch-in | cut | grain,rgb-split,bloom |
| e031 | drop | f12.jpg | 22.015–22.483 | shake | cut | grain,scanline |
| e032 | outro | f14.jpg | 22.483–23.89 | shake | flash | grain,flash |
| e033 | outro | f09.jpg | 23.89–25.296 | zoom-in | cut | grain |
| e034 | outro | f03.jpg | 25.296–26.234 | pan-right | cut | grain |
| e035 | outro | f05.jpg | 26.234–27.64 | zoom-out | cut | grain |
| e036 | outro | f08.jpg | 27.64–28.578 | pan-left | cut | grain |
| e037 | outro | f12.jpg | 28.578–30 | zoom-in | cut | grain |

## 文字 / 排印

- 0.918–3.1180000000000003s · [title] “破晓” · 132px @ y=0.2
- 7.781–9.881s · [lyric] “当世界开始崩坏” · 54px @ y=0.7
- 11.232–11.572s · [impact] “断” · 240px @ y=0.38
- 13.107–13.447s · [impact] “斩” · 240px @ y=0.38
- 14.982–15.322s · [impact] “裂” · 240px @ y=0.38
- 16.858–17.198s · [impact] “燃” · 240px @ y=0.38
- 18.733–19.073s · [impact] “爆” · 240px @ y=0.38
- 20.608–20.948s · [impact] “醒” · 240px @ y=0.38
- 22.883–25.483s · [lyric] “这一次，为自己而战” · 58px @ y=0.74
- 28.6–29.8s · [credits] “DSH · MANJIAN ENGINE” · 40px @ y=0.86

> e001 intro f12.jpg 0-2.325s (beat 0) zoom-in/fade [grain]

生成器：dsh-manjian（导演-剪辑-审片闭环，规则来自 knowledge/ 知识库）
