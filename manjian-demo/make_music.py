"""Synthesize a 128 BPM phonk/EDM-style beat track used to demo the Manjian pipeline.

Outputs a stereo WAV plus a ground-truth beat grid JSON so the beat detector can
be scored against known cut points.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, sosfilt

SR = 44100
BPM = 128
BEAT = 60.0 / BPM
BAR = 4 * BEAT
BARS = 16
DURATION = BARS * BAR  # 30.0 s exactly
N = int(round(DURATION * SR))

rng = np.random.default_rng(7)
t_all = np.arange(N) / SR


def env_decay(n: int, rate: float) -> np.ndarray:
    return np.exp(-np.arange(n) / SR * rate)


def lowpass(x: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
    sos = butter(order, cutoff, btype="low", fs=SR, output="sos")
    return sosfilt(sos, x)


def highpass(x: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
    sos = butter(order, cutoff, btype="high", fs=SR, output="sos")
    return sosfilt(sos, x)


def add(buf: np.ndarray, start: float, sig: np.ndarray) -> None:
    i = int(round(start * SR))
    if i >= len(buf):
        return
    n = min(len(sig), len(buf) - i)
    buf[i:i + n] += sig[:n]


def kick(vel: float = 1.0) -> np.ndarray:
    n = int(0.30 * SR)
    tt = np.arange(n) / SR
    sweep = 95 - 55 * (1 - np.exp(-tt * 30))
    phase = 2 * np.pi * np.cumsum(sweep) / SR
    body = np.sin(phase) * np.exp(-tt * 16)
    click = rng.normal(0, 1, n) * np.exp(-tt * 180) * 0.35
    return (body + click) * vel


def snare(vel: float = 1.0) -> np.ndarray:
    n = int(0.18 * SR)
    tt = np.arange(n) / SR
    noise = highpass(rng.normal(0, 1, n), 1800) * np.exp(-tt * 42)
    tone = np.sin(2 * np.pi * 180 * tt) * np.exp(-tt * 38) * 0.35
    return (noise * 0.8 + tone) * vel


def hat(vel: float = 1.0) -> np.ndarray:
    n = int(0.055 * SR)
    tt = np.arange(n) / SR
    return highpass(rng.normal(0, 1, n), 6500, 6) * np.exp(-tt * 90) * vel


def impact(vel: float = 1.0) -> np.ndarray:
    n = int(0.55 * SR)
    tt = np.arange(n) / SR
    boom = np.sin(2 * np.pi * (42 - 8 * np.exp(-tt * 8)) * tt) * np.exp(-tt * 6)
    noise = lowpass(rng.normal(0, 1, n), 900) * np.exp(-tt * 9) * 0.5
    return (boom + noise) * vel


def riser(dur: float) -> np.ndarray:
    n = int(dur * SR)
    tt = np.arange(n) / SR
    noise = rng.normal(0, 1, n)
    sweep = np.sin(2 * np.pi * (200 + 900 * (tt / dur) ** 2) * tt) * 0.25
    out = np.zeros(n)
    step = int(0.05 * SR)
    for i in range(0, n, step):
        cutoff = 300 + 7000 * (i / n) ** 2
        seg = noise[i:i + step]
        out[i:i + step] = lowpass(seg, cutoff, 2)
    env = (tt / dur) ** 2.2
    return (out * 0.8 + sweep) * env


def saw(freq: np.ndarray | float, n: int, detune: float = 0.0) -> np.ndarray:
    tt = np.arange(n) / SR
    if np.isscalar(freq):
        phase = 2 * np.pi * float(freq) * tt
    else:
        phase = 2 * np.pi * np.cumsum(freq) / SR
    sig = 2 * (phase / (2 * np.pi) % 1.0) - 1
    if detune:
        phase2 = 2 * np.pi * (float(np.mean(freq)) * (1 + detune)) * tt
        sig = sig + (2 * (phase2 / (2 * np.pi) % 1.0) - 1) * 0.6
    return sig


mix = np.zeros(N)

# --- Drums -------------------------------------------------------------
for bar in range(BARS):
  for b in range(4):
    pos = bar * BAR + b * BEAT
    in_drop = 6 <= bar < 12
    in_intro = bar < 4
    vel = 1.0 if in_drop else (0.62 if in_intro else 0.8)
    add(mix, pos, kick(vel))
  for b in (1, 3):
    pos = bar * BAR + b * BEAT
    if bar >= 4 and bar < 12:
        add(mix, pos, snare(0.9))
  for e in range(8):
    pos = bar * BAR + e * BEAT / 2
    if bar < 4 and e % 2 == 1:
        continue
    vel = 0.75 if e % 2 == 0 else 0.5
    add(mix, pos, hat(vel * (1.0 if bar < 12 else 0.6)))

# --- Bass --------------------------------------------------------------
for bar in range(2, 16):
    for e in range(8):
        pos = bar * BAR + e * BEAT / 2
        dur = BEAT / 2 * 0.9
        nn = int(dur * SR)
        root = 55.0 if (bar % 4) < 2 else 65.4
        f = root * (1.0 if e % 4 else 1.0)
        sig = saw(f, nn) * env_decay(nn, 24)
        add(mix, pos, lowpass(sig, 380) * (0.22 if bar < 6 else 0.30))

# --- Lead arpeggio in drop ---------------------------------------------
scale = [220.0, 246.9, 293.7, 349.6]
for bar in range(6, 12):
    for e in range(8):
        if (bar + e) % 3 == 0:
            continue
        pos = bar * BAR + e * BEAT / 2
        nn = int(BEAT / 2 * 0.8 * SR)
        note = scale[(e + bar) % len(scale)]
        sig = saw(note, nn, detune=0.012) * env_decay(nn, 30)
        add(mix, pos, highpass(sig, 180) * 0.10)

# --- Pad ---------------------------------------------------------------
pad = np.zeros(N)
for bar, chord in [(0, [110, 146.8, 174.6]), (4, [110, 138.6, 174.6]), (8, [130.8, 164.8, 196]), (12, [110, 138.6, 174.6])]:
    start = int(bar * BAR * SR)
    end = min(N, start + int(4 * BAR * SR))
    nn = end - start
    tt = np.arange(nn) / SR
    sig = sum(np.sin(2 * np.pi * f * tt + i) for i, f in enumerate(chord))
    sig *= 0.5 * (1 + np.sin(2 * np.pi * 0.25 * tt))
    pad[start:end] += lowpass(sig, 900) * 0.05
mix += pad

# --- Build riser + impacts ---------------------------------------------
add(mix, 4 * BAR, riser(2 * BAR))
for pos in [6 * BAR, 9 * BAR, 12 * BAR]:
    add(mix, pos, impact(1.0))
add(mix, 0.0, impact(0.5))

# --- Sidechain-style ducking on the drop -------------------------------
duck = np.ones(N)
for bar in range(6, 12):
    for b in range(4):
        i = int((bar * BAR + b * BEAT) * SR)
        n = int(0.18 * SR)
        if i < N:
            k = min(n, N - i)
            duck[i:i + k] = np.minimum(duck[i:i + k], 0.55 + 0.45 * np.linspace(0, 1, k))
mix *= duck

# --- Master ------------------------------------------------------------
fade_in = int(0.4 * SR)
mix[:fade_in] *= np.linspace(0, 1, fade_in)
fade_out = int(1.2 * SR)
mix[-fade_out:] *= np.linspace(1, 0, fade_out)
mix = np.tanh(mix / (np.max(np.abs(mix)) + 1e-9) * 1.35) / np.tanh(1.35)
stereo = np.stack([mix, mix], axis=1)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "beat_128.wav")
wavfile.write(out, SR, (stereo * 0.92 * 32767).astype(np.int16))

truth = {
    "source": "synthesized",
    "bpm": BPM,
    "beat": BEAT,
    "bar": BAR,
    "duration": DURATION,
    "beats": [round(i * BEAT, 4) for i in range(int(DURATION / BEAT))],
    "sections": [
        {"name": "intro", "start": 0.0, "end": 4 * BAR},
        {"name": "build", "start": 4 * BAR, "end": 6 * BAR},
        {"name": "drop", "start": 6 * BAR, "end": 12 * BAR},
        {"name": "outro", "start": 12 * BAR, "end": DURATION},
    ],
    "impacts": [round(x, 4) for x in [6 * BAR, 9 * BAR, 12 * BAR]],
}
with open(os.path.join(os.path.dirname(out), "beat_128.truth.json"), "w", encoding="utf-8") as fh:
    json.dump(truth, fh, indent=2, ensure_ascii=False)
print("wrote", out, "duration", DURATION, "beats", len(truth["beats"]))
