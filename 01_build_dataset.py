import os
import numpy as np
import pandas as pd
import wfdb

# 输出目录
os.makedirs("outputs/tables", exist_ok=True)

# MIT-BIH 常用记录：100–124、200–234（最常见集合之一）
# 这样比你之前 100-147 更靠谱，避免很多不存在的记录号导致下载报错。
RECORDS = [f"{i:03d}" for i in range(100, 125)] + [f"{i:03d}" for i in range(200, 235)]

LEFT_SEC = 0.2   # R峰前 0.2s
RIGHT_SEC = 0.4  # R峰后 0.4s

def map_label(sym: str):
    # 5类聚合：N/S/V/F/Q
    if sym in {'N','L','R','e','j'}: return 'N'
    if sym in {'A','a','J','S'}:     return 'S'
    if sym in {'V','E'}:            return 'V'
    if sym == 'F':                  return 'F'
    if sym in {'/','Q','f','?'}:    return 'Q'
    return None

def extract_features(x: np.ndarray):
    # 转 float，去均值
    x = x.astype(np.float32)
    x = x - np.mean(x)
    std = np.std(x) + 1e-8

    feats = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "max": float(np.max(x)),
        "min": float(np.min(x)),
        "ptp": float(np.ptp(x)),  # ✅ NumPy 2.0 兼容修复点
        "rms": float(np.sqrt(np.mean(x**2))),
        "abs_mean": float(np.mean(np.abs(x))),
    }

    # 频域能量（简单但够写论文）
    xn = x / std
    fft = np.abs(np.fft.rfft(xn))**2
    bands = np.array_split(fft, 5)
    for i, b in enumerate(bands):
        feats[f"fft_band_{i}"] = float(np.sum(b))
    feats["fft_total"] = float(np.sum(fft))
    return feats

rows = []
skipped_records = []

for rec in RECORDS:
    try:
        record = wfdb.rdrecord(rec, pn_dir="mitdb")
        ann = wfdb.rdann(rec, "atr", pn_dir="mitdb")
    except Exception as e:
        skipped_records.append((rec, str(e)))
        continue

    fs = int(record.fs)
    sig = record.p_signal[:, 0]  # 先用导联0，最快

    L = int(LEFT_SEC * fs)
    R = int(RIGHT_SEC * fs)

    for s, sym in zip(ann.sample, ann.symbol):
        label = map_label(sym)
        if label is None:
            continue

        start = s - L
        end = s + R
        if start < 0 or end >= len(sig):
            continue

        beat = sig[start:end]
        feats = extract_features(beat)
        feats["label"] = label
        feats["record_id"] = rec
        feats["r_peak_sample"] = int(s)
        rows.append(feats)

df = pd.DataFrame(rows)
out_path = "outputs/tables/beat_features.csv"
df.to_csv(out_path, index=False)

print("✅ Saved:", out_path)
print("✅ Total beats:", len(df))
print("✅ Label counts:\n", df["label"].value_counts())

if skipped_records:
    print("\n⚠️ Skipped records (download/read issues):")
    for rec, msg in skipped_records[:10]:
        print(f"  - {rec}: {msg}")
    if len(skipped_records) > 10:
        print(f"  ... and {len(skipped_records)-10} more")