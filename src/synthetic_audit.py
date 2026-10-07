"""Rà lỗi dữ liệu trong một thư mục định dạng KITTI (viết để tìm các lỗi cài sẵn trong data/synthetic).

    python -m src.synthetic_audit --help
    python -m src.synthetic_audit --data-root data/synthetic

Mỗi kiểm tra so một frame với TRUNG VỊ của cả bộ, nên không cần biết trước giá trị "đúng".
Kết quả: results/synthetic_audit.csv (mỗi dòng là một cờ cảnh báo) và results/figures/synthetic_audit.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from starter.datasets import list_frames, load_points

AZ_BIN_DEG = 5


def audit(data_root: str, nan_thr: float, sector_ratio_thr: float, dt_ratio_thr: float) -> tuple[pd.DataFrame, dict]:
    frames = list_frames(data_root)
    flags, az_hist, n_valid = [], [], []
    for fid in frames:
        pts = load_points(data_root, fid)
        finite = np.isfinite(pts).all(axis=1)
        bad = int((~finite).sum())
        if bad / len(pts) > nan_thr:
            cols = [c for c, k in zip("xyzi", range(4)) if not np.isfinite(pts[:, k]).all()]
            flags.append({"frame_id": fid, "check": "invalid_points", "layer": "I/O",
                          "value": f"{bad}/{len(pts)} điểm ({bad / len(pts):.2%}) có NaN/Inf ở cột {','.join(cols)}",
                          "how_detected": "np.isfinite trên từng dòng; nếu không lọc, phép chiếu sinh NaN và metric sai"})
        p = pts[finite]
        az = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
        az_hist.append(np.histogram(az, bins=360 // AZ_BIN_DEG, range=(-180, 180))[0])
        n_valid.append(len(p))
    az_hist = np.array(az_hist)
    med = np.maximum(np.median(az_hist, axis=0), 1)
    for fid, h, n in zip(frames, az_hist, n_valid):
        low = np.flatnonzero(h / med < sector_ratio_thr)
        if len(low) >= 2:
            lo, hi = -180 + AZ_BIN_DEG * low.min(), -180 + AZ_BIN_DEG * (low.max() + 1)
            flags.append({"frame_id": fid, "check": "sector_dropout", "layer": "I/O",
                          "value": f"azimuth {lo}..{hi} độ chỉ còn {h[low].sum() / med[low].sum():.0%} số điểm so với trung vị "
                                   f"các frame; tổng điểm {n} so với trung vị {int(np.median(n_valid))}",
                          "how_detected": f"histogram azimuth {AZ_BIN_DEG} độ/bin, so từng bin với trung vị của mọi frame"})
    ts_path = Path(data_root) / "training" / "timestamps.txt"
    ts = np.loadtxt(ts_path) if ts_path.exists() else None
    if ts is not None and len(ts) == len(frames):
        dt = np.diff(ts)
        for i in np.flatnonzero(dt > dt_ratio_thr * np.median(dt)):
            flags.append({"frame_id": f"{frames[i]}->{frames[i + 1]}", "check": "time_gap", "layer": "Time",
                          "value": f"dt = {dt[i]:.3f} s, gấp {dt[i] / np.median(dt):.1f} lần chu kỳ trung vị {np.median(dt):.3f} s "
                                   "(mất 1 frame hoặc timestamp sai)",
                          "how_detected": "np.diff(timestamps.txt) so với trung vị"})
    return pd.DataFrame(flags), {"frames": frames, "az_hist": az_hist, "ts": ts}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/synthetic", help="thư mục định dạng KITTI")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--nan-thr", type=float, default=0.0, help="cờ nếu tỉ lệ điểm NaN/Inf vượt ngưỡng này")
    ap.add_argument("--sector-ratio-thr", type=float, default=0.6, help="cờ nếu bin azimuth còn dưới tỉ lệ này so với trung vị")
    ap.add_argument("--dt-ratio-thr", type=float, default=1.5, help="cờ nếu dt vượt số lần này của dt trung vị")
    args = ap.parse_args()

    flags, extra = audit(args.data_root, args.nan_thr, args.sector_ratio_thr, args.dt_ratio_thr)
    out = Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    flags.to_csv(out / "synthetic_audit.csv", index=False)
    print(flags.to_string(index=False) if len(flags) else "Không có cờ nào.")

    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    centers = np.arange(-180, 180, AZ_BIN_DEG) + AZ_BIN_DEG / 2
    for fid, h in zip(extra["frames"], extra["az_hist"]):
        axs[0].plot(centers, h, label=fid)
    axs[0].set(xlabel="azimuth (độ, 0 = phía trước)", ylabel=f"số điểm / {AZ_BIN_DEG} độ", title="Mật độ điểm theo góc quét")
    axs[0].legend()
    if extra["ts"] is not None:
        axs[1].bar(range(len(extra["ts"]) - 1), np.diff(extra["ts"]))
        axs[1].set_xticks(range(len(extra["ts"]) - 1), [f"{a}->{b}" for a, b in zip(extra["frames"][:-1], extra["frames"][1:])],
                          rotation=20)
        axs[1].set(ylabel="dt (s)", title="Khoảng cách thời gian giữa 2 frame liên tiếp")
    fig.tight_layout()
    fig.savefig(out / "figures" / "synthetic_audit.png", dpi=110)
    print(f"-> {out / 'synthetic_audit.csv'}, {out / 'figures' / 'synthetic_audit.png'}")


if __name__ == "__main__":
    main()
