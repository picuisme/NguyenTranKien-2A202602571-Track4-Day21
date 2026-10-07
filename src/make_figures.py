"""Vẽ ảnh minh hoạ cho báo cáo: ảnh demo "gần vs xa" và 2 ảnh failure case.

    python -m src.make_figures --help
    python -m src.make_figures                # vẽ cả 3 ảnh vào results/figures/
    python -m src.make_figures --only fail_01 --roll-frame 000016

Chạy SAU `python -m src.calib_qa all`, vì tiêu đề ảnh failure đọc tỉ lệ phát hiện từ results/drift_detection.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calib_common import inbox_counts, load_frame_with_ring, object_point_sets
from starter.projection import cam_to_image, perturb_extrinsic, velo_to_cam


def _uv(fr, calib, idx):
    """Pixel (M, 2) của các điểm fr["points"][idx] khi chiếu bằng calib, không lọc theo khung ảnh."""
    pc = velo_to_cam(fr["points"][idx, :3], calib)
    proj = np.hstack([pc, np.ones((len(pc), 1))]) @ calib.P2.T
    return proj[:, :2] / proj[:, 2:3]


def _crop_panel(ax, fr, calib, o, title, pad=1.2):
    x1, y1, x2, y2 = o.obj.bbox
    w, h = x2 - x1, y2 - y1
    uv = _uv(fr, calib, o.idx)
    (a, b), = inbox_counts(fr, calib, [o])
    ax.imshow(cv2.cvtColor(fr["image"], cv2.COLOR_BGR2RGB))
    inside = (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
    ax.scatter(uv[inside, 0], uv[inside, 1], s=10, c="lime", label="trong box")
    ax.scatter(uv[~inside, 0], uv[~inside, 1], s=10, c="red", label="ngoài box")
    ax.add_patch(plt.Rectangle((x1, y1), w, h, fill=False, ec="yellow", lw=1.5))
    img_h = fr["image"].shape[0]
    ax.set_facecolor("0.35")                                   # vùng xám = bên ngoài khung ảnh
    ax.set_xlim(x1 - pad * w, x2 + pad * w)
    ax.set_ylim(min(img_h, y2 + pad * h * 0.5), max(0, y1 - pad * h * 0.5))
    ax.set_title(f"{title}\n{o.obj.type} cách {o.dist_m:.0f} m: {a}/{b} điểm trong box ({100 * a / b:.0f}%)", fontsize=10)
    ax.set_xticks([]), ax.set_yticks([])


def demo_near_vs_far(args, out: Path) -> None:
    fr = load_frame_with_ring(args.data_root, args.demo_frame)
    objs = object_point_sets(fr)
    near, far = min(objs, key=lambda o: o.dist_m if o.obj.truncated < 0.5 else 1e9), max(objs, key=lambda o: o.dist_m)
    bad = perturb_extrinsic(fr["calib"], yaw_deg=args.yaw_deg)
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
    _crop_panel(axs[0, 0], fr, fr["calib"], near, "GẦN, calib đúng")
    _crop_panel(axs[0, 1], fr, bad, near, f"GẦN, lệch yaw {args.yaw_deg}°")
    _crop_panel(axs[1, 0], fr, fr["calib"], far, "XA, calib đúng")
    _crop_panel(axs[1, 1], fr, bad, far, f"XA, lệch yaw {args.yaw_deg}°")
    axs[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"KITTI {args.demo_frame}: cùng một độ lệch yaw, vật xa mất nhiều điểm hơn vật gần (nguồn ảnh: KITTI Vision Benchmark Suite)")
    fig.tight_layout()
    path = out / f"demo_yaw{args.yaw_deg:g}deg_near_vs_far_{args.demo_frame}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("->", path)


def _detect_rate(det: pd.DataFrame | None, dataset: str, score: str, axis: str, level: float) -> str:
    if det is None:
        return "?"
    d = det[(det["dataset"] == dataset) & (det["score"] == score) & (det["axis"] == axis) & np.isclose(det["level"], level)]
    return f"{d['detect_rate_pct'].iloc[0]:.0f}%" if len(d) else "?"


def fail_roll(args, out: Path, det) -> None:
    """Failure 1: lệch roll nhỏ gần như không bị 2 score phát hiện, dù điểm ở rìa ảnh đã lệch rõ."""
    fr = load_frame_with_ring(args.data_root, args.roll_frame)
    objs = object_point_sets(fr)
    bad = perturb_extrinsic(fr["calib"], roll_deg=args.roll_deg)
    _, _, m = cam_to_image(velo_to_cam(fr["points"][:, :3], fr["calib"]), fr["calib"].P2, fr["image"].shape)
    idx = np.flatnonzero(m)
    uv0, uv1 = _uv(fr, fr["calib"], idx), _uv(fr, bad, idx)
    shift = np.linalg.norm(uv1 - uv0, axis=1)
    cnt = inbox_counts(fr, bad, objs)
    a, b = sum(x for x, _ in cnt), sum(y for _, y in cnt)
    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.imshow(cv2.cvtColor(fr["image"], cv2.COLOR_BGR2RGB))
    sc = ax.scatter(uv1[:, 0], uv1[:, 1], s=2, c=shift, cmap="turbo", vmin=0)
    for o in objs:
        x1, y1, x2, y2 = o.obj.bbox
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="white", lw=1.5))
    ax.plot(fr["calib"].P2[0, 2], fr["calib"].P2[1, 2], "w+", ms=18, mew=2)
    fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.01, label="độ dịch của điểm so với calib đúng (pixel)")
    name = Path(args.data_root).name
    ax.set_title(f"FAILURE 1 — KITTI {args.roll_frame}, lệch roll {args.roll_deg}°: điểm dịch trung bình {shift.mean():.1f} px, tối đa {shift.max():.1f} px ở rìa ảnh, "
                 f"nhưng {a}/{b} điểm của vật vẫn nằm trong box ({100 * a / b:.0f}%)\n"
                 f"Tỉ lệ phát hiện drift ở mức này: in-box {_detect_rate(det, name, 'inbox_ratio', 'roll', args.roll_deg)}, "
                 f"edge-margin {_detect_rate(det, name, 'edge_margin', 'roll', args.roll_deg)}. Dấu + là tâm ảnh: càng gần tâm, điểm càng ít dịch.",
                 fontsize=10)
    ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    path = out / f"fail_01_roll_{args.roll_deg:g}deg_undetected_{args.roll_frame}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("->", path)


def fail_time(args, out: Path) -> None:
    """Failure 2: calib đúng nhưng bỏ bù chuyển động xe (lỗi Time) -> score in-box báo 'drift' giả."""
    ok = load_frame_with_ring(args.nusc_root, args.time_frame, use_ego_motion=True)
    bad = load_frame_with_ring(args.nusc_root, args.time_frame, use_ego_motion=False)
    objs = object_point_sets(ok)
    cnt = inbox_counts(ok, bad["calib"], objs)
    worst = objs[int(np.argmin([a / b for a, b in cnt]))]
    dt = (ok["timestamp_camera_us"] - ok["timestamp_lidar_us"]) / 1000.0
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    _crop_panel(axs[0], ok, ok["calib"], worst, "Có bù chuyển động xe (đúng)", pad=0.6)
    _crop_panel(axs[1], ok, bad["calib"], worst, "KHÔNG bù chuyển động (coi như 2 sensor chụp cùng lúc)", pad=0.6)
    axs[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"FAILURE 2 — nuScenes {args.time_frame}: calibration KHÔNG lệch, camera chụp trước LiDAR {abs(dt):.0f} ms.\n"
                 "Bỏ bù chuyển động làm điểm của vật gần trượt khỏi box (vùng xám = ngoài khung ảnh), score in-box báo 'drift' giả. Nguồn ảnh: nuScenes (Motional)", fontsize=10)
    fig.tight_layout()
    path = out / f"fail_02_time_no_ego_comp_{args.time_frame}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("->", path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["demo", "fail_01", "fail_02"], help="chỉ vẽ một ảnh (mặc định vẽ cả 3)")
    ap.add_argument("--data-root", default="data/kitti_mini", help="dataset KITTI cho ảnh demo và failure 1")
    ap.add_argument("--nusc-root", default="data/nuscenes_mini_subset", help="dataset nuScenes cho failure 2")
    ap.add_argument("--demo-frame", default="000011")
    ap.add_argument("--yaw-deg", type=float, default=1.0)
    ap.add_argument("--roll-frame", default="000016")
    ap.add_argument("--roll-deg", type=float, default=1.0)
    ap.add_argument("--time-frame", default="scene-0103_008")
    ap.add_argument("--out-dir", default="results/figures")
    ap.add_argument("--detection-csv", default="results/drift_detection.csv")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    det = pd.read_csv(args.detection_csv) if Path(args.detection_csv).exists() else None
    if args.only in (None, "demo"):
        demo_near_vs_far(args, out)
    if args.only in (None, "fail_01"):
        fail_roll(args, out, det)
    if args.only in (None, "fail_02"):
        fail_time(args, out)


if __name__ == "__main__":
    main()
