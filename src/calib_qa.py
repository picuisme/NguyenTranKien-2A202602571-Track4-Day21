"""Topic A: LiDAR-camera projection QA. Công cụ dòng lệnh, dùng lại được cho dataset khác cùng định dạng.

Chạy từ thư mục gốc repo:
    python -m src.calib_qa --help
    python -m src.calib_qa overlay  --data-root data/kitti_mini --frame 000011
    python -m src.calib_qa sweep    --data-root data/kitti_mini
    python -m src.calib_qa detect   --per-frame results/calib_perturb_per_frame_kitti_mini.csv
    python -m src.calib_qa timesync --data-root data/nuscenes_mini_subset
    python -m src.calib_qa latency  --data-root data/kitti_mini
    python -m src.calib_qa all      # chạy lại toàn bộ kết quả trong báo cáo

Mọi phép ngẫu nhiên (bootstrap ở lệnh detect) đều cố định seed, nên chạy lại ra đúng cùng số liệu.
"""
from __future__ import annotations

import argparse
import sys
import platform
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calib_common import (DIST_BINS, depth_edge_weights, edge_alignment_score, image_edge_map,
                              inbox_counts, load_frame_with_ring, mean_pixel_shift, object_point_sets)
from starter.datasets import dataset_type, list_frames
from starter.projection import draw_box2d, overlay_points, perturb_extrinsic, project_velo_to_image

SEED = 0
ROT_LEVELS_DEG = [0.25, 0.5, 1.0, 2.0, 3.0]
TRANS_LEVELS_M = [0.02, 0.05, 0.10, 0.20]
AXES = {  # tên trục -> (tham số của perturb_extrinsic, đơn vị, các mức)
    "yaw": ("yaw_deg", "deg", ROT_LEVELS_DEG),
    "pitch": ("pitch_deg", "deg", ROT_LEVELS_DEG),
    "roll": ("roll_deg", "deg", ROT_LEVELS_DEG),
    "tx": ("t_x", "m", TRANS_LEVELS_M),
    "ty": ("t_y", "m", TRANS_LEVELS_M),
    "tz": ("t_z", "m", TRANS_LEVELS_M),
}
# Tên trục ở trên là tên tham số của perturb_extrinsic, tính trong LiDAR frame. Hai dataset đặt trục LiDAR khác nhau
# (KITTI: x tiến, y trái. nuScenes: x phải, y tiến), nên cùng một tham số lại là chuyển động vật lý khác nhau của xe.
# Bảng này quy về hệ trục của XE để so sánh 2 dataset cho đúng.
VEHICLE_AXIS = {
    "kitti": {"yaw": "yaw", "pitch": "pitch", "roll": "roll", "tx": "forward", "ty": "lateral", "tz": "up", "none": "-"},
    "nuscenes": {"yaw": "yaw", "pitch": "roll", "roll": "pitch", "tx": "lateral", "ty": "forward", "tz": "up", "none": "-"},
}
NEIGHBOR_DEG = 1.0   # edge-margin so calib đang dùng với 6 calib "hàng xóm" lệch +-1 độ quanh nó


def make_perturbed(calib, axis: str, level: float):
    if axis == "none":
        return calib
    key = AXES[axis][0]
    if key.startswith("t_"):
        t = [0.0, 0.0, 0.0]
        t["xyz".index(key[-1])] = level
        return perturb_extrinsic(calib, t_xyz_m=tuple(t))
    return perturb_extrinsic(calib, **{key: level})


def edge_margin(fr, calib, edge_w, edge_map) -> tuple[float, float]:
    """(score của calib, margin = score - trung bình score của 6 calib lệch +-1 độ yaw/pitch/roll).

    Calib đúng nằm ở đỉnh của score nên margin dương rõ rệt. Calib đã lệch nằm ở vùng phẳng nên margin ~ 0.
    Margin không cần label và không phụ thuộc cảnh nhiều như score thô."""
    s = edge_alignment_score(fr, calib, edge_w, edge_map)
    nb = [edge_alignment_score(fr, perturb_extrinsic(calib, **{k: sign * NEIGHBOR_DEG}), edge_w, edge_map)
          for k in ("yaw_deg", "pitch_deg", "roll_deg") for sign in (1, -1)]
    return s, float(s - np.nanmean(nb))


def frame_rows(data_root: str, frame_id: str, configs: list[tuple[str, float]], **load_kwargs) -> list[dict]:
    """Đo mọi metric của một frame cho từng cấu hình (axis, level)."""
    fr = load_frame_with_ring(data_root, frame_id, **load_kwargs)
    objs = object_point_sets(fr)
    edge_w = depth_edge_weights(fr["points"], fr["ring"])
    edge_map = image_edge_map(fr["image"])
    vehicle = VEHICLE_AXIS[dataset_type(data_root)]
    rows = []
    for axis, level in configs:
        calib = make_perturbed(fr["calib"], axis, level)
        _, _, mask = project_velo_to_image(fr["points"], calib, fr["image"].shape)
        row = {"dataset": Path(data_root).name, "frame_id": frame_id, "axis": axis, "vehicle_axis": vehicle[axis],
               "level": level, "unit": "-" if axis == "none" else AXES[axis][1],
               "fov_ratio": float(mask.mean()), "pixel_shift_px": mean_pixel_shift(fr, calib),
               "n_objects": len(objs)}
        counts = inbox_counts(fr, calib, objs)
        for name, _, _ in [("all", 0, 0)] + DIST_BINS:
            sel = [c for c, o in zip(counts, objs) if name == "all" or o.dist_bin == name]
            row[f"inbox_in_{name}"] = sum(a for a, _ in sel)
            row[f"inbox_total_{name}"] = sum(b for _, b in sel)
        row["edge_score"], row["edge_margin"] = edge_margin(fr, calib, edge_w, edge_map)
        rows.append(row)
    return rows


def all_configs() -> list[tuple[str, float]]:
    return [("none", 0.0)] + [(axis, lv) for axis, (_, _, levels) in AXES.items() for lv in levels]


def aggregate(per_frame: pd.DataFrame) -> pd.DataFrame:
    """Gộp theo (dataset, axis, level): tỉ lệ in-box tính trên tổng số điểm (không phải trung bình các frame)."""
    out = []
    for (ds, axis, level), g in per_frame.groupby(["dataset", "axis", "level"], sort=False):
        row = {"dataset": ds, "axis": axis, "vehicle_axis": g["vehicle_axis"].iloc[0], "level": level,
               "unit": g["unit"].iloc[0], "n_frames": len(g),
               "n_objects": int(g["n_objects"].sum()), "fov_ratio_pct": 100 * g["fov_ratio"].mean(),
               "pixel_shift_px": g["pixel_shift_px"].mean()}
        for name in ["all"] + [b[0] for b in DIST_BINS]:
            tot = g[f"inbox_total_{name}"].sum()
            row[f"inbox_pct_{name}"] = 100 * g[f"inbox_in_{name}"].sum() / tot if tot else np.nan
        row["edge_score_mean"] = g["edge_score"].mean()
        row["edge_margin_mean"] = g["edge_margin"].mean()
        out.append(row)
    return pd.DataFrame(out).round(4)


# ------------------------------------------------------------------------------- overlay
def cmd_overlay(args) -> None:
    """Ảnh demo: 3 ảnh overlay cho 3 khoảng cách + (tuỳ chọn) ảnh sau khi làm lệch calibration."""
    fr = load_frame_with_ring(args.data_root, args.frame)
    calib = perturb_extrinsic(fr["calib"], args.roll_deg, args.pitch_deg, args.yaw_deg, (args.tx, args.ty, args.tz))
    uv, depth, _ = project_velo_to_image(fr["points"], calib, fr["image"].shape)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if not any([args.roll_deg, args.pitch_deg, args.yaw_deg, args.tx, args.ty, args.tz]) else \
        f"_r{args.roll_deg}_p{args.pitch_deg}_y{args.yaw_deg}_t{args.tx}_{args.ty}_{args.tz}"
    for name, lo, hi in DIST_BINS:
        sel = (depth >= lo) & (depth < hi)
        vis = overlay_points(fr["image"], uv[sel], depth[sel], max_depth=60.0)
        for obj in fr["labels"]:
            if lo <= obj.location[2] < hi:
                vis = draw_box2d(vis, obj.bbox, label=f"{obj.type} {obj.location[2]:.0f}m")
        cv2.putText(vis, f"{args.frame} | {name} | {int(sel.sum())} diem", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        path = out_dir / f"overlay_{args.frame}_{name}{tag}.png"
        cv2.imwrite(str(path), vis)
        print(f"{name}: {int(sel.sum())} điểm -> {path}")


# ------------------------------------------------------------------------------- sweep
def cmd_sweep(args) -> None:
    """Thí nghiệm chính: làm lệch từng trục (yaw/pitch/roll/tx/ty/tz), mỗi lần một trục, đo trên mọi frame."""
    frames = list_frames(args.data_root)[:: args.every]
    name = Path(args.data_root).name
    rows = []
    for i, fid in enumerate(frames):
        rows += frame_rows(args.data_root, fid, all_configs())
        print(f"\r{name}: {i + 1}/{len(frames)} frame", end="", flush=True)
    print()
    per_frame = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_frame.round(5).to_csv(out / f"calib_perturb_per_frame_{name}.csv", index=False, lineterminator="\n")
    agg = aggregate(per_frame)
    agg.to_csv(out / f"calib_perturb_sweep_{name}.csv", index=False, lineterminator="\n")
    print(agg[["axis", "vehicle_axis", "level", "unit", "pixel_shift_px", "fov_ratio_pct", "inbox_pct_all", "inbox_pct_near_0_15m",
               "inbox_pct_mid_15_30m", "inbox_pct_far_30m_plus", "edge_score_mean", "edge_margin_mean"]].to_string(index=False))
    plot_sweep(agg, name, out / "figures")


def plot_sweep(agg: pd.DataFrame, name: str, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    base = agg[agg["axis"] == "none"].iloc[0]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, axis in zip(axs.ravel(), AXES):
        g = agg[agg["axis"] == axis]
        x = [0.0] + list(g["level"])
        for col, label, style in [("inbox_pct_near_0_15m", "gần < 15 m", "o-"), ("inbox_pct_mid_15_30m", "15–30 m", "s-"),
                                  ("inbox_pct_far_30m_plus", "xa ≥ 30 m", "^-"), ("inbox_pct_all", "mọi vật", "k--")]:
            ax.plot(x, [base[col]] + list(g[col]), style, label=label)
        ax.set_title(f"tham số {axis} = {g['vehicle_axis'].iloc[0]} của xe")
        ax.set_xlabel(f"mức lệch ({g['unit'].iloc[0]})")
        ax.grid(alpha=0.3)
    for ax in axs[:, 0]:
        ax.set_ylabel("% điểm của vật còn nằm trong 2D box")
    axs[0, 0].legend()
    fig.suptitle(f"{name}: % điểm của vật còn nằm trong 2D box khi làm lệch calibration (mỗi lần chỉ lệch 1 trục)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"sweep_inbox_by_distance_{name}.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------------------- detect
def window_detect(clean: np.ndarray, test: np.ndarray, window: int, n_boot: int) -> tuple[float, float]:
    """(ngưỡng, tỉ lệ phát hiện %) của bộ giám sát nhìn `window` frame.

    clean, test: score từng frame (cùng thứ tự frame) khi calib đúng và ở điều kiện cần kiểm tra.
    Ngưỡng = phân vị 1% của trung bình cửa sổ khi calib đúng. Phát hiện = trung bình cửa sổ < ngưỡng."""
    rng = np.random.default_rng(SEED)
    boot = rng.integers(0, len(clean), size=(n_boot, window))
    thr = float(np.percentile(np.nanmean(clean[boot], axis=1), 1))
    return thr, 100 * float((np.nanmean(test[boot], axis=1) < thr).mean())


def cmd_detect(args) -> None:
    """So sánh 2 score phát hiện drift trên cùng dữ liệu: in-box ratio (cần label) và edge-margin (không cần label).

    Mô phỏng một bộ giám sát nhìn cửa sổ `--window` frame: thống kê = trung bình score trong cửa sổ.
    Ngưỡng = phân vị 1% của thống kê khi calib ĐÚNG (bootstrap, báo động giả ~1%).
    Tỉ lệ phát hiện = phần trăm cửa sổ bootstrap có thống kê thấp hơn ngưỡng khi calib bị lệch."""
    out = Path(args.out_dir)
    rows = []
    for path in args.per_frame:
        df = pd.read_csv(path)
        ds = df["dataset"].iloc[0]
        vehicle = dict(zip(df["axis"], df["vehicle_axis"]))
        df["inbox_ratio"] = df["inbox_in_all"] / df["inbox_total_all"].replace(0, np.nan)
        for score in ("inbox_ratio", "edge_margin"):
            wide = df.pivot(index="frame_id", columns=["axis", "level"], values=score)
            wide = wide[wide[("none", 0.0)].notna()]          # chỉ dùng frame đo được score khi calib đúng
            clean = wide[("none", 0.0)].to_numpy()
            for (axis, level) in wide.columns:
                thr, rate = window_detect(clean, wide[(axis, level)].to_numpy(), args.window, args.n_boot)
                rows.append({"dataset": ds, "score": score, "axis": axis, "vehicle_axis": vehicle[axis], "level": level,
                             "window_frames": args.window,
                             "n_frames_used": len(wide), "threshold": thr, "score_mean": float(np.nanmean(wide[(axis, level)])),
                             "detect_rate_pct": rate})
    res = pd.DataFrame(rows).round(4)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "drift_detection.csv", index=False, lineterminator="\n")
    table = res.pivot_table(index=["dataset", "axis", "vehicle_axis", "level"], columns="score", values="detect_rate_pct", sort=False)
    print(table.to_string())
    plot_detect(res, out / "figures")


def plot_detect(res: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    datasets = list(res["dataset"].unique())
    for what, ylabel, fname in [("score_mean", "score trung bình", "score_vs_perturb"),
                                ("detect_rate_pct", "tỉ lệ phát hiện drift (%)", "detect_rate_vs_perturb")]:
        fig, axs = plt.subplots(len(datasets), 2, figsize=(13, 4.2 * len(datasets)), squeeze=False)
        for r, ds in enumerate(datasets):
            for c, unit_axes in enumerate([("yaw", "pitch", "roll"), ("tx", "ty", "tz")]):
                ax = axs[r, c]
                for score, color in [("inbox_ratio", "tab:blue"), ("edge_margin", "tab:red")]:
                    d = res[(res["dataset"] == ds) & (res["score"] == score)]
                    base = d[d["axis"] == "none"].iloc[0]
                    scale = 1.0 if what == "detect_rate_pct" else 1.0 / base["score_mean"]   # chuẩn hoá về 1 khi calib đúng
                    for axis, style in zip(unit_axes, ["o-", "s--", "^:"]):
                        g = d[d["axis"] == axis]
                        ax.plot([0.0] + list(g["level"]), [base[what] * scale] + list(g[what] * scale), style, color=color,
                                label=f"{score} | {axis} ({g['vehicle_axis'].iloc[0]} của xe)")
                    if what == "score_mean":
                        ax.axhline(base["threshold"] * scale, color=color, lw=1, alpha=0.6)
                ax.set_title(f"{ds}: {'xoay' if c == 0 else 'tịnh tiến'}")
                ax.set_xlabel("mức lệch (độ)" if c == 0 else "mức lệch (m)")
                ax.set_ylabel(ylabel + (" (chuẩn hoá, calib đúng = 1)" if what == "score_mean" else ""))
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8)
        if what == "score_mean":
            fig.suptitle("Score theo mức lệch. Đường ngang mảnh = ngưỡng phát hiện (phân vị 1% khi calib đúng)")
        else:
            fig.suptitle("Tỉ lệ phát hiện drift của 2 score, cửa sổ nhiều frame, báo động giả ~1%")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{fname}.png", dpi=110)
        plt.close(fig)


# ------------------------------------------------------------------------------- timesync (nuScenes)
def cmd_timesync(args) -> None:
    """Chỉ nuScenes: bỏ bù chuyển động xe giữa lúc LiDAR quét và lúc camera chụp (calibration vẫn ĐÚNG),
    rồi xem 2 score có báo "drift" hay không. Dùng để phân biệt lỗi Time với lỗi Geometry."""
    if dataset_type(args.data_root) != "nuscenes":
        raise SystemExit("Lệnh timesync chỉ dùng cho nuScenes (cần timestamp và ego pose).")
    frames = list_frames(args.data_root)[:: args.every]
    rows = []
    for i, fid in enumerate(frames):
        fr_ok = load_frame_with_ring(args.data_root, fid, use_ego_motion=True)
        fr_bad = load_frame_with_ring(args.data_root, fid, use_ego_motion=False)
        objs = object_point_sets(fr_ok)                       # tập điểm của vật xác định bằng phép chiếu đúng
        ew, em = depth_edge_weights(fr_ok["points"], fr_ok["ring"]), image_edge_map(fr_ok["image"])
        row = {"frame_id": fid, "scene": fid.rsplit("_", 1)[0],
               "dt_cam_minus_lidar_ms": (fr_ok["timestamp_camera_us"] - fr_ok["timestamp_lidar_us"]) / 1000.0,
               "pixel_shift_px": mean_pixel_shift(fr_ok, fr_bad["calib"]), "n_objects": len(objs)}
        for tag, calib in (("ego_comp", fr_ok["calib"]), ("no_ego_comp", fr_bad["calib"])):
            cnt = inbox_counts(fr_ok, calib, objs)
            tot = sum(b for _, b in cnt)
            row[f"inbox_pct_{tag}"] = 100 * sum(a for a, _ in cnt) / tot if tot else np.nan
            if tag == "no_ego_comp":                          # tách theo khoảng cách để so với lỗi xoay
                for name, _, _ in DIST_BINS:
                    sel = [c for c, o in zip(cnt, objs) if o.dist_bin == name]
                    row[f"noego_in_{name}"], row[f"noego_total_{name}"] = sum(a for a, _ in sel), sum(b for _, b in sel)
            row[f"edge_score_{tag}"], row[f"edge_margin_{tag}"] = edge_margin(fr_ok, calib, ew, em)
        rows.append(row)
        print(f"\rtimesync: {i + 1}/{len(frames)} frame", end="", flush=True)
    print()
    df = pd.DataFrame(rows).round(4)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "timesync_nuscenes.csv", index=False, lineterminator="\n")
    print(df.groupby("scene")[["dt_cam_minus_lidar_ms", "pixel_shift_px", "inbox_pct_ego_comp", "inbox_pct_no_ego_comp",
                               "edge_margin_ego_comp", "edge_margin_no_ego_comp"]].mean().round(3).to_string())
    # Cùng bộ giám sát như lệnh detect: nếu chỉ bỏ bù chuyển động (calib vẫn đúng) thì có bị báo "drift" không?
    alarm = []
    for score, col in (("inbox_ratio", "inbox_pct"), ("edge_margin", "edge_margin")):
        ok = df[f"{col}_ego_comp"].notna()
        thr, rate = window_detect(df.loc[ok, f"{col}_ego_comp"].to_numpy(), df.loc[ok, f"{col}_no_ego_comp"].to_numpy(),
                                  args.window, args.n_boot)
        alarm.append({"score": score, "window_frames": args.window, "threshold": thr,
                      "mean_ego_comp": df.loc[ok, f"{col}_ego_comp"].mean(), "mean_no_ego_comp": df.loc[ok, f"{col}_no_ego_comp"].mean(),
                      "false_drift_alarm_pct": rate})
    alarm = pd.DataFrame(alarm).round(4)
    for name, _, _ in DIST_BINS:                              # % điểm còn trong box khi bỏ bù, theo khoảng cách
        alarm[f"inbox_pct_no_ego_comp_{name}"] = round(100 * df[f"noego_in_{name}"].sum() / df[f"noego_total_{name}"].sum(), 2)
    alarm.to_csv(out / "timesync_false_alarm.csv", index=False, lineterminator="\n")
    print(alarm.to_string(index=False))
    worst = df.sort_values("pixel_shift_px", ascending=False).head(3)
    print("3 frame lệch nhiều nhất khi bỏ bù chuyển động:\n", worst[["frame_id", "dt_cam_minus_lidar_ms", "pixel_shift_px",
                                                                     "inbox_pct_ego_comp", "inbox_pct_no_ego_comp"]].to_string(index=False))


# ------------------------------------------------------------------------------- latency
def cmd_latency(args) -> None:
    """Đo thời gian tính mỗi score trên 1 frame: bỏ lần chạy đầu, lặp `--repeats` lần, báo p50/p95."""
    fr = load_frame_with_ring(args.data_root, args.frame or list_frames(args.data_root)[0])
    objs = object_point_sets(fr)
    ew, em = depth_edge_weights(fr["points"], fr["ring"]), image_edge_map(fr["image"])
    steps = {
        "project_all_points": lambda: project_velo_to_image(fr["points"], fr["calib"], fr["image"].shape),
        "inbox_ratio (cần label)": lambda: inbox_counts(fr, fr["calib"], objs),
        "edge: depth_edge_weights": lambda: depth_edge_weights(fr["points"], fr["ring"]),
        "edge: image_edge_map (Canny + distance transform)": lambda: image_edge_map(fr["image"]),
        "edge: 1 lần chấm score": lambda: edge_alignment_score(fr, fr["calib"], ew, em),
        "edge_margin đầy đủ (7 lần chấm)": lambda: edge_margin(fr, fr["calib"], ew, em),
    }
    rows = []
    for name, fn in steps.items():
        fn()                                                  # lần đầu: khởi tạo, không tính
        t = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            fn()
            t.append((time.perf_counter() - t0) * 1000)
        rows.append({"dataset": Path(args.data_root).name, "frame_id": fr["frame_id"], "n_points": len(fr["points"]),
                     "step": name, "repeats": args.repeats, "p50_ms": np.percentile(t, 50), "p95_ms": np.percentile(t, 95),
                     "cpu": platform.processor() or platform.machine(), "python": platform.python_version()})
    df = pd.DataFrame(rows).round(3)
    out = Path(args.out_dir) / f"latency_{Path(args.data_root).name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, lineterminator="\n")
    print(df[["step", "p50_ms", "p95_ms"]].to_string(index=False), f"\n-> {out}")


# ------------------------------------------------------------------------------- all
def cmd_all(args) -> None:
    """Tái tạo toàn bộ số liệu và hình trong báo cáo (trừ ảnh failure, xem src/failure_cases.py)."""
    ns = argparse.Namespace
    for root, frame in [("data/kitti_mini", "000011"), ("data/nuscenes_mini_subset", "scene-0103_010")]:
        cmd_overlay(ns(data_root=root, frame=frame, out_dir=f"{args.out_dir}/figures", roll_deg=0.0, pitch_deg=0.0,
                       yaw_deg=0.0, tx=0.0, ty=0.0, tz=0.0))
    cmd_overlay(ns(data_root="data/kitti_mini", frame="000011", out_dir=f"{args.out_dir}/figures", roll_deg=0.0,
                   pitch_deg=0.0, yaw_deg=1.0, tx=0.0, ty=0.0, tz=0.0))
    for root in ("data/kitti_mini", "data/nuscenes_mini_subset"):
        cmd_sweep(ns(data_root=root, every=1, out_dir=args.out_dir))
        cmd_latency(ns(data_root=root, frame=None, repeats=30, out_dir=args.out_dir))
    cmd_detect(ns(per_frame=[f"{args.out_dir}/calib_perturb_per_frame_kitti_mini.csv",
                             f"{args.out_dir}/calib_perturb_per_frame_nuscenes_mini_subset.csv"],
                  window=10, n_boot=2000, out_dir=args.out_dir))
    cmd_timesync(ns(data_root="data/nuscenes_mini_subset", every=1, window=10, n_boot=2000, out_dir=args.out_dir))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")   # để in tiếng Việt được trên console Windows
    ap = argparse.ArgumentParser(prog="python -m src.calib_qa", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, frame=False):
        p.add_argument("--data-root", default="data/kitti_mini", help="thư mục KITTI hoặc nuScenes")
        p.add_argument("--out-dir", default="results", help="nơi ghi CSV (hình nằm trong <out-dir>/figures)")
        if frame:
            p.add_argument("--frame", default=None, help="frame id, ví dụ 000011 hoặc scene-0103_010")

    p = sub.add_parser("overlay", help="ảnh overlay theo 3 khoảng cách, có thể kèm lệch calibration")
    p.add_argument("--data-root", default="data/kitti_mini")
    p.add_argument("--frame", default="000011")
    p.add_argument("--out-dir", default="results/figures")
    for flag in ("--roll-deg", "--pitch-deg", "--yaw-deg", "--tx", "--ty", "--tz"):
        p.add_argument(flag, type=float, default=0.0, help="mức lệch giả lập (độ hoặc mét, trong LiDAR frame)")
    p.set_defaults(fn=cmd_overlay)

    p = sub.add_parser("sweep", help="quét mức lệch trên 6 trục, đo in-box ratio theo khoảng cách + edge score")
    common(p)
    p.add_argument("--every", type=int, default=1, help="lấy 1 frame trong mỗi N frame (mặc định dùng hết)")
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("detect", help="so sánh khả năng phát hiện drift của 2 score, tìm ngưỡng")
    p.add_argument("--per-frame", nargs="+", required=True, help="file calib_perturb_per_frame_*.csv do lệnh sweep tạo")
    p.add_argument("--window", type=int, default=10, help="số frame trong một cửa sổ giám sát")
    p.add_argument("--n-boot", type=int, default=2000, help="số cửa sổ bootstrap")
    p.add_argument("--out-dir", default="results")
    p.set_defaults(fn=cmd_detect)

    p = sub.add_parser("timesync", help="nuScenes: bỏ bù chuyển động xe, xem score có báo drift giả không")
    common(p)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--window", type=int, default=10, help="số frame trong một cửa sổ giám sát")
    p.add_argument("--n-boot", type=int, default=2000, help="số cửa sổ bootstrap")
    p.set_defaults(fn=cmd_timesync, data_root="data/nuscenes_mini_subset")

    p = sub.add_parser("latency", help="đo thời gian tính score: p50/p95, bỏ lần chạy đầu")
    common(p, frame=True)
    p.add_argument("--repeats", type=int, default=30)
    p.set_defaults(fn=cmd_latency)

    p = sub.add_parser("all", help="chạy lại toàn bộ thí nghiệm trong báo cáo")
    p.add_argument("--out-dir", default="results")
    p.set_defaults(fn=cmd_all)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
