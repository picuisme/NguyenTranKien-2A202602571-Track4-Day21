"""Hàm dùng chung cho topic A (LiDAR-camera projection QA).

Gồm 3 nhóm:
1. Đọc frame kèm chỉ số ring (để tìm điểm "mép độ sâu" trên từng vòng quét LiDAR).
2. Metric CÓ dùng label: tỉ lệ điểm LiDAR của một vật rơi đúng vào 2D box của vật đó (in-box ratio).
3. Metric KHÔNG dùng label: edge-alignment score, khớp mép độ sâu LiDAR với cạnh ảnh (Canny).

Ý tưởng edge-alignment lấy từ bài báo: J. Levinson, S. Thrun, "Automatic Online Calibration of
Cameras and Lasers", RSS 2013. Code ở đây tự viết lại bằng numpy/OpenCV, không sao chép mã nguồn nào.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from starter import kitti_io, nuscenes_io
from starter.datasets import dataset_type, load_frame
from starter.kitti_io import KittiCalib, KittiObject
from starter.projection import cam_to_image, velo_to_cam

# Ba khoảng cách dùng xuyên suốt báo cáo (mét, theo z_cam của tâm vật).
DIST_BINS = [("near_0_15m", 0.0, 15.0), ("mid_15_30m", 15.0, 30.0), ("far_30m_plus", 30.0, 1e9)]
MIN_POINTS_PER_OBJECT = 10   # vật có ít điểm hơn thì tỉ lệ % không ổn định, bỏ qua
EDGE_JUMP_M = 0.5            # điểm gần hơn hàng xóm cùng ring >= 0.5 m thì coi là mép độ sâu
EDGE_SIGMA_PX = 4.0          # độ rộng "vùng hút" quanh cạnh ảnh, tính bằng pixel


# --------------------------------------------------------------------------- đọc dữ liệu
def load_frame_with_ring(data_root: str, frame_id: str, **kwargs) -> dict:
    """Như starter.datasets.load_frame nhưng thêm fr["ring"] (N,) int, hoặc None nếu không suy ra được.

    - nuScenes: cột thứ 5 của file .pcd.bin chính là ring index (0..31).
    - KITTI thật: file .bin ghi lần lượt từng vòng quét, azimuth tăng dần 0..360 độ rồi quay lại 0.
      Mỗi lần azimuth tụt mạnh (> 180 độ) là sang ring kế tiếp.
    - data/synthetic: thứ tự điểm bị xáo trộn nên không suy ra được ring -> None.
    """
    fr = load_frame(data_root, frame_id, **kwargs)
    pts = fr["points"]
    if dataset_type(data_root) == "nuscenes":
        raw = np.fromfile(nuscenes_io.lidar_path(data_root, frame_id), dtype=np.float32).reshape(-1, 5)
        fr["ring"] = raw[:, 4].astype(np.int32)
    else:
        az = np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) % 360.0
        wraps = np.r_[False, np.diff(az) < -180.0]
        ring = np.cumsum(wraps).astype(np.int32)
        fr["ring"] = ring if 16 <= ring.max() + 1 <= 80 else None
    return fr


# --------------------------------------------------------------------------- metric có label
def points_in_box3d(points_cam: np.ndarray, obj: KittiObject) -> np.ndarray:
    """Mask (N,) các điểm (camera frame) nằm trong 3D box của obj.

    Box KITTI: location là tâm ĐÁY, y hướng xuống, nên thân box nằm ở y trong [-h, 0] của hệ box."""
    h, w, l = obj.dimensions
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])      # box -> camera
    with np.errstate(invalid="ignore"):
        q = (points_cam - obj.location) @ R               # = R^T (p - t): camera -> box
        return (np.abs(q[:, 0]) <= l / 2) & (np.abs(q[:, 2]) <= w / 2) & (q[:, 1] <= 0) & (q[:, 1] >= -h)


@dataclass
class ObjectPoints:
    obj: KittiObject
    idx: np.ndarray      # chỉ số trong fr["points"] của các điểm thuộc vật (xác định bằng calib GỐC)
    dist_m: float        # z_cam của tâm vật
    dist_bin: str


def dist_bin_name(z: float) -> str:
    for name, lo, hi in DIST_BINS:
        if lo <= z < hi:
            return name
    return DIST_BINS[-1][0]


def object_point_sets(fr: dict, min_points: int = MIN_POINTS_PER_OBJECT) -> list[ObjectPoints]:
    """Với mỗi vật có label: tập điểm LiDAR nằm trong 3D box GT VÀ chiếu vào trong ảnh (dùng calib gốc).

    Tập này cố định. Khi làm lệch calibration, ta chiếu lại ĐÚNG các điểm này và đếm còn bao nhiêu
    điểm rơi vào 2D box của vật -> chỉ có một yếu tố thay đổi là calibration."""
    pc = velo_to_cam(fr["points"][:, :3], fr["calib"])
    _, _, mask = cam_to_image(pc, fr["calib"].P2, fr["image"].shape)
    out = []
    for obj in fr["labels"]:
        idx = np.flatnonzero(points_in_box3d(pc, obj) & mask)
        if len(idx) >= min_points:
            z = float(obj.location[2])
            out.append(ObjectPoints(obj, idx, z, dist_bin_name(z)))
    return out


def inbox_counts(fr: dict, calib: KittiCalib, objs: list[ObjectPoints]) -> list[tuple[int, int]]:
    """Mỗi vật: (số điểm của vật rơi trong 2D box sau khi chiếu bằng `calib`, tổng số điểm của vật)."""
    pc = velo_to_cam(fr["points"][:, :3], calib)
    h_img, w_img = fr["image"].shape[:2]
    out = []
    for o in objs:
        p = pc[o.idx]
        front = p[:, 2] > 0.1
        proj = np.hstack([p, np.ones((len(p), 1))]) @ calib.P2.T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = proj[:, :2] / proj[:, 2:3]
        x1, y1, x2, y2 = o.obj.bbox
        inside = front & (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
        inside &= (uv[:, 0] >= 0) & (uv[:, 0] < w_img) & (uv[:, 1] >= 0) & (uv[:, 1] < h_img)
        out.append((int(inside.sum()), len(o.idx)))
    return out


def mean_pixel_shift(fr: dict, calib: KittiCalib) -> float:
    """Độ dịch trung bình (pixel) của các điểm nhìn thấy được ở calib gốc khi chiếu lại bằng `calib`."""
    pts = fr["points"][:, :3]
    uv0, _, m0 = cam_to_image(velo_to_cam(pts, fr["calib"]), fr["calib"].P2, fr["image"].shape)
    pc = velo_to_cam(pts[m0], calib)
    ok = pc[:, 2] > 0.1
    proj = np.hstack([pc[ok], np.ones((int(ok.sum()), 1))]) @ calib.P2.T
    uv1 = proj[:, :2] / proj[:, 2:3]
    return float(np.linalg.norm(uv1 - uv0[ok], axis=1).mean()) if ok.any() else float("nan")


# --------------------------------------------------------------------------- metric không label
def depth_edge_weights(points: np.ndarray, ring: np.ndarray | None,
                       jump_m: float = EDGE_JUMP_M, max_az_gap_deg: float = 1.0) -> np.ndarray:
    """Trọng số (N,) >= 0: điểm nào là "mép độ sâu" của vòng quét thì > 0.

    Trên cùng một ring, điểm i là mép nếu nó GẦN hơn điểm liền trước hoặc liền sau ít nhất `jump_m`
    (tức là mép của vật ở tiền cảnh, phía sau nó là nền xa hơn). Trọng số = sqrt(độ nhảy), chặn ở 5 m.
    Hàng xóm cách quá `max_az_gap_deg` độ azimuth thì không tính (mất tia, không so sánh được)."""
    n = len(points)
    w = np.zeros(n)
    if ring is None:
        return w
    with np.errstate(invalid="ignore"):
        rng = np.linalg.norm(points[:, :3], axis=1)
    az = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    order = np.argsort(ring, kind="stable")           # giữ thứ tự thời gian trong từng ring
    r, a, g = rng[order], az[order], ring[order]
    jump = np.zeros(n)
    for shift in (1, -1):
        r_nb, a_nb, g_nb = np.roll(r, shift), np.roll(a, shift), np.roll(g, shift)
        daz = np.abs((a_nb - a + 180.0) % 360.0 - 180.0)
        valid = (g_nb == g) & (daz <= max_az_gap_deg) & np.isfinite(r) & np.isfinite(r_nb)
        jump = np.maximum(jump, np.where(valid, r_nb - r, 0.0))
    w[order] = np.where(jump >= jump_m, np.sqrt(np.minimum(jump, 5.0)), 0.0)
    return w


def image_edge_map(image_bgr: np.ndarray, sigma_px: float = EDGE_SIGMA_PX) -> np.ndarray:
    """Bản đồ (H, W) float trong [0, 1]: 1 tại cạnh Canny, giảm dần exp(-d / sigma) theo khoảng cách d tới cạnh gần nhất."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    return np.exp(-dist / sigma_px).astype(np.float32)


def edge_alignment_score(fr: dict, calib: KittiCalib, edge_w: np.ndarray, edge_map: np.ndarray) -> float:
    """Trung bình có trọng số của edge_map tại vị trí chiếu của các điểm mép độ sâu. Càng cao càng khớp.

    Trả về NaN nếu sau khi chiếu còn dưới 30 điểm mép trong ảnh (không đủ để kết luận)."""
    sel = np.flatnonzero(edge_w > 0)
    uv, _, mask = cam_to_image(velo_to_cam(fr["points"][sel, :3], calib), calib.P2, fr["image"].shape)
    if len(uv) < 30:
        return float("nan")
    w = edge_w[sel][mask]
    u, v = uv[:, 0].astype(int), uv[:, 1].astype(int)
    return float((w * edge_map[v, u]).sum() / w.sum())
