"""Tự kiểm tra 2 hàm TODO(CP2) bằng số, không nhìn ảnh. Chạy từ gốc repo:

    python -m src.test_projection
"""
from __future__ import annotations

import sys

import numpy as np

from starter.datasets import load_frame
from starter.projection import cam_to_image, project_velo_to_image, velo_to_cam


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    fr = load_frame("data/synthetic", "000000")
    calib, shape = fr["calib"], fr["image"].shape
    pts = np.array([
        [10.0, 0.0, 0.0],      # 10 m phía trước: hợp lệ
        [np.nan, 0.0, 0.0],    # điểm lỗi: phải bị loại
        [-10.0, 0.0, 0.0],     # phía sau xe: z_cam < 0, phải bị loại
        [10.0, 50.0, 0.0],     # 50 m bên trái: ngoài khung hình, phải bị loại
        [np.inf, 1.0, 1.0],    # Inf: phải bị loại
    ])
    cam = velo_to_cam(pts, calib)
    assert cam.shape == (5, 3), cam.shape
    assert abs(cam[0, 2] - 9.73) < 0.01, f"z_cam của (10,0,0) phải là 9.73, đang là {cam[0, 2]}"
    assert cam[2, 2] < 0, "điểm phía sau xe phải có z_cam âm"
    assert cam[3, 0] < -40, "điểm bên trái phải có x_cam âm (trục x camera hướng sang phải)"
    uv, depth, mask = cam_to_image(cam, calib.P2, shape)
    assert mask.tolist() == [True, False, False, False, False], mask
    assert uv.shape == (1, 2) and depth.shape == (1,)
    assert np.allclose(uv[0], [614, 175], atol=1), uv[0]
    # Nếu nhân sai chiều (X @ T thay vì X @ T.T) thì shape vẫn khớp nhưng số điểm trong ảnh sẽ khác 3 con số này.
    for root, frame, expected in [("data/synthetic", "000000", 3910), ("data/kitti_mini", "000011", 19946),
                                  ("data/nuscenes_mini_subset", "scene-0103_010", 3120)]:
        f = load_frame(root, frame)
        _, _, m = project_velo_to_image(f["points"], f["calib"], f["image"].shape)
        assert int(m.sum()) == expected, (root, frame, int(m.sum()))
    print(f"z_cam = {cam[0, 2]:.2f}, pixel = ({uv[0, 0]:.0f}, {uv[0, 1]:.0f})")
    print("✅ CP2 self-test passed")


if __name__ == "__main__":
    main()
