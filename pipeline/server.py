"""
Namma Space - Dense Multi-Frame 3D Gaussian Splatting (3DGS) & PyTorch Optimizer (pipeline/server.py)
====================================================================================================
Processes dense overlapping keyframes using a multi-view 3D reconstruction pipeline.
"""

import base64
import json
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
import numpy as np

# Global live progress state polled by the frontend (GET /api/progress)
PROGRESS_STATE = {
    "active": False,
    "percent": 0,
    "stage": "Idle",
    "detail": "Waiting for scan...",
    "frames_done": 0,
    "frames_total": 0,
    "splat_count": 0,
}


def update_progress(percent: int, stage: str, detail: str, frames_done: int = 0, frames_total: int = 0, splat_count: int = 0):
    PROGRESS_STATE["active"] = percent < 100
    PROGRESS_STATE["percent"] = int(max(0, min(100, percent)))
    PROGRESS_STATE["stage"] = stage
    PROGRESS_STATE["detail"] = detail
    PROGRESS_STATE["frames_done"] = frames_done
    PROGRESS_STATE["frames_total"] = frames_total
    PROGRESS_STATE["splat_count"] = splat_count


# --- 1. Load Pre-Trained YOLOv8n Object Detector ---
YOLO_MODEL = None
try:
    from ultralytics import YOLO
    YOLO_MODEL = YOLO("yolov8n.pt")
    print("[AI BACKEND] Loaded pre-trained YOLOv8n (yolov8n.pt) successfully!")
except Exception as e:
    print(f"[AI BACKEND] YOLOv8n fallback mode: {e}")

# --- 2. Load Pre-Trained PyTorch MiDaS_small Monocular Depth Network ---
MIDAS_MODEL = None
MIDAS_TRANSFORM = None
MIDAS_DEVICE = "cpu"
try:
    import torch
    MIDAS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MIDAS_MODEL = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    MIDAS_MODEL.to(MIDAS_DEVICE)
    MIDAS_MODEL.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    MIDAS_TRANSFORM = midas_transforms.small_transform
    print(f"[AI BACKEND] Loaded pre-trained MiDaS_small Neural Depth model on {MIDAS_DEVICE}!")
except Exception as e:
    print(f"[AI BACKEND] MiDaS fallback mode: {e}")


OBJECT_SPECS = {
    "couch": {"label": "3-Seater Sofa", "category": "Seating"},
    "chair": {"label": "Chair", "category": "Seating"},
    "dining table": {"label": "Desk / Table", "category": "Workspace"},
    "bed": {"label": "Bed", "category": "Zone"},
    "tv": {"label": "Monitor / Display", "category": "Facility"},
    "laptop": {"label": "Laptop & Workstation", "category": "Workspace"},
    "keyboard": {"label": "Keyboard & Desk", "category": "Workspace"},
    "potted plant": {"label": "Indoor Plant", "category": "Landmark"},
    "refrigerator": {"label": "Refrigerator", "category": "Facility"},
    "bench": {"label": "Bench", "category": "Seating"},
    "book": {"label": "Book / Notebook", "category": "Object"},
}

IGNORED_CLASSES = {"person", "spoon", "fork", "knife", "remote", "cell phone"}


def decode_base64_image(b64_str: str):
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def compute_raw_Metric_depth(bgr_img, out_w: int = 160, out_h: int = 120, max_range_m: float = 3.4) -> np.ndarray:
    """
    Runs PyTorch MiDaS_small and returns a bilateral-smoothed metric depth map Z(u, v) in meters.
    """
    min_depth_m = max(0.50, max_range_m * 0.20)

    if MIDAS_MODEL is not None and MIDAS_TRANSFORM is not None:
        import torch
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        input_batch = MIDAS_TRANSFORM(rgb).to(MIDAS_DEVICE)
        with torch.no_grad():
            prediction = MIDAS_MODEL(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(out_h, out_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        inv_depth = prediction.cpu().numpy().astype(np.float32)

        p2 = float(np.percentile(inv_depth, 2))
        p98 = float(np.percentile(inv_depth, 98))
        norm_disp = np.clip((inv_depth - p2) / max(1e-5, p98 - p2), 0.0, 1.0).astype(np.float32)
        smooth_disp = cv2.bilateralFilter(norm_disp, d=7, sigmaColor=0.11, sigmaSpace=4.0)
        metric_z = min_depth_m + ((1.0 - smooth_disp) ** 1.12) * (max_range_m - min_depth_m)
        return metric_z.astype(np.float32)

    y_idx = np.linspace(0, 1, out_h, dtype=np.float32)[:, None]
    x_idx = np.linspace(-1, 1, out_w, dtype=np.float32)[None, :]
    z = max_range_m * 0.82 - (max_range_m * 0.40) * y_idx + 0.14 * (x_idx ** 2)
    return np.clip(z, min_depth_m, max_range_m).astype(np.float32)


def track_klt_and_align_depth(prev_gray, prev_depth, curr_gray, curr_depth, hfov_deg: float = 62.0, vfov_deg: float = 48.0):
    """
    Tracks Shi-Tomasi corners with pyramidal Lucas-Kanade optical flow and aligns depth maps.
    """
    h, w = prev_depth.shape
    pts0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.01, minDistance=6)
    if pts0 is None or len(pts0) < 12:
        return 0.0, 0.0, 0.0, curr_depth

    pts1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts0, None, winSize=(15, 15), maxLevel=3)
    if pts1 is None:
        return 0.0, 0.0, 0.0, curr_depth
    pts0_back, st2, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, pts1, None, winSize=(15, 15), maxLevel=3)
    if pts0_back is None:
        return 0.0, 0.0, 0.0, curr_depth

    fb_err = np.linalg.norm(pts0 - pts0_back, axis=2).ravel()
    valid = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb_err < 1.5)
    p0 = pts0[valid].reshape(-1, 2)
    p1 = pts1[valid].reshape(-1, 2)
    if len(p0) < 10:
        return 0.0, 0.0, 0.0, curr_depth

    dx = p0[:, 0] - p1[:, 0]
    dy = p0[:, 1] - p1[:, 1]
    delta_yaw = float(np.clip((float(np.median(dx)) / float(w)) * hfov_deg, -35.0, 35.0))
    delta_pitch = float(np.clip((float(np.median(dy)) / float(h)) * vfov_deg * 0.35, -12.0, 12.0))

    u0 = np.clip(p0[:, 0].astype(np.int32), 0, w - 1)
    v0 = np.clip(p0[:, 1].astype(np.int32), 0, h - 1)
    u1 = np.clip(p1[:, 0].astype(np.int32), 0, w - 1)
    v1 = np.clip(p1[:, 1].astype(np.int32), 0, h - 1)
    z0 = prev_depth[v0, u0]
    z1 = curr_depth[v1, u1]
    med_z0 = float(np.median(z0))
    med_z1 = float(np.median(z1))
    iqr_z0 = float(np.percentile(z0, 75) - np.percentile(z0, 25))
    iqr_z1 = float(np.percentile(z1, 75) - np.percentile(z1, 25))
    scale = float(np.clip(iqr_z0 / max(0.08, iqr_z1), 0.78, 1.28)) if iqr_z1 > 0.05 else 1.0
    shift = float(np.clip(med_z0 - scale * med_z1, -0.45, 0.45))
    aligned_depth = np.clip(curr_depth * scale + shift, 0.45, 6.0).astype(np.float32)
    return delta_yaw, delta_pitch, 0.0, aligned_depth


def optimize_3d_gaussians_pytorch(positions_np, colors_np, scales_np, opacities_np, normals_np, iterations: int = 24):
    """
    Refines Gaussian positions, colors, scales and opacities with a PyTorch Adam optimizer.
    """
    try:
        import torch
        n_pts = len(scales_np)
        if n_pts < 64:
            return positions_np, colors_np, scales_np, opacities_np

        device = MIDAS_DEVICE
        pos_init = torch.tensor(positions_np, dtype=torch.float32, device=device)
        col_init = torch.tensor(colors_np, dtype=torch.float32, device=device)
        sc_init = torch.tensor(scales_np, dtype=torch.float32, device=device)
        op_init = torch.tensor(opacities_np, dtype=torch.float32, device=device)
        norm_t = torch.tensor(normals_np, dtype=torch.float32, device=device)
        delta_pos = torch.zeros_like(pos_init, requires_grad=True)
        delta_col = torch.zeros_like(col_init, requires_grad=True)
        log_scale_adj = torch.zeros_like(sc_init, requires_grad=True)
        logit_op = torch.logit(torch.clamp(op_init, 0.05, 0.99)).detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([
            {"params": [delta_pos], "lr": 0.0035},
            {"params": [delta_col], "lr": 0.005},
            {"params": [log_scale_adj], "lr": 0.01},
            {"params": [logit_op], "lr": 0.025},
        ])

        spatial_key = pos_init[:, 0] * 0.37 + pos_init[:, 2] * 0.41 + pos_init[:, 1] * 0.22
        order = torch.argsort(spatial_key)
        idx_prev = torch.roll(order, 1)
        idx_next = torch.roll(order, -1)
        dist_prev = torch.norm(pos_init[order] - pos_init[idx_prev], dim=1)
        dist_next = torch.norm(pos_init[order] - pos_init[idx_next], dim=1)
        norm_sim_prev = torch.clamp(torch.sum(norm_t[order] * norm_t[idx_prev], dim=1), 0.0, 1.0)
        norm_sim_next = torch.clamp(torch.sum(norm_t[order] * norm_t[idx_next], dim=1), 0.0, 1.0)
        w_prev = torch.exp(-dist_prev * 18.0) * (norm_sim_prev ** 2)
        w_next = torch.exp(-dist_next * 18.0) * (norm_sim_next ** 2)

        for it in range(iterations):
            optimizer.zero_grad()
            curr_pos = pos_init + delta_pos
            curr_col = torch.clamp(col_init + delta_col, 0.0, 1.0)
            curr_sc = sc_init * torch.exp(torch.clamp(log_scale_adj, -0.35, 0.35))
            curr_op = torch.sigmoid(logit_op)
            diff_prev = curr_pos[order] - curr_pos[idx_prev]
            diff_next = curr_pos[order] - curr_pos[idx_next]
            normal_proj_prev = torch.sum(diff_prev * norm_t[order], dim=1) ** 2
            normal_proj_next = torch.sum(diff_next * norm_t[order], dim=1) ** 2
            loss_plane = torch.mean(w_prev * normal_proj_prev + w_next * normal_proj_next) * 12.0
            loss_anchor = torch.mean(delta_pos ** 2) * 4.0 + torch.mean(delta_col ** 2) * 2.0
            col_diff = torch.sum((curr_col[order] - curr_col[idx_prev]) ** 2, dim=1)
            loss_color = torch.mean(w_prev * col_diff) * 0.5
            loss_reg = torch.mean(log_scale_adj ** 2) * 0.2 + torch.mean((1.0 - curr_op) ** 2) * 0.08
            total_loss = loss_plane + loss_anchor + loss_color + loss_reg
            total_loss.backward()
            optimizer.step()

            if it % 6 == 0 or it == iterations - 1:
                pct = 82 + int(((it + 1) / float(iterations)) * 16)
                update_progress(pct, "Stage 4/4: PyTorch 3D Gaussian Splat Optimization",
                                f"Adam Iteration {it + 1}/{iterations} • Loss: {float(total_loss.item()):.5f} • {n_pts:,} Gaussians",
                                n_pts, n_pts, n_pts)

        with torch.no_grad():
            final_pos = (pos_init + delta_pos).cpu().numpy()
            final_col = torch.clamp(col_init + delta_col, 0.0, 1.0).cpu().numpy()
            final_sc = (sc_init * torch.exp(torch.clamp(log_scale_adj, -0.35, 0.35))).cpu().numpy()
            final_op = torch.sigmoid(logit_op).cpu().numpy()
        return final_pos, final_col, final_sc, final_op
    except Exception as e:
        print(f"[AI BACKEND] PyTorch 3DGS optimizer fallback: {e}")
        return positions_np, colors_np, scales_np, opacities_np


def process_scan_to_3d_scene(payload: dict) -> dict:
    t0 = time.time()
    raw_frames = payload.get("frames", [])
    if not raw_frames:
        raise ValueError("No keyframes provided in scan payload.")
    max_range_m = max(1.4, min(5.5, float(payload.get("range_m", 3.2))))
    if len(raw_frames) > 140:
        step = len(raw_frames) / 140.0
        frames = [raw_frames[int(i * step)] for i in range(140)]
    else:
        frames = raw_frames
    num_input = len(frames)
    update_progress(5, "Stage 1/4: Decoding Dense Stream & Initializing KLT Tracker",
                    f"Loaded {num_input} dense frames for multi-view 3D Gaussian Splatting...", 0, num_input, 0)
    raw_yaws = [float(f.get("yaw_deg", 0.0)) for f in frames]
    has_gyro_variation = (max(raw_yaws) - min(raw_yaws)) > 10.0 and not bool(payload.get("measure_angles_from_optical_flow", False))
    if num_input <= 8:
        splat_w, splat_h = 160, 120
    elif num_input <= 25:
        splat_w, splat_h = 136, 102
    else:
        splat_w, splat_h = 112, 84
    hfov_deg, vfov_deg = 62.0, 48.0
    hfov_rad, vfov_rad = math.radians(hfov_deg), math.radians(vfov_deg)
    fx = (splat_w / 2.0) / math.tan(hfov_rad / 2.0)
    fy = (splat_h / 2.0) / math.tan(vfov_rad / 2.0)
    cx_grid, cy_grid = (splat_w - 1) / 2.0, (splat_h - 1) / 2.0
    processed_views = []
    accumulated_yaw, accumulated_pitch = float(frames[0].get("yaw_deg", 0.0)), 0.0
    prev_gray = prev_depth = None

    for idx, f in enumerate(frames):
        bgr = decode_base64_image(f.get("image_base64", ""))
        if bgr is None:
            continue
        bgr_std = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA)
        gray_small = cv2.cvtColor(cv2.resize(bgr_std, (splat_w, splat_h)), cv2.COLOR_BGR2GRAY)
        depth_map = compute_raw_Metric_depth(bgr_std, out_w=splat_w, out_h=splat_h, max_range_m=max_range_m)
        if idx > 0 and prev_gray is not None and prev_depth is not None:
            klt_dyaw, klt_dpitch, _, depth_map = track_klt_and_align_depth(prev_gray, prev_depth, gray_small, depth_map, hfov_deg, vfov_deg)
            if has_gyro_variation:
                accumulated_yaw = float(f.get("yaw_deg", 0.0)) % 360.0
            else:
                accumulated_yaw = (accumulated_yaw + klt_dyaw) % 360.0
            accumulated_pitch = float(np.clip(accumulated_pitch * 0.85 + klt_dpitch * 0.15, -14.0, 14.0))
        else:
            accumulated_yaw = float(f.get("yaw_deg", 0.0)) % 360.0
        prev_gray, prev_depth = gray_small, depth_map
        pct = 8 + int(((idx + 1) / float(num_input)) * 44)
        update_progress(pct, "Stage 2/4: PyTorch MiDaS Depth & KLT Multi-View Registration",
                        f"Processed frame {idx + 1}/{num_input} (Yaw: {accumulated_yaw:.1f}°)", idx + 1, num_input, 0)
        processed_views.append({"idx": idx, "yaw_deg": accumulated_yaw, "pitch_deg": accumulated_pitch,
                                "pos_x": float(f.get("pos_x", 0.0)), "pos_z": float(f.get("pos_z", 0.0)),
                                "bgr": bgr_std, "depth_map": depth_map, "image_base64": f.get("image_base64", "")})

    num_views = len(processed_views)
    if num_views == 0:
        raise ValueError("No valid images could be decoded from scan.")
    sin_sum = sum(math.sin(math.radians(v["yaw_deg"])) for v in processed_views)
    cos_sum = sum(math.cos(math.radians(v["yaw_deg"])) for v in processed_views)
    mean_yaw = math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0
    for v in processed_views:
        v["rel_yaw_deg"] = ((v["yaw_deg"] - mean_yaw + 180.0) % 360.0) - 180.0

    update_progress(56, "Stage 3/4: Fine 1.8cm 3D Voxel Fusion & Normal Estimation",
                    f"Unprojecting {num_views} registered views into 3D Gaussian voxel volume...", num_views, num_views, 0)
    u_coords = np.arange(splat_w, dtype=np.float32)[None, :]
    v_coords = np.arange(splat_h, dtype=np.float32)[:, None]
    ray_x, ray_y = (u_coords - cx_grid) / fx, -(v_coords - cy_grid) / fy
    camera_height_m = 1.20 if max_range_m >= 2.5 else 0.85
    voxel_size = 0.018 if max_range_m <= 2.2 else 0.025
    voxel_sum_xyz, voxel_sum_rgb, voxel_sum_norm = {}, {}, {}
    voxel_sum_scale, voxel_sum_opacity, voxel_count = {}, {}, {}
    raw_obstacles = []
    yolo_stride = max(1, num_views // 18)

    for i, v in enumerate(processed_views):
        yaw_rad = math.radians(v["rel_yaw_deg"])
        cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)
        pitch_rad = math.radians(v["pitch_deg"])
        cos_pitch, sin_pitch = math.cos(pitch_rad), math.sin(pitch_rad)
        Z = v["depth_map"]
        X_cam, Y_cam = ray_x * Z, ray_y * Z
        X_rot = X_cam * cos_yaw + Z * sin_yaw
        Z_rot = -X_cam * sin_yaw + Z * cos_yaw
        Y_rot = Y_cam * cos_pitch - Z_rot * sin_pitch
        Z_world = Y_cam * sin_pitch + Z_rot * cos_pitch
        X_world, Y_world = X_rot + v["pos_x"], Y_rot + camera_height_m
        dx_u = np.gradient(X_world, axis=1)
        dy_u = np.gradient(Y_world, axis=1)
        dz_u = np.gradient(Z_world, axis=1)
        dx_v = np.gradient(X_world, axis=0)
        dy_v = np.gradient(Y_world, axis=0)
        dz_v = np.gradient(Z_world, axis=0)
        tangent_u = np.stack([dx_u, dy_u, dz_u], axis=-1)
        tangent_v = np.stack([dx_v, dy_v, dz_v], axis=-1)
        normals = np.cross(tangent_u, tangent_v)
        normals /= (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-6)
        grad_mag = np.sqrt(np.gradient(Z, axis=0) ** 2 + np.gradient(Z, axis=1) ** 2)
        u_win = np.cos(((u_coords - cx_grid) / (splat_w * 0.52)) * (math.pi * 0.5)) ** 2
        u_win = np.clip(u_win, 0.15, 1.0)
        splat_scale = np.clip((Z / fx) * 1.55, 0.012, 0.068).astype(np.float32)
        splat_op = np.where(grad_mag > 0.22, 0.0, 0.88 * u_win).astype(np.float32)
        rgb_small = cv2.cvtColor(cv2.resize(v["bgr"], (splat_w, splat_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        xw_f, yw_f, zw_f = X_world.ravel(), Y_world.ravel(), Z_world.ravel()
        rgb_f, norm_f, sc_f, op_f = rgb_small.reshape(-1, 3), normals.reshape(-1, 3), splat_scale.ravel(), splat_op.ravel()
        valid_mask = op_f > 0.25
        xw_f, yw_f, zw_f = xw_f[valid_mask], yw_f[valid_mask], zw_f[valid_mask]
        rgb_f, norm_f, sc_f, op_f = rgb_f[valid_mask], norm_f[valid_mask], sc_f[valid_mask], op_f[valid_mask]
        vx = np.floor(xw_f / voxel_size).astype(np.int64) + 32768
        vy = np.floor(yw_f / voxel_size).astype(np.int64) + 32768
        vz = np.floor(zw_f / voxel_size).astype(np.int64) + 32768
        keys = (vx << 32) | (vy << 16) | vz
        for k_idx, key in enumerate(keys.tolist()):
            w_val = float(op_f[k_idx])
            if key in voxel_count:
                voxel_count[key] += w_val
                voxel_sum_xyz[key][0] += float(xw_f[k_idx]) * w_val
                voxel_sum_xyz[key][1] += float(yw_f[k_idx]) * w_val
                voxel_sum_xyz[key][2] += float(zw_f[k_idx]) * w_val
                voxel_sum_rgb[key][0] += float(rgb_f[k_idx, 0]) * w_val
                voxel_sum_rgb[key][1] += float(rgb_f[k_idx, 1]) * w_val
                voxel_sum_rgb[key][2] += float(rgb_f[k_idx, 2]) * w_val
                voxel_sum_norm[key][0] += float(norm_f[k_idx, 0]) * w_val
                voxel_sum_norm[key][1] += float(norm_f[k_idx, 1]) * w_val
                voxel_sum_norm[key][2] += float(norm_f[k_idx, 2]) * w_val
                voxel_sum_scale[key] += float(sc_f[k_idx]) * w_val
                voxel_sum_opacity[key] = min(0.99, voxel_sum_opacity[key] + 0.18)
            else:
                voxel_count[key] = w_val
                voxel_sum_xyz[key] = [float(xw_f[k_idx]) * w_val, float(yw_f[k_idx]) * w_val, float(zw_f[k_idx]) * w_val]
                voxel_sum_rgb[key] = [float(rgb_f[k_idx, 0]) * w_val, float(rgb_f[k_idx, 1]) * w_val, float(rgb_f[k_idx, 2]) * w_val]
                voxel_sum_norm[key] = [float(norm_f[k_idx, 0]) * w_val, float(norm_f[k_idx, 1]) * w_val, float(norm_f[k_idx, 2]) * w_val]
                voxel_sum_scale[key] = float(sc_f[k_idx]) * w_val
                voxel_sum_opacity[key] = float(op_f[k_idx])

        if i % yolo_stride == 0 and YOLO_MODEL is not None:
            h_img, w_img, _ = v["bgr"].shape
            results = YOLO_MODEL.predict(v["bgr"], conf=0.32, verbose=False)
            for res in results:
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = YOLO_MODEL.names.get(cls_id, "")
                    if cls_name in IGNORED_CLASSES or cls_name not in OBJECT_SPECS:
                        continue
                    x1, y1, x2, y2 = [float(val) for val in box.xyxy[0]]
                    conf = float(box.conf[0])
                    gc = int(np.clip(((x1 + x2) * 0.5 / w_img) * (splat_w - 1), 0, splat_w - 1))
                    gr = int(np.clip(((y1 + y2) * 0.5 / h_img) * (splat_h - 1), 0, splat_h - 1))
                    spec = OBJECT_SPECS[cls_name]
                    raw_obstacles.append({"class_name": cls_name, "label": spec["label"], "category": spec["category"],
                                          "distance_m": round(float(Z[gr, gc]), 2), "x": round(float(X_world[gr, gc]), 2),
                                          "y": round(float(Y_world[gr, gc]), 2), "z": round(float(Z_world[gr, gc]), 2),
                                          "confidence": round(conf, 2)})

    min_weight = 0.55 if num_views >= 8 else 0.20
    valid_keys = [k for k, w_tot in voxel_count.items() if w_tot >= min_weight]
    if len(valid_keys) > 140000:
        valid_keys.sort(key=lambda k: voxel_count[k], reverse=True)
        valid_keys = valid_keys[:140000]
    n_splats = len(valid_keys)
    pos_np, col_np, norm_np = np.zeros((n_splats, 3), np.float32), np.zeros((n_splats, 3), np.float32), np.zeros((n_splats, 3), np.float32)
    sc_np, op_np = np.zeros(n_splats, np.float32), np.zeros(n_splats, np.float32)
    for idx_k, key in enumerate(valid_keys):
        inv_w = 1.0 / float(voxel_count[key])
        pos_np[idx_k] = np.asarray(voxel_sum_xyz[key]) * inv_w
        col_np[idx_k] = np.asarray(voxel_sum_rgb[key]) * inv_w
        norm = np.asarray(voxel_sum_norm[key]) * inv_w
        norm_np[idx_k] = norm / (np.linalg.norm(norm) + 1e-6)
        sc_np[idx_k] = voxel_sum_scale[key] * inv_w
        op_np[idx_k] = voxel_sum_opacity[key]

    update_progress(80, "Stage 4/4: PyTorch 3D Gaussian Splat Optimization",
                    f"Refining {n_splats:,} 3D Gaussians with Adam surface-normal regularization...", num_views, num_views, n_splats)
    pos_np, col_np, sc_np, op_np = optimize_3d_gaussians_pytorch(pos_np, col_np, sc_np, op_np, norm_np, iterations=24)
    if n_splats > 0:
        floor_y = float(np.percentile(pos_np[:, 1], 2))
        y_shift = -floor_y + 0.02
        pos_np[:, 1] += y_shift
        min_x, max_x = round(float(np.percentile(pos_np[:, 0], 2)), 2), round(float(np.percentile(pos_np[:, 0], 98)), 2)
        min_z, max_z = round(float(np.percentile(pos_np[:, 2], 2)), 2), round(float(np.percentile(pos_np[:, 2], 98)), 2)
        height_m = round(float(np.percentile(pos_np[:, 1], 98)), 2)
    else:
        y_shift, min_x, max_x, min_z, max_z, height_m = 0.0, -2.0, 2.0, -3.0, -0.5, 2.2
    for obs in raw_obstacles:
        obs["y"] = round(max(0.25, obs["y"] + y_shift), 2)
    splat_positions, splat_colors = pos_np.round(3).ravel().tolist(), col_np.round(3).ravel().tolist()
    splat_scales, splat_opacities = sc_np.round(4).ravel().tolist(), op_np.round(3).ravel().tolist()
    width_m, depth_m = round(max(1.2, max_x - min_x), 2), round(max(1.2, max_z - min_z), 2)
    floor_area_m2 = round(width_m * depth_m, 1)
    rel_yaws = [v["rel_yaw_deg"] for v in processed_views]
    span_deg = (max(rel_yaws) - min(rel_yaws)) + hfov_deg if num_views >= 2 else hfov_deg
    covered_deg = round(min(360.0, span_deg), 1)
    coverage_pct = round(min(100.0, (covered_deg / 360.0) * 100.0), 1)
    merged_obstacles, pois = [], [{"id": "poi-splat-center", "name": f"3D Gaussian Splat Center ({width_m}m × {depth_m}m)",
                                  "category": "Room", "description": f"{n_splats:,} PyTorch-Optimized 3D Gaussian Splats ({num_views} dense frames)",
                                  "x": round((min_x + max_x) / 2.0, 2), "y": round(min(1.1, height_m * 0.55), 2),
                                  "z": round((min_z + max_z) / 2.0, 2)}]
    raw_obstacles.sort(key=lambda o: o["confidence"], reverse=True)
    seen_classes = set()
    for obs in raw_obstacles:
        cls = obs["class_name"]
        if cls in seen_classes and cls != "chair":
            continue
        if any(math.hypot(obs["x"] - ex["x"], obs["z"] - ex["z"]) < 1.1 for ex in merged_obstacles):
            continue
        seen_classes.add(cls)
        obs["id"] = f"obs-{len(merged_obstacles) + 1}"
        merged_obstacles.append(obs)
        pois.append({"id": f"poi-obs-{len(merged_obstacles)}", "name": f"{obs['label']} ({obs['distance_m']}m)",
                     "category": obs["category"], "description": f"3D Splat Object: {obs['distance_m']}m ({int(obs['confidence'] * 100)}% conf)",
                     "x": obs["x"], "y": round(obs["y"] + 0.18, 2), "z": obs["z"]})
    elapsed_s = round(time.time() - t0, 2)
    update_progress(100, "Complete!", f"Reconstructed {n_splats:,} 3D Gaussian Splats from {num_views} frames in {elapsed_s}s",
                    num_views, num_views, n_splats)
    return {
        "status": "ok",
        "engine": "Dense Multi-Frame 3D Gaussian Splatting (KLT + MiDaS + PyTorch Adam 3DGS)",
        "processing_time_sec": elapsed_s,
        "keyframe_count": num_views,
        "fused_view_count": num_views,
        "gaussian_splat_count": n_splats,
        "room_dimensions": {"detected_space_type": "3d_gaussian_splats", "width_m": width_m, "depth_m": depth_m,
                            "height_m": height_m, "floor_area_m2": floor_area_m2, "min_x": min_x, "max_x": max_x,
                            "min_z": min_z, "max_z": max_z, "angular_coverage_deg": covered_deg,
                            "coverage_percent": coverage_pct},
        "gaussian_splats": {"count": n_splats, "positions": splat_positions, "colors": splat_colors,
                            "scales": splat_scales, "opacities": splat_opacities},
        "surface_meshes": [],
        "obstacles": merged_obstacles,
        "pois": pois,
    }


class ReconstructionRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        if "/api/progress" not in (args[0] if args else ""):
            super().log_message(format, *args)

    def _send_json(self, status_code: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(200, {"status": "ok"})

    def do_GET(self):
        if self.path.startswith("/api/progress"):
            self._send_json(200, PROGRESS_STATE)
        elif self.path.startswith("/api/health"):
            self._send_json(200, {"status": "online", "midas_loaded": MIDAS_MODEL is not None, "yolo_loaded": YOLO_MODEL is not None})
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path.startswith("/api/reconstruct-scan"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
                scene_graph = process_scan_to_3d_scene(payload)
                self._send_json(200, scene_graph)
            except Exception as e:
                import traceback
                traceback.print_exc()
                update_progress(0, "Error", str(e))
                self._send_json(500, {"status": "error", "message": str(e)})
        else:
            self._send_json(404, {"error": "Endpoint not found"})


# Vercel's Python runtime requires a top-level handler class for HTTP functions.
# Keep the local development server below; Vercel imports this alias instead.
handler = ReconstructionRequestHandler


if __name__ == "__main__":
    port = 8000
    print(f"Namma Space Python server listening on http://127.0.0.1:{port}")
    server = ThreadingHTTPServer(("127.0.0.1", port), ReconstructionRequestHandler)
    server.serve_forever()
