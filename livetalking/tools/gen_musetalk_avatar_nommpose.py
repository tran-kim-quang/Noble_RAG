"""Generate MuseTalk avatar assets from a video without mmpose dependency.

Outputs avatar package under:
  data/avatars/<avatar_id>/
    - full_imgs/*.png
    - mask/*.png
    - coords.pkl
    - mask_coords.pkl
    - latents.pt
    - avator_info.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import torch
import numpy as np
from tqdm import tqdm

# Ensure repo root is importable when script is executed directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musetalk.utils.blending import get_crop_box
from musetalk.utils.face_detection import FaceAlignment, LandmarksType
from musetalk.utils.utils import load_all_model

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _detect_face_bbox(
    detector: FaceAlignment,
    frame_bgr,
) -> Optional[Tuple[int, int, int, int]]:
    # detector expects ndarray batch [N, H, W, C]
    batch = np.asarray([frame_bgr])
    boxes = detector.get_detections_for_batch(batch)
    if not boxes:
        return None
    box = boxes[0]
    if box is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in box]
    if x2 <= x1 or y2 <= y1:
        return None
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(1, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _extract_frames(video_path: Path, max_frames: int) -> List:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    frames = []
    try:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError("no frame extracted from video")
    return frames


def _build_mask_and_crop_box(
    frame_shape: Tuple[int, int, int],
    bbox: Tuple[int, int, int, int],
    upper_boundary_ratio: float = 0.55,
    expand: float = 1.5,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop_box, _ = get_crop_box([x1, y1, x2, y2], expand)
    x_s, y_s, x_e, y_e = crop_box

    x_s = max(0, min(x_s, w - 1))
    y_s = max(0, min(y_s, h - 1))
    x_e = max(x_s + 1, min(x_e, w))
    y_e = max(y_s + 1, min(y_e, h))
    crop_box = (x_s, y_s, x_e, y_e)

    crop_w = x_e - x_s
    crop_h = y_e - y_s

    fx1 = max(0, min(crop_w, x1 - x_s))
    fx2 = max(0, min(crop_w, x2 - x_s))
    fy1 = max(0, min(crop_h, y1 - y_s))
    fy2 = max(0, min(crop_h, y2 - y_s))

    top_boundary = fy1 + int(max(0, fy2 - fy1) * upper_boundary_ratio)
    top_boundary = max(fy1, min(top_boundary, fy2))

    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    if fx2 > fx1 and fy2 > fy1:
        mask[top_boundary:fy2, fx1:fx2] = 255

    blur_kernel_size = int(0.1 * crop_w // 2 * 2) + 1
    mask = cv2.GaussianBlur(mask, (blur_kernel_size, blur_kernel_size), 0)
    return mask, crop_box


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="input video path")
    parser.add_argument("--avatar_id", required=True, help="target avatar id")
    parser.add_argument("--max_frames", type=int, default=600, help="max frames to process")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id when CUDA is available")
    parser.add_argument("--parsing_mode", type=str, default="jaw", help="face parsing mode")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    video_path = (root / args.video).resolve() if not os.path.isabs(args.video) else Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    avatar_root = root / "data" / "avatars" / args.avatar_id
    full_imgs_dir = avatar_root / "full_imgs"
    mask_dir = avatar_root / "mask"
    _ensure_dir(full_imgs_dir)
    _ensure_dir(mask_dir)

    # Clean old generated png files for deterministic output
    for p in full_imgs_dir.glob("*.png"):
        p.unlink(missing_ok=True)
    for p in mask_dir.glob("*.png"):
        p.unlink(missing_ok=True)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    vae, _unet, _pe = load_all_model(device=device)
    vae.vae = vae.vae.half().to(device)

    detector_device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = FaceAlignment(LandmarksType._2D, flip_input=False, device=detector_device)

    frames = _extract_frames(video_path, max_frames=max(1, int(args.max_frames)))

    coord_list: List[Tuple[int, int, int, int]] = []
    mask_coords_list: List[Tuple[int, int, int, int]] = []
    latents_list: List[torch.Tensor] = []
    kept = 0
    dropped = 0

    for frame in tqdm(frames, desc="build-avatar"):
        bbox = _detect_face_bbox(detector, frame)
        if bbox is None:
            dropped += 1
            continue

        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            dropped += 1
            continue

        resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latent = vae.get_latents_for_unet(resized)
        mask, crop_box = _build_mask_and_crop_box(frame.shape, (x1, y1, x2, y2))

        out_name = f"{kept:08d}.png"
        cv2.imwrite(str(full_imgs_dir / out_name), frame)
        cv2.imwrite(str(mask_dir / out_name), mask)

        coord_list.append((x1, y1, x2, y2))
        mask_coords_list.append(crop_box)
        latents_list.append(latent)
        kept += 1

    if kept == 0:
        raise RuntimeError("no usable frame with detectable face")

    with open(avatar_root / "coords.pkl", "wb") as f:
        pickle.dump(coord_list, f)
    with open(avatar_root / "mask_coords.pkl", "wb") as f:
        pickle.dump(mask_coords_list, f)
    torch.save(latents_list, avatar_root / "latents.pt")

    info = {
        "avatar_id": args.avatar_id,
        "video_path": str(video_path),
        "kept_frames": kept,
        "dropped_frames": dropped,
        "max_frames": int(args.max_frames),
        "generator": "gen_musetalk_avatar_nommpose",
    }
    with open(avatar_root / "avator_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"Avatar generated at: {avatar_root}")
    print(f"kept_frames={kept} dropped_frames={dropped}")


if __name__ == "__main__":
    main()
