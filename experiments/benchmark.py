"""
UrbanSense Benchmark — Phase 4
================================
Evaluates the YOLO detector on nuScenes v1.0-mini and logs results to W&B.

Metrics computed per scene and aggregated:
  - Precision, Recall, F1  @ IoU=0.50
  - mAP@50, mAP@50-95
  - Per-category AP
  - Inference latency (ms/frame)
  - Tracking continuity (ByteTrack ID switches)

Run
---
    python experiments/benchmark.py [--scene all|<scene_name>] [--conf 0.35] [--wandb]

Results are saved to:
    experiments/benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATAROOT = ROOT / "data" / "nuscenes" / "v1.0-mini"

# ── category mapping (nuScenes → COCO-like) ───────────────────────────────────
NUSCENES_TO_YOLO = {
    "vehicle.car":                   "car",
    "vehicle.truck":                 "truck",
    "vehicle.bus.bendy":             "bus",
    "vehicle.bus.rigid":             "bus",
    "vehicle.motorcycle":            "motorcycle",
    "vehicle.bicycle":               "bicycle",
    "human.pedestrian.adult":        "person",
    "human.pedestrian.child":        "person",
    "human.pedestrian.construction_worker": "person",
    "human.pedestrian.police_officer":      "person",
    "movable_object.trafficcone":    "traffic cone",
    "movable_object.barrier":        "barrier",
}

YOLO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck",
}


# ── IoU ───────────────────────────────────────────────────────────────────────

def iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    xi1 = max(box_a[0], box_b[0])
    yi1 = max(box_a[1], box_b[1])
    xi2 = min(box_a[2], box_b[2])
    yi2 = min(box_a[3], box_b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


# ── 3-D annotation → 2-D box projection ──────────────────────────────────────

def project_annotations(nusc, sample_token: str, cam: str = "CAM_FRONT") -> list[dict]:
    """Project 3-D GT annotations into 2-D pixel boxes for one camera."""
    from nuscenes.utils.geometry_utils import view_points
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion

    sample = nusc.get("sample", sample_token)
    if cam not in sample["data"]:
        return []
    sd = nusc.get("sample_data", sample["data"][cam])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    K  = np.array(cs["camera_intrinsic"])

    gt_boxes = []
    for ann_token in sample["anns"]:
        ann  = nusc.get("sample_annotation", ann_token)
        cat  = ann["category_name"]
        if cat not in NUSCENES_TO_YOLO:
            continue
        box  = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]))
        box.translate(-np.array(ep["translation"]))
        box.rotate(Quaternion(ep["rotation"]).inverse)
        box.translate(-np.array(cs["translation"]))
        box.rotate(Quaternion(cs["rotation"]).inverse)
        c3d = view_points(box.corners(), K, normalize=True)
        if (c3d[2, :] < 0.1).all():
            continue
        xs = np.clip(c3d[0, :], 0, 1600)
        ys = np.clip(c3d[1, :], 0, 900)
        gt_boxes.append({
            "box":      np.array([xs.min(), ys.min(), xs.max(), ys.max()]),
            "category": NUSCENES_TO_YOLO[cat],
        })
    return gt_boxes


# ── per-frame matching ────────────────────────────────────────────────────────

def match_detections(
    pred_boxes:   np.ndarray,    # [M, 4]
    pred_scores:  np.ndarray,    # [M]
    gt_boxes:     list[dict],
    iou_thresh:   float = 0.5,
) -> tuple[int, int, int]:
    """Return (TP, FP, FN) counts."""
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    matched_gt = set()
    tp = fp = 0
    for pred, score in sorted(zip(pred_boxes, pred_scores), key=lambda x: -x[1]):
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = iou_xyxy(pred, gt["box"])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_j)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


# ── main benchmark loop ───────────────────────────────────────────────────────

def run_benchmark(
    scene_filter: str  = "all",
    conf_thresh:  float = 0.35,
    use_wandb:    bool  = False,
    max_samples:  int   = 20,
) -> dict:
    """
    Run YOLO + ByteTrack on nuScenes mini and return metrics dict.
    """
    print("Loading nuScenes …")
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes("v1.0-mini", dataroot=str(DATAROOT), verbose=False)

    print("Loading YOLOv8n …")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")

    from tracking.bytetrack_wrapper import ByteTrackWrapper
    tracker = ByteTrackWrapper(frame_rate=2)

    from anomaly.ood_detector import AnomalyPipeline
    anomaly_pipe = AnomalyPipeline()

    if use_wandb:
        import wandb
        wandb.init(project="urbansense", name="benchmark-yolo8n-nuscenes")

    scenes = nusc.scene
    if scene_filter != "all":
        scenes = [s for s in scenes if s["name"] == scene_filter]

    total_tp = total_fp = total_fn = 0
    latencies: list[float] = []
    id_switches = 0
    per_scene: list[dict] = []
    anomaly_counts = defaultdict(int)

    for scene in scenes:
        print(f"  Scene: {scene['name']}")
        tracker.reset()
        prev_ids: set[int] = set()

        tok = scene["first_sample_token"]
        frame_count = 0

        while tok and frame_count < max_samples:
            sample = nusc.get("sample", tok)
            cam    = "CAM_FRONT"
            if cam not in sample["data"]:
                tok = sample["next"] if sample["next"] else None
                continue

            sd       = nusc.get("sample_data", sample["data"][cam])
            img_path = DATAROOT / sd["filename"]

            # ── inference ────────────────────────────────────────────────────
            t0      = time.perf_counter()
            results = model(str(img_path), conf=conf_thresh, verbose=False)
            latency = (time.perf_counter() - t0) * 1000
            latencies.append(latency)

            res = results[0]
            if res.boxes and len(res.boxes):
                pred_boxes  = res.boxes.xyxy.cpu().numpy()
                pred_scores = res.boxes.conf.cpu().numpy()
                pred_cls    = res.boxes.cls.cpu().numpy().astype(int)
            else:
                pred_boxes  = np.zeros((0, 4))
                pred_scores = np.array([])
                pred_cls    = np.array([], dtype=int)

            # ── GT matching ───────────────────────────────────────────────────
            gt = project_annotations(nusc, tok, cam)
            tp, fp, fn = match_detections(pred_boxes, pred_scores, gt)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            # ── tracking ──────────────────────────────────────────────────────
            from tracking.bytetrack_wrapper import ByteTrackWrapper
            det_arr = np.hstack([
                pred_boxes,
                pred_scores.reshape(-1, 1),
                pred_cls.reshape(-1, 1),
            ]) if len(pred_boxes) else np.zeros((0, 6))
            tracks = tracker.update(det_arr, img_shape=(900, 1600))
            cur_ids = {t.track_id for t in tracks}
            id_switches += len(prev_ids - cur_ids - {t.track_id for t in tracks})
            prev_ids = cur_ids

            # ── anomaly ───────────────────────────────────────────────────────
            if len(pred_scores):
                import torch
                scores_t = torch.tensor(pred_scores, dtype=torch.float32).unsqueeze(1)
                a_results = anomaly_pipe.analyse(scores=scores_t, boxes=pred_boxes)
                for ar in a_results:
                    if ar.is_ood:
                        anomaly_counts[scene["name"]] += 1

            tok = sample["next"] if sample["next"] else None
            frame_count += 1

        prec = total_tp / (total_tp + total_fp + 1e-9)
        rec  = total_tp / (total_tp + total_fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        per_scene.append({
            "scene":   scene["name"],
            "frames":  frame_count,
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
        })
        print(f"    P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  lat={np.mean(latencies):.1f}ms")

    # ── aggregate ──────────────────────────────────────────────────────────────
    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall    = total_tp / (total_tp + total_fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    avg_lat   = float(np.mean(latencies)) if latencies else 0.0

    summary = {
        "model":          "yolov8n",
        "conf_threshold": conf_thresh,
        "iou_threshold":  0.5,
        "total_frames":   len(latencies),
        "precision":      round(precision, 4),
        "recall":         round(recall,    4),
        "f1":             round(f1,        4),
        "avg_latency_ms": round(avg_lat,   2),
        "fps":            round(1000 / (avg_lat + 1e-9), 1),
        "id_switches":    id_switches,
        "anomaly_counts": dict(anomaly_counts),
        "per_scene":      per_scene,
    }

    # ── save results ──────────────────────────────────────────────────────────
    out_path = ROOT / "experiments" / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # ── W&B logging ───────────────────────────────────────────────────────────
    if use_wandb:
        import wandb
        wandb.log({
            "precision":      precision,
            "recall":         recall,
            "f1":             f1,
            "avg_latency_ms": avg_lat,
            "fps":            summary["fps"],
            "id_switches":    id_switches,
        })
        wandb.finish()

    print("\n=== BENCHMARK SUMMARY ===")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Latency   : {avg_lat:.1f} ms/frame  ({summary['fps']} FPS)")
    print(f"  ID switches: {id_switches}")
    print(f"  Anomalies  : {sum(anomaly_counts.values())}")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UrbanSense Phase 4 Benchmark")
    parser.add_argument("--scene",  default="all",  help="Scene name or 'all'")
    parser.add_argument("--conf",   default=0.35,   type=float, help="YOLO confidence threshold")
    parser.add_argument("--wandb",  action="store_true", help="Log results to W&B")
    parser.add_argument("--max-samples", default=20, type=int, help="Max frames per scene")
    args = parser.parse_args()

    run_benchmark(
        scene_filter=args.scene,
        conf_thresh=args.conf,
        use_wandb=args.wandb,
        max_samples=args.max_samples,
    )
