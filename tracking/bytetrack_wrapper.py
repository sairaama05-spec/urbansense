"""
ByteTrack wrapper for UrbanSense.

Wraps the `bytetracker` library to provide a clean per-scene tracking API
that integrates with YOLO / DETR detections and the nuScenes sample loop.

Input detection format: numpy array [N, 6] = [x1, y1, x2, y2, score, class_id]
Output tracks: list of Track objects with .track_id, .tlbr, .score, .class_id
"""

from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List

try:
    from bytetracker import BYTETracker
except ImportError:  # CI / environments without lap compiled
    BYTETracker = None  # type: ignore[assignment,misc]


@dataclass
class Track:
    """Simplified track object returned by the wrapper."""
    track_id:  int
    bbox:      np.ndarray   # [x1, y1, x2, y2]
    score:     float
    class_id:  int
    age:       int          # frames since first seen
    hits:      int          # consecutive detections


class ByteTrackWrapper:
    """
    Scene-level ByteTracker that resets state between scenes.

    Usage
    -----
    tracker = ByteTrackWrapper()
    for sample in scene_samples:
        dets = run_detector(sample)         # np.ndarray [N, 6]
        tracks = tracker.update(dets, img_shape=(H, W))
        for t in tracks:
            print(t.track_id, t.bbox, t.class_id)
    tracker.reset()                         # start next scene fresh
    """

    def __init__(
        self,
        track_thresh:  float = 0.45,
        track_buffer:  int   = 25,
        match_thresh:  float = 0.8,
        frame_rate:    int   = 2,        # nuScenes keyframes ~2 Hz
    ):
        self.cfg = dict(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate,
        )
        self._tracker: BYTETracker | None = None
        self.reset()

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self):
        """Reset tracker state (call between scenes)."""
        if BYTETracker is not None:
            self._tracker = BYTETracker(**self.cfg)
        else:
            self._tracker = None
        self.frame_id = 0

    def update(
        self,
        detections: np.ndarray,
        img_shape:  tuple[int, int] = (900, 1600),
    ) -> List[Track]:
        """
        Update tracker with detections from one frame.

        Parameters
        ----------
        detections : np.ndarray [N, 5 or 6]
            Each row: [x1, y1, x2, y2, score] or [x1, y1, x2, y2, score, class_id]
        img_shape  : (H, W)

        Returns
        -------
        List[Track]
        """
        self.frame_id += 1

        if detections is None or len(detections) == 0:
            return []

        dets = np.asarray(detections, dtype=np.float32)

        # ensure 6-column format (add class_id=0 if missing)
        if dets.ndim == 1:
            dets = dets[np.newaxis, :]
        if dets.shape[1] == 5:
            dets = np.hstack([dets, np.zeros((len(dets), 1))])

        # bytetracker expects torch tensors
        dets_t = torch.from_numpy(dets)

        try:
            if self._tracker is None:
                raw_tracks = []
            else:
                raw_tracks = self._tracker.update(dets_t, img_shape)
        except Exception:
            # tolerate internal bytetracker numpy/torch quirks
            raw_tracks = []

        tracks: List[Track] = []
        for t in raw_tracks:
            try:
                tracks.append(Track(
                    track_id=int(t.track_id),
                    bbox=np.array(t.tlbr, dtype=np.float32),
                    score=float(t.score),
                    class_id=int(getattr(t, "cls", 0)),
                    age=int(getattr(t, "frame_id", self.frame_id)),
                    hits=int(getattr(t, "tracklet_len", 1)),
                ))
            except Exception:
                continue

        return tracks

    # ── convenience ───────────────────────────────────────────────────────────

    @staticmethod
    def detections_from_yolo(yolo_result) -> np.ndarray:
        """
        Convert an Ultralytics YOLO result object to the [N, 6] array
        expected by `update()`.
        """
        boxes = yolo_result.boxes
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        xyxy   = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy().reshape(-1, 1)
        clsids = boxes.cls.cpu().numpy().reshape(-1, 1)
        return np.hstack([xyxy, scores, clsids]).astype(np.float32)

    @staticmethod
    def detections_from_detr(boxes, labels, scores) -> np.ndarray:
        """
        Convert DETR output tensors to [N, 6] detection array.

        boxes  : Tensor [N, 4] xyxy
        labels : Tensor [N]
        scores : Tensor [N]
        """
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        b = boxes.cpu().numpy()
        s = scores.cpu().numpy().reshape(-1, 1)
        c = labels.cpu().numpy().reshape(-1, 1)
        return np.hstack([b, s, c]).astype(np.float32)
