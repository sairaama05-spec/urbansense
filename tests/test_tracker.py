"""
tests/test_tracker.py
=====================
Unit tests for tracking/bytetrack_wrapper.py.

bytetracker (and its C-extension dep `lap`) are mocked at the top of this
file so that all tests run in CI without needing a compiled bytetracker.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest


# ── inject a fake `bytetracker` before any project code is imported ───────────

def _install_bytetracker_mock() -> None:
    """Put a lightweight stub in sys.modules so bytetrack_wrapper imports cleanly."""
    if "bytetracker" in sys.modules:
        return  # already present (real or stub)

    fake_pkg = types.ModuleType("bytetracker")

    class _FakeBYTETracker:
        def __init__(self, **kwargs):
            pass

        def update(self, dets, img_shape):
            return []   # no tracks — sufficient for unit testing

    fake_pkg.BYTETracker = _FakeBYTETracker
    sys.modules["bytetracker"] = fake_pkg


_install_bytetracker_mock()


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_dets(n: int) -> np.ndarray:
    """Generate n random [x1, y1, x2, y2, score, class_id] detections."""
    if n == 0:
        return np.zeros((0, 6), dtype=np.float32)
    rng   = np.random.default_rng(seed=42)
    x1y1  = rng.uniform(0,   800, (n, 2)).astype(np.float32)
    x2y2  = x1y1 + rng.uniform(20, 200, (n, 2)).astype(np.float32)
    score = rng.uniform(0.5, 1.0, (n, 1)).astype(np.float32)
    cls   = rng.integers(0, 5,   (n, 1)).astype(np.float32)
    return np.hstack([x1y1, x2y2, score, cls])


# ── tests ─────────────────────────────────────────────────────────────────────

class TestByteTrackWrapper:

    # ── 1. import test ────────────────────────────────────────────────────────
    def test_tracker_imports(self):
        """ByteTrackWrapper can be imported and is not None."""
        from tracking.bytetrack_wrapper import ByteTrackWrapper
        assert ByteTrackWrapper is not None

    # ── 2. empty-input test ───────────────────────────────────────────────────
    def test_tracker_empty_input(self):
        """update() with zero detections must return an empty list (0 tracks)."""
        from tracking.bytetrack_wrapper import ByteTrackWrapper

        tracker = ByteTrackWrapper(frame_rate=2)
        empty   = _make_dets(0)
        result  = tracker.update(empty, img_shape=(900, 1600))

        assert isinstance(result, list), "update() should return a list"
        assert len(result) == 0, f"Expected 0 tracks, got {len(result)}"

    # ── 3. none-input test ────────────────────────────────────────────────────
    def test_tracker_none_input(self):
        """update() with None detections must return an empty list."""
        from tracking.bytetrack_wrapper import ByteTrackWrapper

        tracker = ByteTrackWrapper(frame_rate=2)
        result  = tracker.update(None, img_shape=(900, 1600))

        assert isinstance(result, list)
        assert len(result) == 0

    # ── 4. track dataclass fields ─────────────────────────────────────────────
    def test_track_dataclass_fields(self):
        """Track dataclass has all required fields."""
        from tracking.bytetrack_wrapper import Track

        t = Track(
            track_id=1,
            bbox=np.array([10.0, 20.0, 100.0, 200.0]),
            score=0.92,
            class_id=2,
            age=5,
            hits=3,
        )
        assert t.track_id == 1
        assert t.bbox.shape == (4,)
        assert 0.0 <= t.score <= 1.0
        assert isinstance(t.class_id, int)

    # ── 5. reset clears state ─────────────────────────────────────────────────
    def test_tracker_reset(self):
        """reset() sets frame_id back to 0."""
        from tracking.bytetrack_wrapper import ByteTrackWrapper

        tracker = ByteTrackWrapper(frame_rate=2)
        tracker.update(_make_dets(3), img_shape=(900, 1600))
        assert tracker.frame_id == 1

        tracker.reset()
        assert tracker.frame_id == 0

    # ── 6. helper: detections_from_detr ──────────────────────────────────────
    def test_detections_from_detr_empty(self):
        """detections_from_detr returns (0,6) array for empty tensors."""
        import torch
        from tracking.bytetrack_wrapper import ByteTrackWrapper

        empty_boxes  = torch.zeros((0, 4))
        empty_labels = torch.zeros((0,), dtype=torch.long)
        empty_scores = torch.zeros((0,))

        arr = ByteTrackWrapper.detections_from_detr(empty_boxes, empty_labels, empty_scores)
        assert arr.shape == (0, 6)

    # ── 7. helper: detections_from_yolo ──────────────────────────────────────
    def test_detections_from_yolo_empty(self):
        """detections_from_yolo returns (0,6) array for empty YOLO result."""
        from tracking.bytetrack_wrapper import ByteTrackWrapper

        mock_result = type("R", (), {"boxes": None})()
        arr = ByteTrackWrapper.detections_from_yolo(mock_result)
        assert arr.shape == (0, 6)
