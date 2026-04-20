"""
tests/test_detector.py
======================
Unit tests for models/detection/detr_detector.py.

The HuggingFace model is mocked so these tests run fast in CI
without downloading ~160 MB of weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_mock_processor(n_dets: int = 6, img_w: int = 640, img_h: int = 480):
    """Return a mock DetrImageProcessor that produces realistic outputs."""
    proc = MagicMock()

    # processor(images=..., return_tensors="pt") → dict-like inputs
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    proc.return_value = mock_inputs

    # post_process_object_detection → list of result dicts
    boxes  = torch.rand(n_dets, 4) * torch.tensor([img_w, img_h, img_w, img_h])
    labels = torch.randint(0, 80, (n_dets,))
    scores = torch.rand(n_dets)          # [0, 1) — all in valid range

    proc.post_process_object_detection.return_value = [
        {"boxes": boxes, "labels": labels, "scores": scores}
    ]
    return proc


def _make_mock_model():
    """Return a mock DetrForObjectDetection."""
    model = MagicMock()
    model.to.return_value = model       # .to(device) → self
    model.eval.return_value = model     # .eval() → self
    model.config.id2label = {i: f"cls_{i}" for i in range(80)}
    model.return_value = MagicMock()    # model(**inputs) → outputs object
    return model


# ── tests ─────────────────────────────────────────────────────────────────────

class TestDETRDetector:

    # ── 1. import test ────────────────────────────────────────────────────────
    def test_detr_imports(self):
        """DETRDetector can be imported and the class object is not None."""
        from models.detection.detr_detector import DETRDetector
        assert DETRDetector is not None

    # ── 2. output shape test ─────────────────────────────────────────────────
    @patch("models.detection.detr_detector.DetrForObjectDetection.from_pretrained")
    @patch("models.detection.detr_detector.DetrImageProcessor.from_pretrained")
    def test_detr_output_shape(self, mock_proc_cls, mock_model_cls):
        """predict() returns boxes tensor with shape (N, 4)."""
        n_dets = 6
        img_w, img_h = 640, 480
        blank = Image.new("RGB", (img_w, img_h), color=(128, 128, 128))

        mock_proc_cls.return_value  = _make_mock_processor(n_dets, img_w, img_h)
        mock_model_cls.return_value = _make_mock_model()

        from models.detection.detr_detector import DETRDetector
        detector = DETRDetector(device="cpu", threshold=0.01)

        boxes, labels, scores = detector.predict(blank, threshold=0.01)

        assert boxes.ndim == 2,       f"Expected 2-D boxes, got {boxes.ndim}-D"
        assert boxes.shape[1] == 4,   f"Expected 4 columns (xyxy), got {boxes.shape[1]}"
        assert boxes.shape[0] == n_dets

    # ── 3. scores range test ─────────────────────────────────────────────────
    @patch("models.detection.detr_detector.DetrForObjectDetection.from_pretrained")
    @patch("models.detection.detr_detector.DetrImageProcessor.from_pretrained")
    def test_detr_scores_range(self, mock_proc_cls, mock_model_cls):
        """All returned confidence scores must be in [0, 1]."""
        blank = Image.new("RGB", (640, 480), color=0)

        mock_proc_cls.return_value  = _make_mock_processor(n_dets=10)
        mock_model_cls.return_value = _make_mock_model()

        from models.detection.detr_detector import DETRDetector
        detector = DETRDetector(device="cpu", threshold=0.01)

        _, _, scores = detector.predict(blank, threshold=0.01)

        assert scores.min().item() >= 0.0, "Score below 0 detected"
        assert scores.max().item() <= 1.0, "Score above 1 detected"

    # ── 4. label_name helper ─────────────────────────────────────────────────
    @patch("models.detection.detr_detector.DetrForObjectDetection.from_pretrained")
    @patch("models.detection.detr_detector.DetrImageProcessor.from_pretrained")
    def test_detr_label_name(self, mock_proc_cls, mock_model_cls):
        """label_name() returns a string for known and unknown indices."""
        mock_proc_cls.return_value  = _make_mock_processor()
        mock_model_cls.return_value = _make_mock_model()

        from models.detection.detr_detector import DETRDetector
        detector = DETRDetector(device="cpu")

        name = detector.label_name(0)
        assert isinstance(name, str) and len(name) > 0

        unknown = detector.label_name(9999)
        assert isinstance(unknown, str)
