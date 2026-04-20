"""
tests/test_detector.py
======================
Unit tests for models/detection/detr_detector.py.

`transformers` is stubbed in sys.modules before any project code loads so
these tests run in CI without downloading weights or needing the package.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image


# ── build a realistic fake `transformers` and inject it ──────────────────────

def _make_processor_instance(n_dets: int = 6, img_w: int = 640, img_h: int = 480):
    proc = MagicMock()
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    proc.return_value = mock_inputs          # proc(images=...) → inputs

    boxes  = torch.rand(n_dets, 4) * torch.tensor([img_w, img_h, img_w, img_h])
    labels = torch.randint(0, 80, (n_dets,))
    scores = torch.rand(n_dets)
    proc.post_process_object_detection.return_value = [
        {"boxes": boxes, "labels": labels, "scores": scores}
    ]
    return proc


def _make_model_instance():
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model
    model.config.id2label = {i: f"cls_{i}" for i in range(80)}
    model.return_value = MagicMock()
    return model


def _build_fake_transformers(n_dets: int = 6, img_w: int = 640, img_h: int = 480):
    """Return a fake `transformers` module with from_pretrained already wired."""
    fake = types.ModuleType("transformers")

    proc_instance  = _make_processor_instance(n_dets, img_w, img_h)
    model_instance = _make_model_instance()

    FakeProcessor = MagicMock()
    FakeProcessor.from_pretrained.return_value = proc_instance

    FakeModel = MagicMock()
    FakeModel.from_pretrained.return_value = model_instance

    fake.DetrImageProcessor      = FakeProcessor
    fake.DetrForObjectDetection  = FakeModel
    return fake, proc_instance, model_instance


def _inject_transformers(n_dets=6, img_w=640, img_h=480):
    """Overwrite sys.modules['transformers'] with fresh mocks and reload detr_detector."""
    import importlib

    fake, proc, model = _build_fake_transformers(n_dets, img_w, img_h)
    sys.modules["transformers"] = fake

    # Force the module to re-import with the new mock
    if "models.detection.detr_detector" in sys.modules:
        del sys.modules["models.detection.detr_detector"]

    return proc, model


# Perform the initial injection at import time so test_detr_imports works
_inject_transformers()


# ── tests ─────────────────────────────────────────────────────────────────────

class TestDETRDetector:

    # ── 1. import test ────────────────────────────────────────────────────────
    def test_detr_imports(self):
        """DETRDetector can be imported and the class object is not None."""
        from models.detection.detr_detector import DETRDetector
        assert DETRDetector is not None

    # ── 2. output shape test ─────────────────────────────────────────────────
    def test_detr_output_shape(self):
        """predict() returns boxes tensor with shape (N, 4)."""
        n_dets, img_w, img_h = 6, 640, 480
        _inject_transformers(n_dets, img_w, img_h)

        from models.detection.detr_detector import DETRDetector
        detector = DETRDetector(device="cpu", threshold=0.01)

        blank = Image.new("RGB", (img_w, img_h), color=(128, 128, 128))
        boxes, labels, scores = detector.predict(blank, threshold=0.01)

        assert boxes.ndim == 2,     f"Expected 2-D boxes, got {boxes.ndim}-D"
        assert boxes.shape[1] == 4, f"Expected 4 cols (xyxy), got {boxes.shape[1]}"
        assert boxes.shape[0] == n_dets

    # ── 3. scores range test ─────────────────────────────────────────────────
    def test_detr_scores_range(self):
        """All returned confidence scores must be in [0, 1]."""
        _inject_transformers(n_dets=10)

        from models.detection.detr_detector import DETRDetector
        detector = DETRDetector(device="cpu", threshold=0.01)

        blank = Image.new("RGB", (640, 480), color=0)
        _, _, scores = detector.predict(blank, threshold=0.01)

        assert scores.min().item() >= 0.0, "Score below 0 detected"
        assert scores.max().item() <= 1.0, "Score above 1 detected"

    # ── 4. label_name helper ─────────────────────────────────────────────────
    def test_detr_label_name(self):
        """label_name() returns a string for known and unknown indices."""
        _inject_transformers()

        from models.detection.detr_detector import DETRDetector
        detector = DETRDetector(device="cpu")

        name    = detector.label_name(0)
        unknown = detector.label_name(9999)

        assert isinstance(name,    str) and len(name) > 0
        assert isinstance(unknown, str)
