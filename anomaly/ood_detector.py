"""
Out-of-Distribution (OOD) / Anomaly Detector for UrbanSense.

Implements three complementary anomaly signals that work without retraining:

1. Energy score  – low energy  → in-distribution (normal)
                   high energy → OOD / anomalous
2. Confidence score – model's max softmax probability
                      low confidence → uncertain / anomalous
3. Spatial outlier  – detections far from the ego vehicle or clustered
                      in unexpected image regions

All scorers operate on raw model outputs and require no labelled anomaly data.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AnomalyResult:
    """Per-detection anomaly assessment."""
    detection_idx:  int
    energy_score:   float           # higher = more anomalous
    confidence:     float           # lower = more anomalous
    is_ood:         bool
    reason:         str             # human-readable explanation


# ── energy-based OOD ─────────────────────────────────────────────────────────

class EnergyOODDetector:
    """
    Energy-based OOD detection (Liu et al., 2020 — "Energy-based OOD Detection").

    Energy(x) = -T · log Σ exp(f_k(x) / T)

    High energy = model is uncertain = likely OOD.
    Threshold is set at initialisation time; call `calibrate()` on
    in-distribution logits to set it automatically.

    Usage
    -----
    detector = EnergyOODDetector(temperature=1.0, threshold=20.0)
    results  = detector.score_logits(logits)          # [N] energy scores
    flags    = detector.flag_ood(logits)               # [N] bool
    """

    def __init__(self, temperature: float = 1.0, threshold: float = 20.0):
        self.T         = temperature
        self.threshold = threshold

    def energy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute per-sample energy.  logits: [N, C] → energy: [N]"""
        return -self.T * torch.logsumexp(logits / self.T, dim=-1)

    def flag_ood(self, logits: torch.Tensor) -> torch.Tensor:
        """Return bool tensor [N] — True where sample is OOD."""
        return self.energy(logits) > self.threshold

    def calibrate(self, in_dist_logits: torch.Tensor, percentile: float = 95.0):
        """
        Set threshold automatically from in-distribution logits.
        Sets threshold = percentile-th energy value of in-dist data.
        """
        energies = self.energy(in_dist_logits).cpu().numpy()
        self.threshold = float(np.percentile(energies, percentile))
        return self.threshold


# ── confidence-based OOD ──────────────────────────────────────────────────────

class ConfidenceOODDetector:
    """
    Simple maximum-softmax-probability (MSP) OOD detector (Hendrycks & Gimpel, 2017).

    Detections with max confidence below `threshold` are flagged as anomalous.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def score(self, logits_or_probs: torch.Tensor) -> torch.Tensor:
        """Return max confidence per detection. [N, C] → [N]"""
        if logits_or_probs.shape[-1] > 1:
            probs = F.softmax(logits_or_probs, dim=-1)
        else:
            probs = logits_or_probs
        return probs.max(dim=-1).values

    def flag_ood(self, logits_or_probs: torch.Tensor) -> torch.Tensor:
        """Return bool tensor [N] — True where detection is low-confidence."""
        return self.score(logits_or_probs) < self.threshold


# ── spatial outlier detector ──────────────────────────────────────────────────

class SpatialAnomalyDetector:
    """
    Heuristic detector that flags detections with unusual spatial properties:

    - Boxes that are extremely large (> max_area_frac of image area)
    - Boxes entirely in the sky region (top 20% of image, expected for vehicles)
    - Isolated singleton detections far from all other detections
    """

    def __init__(
        self,
        img_shape:       tuple[int, int] = (900, 1600),  # H, W
        max_area_frac:   float           = 0.25,
        isolation_dist:  float           = 300.0,        # pixels
        sky_frac:        float           = 0.20,
    ):
        self.H, self.W  = img_shape
        self.max_area   = max_area_frac * self.H * self.W
        self.iso_dist   = isolation_dist
        self.sky_thresh = sky_frac * self.H

    def flag(self, boxes: np.ndarray) -> List[dict]:
        """
        Parameters
        ----------
        boxes : np.ndarray [N, 4]  xyxy pixel coords

        Returns
        -------
        List of dicts: {idx, is_anomaly, reason}
        """
        results = []
        if len(boxes) == 0:
            return results

        centres = np.stack([
            (boxes[:, 0] + boxes[:, 2]) / 2,
            (boxes[:, 1] + boxes[:, 3]) / 2,
        ], axis=1)

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        for i, box in enumerate(boxes):
            reasons = []

            # oversized box
            if areas[i] > self.max_area:
                reasons.append(f"box area {areas[i]:.0f}px² > limit {self.max_area:.0f}px²")

            # box entirely in sky region
            if box[3] < self.sky_thresh:
                reasons.append(f"box bottom y={box[3]:.0f} in sky region (<{self.sky_thresh:.0f})")

            # spatial isolation (nearest neighbour distance)
            if len(boxes) > 1:
                dists = np.linalg.norm(centres - centres[i], axis=1)
                dists[i] = np.inf
                nn_dist = dists.min()
                if nn_dist > self.iso_dist:
                    reasons.append(f"isolated detection (nearest neighbour {nn_dist:.0f}px away)")

            results.append({
                "idx":        i,
                "is_anomaly": len(reasons) > 0,
                "reason":     "; ".join(reasons) if reasons else "normal",
            })

        return results


# ── unified pipeline ──────────────────────────────────────────────────────────

class AnomalyPipeline:
    """
    Combines energy, confidence, and spatial signals into one
    per-detection anomaly assessment.

    Usage
    -----
    pipeline = AnomalyPipeline()
    results  = pipeline.analyse(logits=logits, boxes=boxes_np)
    for r in results:
        print(r.detection_idx, r.is_ood, r.reason)
    """

    def __init__(
        self,
        energy_thresh:     float = 20.0,
        confidence_thresh: float = 0.50,
        temperature:       float = 1.0,
        img_shape:         tuple[int, int] = (900, 1600),
    ):
        self.energy_det    = EnergyOODDetector(temperature=temperature, threshold=energy_thresh)
        self.conf_det      = ConfidenceOODDetector(threshold=confidence_thresh)
        self.spatial_det   = SpatialAnomalyDetector(img_shape=img_shape)

    def analyse(
        self,
        logits: Optional[torch.Tensor]  = None,
        scores: Optional[torch.Tensor]  = None,
        boxes:  Optional[np.ndarray]    = None,
    ) -> List[AnomalyResult]:
        """
        Analyse detections for anomalies.

        Parameters
        ----------
        logits  : Tensor [N, C]   raw model logits (used for energy + confidence)
        scores  : Tensor [N]      pre-computed confidence scores (used if logits=None)
        boxes   : np.ndarray [N, 4]  xyxy pixel coords (used for spatial check)

        Returns
        -------
        List[AnomalyResult]
        """
        N = 0
        if logits is not None:
            N = logits.shape[0]
        elif scores is not None:
            N = len(scores)
        elif boxes is not None:
            N = len(boxes)

        if N == 0:
            return []

        # energy + confidence flags
        energy_scores = np.full(N, 0.0)
        conf_scores   = np.full(N, 1.0)
        energy_ood    = np.zeros(N, dtype=bool)
        conf_ood      = np.zeros(N, dtype=bool)

        if logits is not None:
            energy_scores = self.energy_det.energy(logits).cpu().numpy()
            energy_ood    = self.energy_det.flag_ood(logits).cpu().numpy()
            conf_scores   = self.conf_det.score(logits).cpu().numpy()
            conf_ood      = self.conf_det.flag_ood(logits).cpu().numpy()
        elif scores is not None:
            s = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
            conf_scores = s
            conf_ood    = s < self.conf_det.threshold

        # spatial flags
        spatial_flags = [{} for _ in range(N)]
        if boxes is not None and len(boxes) == N:
            spatial_flags = self.spatial_det.flag(np.asarray(boxes))

        results: List[AnomalyResult] = []
        for i in range(N):
            reasons = []
            if energy_ood[i]:
                reasons.append(f"high energy={energy_scores[i]:.2f}")
            if conf_ood[i]:
                reasons.append(f"low confidence={float(conf_scores[i]):.3f}")
            sp = spatial_flags[i] if i < len(spatial_flags) else {}
            if sp.get("is_anomaly"):
                reasons.append(sp["reason"])

            results.append(AnomalyResult(
                detection_idx=i,
                energy_score=float(energy_scores[i]),
                confidence=float(conf_scores[i]),
                is_ood=len(reasons) > 0,
                reason="; ".join(reasons) if reasons else "in-distribution",
            ))

        return results
