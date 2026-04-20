"""
Visualisation utilities for UrbanSense.

All functions operate on PIL Images and return annotated PIL Images so they
compose naturally with Streamlit's st.image() and matplotlib.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# colour palette: track_id → (R, G, B)
_TRACK_PALETTE: dict[int, Tuple[int, int, int]] = {}

# category palette
CATEGORY_COLORS: dict[str, Tuple[int, int, int]] = {
    "car":          ( 30, 144, 255),
    "truck":        (255, 127,  80),
    "bus":          (148,   0, 211),
    "motorcycle":   (  0, 191, 255),
    "bicycle":      (135, 206, 235),
    "pedestrian":   (255,  69,   0),
    "trafficcone":  (255, 165,   0),
    "barrier":      (210, 180, 140),
    "construction": (255,  20, 147),
    "animal":       ( 50, 205,  50),
}
_DEFAULT_COLOR = (200, 200, 200)


# ── helpers ───────────────────────────────────────────────────────────────────

def _track_color(track_id: int) -> Tuple[int, int, int]:
    """Assign a stable random colour to a track ID."""
    if track_id not in _TRACK_PALETTE:
        rng = random.Random(track_id)
        _TRACK_PALETTE[track_id] = (
            rng.randint(80, 255),
            rng.randint(80, 255),
            rng.randint(80, 255),
        )
    return _TRACK_PALETTE[track_id]


def _category_color(name: str) -> Tuple[int, int, int]:
    """Look up category colour by short name (case-insensitive)."""
    key = name.lower().split(".")[-1]
    for cat, col in CATEGORY_COLORS.items():
        if cat in key:
            return col
    return _DEFAULT_COLOR


def _try_font(size: int = 12) -> ImageFont.ImageFont:
    """Return a TrueType font if available, else default."""
    for face in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(face, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


# ── bounding box drawing ──────────────────────────────────────────────────────

def draw_boxes(
    image:    Image.Image,
    boxes:    np.ndarray,
    labels:   Optional[List[str]] = None,
    scores:   Optional[np.ndarray] = None,
    colors:   Optional[List[Tuple[int, int, int]]] = None,
    line_width: int = 2,
    font_size:  int = 12,
) -> Image.Image:
    """
    Draw 2-D bounding boxes on a PIL image.

    Parameters
    ----------
    image   : PIL Image (RGB)
    boxes   : [N, 4] xyxy pixel coords (numpy or list)
    labels  : list of N label strings
    scores  : [N] confidence float array
    colors  : list of N (R, G, B) tuples; auto-assigned if None
    line_width, font_size

    Returns
    -------
    Annotated PIL Image (original not modified)
    """
    img  = image.copy()
    draw = ImageDraw.Draw(img)
    font = _try_font(font_size)
    boxes = np.asarray(boxes)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        color = colors[i] if colors else _category_color(labels[i] if labels else "")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        parts = []
        if labels:
            parts.append(labels[i])
        if scores is not None:
            parts.append(f"{scores[i]:.2f}")
        if parts:
            text = " ".join(parts)
            draw.text((x1 + 2, y1 + 2), text, fill=color, font=font)

    return img


# ── track drawing ─────────────────────────────────────────────────────────────

def draw_tracks(
    image:      Image.Image,
    tracks,                             # List[tracking.bytetrack_wrapper.Track]
    show_id:    bool = True,
    show_score: bool = False,
    line_width: int  = 2,
    font_size:  int  = 12,
) -> Image.Image:
    """
    Draw ByteTrack track boxes colour-coded by track ID.

    Parameters
    ----------
    image   : PIL Image (RGB)
    tracks  : list of Track objects from ByteTrackWrapper.update()
    """
    img  = image.copy()
    draw = ImageDraw.Draw(img)
    font = _try_font(font_size)

    for t in tracks:
        x1, y1, x2, y2 = [float(v) for v in t.bbox]
        color = _track_color(t.track_id)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        label_parts = []
        if show_id:
            label_parts.append(f"#{t.track_id}")
        if show_score:
            label_parts.append(f"{t.score:.2f}")
        if label_parts:
            draw.text((x1 + 2, y1 + 2), " ".join(label_parts), fill=color, font=font)

    return img


# ── anomaly overlay ───────────────────────────────────────────────────────────

def draw_anomalies(
    image:   Image.Image,
    boxes:   np.ndarray,
    results,                            # List[AnomalyResult]
    line_width: int = 3,
    font_size:  int = 11,
) -> Image.Image:
    """
    Overlay anomaly detection results on an image.
    OOD detections are drawn in red with their reason; normal ones in green.

    Parameters
    ----------
    image   : PIL Image (RGB)
    boxes   : [N, 4] xyxy
    results : list of AnomalyResult from AnomalyPipeline.analyse()
    """
    img  = image.copy()
    draw = ImageDraw.Draw(img)
    font = _try_font(font_size)
    boxes = np.asarray(boxes)

    for r in results:
        if r.detection_idx >= len(boxes):
            continue
        x1, y1, x2, y2 = [float(v) for v in boxes[r.detection_idx][:4]]
        color = (220, 30, 30) if r.is_ood else (50, 200, 50)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        tag = "OOD" if r.is_ood else "OK"
        draw.text((x1 + 2, y1 + 2), tag, fill=color, font=font)

    return img


# ── instance segmentation overlay ────────────────────────────────────────────

def draw_masks(
    image:      Image.Image,
    seg_result: dict,
    alpha:      float = 0.45,
    label_fn=None,
) -> Image.Image:
    """
    Overlay Mask2Former instance segmentation on an image.

    Parameters
    ----------
    image      : PIL Image (RGB)
    seg_result : dict from SegmentationModel.segment()
                 keys: 'segmentation' (H×W tensor), 'segments_info' (list of dicts)
    alpha      : mask opacity
    label_fn   : callable(label_id) → str for legend text
    """
    img   = image.copy().convert("RGBA")
    seg   = seg_result.get("segmentation")
    segs  = seg_result.get("segments_info", [])

    if seg is None:
        return image

    seg_np = seg.cpu().numpy() if hasattr(seg, "cpu") else np.asarray(seg)
    overlay = Image.fromarray(np.zeros((*seg_np.shape[:2], 4), dtype=np.uint8), "RGBA")
    draw    = ImageDraw.Draw(overlay)
    font    = _try_font(11)

    for info in segs:
        seg_id   = info["id"]
        label_id = info.get("label_id", 0)
        mask     = (seg_np == seg_id)
        if not mask.any():
            continue
        color = _track_color(label_id)
        rgba  = (*color, int(alpha * 255))

        # fill mask pixels
        ys, xs = np.where(mask)
        for y, x in zip(ys[::4], xs[::4]):  # stride for speed
            overlay.putpixel((int(x), int(y)), rgba)

        # label at centroid
        cy, cx = int(ys.mean()), int(xs.mean())
        label  = label_fn(label_id) if label_fn else str(label_id)
        draw.text((cx, cy), label, fill=(255, 255, 255, 230), font=font)

    img = Image.alpha_composite(img, overlay).convert("RGB")
    return img


# ── multi-camera grid ─────────────────────────────────────────────────────────

def make_camera_grid(
    images: List[Image.Image],
    labels: List[str],
    cols:   int  = 3,
    thumb_w: int = 640,
    thumb_h: int = 360,
) -> Image.Image:
    """
    Stitch multiple camera images into a labelled grid.

    Parameters
    ----------
    images  : list of PIL Images (RGB)
    labels  : list of camera-channel names
    cols    : images per row
    thumb_w, thumb_h : thumbnail size

    Returns
    -------
    Single PIL Image containing the grid
    """
    n    = len(images)
    rows = (n + cols - 1) // cols
    font = _try_font(14)

    grid = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (20, 20, 20))
    draw = ImageDraw.Draw(grid)

    for idx, (img, lbl) in enumerate(zip(images, labels)):
        thumb = img.resize((thumb_w, thumb_h), Image.LANCZOS)
        row, col = divmod(idx, cols)
        x, y = col * thumb_w, row * thumb_h
        grid.paste(thumb, (x, y))
        draw.text((x + 6, y + 4), lbl, fill=(255, 255, 255), font=font)

    return grid
