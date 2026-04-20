try:
    from transformers import DetrImageProcessor, DetrForObjectDetection
except ImportError:  # CI / environments without transformers installed
    DetrImageProcessor = None       # type: ignore[assignment,misc]
    DetrForObjectDetection = None   # type: ignore[assignment,misc]

import torch


class DETRDetector:
    """
    Wrapper around facebook/detr-resnet-50 for object detection.

    Returns boxes (xyxy), labels, and scores for a PIL image.
    """

    MODEL_ID = "facebook/detr-resnet-50"

    def __init__(self, device: str = "cpu", threshold: float = 0.7):
        self.device = device
        self.threshold = threshold
        self.processor = DetrImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = (
            DetrForObjectDetection.from_pretrained(self.MODEL_ID)
            .to(device)
        )
        self.model.eval()

    @torch.no_grad()
    def predict(self, pil_image, threshold: float | None = None):
        """
        Run inference on a PIL image.

        Args:
            pil_image : PIL.Image  – input RGB image
            threshold : float      – confidence threshold (overrides instance default)

        Returns:
            boxes  : Tensor [N, 4]  xyxy pixel coords
            labels : Tensor [N]     COCO class indices
            scores : Tensor [N]     confidence scores
        """
        thresh = threshold if threshold is not None else self.threshold
        inputs = self.processor(
            images=pil_image, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=[pil_image.size[::-1]],
            threshold=thresh,
        )[0]
        return results["boxes"], results["labels"], results["scores"]

    def label_name(self, label_idx: int) -> str:
        """Convert COCO label index to human-readable name."""
        return self.model.config.id2label.get(label_idx, f"cls_{label_idx}")
