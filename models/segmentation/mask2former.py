from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch


class SegmentationModel:
    """
    Wrapper around facebook/mask2former-swin-base-coco-instance
    for panoptic / instance segmentation.

    Returns the post-processed segmentation result dict for a PIL image.
    """

    MODEL_ID = "facebook/mask2former-swin-base-coco-instance"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = (
            Mask2FormerForUniversalSegmentation.from_pretrained(self.MODEL_ID)
            .to(device)
        )
        self.model.eval()

    @torch.no_grad()
    def segment(self, pil_image):
        """
        Run instance segmentation on a PIL image.

        Args:
            pil_image : PIL.Image – input RGB image

        Returns:
            result dict with keys:
                'segmentation'  – H×W label map (tensor)
                'segments_info' – list of dicts with id, label_id, score
        """
        inputs = self.processor(
            images=pil_image, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[pil_image.size[::-1]],
        )[0]
        return results

    def label_name(self, label_id: int) -> str:
        """Convert label id to human-readable category name."""
        return self.model.config.id2label.get(label_id, f"cls_{label_id}")
