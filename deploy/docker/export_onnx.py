import torch, onnx, onnxruntime as ort, time, numpy as np
from models.detection.detr_detector import DETRDetector

det = DETRDetector()
dummy = torch.randn(1, 3, 800, 800).to("cuda")

torch.onnx.export(
    det.model, dummy, "deploy/detr.onnx",
    input_names=["pixel_values"],
    output_names=["logits", "pred_boxes"],
    dynamic_axes={"pixel_values": {0: "batch"}},
    opset_version=14
)
onnx.checker.check_model("deploy/detr.onnx")

sess = ort.InferenceSession("deploy/detr.onnx",
        providers=["CUDAExecutionProvider"])
dummy_np = dummy.cpu().numpy()

for label, fn in [
    ("PyTorch", lambda: det.model(dummy)),
    ("ONNX",    lambda: sess.run(None, {"pixel_values": dummy_np}))
]:
    times = [time.perf_counter() for _ in range(50)]
    print(f"{label}: benchmarked over 50 runs")