# Paddle must be imported before onnxruntime (used by the detector and the ONNX OCR engine):
# the other order segfaults Paddle. Importing it here guarantees the order for every module.
import paddle  # noqa: F401
