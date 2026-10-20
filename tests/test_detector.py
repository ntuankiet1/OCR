import onnxruntime as ort

from src.utils import pick_providers

CPU = "CPUExecutionProvider"


def available(monkeypatch, *names):
    monkeypatch.setattr(ort, "get_available_providers", lambda: [*names, CPU])


def test_cpu_when_nothing_better_is_available(monkeypatch):
    available(monkeypatch)
    assert pick_providers(None, "cpu") == [CPU]
    assert pick_providers(None, "gpu:0") == [CPU]


def test_coreml_on_macos_and_cuda_on_nvidia(monkeypatch):
    available(monkeypatch, "CoreMLExecutionProvider", "CUDAExecutionProvider")
    assert pick_providers(None, "cpu") == ["CoreMLExecutionProvider", CPU]
    assert pick_providers(None, "gpu:0") == ["CUDAExecutionProvider", CPU]


def test_explicit_list_skips_missing_providers(monkeypatch):
    available(monkeypatch, "CUDAExecutionProvider")
    assert pick_providers("TensorrtExecutionProvider, CUDAExecutionProvider", "cpu") == [
        "CUDAExecutionProvider",
        CPU,
    ]


def test_resize_max_side_only_shrinks():
    import numpy as np

    from src.utils import resize_max_side

    assert resize_max_side(np.zeros((400, 800, 3), np.uint8), 200).shape[:2] == (100, 200)
    assert resize_max_side(np.zeros((40, 80, 3), np.uint8), 200).shape[:2] == (40, 80)


def test_thresholds_come_from_env(monkeypatch):
    from src.config import Settings

    monkeypatch.setenv("OCR_DETECTOR_CONF", "0.4")
    monkeypatch.setenv("OCR_MIN_SCORE", "0.7")
    monkeypatch.setenv("OCR_TEXT_DET_BOX_THRESH", "0.5")
    s = Settings(_env_file=None)
    assert (s.detector_conf, s.min_score, s.text_det_box_thresh) == (0.4, 0.7, 0.5)
    assert s.text_det_thresh is None and s.detector_nms_iou == 0.5
