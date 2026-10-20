import cv2
import numpy as np

from src.locate import locate


def test_finds_white_label_on_dark_background():
    img = np.full((600, 800, 3), 40, np.uint8)
    cv2.fillPoly(img, [np.array([[150, 100], [650, 130], [630, 500], [170, 480]])], (240, 240, 240))
    flat, quad = locate(img)
    assert quad is not None
    assert 400 < flat.shape[1] < 560 and 300 < flat.shape[0] < 420


def test_falls_back_to_whole_image():
    img = np.full((200, 300, 3), 128, np.uint8)
    flat, quad = locate(img)
    assert quad is None and flat.shape == img.shape
