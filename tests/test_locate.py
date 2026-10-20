import cv2
import numpy as np

from src.locate import LabelLocator


def label_photo():
    img = np.full((600, 800, 3), 40, np.uint8)
    cv2.fillPoly(img, [np.array([[150, 100], [650, 130], [630, 500], [170, 480]])], (240, 240, 240))
    return img


class FakeDetector:
    def detect(self, img):
        return [([100.0, 80.0, 700.0, 520.0], 0.9)]


def test_finds_white_label_on_dark_background():
    label = LabelLocator().locate(label_photo())
    assert label.quad is not None
    assert 400 < label.image.shape[1] < 560 and 300 < label.image.shape[0] < 420


def test_falls_back_to_whole_image():
    img = np.full((200, 300, 3), 128, np.uint8)
    label = LabelLocator().locate(img)
    assert label.quad is None and label.image.shape == img.shape


def test_detector_box_is_cropped_and_corners_map_back_to_the_photo():
    label = LabelLocator(FakeDetector()).locate(label_photo())
    quad = label.quad
    assert quad is not None and 130 < quad[0][0] < 170 and 90 < quad[0][1] < 110
    assert 400 < label.image.shape[1] < 560
    # a point at the straightened label's top-left maps back to the label's corner in the photo
    x, y = label.to_photo(np.array([[0, 0]]))[0]
    assert abs(x - quad[0][0]) < 2 and abs(y - quad[0][1]) < 2
