import cv2
import numpy as np
from PIL import Image


class DocumentPreprocessor:
    def process(self, image: Image.Image) -> Image.Image:
        img = np.array(image.convert('RGB'))
        img = self._upscale_if_needed(img)
        img = self._deskew(img)
        img = self._denoise(img)
        img = self._enhance_contrast(img)
        return Image.fromarray(img)

    def _upscale_if_needed(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if min(h, w) < 800:
            scale = 800 / min(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)
        return img

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                                minLineLength=100, maxLineGap=10)
        if lines is None:
            return img
        angles = []
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:
                angles.append(angle)
        if not angles:
            return img
        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:
            return img
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        lab     = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l       = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
