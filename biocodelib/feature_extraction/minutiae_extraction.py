# biocodelib/feature_extraction/minutiae_extraction.py
import cv2
import numpy as np
from skimage.morphology import skeletonize

def extract_minutiae(image):
    """
    استخراج Minutiae (ending + bifurcation) — مطابق PDF
    خروجی: آرایه 2D با شکل (N, 3): [x, y, type]
    """
    # باینری کردن
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # اسکلت‌سازی
    skeleton = skeletonize(binary > 127).astype(np.uint8) * 255

    minutiae = []
    h, w = skeleton.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] == 255:
                neighbors = [
                    skeleton[y-1, x-1], skeleton[y-1, x], skeleton[y-1, x+1],
                    skeleton[y,   x-1],                   skeleton[y,   x+1],
                    skeleton[y+1, x-1], skeleton[y+1, x], skeleton[y+1, x+1]
                ]
                count = sum(1 for n in neighbors if n == 255)
                if count == 1:
                    minutiae.append([x, y, 0])  # 0 = ending
                elif count == 3:
                    minutiae.append([x, y, 1])  # 1 = bifurcation

    return np.array(minutiae, dtype=float)  # خروجی: (N, 3)