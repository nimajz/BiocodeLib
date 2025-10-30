import cv2
import numpy as np
from skimage.morphology import skeletonize

def extract_minutiae(image):
    """
    Extract minutiae points from fingerprint image.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    skeleton = skeletonize(binary // 255).astype(np.uint8) * 255
    minutiae = []
    rows, cols = skeleton.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if skeleton[i, j] == 255:
                neighbors = np.sum(skeleton[i-1:i+2, j-1:j+2]) // 255 - 1
                if neighbors == 1 or neighbors == 3:
                    minutiae.append((j, i))
    return np.array(minutiae).flatten() if minutiae else np.array([])