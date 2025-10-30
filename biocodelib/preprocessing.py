import cv2
import numpy as np

def preprocess_image(image, target_size=(500, 500)):
    """
    Preprocess biometric image.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    resized = cv2.resize(gray, target_size)
    equalized = cv2.equalizeHist(resized)
    denoised = cv2.GaussianBlur(equalized, (5, 5), 0)
    return denoised