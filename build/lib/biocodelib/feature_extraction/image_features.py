import cv2
import numpy as np

def extract_image_features(image):
    """
    Extract general features using SIFT.
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors.flatten() if descriptors is not None else np.array([])