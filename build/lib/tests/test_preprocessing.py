import pytest
import cv2
import numpy as np
from biocodelib.preprocessing import preprocess_image

def test_preprocess_image():
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    processed = preprocess_image(img)
    assert processed.shape == (500, 500)