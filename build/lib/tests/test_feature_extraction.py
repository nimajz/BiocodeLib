import pytest
import numpy as np
from biocodelib.feature_extraction.minutiae_extraction import extract_minutiae

def test_extract_minutiae():
    img = np.zeros((100, 100), dtype=np.uint8)
    minutiae = extract_minutiae(img)
    assert len(minutiae) == 0