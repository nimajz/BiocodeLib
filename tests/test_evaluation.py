import pytest
import numpy as np
from biocodelib.evaluation import compare_algorithms

def test_compare_algorithms():
    features = np.random.rand(128)
    results, best = compare_algorithms(features)
    assert len(results) == 3
    assert best in ["BioHashing", "IoM Hashing", "XOR Encryption"]