import pytest
import numpy as np
from biocodelib.algorithms.biohashing import apply_biohashing
from biocodelib.algorithms.iom_hashing import apply_iom_hashing
from biocodelib.algorithms.xor_encryption import apply_xor_encryption

def test_apply_biohashing():
    features = np.random.rand(100)
    key = np.random.rand(100)
    code = apply_biohashing(features, key)
    assert len(code) == 128

def test_apply_iom_hashing():
    features = np.random.rand(128)
    key = np.random.rand(100)
    code = apply_iom_hashing(features, key)
    assert len(code) == 128  # 32 groups * 4

def test_apply_xor_encryption():
    features = np.random.randint(0, 2, 100)
    key = np.random.randint(0, 2, 100)
    code = apply_xor_encryption(features, key)
    assert len(code) == 100