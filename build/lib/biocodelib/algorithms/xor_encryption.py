import numpy as np

def apply_xor_encryption(features, key):
    """
    Simple XOR encryption for biometric features.
    - features: Integer array (e.g., binarized).
    - key: Key of same length.
    """
    if len(features) != len(key):
        key = np.resize(key, len(features))
    return np.bitwise_xor(features.astype(int), key.astype(int))