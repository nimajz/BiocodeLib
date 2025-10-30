import numpy as np

def apply_biohashing(features, key, m=128):
    """
    Apply BioHashing: Random projection and binarization.
    - features: Flattened feature vector.
    - key: Random vector for projection.
    - m: Length of the output binary code.
    """
    n = len(features)
    np.random.seed(hash(tuple(key)))
    projection_matrix = np.random.uniform(-0.5, 0.5, (n, m))
    projected = np.dot(features, projection_matrix)
    binary_code = (projected >= 0).astype(int)
    return binary_code