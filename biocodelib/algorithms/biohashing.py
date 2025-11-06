import numpy as np
import hashlib

def apply_biohashing(features, key, m=128):
    """
    Apply BioHashing: Random projection and binarization.
    - features: Flattened feature vector.
    - key: Random vector for projection.
    - m: Length of the output binary code.
    """
    n = len(features)
    # Derive a stable 32-bit seed from the key
    key_arr = np.asarray(key, dtype=np.uint8)
    digest = hashlib.blake2b(key_arr.tobytes(), digest_size=8).digest()
    seed = int.from_bytes(digest, 'little') % (2**32)
    np.random.seed(seed)
    projection_matrix = np.random.uniform(-0.5, 0.5, (n, m))
    projected = np.dot(features, projection_matrix)
    binary_code = (projected >= 0).astype(int)
    return binary_code