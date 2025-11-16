import numpy as np
import hashlib

def apply_iom_hashing(features, key, groups=32, group_size=4):
    """
    Index-of-Maximum (IoM) Hashing with random sparse encoding (inspired by RSBE-IoM).
    - features: Flattened feature vector.
    - key: Random key for permutation.
    - groups: Number of groups.
    - group_size: Size of each group.
    
    Divides features into groups, finds index of max in each, encodes sparsely.
    """
    # Ensure we have enough features for the requested groups and group_size
    required_size = groups * group_size
    if len(features) < required_size:
        # Dynamically adjust groups to fit available features
        groups = len(features) // group_size
        required_size = groups * group_size
        if groups == 0:
            raise ValueError(f"Feature vector too small: need at least {group_size} elements, got {len(features)}")
    
    # Derive a stable 32-bit seed from the key
    key_arr = np.asarray(key, dtype=np.uint8)
    digest = hashlib.blake2b(key_arr.tobytes(), digest_size=8).digest()
    seed = int.from_bytes(digest, 'little') % (2**32)
    np.random.seed(seed)
    permuted = features[np.random.permutation(len(features))]
    reshaped = permuted[:required_size].reshape(groups, group_size)
    indices = np.argmax(reshaped, axis=1)
    # Sparse binary encoding (one-hot like, but concatenated)
    code = np.zeros(groups * group_size, dtype=int)
    for i, idx in enumerate(indices):
        code[i * group_size + idx] = 1
    return code