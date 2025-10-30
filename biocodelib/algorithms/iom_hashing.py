import numpy as np

def apply_iom_hashing(features, key, groups=32, group_size=4):
    """
    Index-of-Maximum (IoM) Hashing with random sparse encoding (inspired by RSBE-IoM).
    - features: Flattened feature vector.
    - key: Random key for permutation.
    - groups: Number of groups.
    - group_size: Size of each group.
    
    Divides features into groups, finds index of max in each, encodes sparsely.
    """
    np.random.seed(hash(tuple(key)))
    permuted = features[np.random.permutation(len(features))]
    reshaped = permuted[:groups * group_size].reshape(groups, group_size)
    indices = np.argmax(reshaped, axis=1)
    # Sparse binary encoding (one-hot like, but concatenated)
    code = np.zeros(groups * group_size, dtype=int)
    for i, idx in enumerate(indices):
        code[i * group_size + idx] = 1
    return code