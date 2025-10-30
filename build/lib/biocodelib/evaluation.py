import time
import numpy as np
from scipy.stats import entropy
from .algorithms import apply_biohashing, apply_xor_encryption, apply_iom_hashing
from .algorithms.encryption_utils import generate_random_key

def evaluate_algorithm(alg_func, features, key, name):
    start = time.time()
    code = alg_func(features, key)
    runtime = time.time() - start
    # Simulated security: Entropy as proxy for non-invertibility (higher = better)
    sec_score = entropy(code + 1e-10) / np.log2(len(code)) if len(code) > 0 else 0
    return {
        "name": name,
        "code": code,
        "code_length": len(code),
        "runtime": runtime,
        "security_score": sec_score
    }

def compare_algorithms(features):
    results = []
    key = generate_random_key(128)  # Common key for fairness

    # BioHashing
    results.append(evaluate_algorithm(apply_biohashing, features, key, "BioHashing"))

    # IoM Hashing
    results.append(evaluate_algorithm(apply_iom_hashing, features, key, "IoM Hashing"))

    # XOR Encryption (binarize features first)
    bin_features = (features > np.mean(features)).astype(int)
    results.append(evaluate_algorithm(apply_xor_encryption, bin_features, key, "XOR Encryption"))

    # Select best: Highest security, then lowest length + runtime
    results.sort(key=lambda x: (-x["security_score"], x["code_length"] + x["runtime"] * 1000))
    best = results[0]["name"]
    return results, best