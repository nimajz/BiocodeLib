import time
import numpy as np
from scipy.stats import entropy
from .algorithms import apply_biohashing, apply_xor_encryption, apply_iom_hashing
from .algorithms.encryption_utils import generate_random_key

def evaluate_algorithm(alg_func, features, key, name):
    """
    Evaluate an algorithm based on multiple criteria.
    
    Metrics:
    - Security Score: Normalized entropy (0-1), higher is better
      Measures randomness/non-invertibility of the code
    - Code Length: Number of bits in output code, lower is better
    - Runtime: Execution time in seconds, lower is better
    
    Returns:
        Dictionary with algorithm metrics and scores
    """
    start = time.time()
    code = alg_func(features, key)
    runtime = time.time() - start
    # Security: Normalized entropy as proxy for non-invertibility (0-1, higher = better)
    # Calculate entropy: H = -Σ[p(i) × log2(p(i))]
    # For binary codes: max entropy = log2(2) = 1.0
    if len(code) > 0:
        # Get unique values and their counts
        unique_vals, counts = np.unique(code, return_counts=True)
        # Calculate probabilities
        probs = counts / len(code)
        # Calculate entropy in base 2
        ent = entropy(probs, base=2)
        # Normalize by max entropy for binary (log2(2) = 1.0)
        # This gives us a score between 0 (predictable) and 1 (random)
        max_entropy_binary = np.log2(len(unique_vals)) if len(unique_vals) > 1 else 1.0
        sec_score = ent / max_entropy_binary if max_entropy_binary > 0 else 0
    else:
        sec_score = 0
    return {
        "name": name,
        "code": code,
        "code_length": len(code),
        "runtime": runtime,
        "security_score": sec_score
    }

def compare_algorithms(features, security_weight=0.5, efficiency_weight=0.3, compactness_weight=0.2):
    """
    Compare multiple algorithms and select the best one.
    
    Selection Criteria (Weighted Scoring):
    --------------------------------------
    1. Security Score (weight: security_weight, default: 0.5 = 50%)
       - Normalized entropy (0-1)
       - Higher is better
       - Impact: HIGHEST - Security is the primary concern
    
    2. Efficiency Score (weight: efficiency_weight, default: 0.3 = 30%)
       - Based on runtime (inverse, normalized)
       - Lower runtime = higher score
       - Impact: MEDIUM - Performance matters for real-time applications
    
    3. Compactness Score (weight: compactness_weight, default: 0.2 = 20%)
       - Based on code length (inverse, normalized)
       - Shorter codes = higher score
       - Impact: LOW-MEDIUM - Storage/bandwidth optimization
    
    Total Score = (security * security_weight) + 
                  (efficiency * efficiency_weight) + 
                  (compactness * compactness_weight)
    
    The algorithm with the HIGHEST total score is selected as best.
    
    Args:
        features: Input feature vector
        security_weight: Weight for security score (default: 0.5)
        efficiency_weight: Weight for efficiency/runtime (default: 0.3)
        compactness_weight: Weight for code length (default: 0.2)
    
    Returns:
        results: List of algorithm results with scores
        best: Name of the best algorithm
    """
    results = []
    key = generate_random_key(128)  # Common key for fairness

    # Evaluate all algorithms
    results.append(evaluate_algorithm(apply_biohashing, features, key, "BioHashing"))
    results.append(evaluate_algorithm(apply_iom_hashing, features, key, "IoM Hashing"))
    
    # XOR Encryption (binarize features first)
    bin_features = (features > np.mean(features)).astype(int)
    results.append(evaluate_algorithm(apply_xor_encryption, bin_features, key, "XOR Encryption"))

    # Normalize scores (0-1 scale, higher is better for all)
    security_scores = [r["security_score"] for r in results]
    runtimes = [r["runtime"] for r in results]
    code_lengths = [r["code_length"] for r in results]
    
    # Normalize security (already 0-1, but ensure no division by zero)
    max_sec = max(security_scores) if max(security_scores) > 0 else 1
    norm_security = [s / max_sec for s in security_scores]
    
    # Normalize efficiency (inverse of runtime - lower runtime = higher score)
    max_runtime = max(runtimes) if max(runtimes) > 0 else 1
    min_runtime = min(runtimes) if min(runtimes) > 0 else max_runtime
    if max_runtime == min_runtime:
        norm_efficiency = [1.0] * len(runtimes)
    else:
        # Linear normalization: fastest gets 1.0, slowest gets 0.0
        norm_efficiency = [1.0 - (rt - min_runtime) / (max_runtime - min_runtime) for rt in runtimes]
    
    # Normalize compactness (inverse of length - shorter codes = higher score)
    max_length = max(code_lengths) if max(code_lengths) > 0 else 1
    min_length = min(code_lengths) if min(code_lengths) > 0 else max_length
    if max_length == min_length:
        norm_compactness = [1.0] * len(code_lengths)
    else:
        # Linear normalization: shortest gets 1.0, longest gets 0.0
        norm_compactness = [1.0 - (cl - min_length) / (max_length - min_length) for cl in code_lengths]
    
    # Calculate weighted total scores
    for i, r in enumerate(results):
        total_score = (norm_security[i] * security_weight +
                      norm_efficiency[i] * efficiency_weight +
                      norm_compactness[i] * compactness_weight)
        
        r["normalized_security"] = norm_security[i]
        r["normalized_efficiency"] = norm_efficiency[i]
        r["normalized_compactness"] = norm_compactness[i]
        r["total_score"] = total_score
        r["security_contribution"] = norm_security[i] * security_weight
        r["efficiency_contribution"] = norm_efficiency[i] * efficiency_weight
        r["compactness_contribution"] = norm_compactness[i] * compactness_weight
    
    # Sort by total score (highest first)
    results.sort(key=lambda x: -x["total_score"])
    best = results[0]["name"]
    
    return results, best