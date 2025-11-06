# usege.py
import cv2
import numpy as np
from biocodelib.preprocessing import preprocess_image
from biocodelib.feature_extraction.minutiae_extraction import extract_minutiae
from biocodelib.evaluation import compare_algorithms

print("Unified Framework for Converting Biometric Images to Code")
print("="*70)

# 1. Load image
img = cv2.imread("fingerprint.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Image not found → simulating")
    img = np.random.randint(0, 255, (600, 600), dtype=np.uint8)
else:
    print(f"Image loaded: {img.shape}")

# 2. Preprocessing
preprocessed = preprocess_image(img)
print("Preprocessing: 500x500 + normalization + noise removal")

# 3. Minutiae
minutiae = extract_minutiae(preprocessed)
minutiae = np.asarray(minutiae)
print(f"Number of Minutiae: {len(minutiae)}")

if len(minutiae) == 0:
    print("Warning: No Minutiae → simulating data")
    minutiae = np.array([[100, 100, 0], [300, 300, 1]], dtype=float)

# 4. Numerical features (only x, y)
# Compatible with different extract_minutiae output formats
def _to_xy_array(minu):
    # If standard 2D array
    if isinstance(minu, np.ndarray) and minu.ndim == 2 and minu.shape[1] >= 2:
        return minu[:, :2].astype(float)

    # Structured array with fields
    if isinstance(minu, np.ndarray) and getattr(minu.dtype, 'names', None):
        field_names = minu.dtype.names
        for kx, ky in (("x", "y"), ("X", "Y"), ("col", "row"), ("cx", "cy"), ("j", "i")):
            if kx in field_names and ky in field_names:
                return np.vstack([minu[kx], minu[ky]]).T.astype(float)

    # If object-array/1D list (tuple/list/dictionary)
    if isinstance(minu, np.ndarray) and minu.ndim == 1:
        items = minu.tolist()
    else:
        items = minu

    if not items:
        return np.empty((0, 2), dtype=float)

    first = items[0]

    # Dictionary with x,y keys
    if isinstance(first, dict):
        if all(k in first for k in ("x", "y")):
            return np.array([[it["x"], it["y"]] for it in items], dtype=float)
        # Second attempt for alternative keys
        for kx, ky in (("X", "Y"), ("col", "row")):
            if kx in first and ky in first:
                return np.array([[it[kx], it[ky]] for it in items], dtype=float)
        raise ValueError("Minutiae structure is dictionary but x,y keys not found.")

    # Object with x,y attributes
    if hasattr(first, 'x') and hasattr(first, 'y'):
        return np.array([[getattr(it, 'x'), getattr(it, 'y')] for it in items], dtype=float)
    if hasattr(first, 'X') and hasattr(first, 'Y'):
        return np.array([[getattr(it, 'X'), getattr(it, 'Y')] for it in items], dtype=float)

    # Sequences/tuples with length >= 2
    if hasattr(first, "__len__") and len(first) >= 2:
        return np.array([[it[0], it[1]] for it in items], dtype=float)

    # Flat numeric array/list with even length
    try:
        arr = np.asarray(items, dtype=float)
        if arr.ndim == 1 and arr.size % 2 == 0:
            return arr.reshape(-1, 2)
    except Exception:
        pass

    raise ValueError("Unknown Minutiae structure; expected list of (x,y,...) or dictionaries.")

xy = _to_xy_array(minutiae)
features = xy.flatten()
print(f"Numerical features: {features.shape}")

# 5. Comparison
print("\n" + "="*70)
print("Classical Algorithm Comparison")
print("="*70)

results, best = compare_algorithms(features)

for r in results:
    code_str = ''.join(map(str, r['code'][:30]))
    print(f"\n{r['name']}")
    print(f"   Code: {code_str}... (length: {r['code_length']})")
    print(f"   Time: {r['runtime']:.5f}s")
    print(f"   Security: {r['security_score']:.3f}")

print("\n" + "="*70)
print(f"Best algorithm: {best}")
print("="*70)