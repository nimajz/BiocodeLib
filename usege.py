# usege.py
import cv2
import numpy as np
from biocodelib.preprocessing import preprocess_image
from biocodelib.feature_extraction.minutiae_extraction import extract_minutiae
from biocodelib.evaluation import compare_algorithms

print("چارچوب یکپارچه تبدیل تصویر بیومتریک به کد")
print("="*70)

# 1. بارگذاری تصویر
img = cv2.imread("fingerprint.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("تصویر پیدا نشد → شبیه‌سازی")
    img = np.random.randint(0, 255, (600, 600), dtype=np.uint8)
else:
    print(f"تصویر بارگذاری شد: {img.shape}")

# 2. پیش‌پردازش (مطابق PDF)
preprocessed = preprocess_image(img)
print("پیش‌پردازش: 500x500 + نرمال‌سازی + حذف نویز")

# 3. Minutiae (مطابق PDF)
minutiae = extract_minutiae(preprocessed)
print(f"تعداد Minutiae: {len(minutiae)}")

if len(minutiae) == 0:
    minutiae = np.array([[100, 100, 'ending'], [300, 300, 'bifurcation']])

features = minutiae[:, :2].astype(float).flatten()

# 4. مقایسه الگوریتم‌ها
print("\n" + "="*70)
print("مقایسه الگوریتم‌های کلاسیک")
print("="*70)

results, best = compare_algorithms(features)

for r in results:
    code_str = ''.join(map(str, r['code'][:30]))
    print(f"\n{r['name']}")
    print(f"   کد: {code_str}... (طول: {r['length']})")
    print(f"   زمان: {r['runtime']:.5f}s")
    print(f"   امنیت: {r['security']:.3f}")

print("\n" + "="*70)
print(f"بهترین الگوریتم: {best}")
print("="*70)