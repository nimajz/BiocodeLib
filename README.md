# BioCodeLib

A unified Python library for converting biometric images (e.g., fingerprints) to secure, compressed codes using classical algorithms. This library integrates traditional methods for biometric encryption, evaluation, and comparison. It supports preprocessing, feature extraction, and encryption based on methods like BioHashing, IoM Hashing (inspired by RSBE-IoM), and simple XOR encryption.

## Features
- **Preprocessing**: Image loading, grayscale conversion, normalization, noise removal
- **Feature Extraction**: Minutiae extraction for fingerprints, general image features
- **Algorithms**: BioHashing, IoM Hashing, XOR encryption
- **Evaluation**: Compare algorithms based on runtime, code length, and simulated security (non-invertibility score)
- **Flexible**: Automatically adjusts parameters for different feature vector sizes
- Open-source and extensible. Deep learning models are not included as per priority.

## Installation

### From PyPI (when published)
```bash
pip install biocodelib
```

### From GitHub
```bash
pip install git+https://github.com/nimajz/BiocodeLib.git
```

### From source
```bash
git clone https://github.com/nimajz/BiocodeLib.git
cd BiocodeLib
pip install -r requirements.txt
pip install .
```

## Quick Start

```python
from biocodelib import preprocess_image, extract_minutiae, compare_algorithms
import numpy as np

# Load and preprocess image
image = preprocess_image('fingerprint.jpg')

# Extract features
features = extract_minutiae(image)

# Compare algorithms and get best one
results, best_algorithm = compare_algorithms(features)
print(f"Best algorithm: {best_algorithm}")
```

## Documentation

For detailed documentation and examples, see the [ALGORITHMS_GUIDE_FA.md](ALGORITHMS_GUIDE_FA.md) file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Nima jzzz - nimajaberzadeh@gmail.com

## Repository

https://github.com/nimajz/BiocodeLib