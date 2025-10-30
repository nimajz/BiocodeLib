# BioCodeLib

A unified Python library for converting biometric images (e.g., fingerprints) to secure, compressed codes using classical algorithms. This library integrates traditional methods for biometric encryption, evaluation, and comparison. It supports preprocessing, feature extraction, and encryption based on methods like BioHashing, IoM Hashing (inspired by RSBE-IoM), and simple XOR encryption.

## Features
- Preprocessing: Image loading, grayscale conversion, normalization, noise removal.
- Feature Extraction: Minutiae extraction for fingerprints, general image features.
- Algorithms: BioHashing, IoM Hashing, XOR encryption.
- Evaluation: Compare algorithms based on runtime, code length, and simulated security (non-invertibility score).
- Prints output codes for each algorithm and selects the best based on criteria.
- Open-source and extensible. Deep learning models are not included as per priority.

## Installation
```bash
pip install -r requirements.txt
pip install .