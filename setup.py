from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='biocodelib',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'scikit-image',
        'scipy',
    ],
    python_requires='>=3.7',
    description='A unified Python library for converting biometric images to secure codes using classical algorithms like BioHashing, IoM Hashing, and XOR encryption.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Nima jzzz',
    author_email='nimajaberzadeh@gmail.com',
    url='https://github.com/nimajz/BiocodeLib',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Security',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    keywords='biometric, fingerprint, hashing, encryption, biohashing, iom-hashing, security',
    project_urls={
        'Bug Reports': 'https://github.com/nimajz/BiocodeLib/issues',
        'Source': 'https://github.com/nimajz/BiocodeLib',
        'Documentation': 'https://github.com/nimajz/BiocodeLib#readme',
    },
)