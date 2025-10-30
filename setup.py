from setuptools import setup, find_packages

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
    description='A unified Python library for converting biometric images to secure codes using classical algorithms like BioHashing, IoM Hashing, and XOR encryption.',
    author='Nima jzzz',
    author_email='nimajaberzadeh@gmail.com',
    url='https://github.com/yourusername/biocodelib',
    license='MIT',
)