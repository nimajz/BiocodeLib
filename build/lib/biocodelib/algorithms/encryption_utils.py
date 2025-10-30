import numpy as np

def generate_random_key(length=128):
    """Generate a random binary key."""
    return np.random.randint(0, 2, size=length)