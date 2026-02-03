import numpy as np

def preprocess_data(X):
    # Convert to float32 (required for deep learning)
    X = X.astype(np.float32)

    # Normalize values between 0 and 1
    X_min = X.min()
    X_max = X.max()
    X = (X - X_min) / (X_max - X_min + 1e-8)

    # Add channel dimension (for CNN)
    X = X[..., np.newaxis]

    return X
