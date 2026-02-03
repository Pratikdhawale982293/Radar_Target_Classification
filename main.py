import os
import numpy as np

from src.load_data import load_radar_data
from src.preprocess import preprocess_data
from src.train_model import build_cnn_model

from sklearn.metrics import classification_report, confusion_matrix


# --------------------------------------------------
# Step 1: Load Radar Dataset (limited samples for testing)
# --------------------------------------------------
X, y = load_radar_data("data", max_files_per_class=1500)

print("Raw data shape:", X.shape)
print("Labels shape:", y.shape)
print("Unique labels:", np.unique(y))


# --------------------------------------------------
# Step 2: Preprocess Data
# --------------------------------------------------
X = preprocess_data(X)

print("Preprocessed data shape:", X.shape)
print("Data range:", X.min(), "to", X.max())


# --------------------------------------------------
# Step 3: Build CNN Model
# --------------------------------------------------
input_shape = X.shape[1:]   # (height, width, channels)
num_classes = 3

model = build_cnn_model(input_shape, num_classes)

print("\nModel Summary:")
model.summary()


# --------------------------------------------------
# Step 4: Train Model
# --------------------------------------------------
history = model.fit(
    X,
    y,
    epochs=60,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# --------------------------------------------------
# Step 5: Model Evaluation
# --------------------------------------------------
y_pred = model.predict(X)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(
    classification_report(
        y,
        y_pred_classes,
        target_names=["Cars", "Drones", "People"]
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y, y_pred_classes))


# --------------------------------------------------
# End of main.py
# --------------------------------------------------
# --------------------------------------------------
# Save trained model for manual testing
# --------------------------------------------------
model.save("radar_cnn_model.h5")
print("Model saved as radar_cnn_model.h5")
