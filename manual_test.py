import numpy as np
from tensorflow.keras.models import load_model
from src.preprocess import preprocess_data


# ---------------------------------
# Load trained CNN model
# ---------------------------------
model = load_model("radar_cnn_model.h5")



# ---------------------------------
# Manually select one radar CSV file
# ---------------------------------
file_path = r"data\Cars\15-55m\011.csv"   # change file path as needed

sample = np.genfromtxt(file_path, delimiter=",")

# Add batch dimension
sample = sample[np.newaxis, ...]

# Preprocess sample
sample = preprocess_data(sample)


# ---------------------------------
# Predict target class
# ---------------------------------
prediction = model.predict(sample)
predicted_class = np.argmax(prediction)

class_map = {
    0: "Cars",
    1: "Drones",
    2: "People"
}

print("Predicted Target:", class_map[predicted_class])
print("Prediction Confidence:", prediction)
