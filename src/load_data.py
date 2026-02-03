import os
import numpy as np


def load_radar_data(base_path, max_files_per_class=1500):
    data = []
    labels = []

    class_map = {
        "Cars": 0,
        "Drones": 1,
        "People": 2
    }

    for class_name, label in class_map.items():
        class_path = os.path.join(base_path, class_name)
        file_count = 0

        for root, _, files in os.walk(class_path):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    try:
                        matrix = np.loadtxt(file_path, delimiter=",")
                        data.append(matrix)
                        labels.append(label)
                        file_count += 1
                    except:
                        continue

                    if file_count >= max_files_per_class:
                        break
            if file_count >= max_files_per_class:
                break

        print(f"{class_name}: Loaded {file_count} samples")

    return np.array(data), np.array(labels)
