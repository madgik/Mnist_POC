import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import os
from PIL import Image
import glob


data_locations = [
    "imaging_data/MNIST/training/node1",
    "imaging_data/MNIST/training/node2",
]

def load_and_process_images(folder, target_size=(28, 28)):
    images = []
    labels = []
    for folder_name in os.listdir(folder):
        folder_path = os.path.join(folder, folder_name)
        if os.path.isdir(folder_path):
            for file_path in glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png")) + glob.glob(os.path.join(folder_path, "*.jpeg")):
                try:
                    img = Image.open(file_path).convert("L").resize(target_size)
                    images.append(np.array(img))
                    labels.append(folder_name)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    return np.array(images), np.array(labels)

def get_all_data(data_locations):
    all_data = [load_and_process_images(loc) for loc in data_locations]
    images = np.concatenate([d[0] for d in all_data])
    labels = np.concatenate([d[1] for d in all_data])
    return images, labels

images, labels = get_all_data(data_locations)
images = images.reshape(images.shape[0], -1) / 255.0  # Flatten and normalize
labels = labels.astype('int')

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def split_data_among_clients(X, y, num_clients):
    split_size = len(X) // num_clients
    return [(X[i*split_size:(i+1)*split_size], y[i*split_size:(i+1)*split_size]) for i in range(num_clients)]

def train_local_model(X, y):
    model = LogisticRegression(solver="saga", penalty="l2", max_iter=1000, warm_start=True)
    model.fit(X, y)
    return model

def federated_averaging(models):
    global_model = LogisticRegression(solver="saga", penalty="l2", max_iter=1000, warm_start=True)
    global_model.coef_ = np.mean([model.coef_ for model in models], axis=0)
    global_model.intercept_ = np.mean([model.intercept_ for model in models], axis=0)
    return global_model

clients_data = split_data_among_clients(X_train_scaled, y_train, num_clients=5)

for round in range(5):
    local_models = [train_local_model(X, y) for X, y in clients_data]
    global_model = federated_averaging(local_models)
    global_model.classes_ = np.unique(y_train)

    y_pred = global_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Round {round + 1}: Global Model Test Accuracy: {accuracy}")

final_y_pred = global_model.predict(X_test_scaled)
final_accuracy = accuracy_score(y_test, final_y_pred)
print(f"Final Global Model Test Accuracy: {final_accuracy}")
