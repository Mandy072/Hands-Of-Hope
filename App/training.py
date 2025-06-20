from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import models, layers
import numpy as np
import os
import yaml
import json


def load_data(directory):
    data = []
    labels = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".yml"):
            with open(os.path.join(directory, filename), 'r') as yaml_file:
                hand_data = yaml.safe_load(yaml_file)
                landmarks = [point['x'] for point in hand_data['hand_landmarks']]
                data.append(landmarks)
                labels.append(directory.split("/")[-1])  # Assuming the class label is the last part of the directory

    return np.array(data), labels

def train_classes():
    try:
        signs_dir = "Dataset/Signs"
        classes = os.listdir(signs_dir)
        all_data = []
        all_labels = []

        # Convert list to dictionary
        my_dict = {i: classes[i] for i in range(len(classes))}

        # Save the dictionary to a JSON file
        with open('Dataset/labels_info.json', 'w') as json_file:
            json.dump(my_dict, json_file)

        for sign_class in classes:
            data, labels = load_data(os.path.join(signs_dir, sign_class))
            all_data.extend(data)
            all_labels.extend(labels)
            
        # Convert labels to integers
            
        label_to_int = {label: i for i, label in enumerate(set(all_labels))}
        all_labels = [label_to_int[label] for label in all_labels]

        print(all_labels)
        # Convert data to numpy array
        X_data = np.array(all_data)
        y_labels = np.array(all_labels)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_labels, test_size=0.2, random_state=42)

        # Build the model
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(set(all_labels)), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

        # Evaluate the model
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_acc}")

        model.save('Dataset/sign_language_model.h5')

        label_to_class_name = {label_to_int[label]: label for label in label_to_int}
        with open('Dataset/label_mapping.json', 'w') as json_file:
            json.dump(label_to_class_name, json_file)
        
        return "Success"

    except Exception as e:
        return f"Error: {str(e)}"