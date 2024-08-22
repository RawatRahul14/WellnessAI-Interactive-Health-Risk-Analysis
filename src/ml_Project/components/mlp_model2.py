import numpy as np
import pandas as pd
import json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, add
from tensorflow.keras.utils import to_categorical
from ml_Project.utils.common import create_directories
import os

class ModelTrainer:
    def __init__(self, train_data_path, test_data_path, target_col, dir_path, history_filepath):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.target_col = target_col
        self.dir_path = dir_path
        self.history_filepath = history_filepath

    def create_dir(self):
        create_directories([self.dir_path])
        create_directories([self.history_filepath])

    def load_data(self):
        train_data = pd.read_csv(self.train_data_path)
        test_data = pd.read_csv(self.test_data_path)
        return train_data, test_data

    def preprocess_data(self, train_data, test_data):
        X_train = train_data.drop([self.target_col], axis = 1)
        X_test = test_data.drop([self.target_col], axis = 1)
        y_train = train_data[[self.target_col]]
        y_test = test_data[[self.target_col]]

        # One-hot encode the target variable
        y_train = to_categorical(y_train, num_classes=3)
        y_test = to_categorical(y_test, num_classes=3)

        return X_train, X_test, y_train, y_test

    def build_custom_model(self, input_shape):
        # Input layer
        inputs = Input(shape=input_shape)

        # Layer 1
        x1 = Dense(256, activation = "relu")(inputs)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.4)(x1)

        # Layer 2
        x2 = Dense(256, activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.5)(x2)
        x2 = add([x2, x1])  # Skip connection

        # Hidden Layer 3
        x3 = Dense(256, activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = Dropout(0.5)(x3)
        
        # Hidden Layer 4 with Skip Connection
        x4 = Dense(256, activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = Dropout(0.5)(x4)
        x4 = add([x4, x2])  # Skip connection
        
        # Fully connected layers
        x = Dense(256, activation = "relu")(x4)
        outputs = Dense(3, activation = "softmax")(x)

        # Build model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

        return model

    def train(self):

        self.create_dir()

        # Load data
        train_data, test_data = self.load_data()

        # Preprocess data
        X_train, X_test, y_train, y_test = self.preprocess_data(train_data, test_data)

        # Build and train model
        model = self.build_custom_model((X_train.shape[1], ))
        history = model.fit(X_train, y_train, epochs = 25, batch_size = 128, validation_data = (X_test, y_test))

        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Model Accuracy: {accuracy:.4f}")

        # Save model weights
        model.save_weights(self.dir_path + "/model_2_mlp.h5")

        # Save history
        history_dict = history.history
        history_path = os.path.join(self.history_filepath, "training_history_model_2.json")
        with open(history_path, 'w') as f:
            json.dump(history_dict, f)

        print(f"Training history saved to {history_path}")