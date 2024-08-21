import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.utils import to_categorical
from ml_Project import logger
from ml_Project.utils.common import create_directories
import json
import os

class ModelTrainer:
    def __init__(self, train_data_path, test_data_path, target_col, dir_path):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.target_col = target_col
        self.dir_path = dir_path

    def create_dir(self):
        create_directories([self.dir_path])

    def load_data(self):
        train_data = pd.read_csv(self.train_data_path)
        test_data = pd.read_csv(self.test_data_path)
        logger.info("Data is loaded.")
        return train_data, test_data

    def preprocess_data(self, train_data, test_data):
        X_train = train_data.drop([self.target_col], axis = 1)
        X_test = test_data.drop([self.target_col], axis = 1)
        y_train = train_data[[self.target_col]]
        y_test = test_data[[self.target_col]]

        # One-hot encode the target variable
        y_train = to_categorical(y_train, num_classes=3)
        y_test = to_categorical(y_test, num_classes=3)

        logger.info("Preprocessing done.")
        return X_train, X_test, y_train, y_test

    def build_custom_model(self, input_shape):
        # Input layer
        inputs = Input(shape=input_shape)

        # Branch 1
        x1 = Dense(128, activation = "relu")(inputs)
        x1 = Dense(128, activation = "relu")(x1)
        x1 = Dense(64, activation = "relu")(x1)

        # Branch 2
        x2 = Dense(128, activation = "relu")(inputs)
        x2 = Dense(128, activation = "relu")(x2)
        x2 = Dense(64, activation = "relu")(x2)

        # Branch 3
        x3 = Dense(128, activation = "relu")(inputs)
        x3 = Dense(128, activation = "relu")(x3)
        x3 = Dense(64, activation = "relu")(x3)

        # Branch 4
        x4 = Dense(128, activation = "relu")(inputs)
        x4 = Dense(128, activation = "relu")(x4)
        x4 = Dense(64, activation = "relu")(x4)

        # Combine branches
        combined = Concatenate()([x1, x2, x3, x4])

        # Fully connected layers
        x = Dense(256, activation = "relu")(combined)
        x = Dense(256, activation = "relu")(x)
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
        logger.info(f"Model Accuracy: {accuracy:.4f}")

        # Save model weights
        model.save_weights("Model_weights/Model_selection/model_mlp.h5")
        logger.info("Model saved.")

        # Save history
        history_dict = history.history
        history_path = os.path.join(self.history_filepath, "training_history_model.json")
        with open(history_path, 'w') as f:
            json.dump(history_dict, f)

        print(f"Training history saved to {history_path}")