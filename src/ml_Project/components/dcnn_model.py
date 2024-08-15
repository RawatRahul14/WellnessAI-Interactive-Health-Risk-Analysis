import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from ml_Project import logger
from ml_Project.utils.common import create_directories

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
        X_train = train_data.drop([self.target_col], axis=1)
        X_test = test_data.drop([self.target_col], axis=1)
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

        x = Conv2D(32, (1, 1), activation = "relu", padding = "same")(inputs)
        x = Conv2D(64, (1, 1), activation = "relu", padding = "same")(x)
        x = Conv2D(128, (1, 1), activation = "relu", padding = "same")(x)

        x = Flatten()(x)

        x = Dense(128, activation = "relu")(x)
        outputs = Dense(3, activation = "softmax")(x)

        # Build model
        model = Model(inputs = inputs, outputs = outputs)
        model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

        return model

    def train(self):
        self.create_dir()

        # Load data
        train_data, test_data = self.load_data()

        # Preprocess data
        X_train, X_test, y_train, y_test = self.preprocess_data(train_data, test_data)

        X_train = X_train.values.reshape(-1, X_train.shape[1], 1, 1)
        X_test = X_test.values.reshape(-1, X_test.shape[1], 1, 1)

        # Build and train model
        model = self.build_custom_model(X_train.shape[1:])
        model.fit(X_train, y_train, epochs = 10, batch_size = 128, validation_data = (X_test, y_test))

        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test)
        logger.info(f"Model Accuracy: {accuracy:.4f}")

        # Save model weights
        model.save_weights(f"{self.dir_path}/model_cnn.h5")
        logger.info("Model saved.")