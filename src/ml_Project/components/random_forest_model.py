from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
from ml_Project import logger
from ml_Project.utils.common import create_directories
import os
import json

class ModelTrainer:
    def __init__(self, 
                 train_data_path, 
                 test_data_path, 
                 target_col, 
                 dir_path,
                 history_filepath):
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
        logger.info("Data is loaded.")
        return train_data, test_data

    def preprocess_data(self, train_data, test_data):
        X_train = train_data.drop([self.target_col], axis=1)
        X_test = test_data.drop([self.target_col], axis=1)
        y_train = train_data[self.target_col]
        y_test = test_data[self.target_col]

        logger.info("Preprocessing done.")
        return X_train, X_test, y_train, y_test

    def build_random_forest_model(self):
        # Initialize the RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        return model

    def train(self):
        self.create_dir()

        # Load data
        train_data, test_data = self.load_data()

        # Preprocess data
        X_train, X_test, y_train, y_test = self.preprocess_data(train_data, test_data)

        # Build and train the Random Forest model
        model = self.build_random_forest_model()
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Random Forest Model Accuracy: {accuracy:.4f}")

        # Save model
        model_path = os.path.join(self.dir_path, "model_rf.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Random Forest model saved to {model_path}")

        # Save training history (for Random Forest, just saving the accuracy as an example)
        history_dict = {"accuracy": accuracy}
        history_path = os.path.join(self.history_filepath, "training_history_rf.json")
        with open(history_path, 'w') as f:
            json.dump(history_dict, f)

        print(f"Training history saved to {history_path}")