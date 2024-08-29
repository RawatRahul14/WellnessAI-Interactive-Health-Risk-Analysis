import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from ml_Project.utils.common import create_directories
from ml_Project import logger

class DataTransformation:
    def __init__(self, data_transform_dir, data_path):
        self.data_transform_dir = data_transform_dir
        self.data_path = data_path
    
    def create_dir(self):
        create_directories([self.data_transform_dir])

    def transform_data(self):
        data = pd.read_csv(self.data_path)

        data = data.drop_duplicates()

        col_drops = ["Income", "Education", "AnyHealthcare"]

        logger.info("Removed the duplicated data.")

        data.drop(columns = col_drops, axis = 1, inplace = True)

        logger.info("Removed the unnecessary columns.")

        X = data.drop("Diabetes_012", axis = 1)
        y = data["Diabetes_012"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
        smote = SMOTE(random_state = 42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        logger.info("Resampled the data to balance the Labels.")

        cols = ["HighBP", "HighChol", "Sex", "DiffWalk", "Veggies", 
                "Fruits", "HeartDiseaseorAttack", "Stroke"]
        for col in cols:
            X_resampled[col] = (X_resampled[col] >= 0.5).astype(float)

        merged_df = pd.concat([X_resampled, pd.DataFrame(y_resampled)], axis = 1)

        train, test = train_test_split(merged_df, test_size = 0.2, random_state = 42)

        train.to_csv(os.path.join(self.data_transform_dir, "train.csv"), index = False)
        test.to_csv(os.path.join(self.data_transform_dir, "test.csv"), index = False)
        logger.info("Splitted data into train and test sets.")