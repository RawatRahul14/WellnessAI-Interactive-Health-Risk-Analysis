from ml_Project import logger
from ml_Project.components.random_forest_model import ModelTrainer

STAGE_NAME = "Random Forest Model Training"

train_data_path = "Dataset/Data_transformation/train.csv"
test_data_path = "Dataset/Data_transformation/test.csv"
target_col = "Diabetes_012"
dir_path = "Model_weights/Model_selection"
history_filepath = "History"

class RandomForestModel_Pipeline:
    def __init__(self):
        pass
    def main(self):
        try:
            rf_trainer = ModelTrainer(train_data_path,
                                      test_data_path,
                                      target_col,
                                      dir_path,
                                      history_filepath)
            rf_trainer.train()
        except Exception as e:
            raise e
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> Stage: {STAGE_NAME} started <<<<<<<<<")
        obj = RandomForestModel_Pipeline()
        obj.main()
        logger.info(f">>>>>>>> Stage: {STAGE_NAME} completed <<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e