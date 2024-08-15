from ml_Project.components.dcnn_model import ModelTrainer
from ml_Project import logger

STAGE_NAME = "CNN Model Training"

train_data_path = "Dataset/Data_transformation/train.csv"
test_data_path = "Dataset/Data_transformation/test.csv"
target_col = "Diabetes_012"
dir_path = "Dataset/Model_selection"

class CNNModelPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            trainer = ModelTrainer(train_data_path,
                                   test_data_path,
                                   target_col,
                                   dir_path)
            
            trainer.train()
        except Exception as e:
            raise e
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> Stage: {STAGE_NAME} started <<<<<<<<<")
        obj = CNNModelPipeline()
        obj.main()
        logger.info(f">>>>>>>> Stage: {STAGE_NAME} completed <<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e