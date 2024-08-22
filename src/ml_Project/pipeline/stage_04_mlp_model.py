from ml_Project import logger
from ml_Project.components.mlp_model import ModelTrainer

STAGE_NAME = "MLP Model Training"

train_data_path = "Dataset/Data_transformation/train.csv"
test_data_path = "Dataset/Data_transformation/test.csv"
target_col = "Diabetes_012"
dir_path = "Model_weights/Model_selection"
history_filepath = "History"

class MLPModelPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            mlp_trainer = ModelTrainer(train_data_path,
                                       test_data_path,
                                       target_col,
                                       dir_path)
            
            mlp_trainer.train()
        except Exception as e:
            raise e
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> Stage: {STAGE_NAME} started <<<<<<<<<")
        obj = MLPModelPipeline()
        obj.main()
        logger.info(f">>>>>>>> Stage: {STAGE_NAME} completed <<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e