from ml_Project import logger
from ml_Project.components.mlp_model2 import ModelTrainer

STAGE_NAME = "2nd MLP Model Training"

train_data_path = "Dataset/Data_transformation/train.csv"
test_data_path = "Dataset/Data_transformation/test.csv"
target_col = "Diabetes_012"
dir_path = "Model_weights/Model_selection"
history_filepath = "History"

class MLPModel_2_Pipeline:
    def __init__(self):
        pass
    def main(self):
        try:
            mlp_2_trainer = ModelTrainer(train_data_path,
                                         test_data_path,
                                         target_col,
                                         dir_path,
                                         history_filepath)
            mlp_2_trainer.train()
        except Exception as e:
            raise e
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> Stage: {STAGE_NAME} started <<<<<<<<<")
        obj = MLPModel_2_Pipeline()
        obj.main()
        logger.info(f">>>>>>>> Stage: {STAGE_NAME} completed <<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e