from ml_Project import logger
from ml_Project.components.data_transformation import DataTransformation

STAGE_NAME = "Data Transformation"

data_transform_dir = "Dataset/Data_transformation"
data_path = "Dataset/Data_Ingestion/diabetes_data.csv"

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            data_transformation = DataTransformation(data_transform_dir = data_transform_dir,
                                                    data_path = data_path)
            
            data_transformation.create_dir()
            data_transformation.transform_data()
        except Exception as e:
            raise e
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> Stage: {STAGE_NAME} started <<<<<<<<<")
        obj = DataTransformationPipeline()
        obj.main()
        logger.info(f">>>>>>>> Stage: {STAGE_NAME} completed <<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e