from ml_Project.components.data_ingestion import DataIngestion
from ml_Project import logger

STAGE_NAME = "Data Ingestion Stage"

data_dir = "Dataset/Data_Ingestion"
file_path = "Dataset/Data_Ingestion/data.zip"
url = "https://github.com/RawatRahul14/diabetes_dataset/raw/main/diabetes_data.zip"
unzip_filepath = "Dataset/Data_Ingestion"

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            data_ingestion = DataIngestion(data_dir = data_dir,
                                           file_path = file_path,
                                           url = url,
                                           unzip_filepath = unzip_filepath)
            
            data_ingestion.create_dir()
            data_ingestion.download_file()
            data_ingestion.extract_zip_file()
        except Exception as e:
            raise e
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> Stage: {STAGE_NAME} started <<<<<<<<<")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f">>>>>>>> Stage: {STAGE_NAME} completed <<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e