from ml_Project import logger
from ml_Project.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from ml_Project.pipeline.stage_02_data_transformation import DataTransformationPipeline
from ml_Project.pipeline.stage_04_mlp_model import MLPModelPipeline
from ml_Project.pipeline.stage_05_cnn_model import CNNModelPipeline
from ml_Project.pipeline.stage_06_rnn_model import RNNModelPipeline

STAGE_NAME = "Data ingestion stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
    
except Exception as e:
    logger.exception(e)
    raise e

logger.info("-"*40)

STAGE_NAME = "Data Transformation"
try:
    logger.info(f">>>>>>>> Stage: {STAGE_NAME} started <<<<<<<<<")
    obj = DataTransformationPipeline()
    obj.main()
    logger.info(f">>>>>>>> Stage: {STAGE_NAME} completed <<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

logger.info("-"*40)

STAGE_NAME = "MLP Model Training"
try:
    logger.info(f">>>>>>>> Stage: {STAGE_NAME} started <<<<<<<<<")
    obj = MLPModelPipeline()
    obj.main()
    logger.info(f">>>>>>>> Stage: {STAGE_NAME} completed <<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

logger.info("-"*40)

STAGE_NAME = "CNN Model Training"
try:
    logger.info(f">>>>>>>> Stage: {STAGE_NAME} started <<<<<<<<<")
    obj = CNNModelPipeline()
    obj.main()
    logger.info(f">>>>>>>> Stage: {STAGE_NAME} completed <<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

logger.info("-"*40)

STAGE_NAME = "RNN Model Training"

try:
    logger.info(f">>>>>>>> Stage: {STAGE_NAME} started <<<<<<<<<")
    obj = RNNModelPipeline()
    obj.main()
    logger.info(f">>>>>>>> Stage: {STAGE_NAME} completed <<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e