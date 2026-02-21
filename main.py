from networksecurity.componenets.data_ingestion import DataIngestion
from networksecurity.componenets.data_validation import DataValidation
from networksecurity.componenets.model_trainer import ModelTrainer
from networksecurity.componenets.data_transformation import DataTransformation
from networksecurity.exceptions.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entiity.config_entity import (DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig)
import sys


if __name__=='__main__':
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        dataingestion=DataIngestion(dataingestionconfig)
        logging.info("initiate the data ingesiton")
        dataingestionartifact= dataingestion.initiate_data_ingestion()
        logging.info("initiate the data ingesiton Completed")
        
        datavalidationconfig=DataValidationConfig(trainingpipelineconfig)
        datavalidation=DataValidation(dataingestionartifact,datavalidationconfig)
        datavalidationartifact=datavalidation.initiate_data_validation()
        print(dataingestionartifact)
        
        datatransformationconfig=DataTransformationConfig(trainingpipelineconfig)
        datatransformation=DataTransformation(datavalidationartifact,datatransformationconfig)
        datatransformationartifact=datatransformation.initiate_data_transformation()
        print(datatransformationartifact)
        
        
        modeltrainerconfig=ModelTrainerConfig(trainingpipelineconfig)
        modeltrainer=ModelTrainer(modeltrainerconfig, datatransformationartifact)
        modeltrainerartifact=modeltrainer.initiate_model_trainer()
        print(modeltrainerartifact)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
