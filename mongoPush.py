import os
import json
import sys
from dotenv import load_dotenv

load_dotenv()

MONGO_DB_URI=os.getenv("MONGOURI")

import certifi
ca=certifi.where()

import pandas as pd
import numpy as np
import pymongo
from networksecurity.exceptions.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def cv_to_json_converter(self, file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    def insert_data_mongodb(self, records, database, collection):
        try:
            self.database=database
            self.records=records
            self.collection=collection
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URI)
            self.database=self.mongo_client[self.database]
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return (len(self.records))
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

if __name__=='__main__':
    FILE_PATH="NetworkData/phisingData.csv"
    DATABASE="NetworkData"
    COLLECTION='NetworkData'
    networkobj=NetworkDataExtract()
    records=networkobj.cv_to_json_converter(FILE_PATH)
    print(f"records: \n {records}")
    no_of_records=networkobj.insert_data_mongodb(records,DATABASE,COLLECTION)
    print(f"no. of records: {no_of_records}")