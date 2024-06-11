from fastapi import FastAPI, Depends
import pandas as pd
from darts import TimeSeries
import pickle
from pydantic import BaseModel
from typing import Dict, Union, List
from fastapi.middleware.cors import CORSMiddleware
import logging
import pymongo
from configparser import ConfigParser
import holidays
from sklearn.preprocessing import LabelEncoder
import utils

app = FastAPI()

# Your MongoDB and configuration setup
config_file = "config.ini"
parser = ConfigParser()
parser.read(config_file)
host = parser.get('mongodb', 'host')
port = int(parser.get('mongodb', 'port'))
database = parser.get('mongodb', 'database')
config_collection = parser.get('mongodb', 'config_collection')
frequency = parser.get('mongodb', 'frequency')
feature_engineered_collection_name = parser.get('mongodb', 'feature_engineered_collection_name')
tenants_collection_name = parser.get('mongodb', 'tenants_collection_name')
client = pymongo.MongoClient("mongodb://localhost:27017/")

class MongoDBManager:
    __instance = None

    @staticmethod
    def get_instance():
        if MongoDBManager.__instance is None:
            MongoDBManager.__instance = MongoDBManager()
        return MongoDBManager.__instance

    def _init_(self):
        db = utils.get_mongodb_connection(host, port, database)
        if db is not None:
            logging.info("MongoDB database connected successfully!")
            collection = db[feature_engineered_collection_name + frequency]
            tenants_list = utils.fetch_tenants_list(tenants_collection_name, db)
            site_ids = utils.extract_site_ids(tenants_list)
            query = {"site_id": site_ids[1]}
            result = collection.find(query)
            records = list(result)
            df = pd.DataFrame(records)
            collection = db[config_collection]
            result = collection.find({})
            configurations = utils.fetch_configurations(config_collection, db, site_ids[1])
            future_features_list = configurations["future_features"]
            past_features_list = configurations["past_features"]
            selected_columns = [col for col in df.columns if col in future_features_list or col == 'creation_date' or col == 'product_item_sku_id' or col == 'sales']
            df = df[selected_columns]
            self.daily_data = df
            self.future_list = future_features_list
            self.past_list = past_features_list
        else:
            logging.error("Failed to connect to MongoDB database.")

mongo_db_manager = MongoDBManager.get_instance()

class PredictionRequest(BaseModel):
    sku_id_str: str
    pred_chunk: int

class PredictionResponse(BaseModel):
    data: List[Dict[str, Union[str, float]]]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict_sales(
    request: PredictionRequest,
    daily_data: pd.DataFrame = Depends(lambda: mongo_db_manager.daily_data),
    future_features_list=Depends(lambda: mongo_db_manager.future_list),
    past_features_list=Depends(lambda: mongo_db_manager.past_list)
):
    site_id = "8d3ea3bc-f65b-4227-9fa6-6fae40e4575a"
    sku_id_str = request.sku_id_str
    pred_chunk = request.pred_chunk
    sku_mapping = {
        'FO-DMUN2Q': 0,
        'MA-DMDCNPLQ': 1,
        'MA-DMSUFQ': 2,
        'MA-DMSUFT': 3,
        'PI-DMDCFMJ': 4,
        'PI-DMDCFMK': 5,
        'PI-DMDCMEJ': 6,
        'PI-DMDCMEK': 7,
        'PI-DMDCSOJ': 8,
        'PI-DMDCSOK': 9
    }
    sku_id = sku_mapping.get(sku_id_str)
    data = pd.DataFrame()
    data = daily_data
    model_filename = "catboost_dynamic_pricing_model_D_8d3ea3bc-f65b-4227-9fa6-6fae40e4575a.pkl"
    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)
    data_appended = utils.append_rows(daily_data, pred_chunk)
    fr = "D"
    label_encoder = LabelEncoder()
    data['product_item_sku_id_encoded'] = label_encoder.fit_transform(data['product_item_sku_id'])
    data_appended['product_item_sku_id_encoded'] = label_encoder.fit_transform(data_appended['product_item_sku_id'])
    train_time_series = TimeSeries.from_group_dataframe(data, group_cols="product_item_sku_id_encoded", time_col='creation_date', fill_missing_dates=False, freq=fr, value_cols=['sales'])
    future_covariates_series = TimeSeries.from_group_dataframe(data_appended, group_cols="product_item_sku_id_encoded", time_col='creation_date', fill_missing_dates=False, freq=fr, value_cols=future_features_list)
    past_covariates_series = TimeSeries.from_group_dataframe(data_appended, group_cols="product_item_sku_id_encoded", time_col='creation_date', fill_missing_dates=False, freq=fr, value_cols=past_features_list)
    pred = loaded_model.predict(pred_chunk, series=train_time_series[sku_id], future_covariates=future_covariates_series[sku_id], past_covariates=past_covariates_series[sku_id])
    df1 = pred.pd_dataframe()
    df1['sales'] = df1['sales'].apply(lambda x: max(0, round(x))).astype(int).tolist()
    df1['flag'] = 0
    df2 = train_time_series[sku_id].pd_dataframe().tail(30)
    df2['flag'] = 1
    result_df = pd.concat([df2, df1])
    data = [{'date': date.strftime('%Y-%m-%d'), 'sales': sales, 'flag': flag} for date, sales, flag in zip(result_df.index, result_df['sales'], result_df['flag'])]
    return PredictionResponse(data=data)

# Ensure the app is callable as 'app'
if _name_ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)