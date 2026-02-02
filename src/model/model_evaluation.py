import numpy as np
import pandas as pd

import pickle
import json
import logging
import yaml

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

import mlflow
import dagshub
dagshub.init(repo_owner='Sudeep1245', repo_name='Mlops-dagshub-demo', mlflow=True)

mlflow.set_tracking_uri('https://github.com/Sudeep1245/Mlops-dagshub-demo.git')

logger = logging.getLogger('model_Evaluation')
logger.setLevel('DEBUG')

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel('DEBUG')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def loading_neccesity(model_path :str,data_path :str) ->tuple[BaseEstimator,pd.DataFrame]:
    try:
        clf = pickle.load(open(model_path,'rb'))
        logger.debug('Model Loaded Successfully.')
        # import data
        test_data = pd.read_csv(data_path)
        logger.debug(f'Test data is retrived from {data_path}')
    except FileNotFoundError as e:
        logger.error(f'File not found: {e}')
        raise
    except Exception as e:
        logger.error(f'Loading failed: {e}')
        raise
    else:
        return clf,test_data


def X_y_split(test_data :pd.DataFrame) ->tuple[pd.DataFrame,pd.Series]:
    try:
        X_test_bow = test_data.iloc[:,0:-1]
        y_test = test_data.iloc[:,-1]
        logger.debug(f'Test features created: {X_test_bow.shape}')
        logger.debug(f'Test label data : {y_test.shape}')

    except TypeError:
        logger.error('Value you provided is not a dataframe.')
        raise
    else:
        return X_test_bow,y_test


# Calculate evaluation metrics
def metric_cal(y_test :pd.Series,y_pred :pd.Series,y_pred_proba :pd.Series) ->tuple[float,float,float,float]:
    try:
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test, y_pred,average='binary')
        recall = recall_score(y_test, y_pred,average='binary')
        auc = roc_auc_score(y_test, y_pred_proba)
    except ImportError:
        logger.error('Check your Imports.')
        raise
    else:    
        return accuracy,precision,recall,auc

# Created dvc live experiment tracking code    
with open('params.yaml','r') as f:
        params = yaml.safe_load(f)

def dagshub_tracking(accuracy,precision,recall,auc,params):

    with mlflow.start_run() as exp :
        exp.log_metric('accuracy',accuracy)
        exp.log_metric('precision',precision)
        exp.log_metric('recall',recall)
        exp.log_metric('auc',auc)

        for file,value in params.items():
            for key,val in value.items():
                exp.log_param(key,val)


def model_prediction():
    try:
        logger.debug('Model Evaluation Started.')
        clf,test_data = loading_neccesity(model_path='./models/model.pkl',
                      data_path='./data/processed/test_bow.csv')
        X_test_bow,y_test =X_y_split(test_data=test_data)

        y_pred = clf.predict(X_test_bow)
        logger.debug('Model prediction done.')
        y_pred_proba = clf.predict_proba(X_test_bow)[:, 1]

        accuracy,precision,recall,auc=metric_cal(y_test,y_pred,y_pred_proba)
        dagshub_tracking(accuracy=accuracy,precision=precision,recall=recall,auc=auc,params=params)
        metric_dic = {
        'accuracy':accuracy,
        'precision' : precision,
        'recall':recall,
        'auc' : auc
        }
    except Exception as e:
        logger.error(f'Model evaluation failed: {e}')
        raise
    else:
        with open('./reports/metrics.json','w') as file:
            json.dump(metric_dic,file,indent=4)
        logger.debug('Model Metric score is stored at reports Folders.')

if __name__=='__main__':
    model_prediction()