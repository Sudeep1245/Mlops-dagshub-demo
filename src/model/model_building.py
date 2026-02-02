import numpy as np
import pandas as pd

import pickle
import yaml
import logging

from sklearn.ensemble import GradientBoostingClassifier

logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel('DEBUG')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

try:
    with open('params.yaml','r') as f:
        model_building_params = yaml.safe_load(f)['model_building']
    logger.debug('Params file loaded Sucessfully.')

    n_estimators = model_building_params['n_estimators']
    learning_rate = model_building_params['learning_rate']
    logger.debug('Params file parameter retrived sucessfully.')

except FileNotFoundError:
    logger.error('File Not Found ,Please Check You File Name.')
    raise
except KeyError:
    logger.error('Value You are Looking Is Not In The File.')
    raise



def data_loading(train_path :str) ->pd.DataFrame:
    try :
        df = pd.read_csv(train_path)
        logger.debug(f'Train Data is retreived from {train_path}')
        return df

    except FileNotFoundError:
        logger.error('File is not found please check your input')
        raise

def X_y_split(train_data :pd.DataFrame) ->tuple[np.ndarray,np.ndarray]:
    try:
        X_train = train_data.iloc[:,0:-1].values #.values removes column names
        y_train = train_data.iloc[:,-1].values 
        logger.debug(f'X_train data created successfully :{(X_train.shape)}')
        logger.debug(f'y_train data created successfully :{(y_train.shape)}')

    except TypeError:
        logger.error('Value you provided is not a dataframe.')
        raise
    else:
        return X_train,y_train

# train the model

def model_building():
    try :
        logger.debug('Model training is started.')
        train_data = data_loading(train_path='./data/processed/train_bow.csv')

        # train data split
        X_train,y_train = X_y_split(train_data=train_data)

        # model building 
        clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,random_state=42)
        logger.debug('Model is Initiated.')
        clf.fit(X_train,y_train)
        logger.debug('Model fitted on train data.')

    except Exception as e:
        logger.error(f'Model Training failed :{e}.')
        raise
    else:
        # Store model
        pickle.dump(clf,open('./models/model.pkl','wb'))
        logger.debug('Model dump at model folder.')

if __name__ =='__main__':
    model_building()