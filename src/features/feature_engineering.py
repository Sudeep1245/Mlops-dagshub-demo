import numpy as np
import pandas as pd 
from sklearn.base import BaseEstimator

import os
import yaml
import logging

from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger('feature_engg')
logger.setLevel('DEBUG')

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel('DEBUG')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

try:
    with open('params.yaml','r') as f:
        params = yaml.safe_load(f)
    
    logger.debug(f"Params file loaded successfully.")
    
    max_features = params['feature_engineering']['max_features']
    logger.debug(f"Max feature retrived successfully.")

except FileNotFoundError:
    logger.error('File Not Found ,Please Check You File Name.')
    raise
except KeyError:
    logger.error('Value You are Looking Is Not In The File.')
    raise


# data loading
def data_loading(train_path :str,test_path :str) ->tuple[pd.DataFrame,pd.DataFrame]:
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        logger.debug(f'Train data retrived : {train_data.shape}')
        logger.debug(f'Test data retrived : {test_data.shape}')

    except FileNotFoundError:
        logger.error('File is not found please check your input')
        raise
    except Exception as e:
        logger.error(f'Some eror happend : {e}')
        raise
    else:
        return train_data, test_data


# Feature Engeneering
def data_split(train_data :pd.DataFrame,test_data :pd.DataFrame) ->tuple[pd.DataFrame,pd.Series,pd.DataFrame,pd.Series]:
    try:
        train_data = train_data.fillna('')
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        logger.debug(f'Successful extracted X_train :{X_train.shape}, y_train : {y_train.shape}')

        test_data = test_data.fillna('')
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values
        logger.debug(f'Successful extracted X_test :{X_test.shape}, y_test :{y_test.shape}')

    except TypeError:
        logger.error('Value you provided is not a dataframe.')
        raise
    else:
        return X_train, y_train, X_test, y_test



def bag_of_word(model : CountVectorizer, X_train :pd.DataFrame, y_train :pd.DataFrame, X_test :pd.DataFrame, y_test :pd.DataFrame) ->tuple[pd.DataFrame,pd.DataFrame]:
    try:
        # Fit the vectorizer on the training data and transform it
        X_train_bow = model.fit_transform(X_train)
        logger.debug(f'Transfomed X_train -> X_train bow ({X_train_bow.shape})')
    
        # Transform the test data using the same vectorizer
        X_test_bow = model.transform(X_test)
        logger.debug(f'Transfomed X_test -> X_test bow ({X_test_bow.shape})')
        # adding output column in train data

        train_df = pd.DataFrame(X_train_bow.toarray())

        train_df['label'] = y_train

        # adding output column  in test data
        test_df = pd.DataFrame(X_test_bow.toarray())

        test_df['label'] = y_test

    except TypeError:
        logger.error('Value you provided is not a dataframe.')
        raise
    except Exception as e:
        logger.error(f"Bag of Words failed: {e}")
        raise
    else:
        logger.debug('Successfully Applied Bag of words on train_data,test_data')
        return train_df, test_df

def data_storing(train_df :pd.DataFrame,test_df :pd.DataFrame) ->None:# Storing the transform data
    try : 
        data_path = os.path.join('data','processed')
        os.makedirs(data_path,exist_ok=True)
    except Exception as e:
        logger.error(e)
        raise
    else:
        train_df.to_csv(os.path.join(data_path,'train_bow.csv'),index=False)
        test_df.to_csv(os.path.join(data_path,'test_bow.csv'),index=False)


def complete_feature_eng():
    try :
        logger.debug('Feature engineering started.')
        train, test = data_loading(train_path='./data/interim/train_pre_processed.csv',
                               test_path='./data/interim/test_pre_processed.csv')


        X_train, y_train, X_test, y_test=data_split(train_data=train, test_data=test)

        vectorizer = CountVectorizer(max_features=max_features) # creating bag of word model
        train_df, test_df = bag_of_word(model=vectorizer,
                                   X_train=X_train, y_train=y_train,
                                   X_test=X_test, y_test=y_test)
    except ImportError:
        logger.error('Check Your Imports.')
        raise
    else:    
        data_storing(train_df=train_df, test_df=test_df)# Dumping data
        logger.debug(f'Train bow :{train_df.shape}, Test bow :{test_df.shape} is dumped after processing at data/processed.')

if __name__=='__main__':
    complete_feature_eng()