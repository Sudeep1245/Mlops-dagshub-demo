#-----Import Section------
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import logging
import os
import yaml

# Creating Logger
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

if not logger.handlers:# Create logger only when it is not created.
    # Creating handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel('DEBUG')

    #Creating Fomatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    #Connecting handler,formatter,handler with each other
    console_handler.setFormatter(formatter) #attached the formatter with handler
    logger.addHandler(console_handler) # attached the handler with logger

try :
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    logger.debug(f"Params file loaded successfully")

    test_size = params['data_ingestion']['test_size']
    logger.debug('Test Size is retrieved')
except FileNotFoundError:
    logger.error('File not Found Please check your yaml File path')
    raise
except KeyError:
    logger.error('Value You are Looking Is Not In The File.')
    raise
except Exception as e:
    logger.error(f'Some Error :{e}')
    raise

# -----Function_Section-----
def extract_url_data(url : str) -> pd.DataFrame:
    # Created to extract data from url
    try :
        logger.debug(f"Reading data from URL: {url}")
        df = pd.read_csv(url)
        logger.debug(f"Data successfully loaded with shape {df.shape}")
        return df
    except UnicodeDecodeError:
        logger.error('File Encoding Is Not In UTF-8.')
        raise
    except TypeError:
        logger.error('Maybe Url Is Not In String format.')
        raise



def basic_filtering(data :pd.DataFrame) -> pd.DataFrame: 
    # Created to do dropping irrelevant columns,filter on required label data, and converting label in numerical
    try : 
        logger.debug(f"Initial data shape: {data.shape}")

        data.drop(columns=['tweet_id'],inplace=True)
        logger.debug("Dropped column: tweet_id")

        final_df = data[data['sentiment'].isin(['happiness','sadness'])]
        logger.debug(f"Filtered data shape: {final_df.shape}")

    except KeyError:
        logger.error('Please Check Your Column Name or Label Name')
        raise
    else:    
        return final_df.replace({'sentiment':{'happiness':1,'sadness':0}})

def data_dump(train_data : pd.DataFrame,test_data :pd.DataFrame) -> None:
    try:
        # Created to Train and test data into data/raw folder 
        data_path = os.path.join('data','raw')
        os.makedirs(data_path,exist_ok=True) # exist_ok save from throwing error if file already exists
        logger.debug(f"Created directory: {data_path}")
    except Exception as e:
        logger.error(f'There an issue with data dumping :{e}')
        raise
    else: 
        train_data.to_csv(os.path.join(data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(data_path,'test.csv'),index=False)

def function_execution() ->None:
    try : 
        logger.debug("Starting data ingestion pipeline")
        data = extract_url_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

        data = basic_filtering(data)

        logger.debug("Performing train-test split")
        train_data, test_data = train_test_split(data,test_size=test_size,random_state=42)

        logger.debug(
            f"Train shape: {train_data.shape}, Test shape: {test_data.shape}"
        )

    except Exception as e:
        logger.error(f'There is an issue with main function of data ingestion : {e}')
        raise

    else :
        data_dump(train_data=train_data,test_data=test_data)# data dumping


#-----Code Execution-----
if __name__  == '__main__':
    function_execution()

