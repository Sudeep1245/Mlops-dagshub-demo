import numpy as np
import pandas as pd

import os
import logging

import re
import nltk
import string
from nltk.corpus import stopwords,wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer

logger = logging.getLogger('data_preprocessing')# creating logger
logger.setLevel('DEBUG')# set Logger Level

if not logger.handlers:
    # Creating Handler
    consoler_handler = logging.StreamHandler()
    consoler_handler.setLevel('DEBUG')# setting handler level

    # creating formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # connecting formatter,handler,logger with each other
    consoler_handler.setFormatter(formatter)# connecting formatter with handler
    logger.addHandler(consoler_handler) # connecting handler with logger

try:
    wordnet.ensure_loaded()
    logger.debug('NLTK wordnet is already downloaded')
except LookupError:
    nltk.download("wordnet")
    logger.info('NLTK wordnet is Downloaded.')

try:
    stopwords.words("english")
    logger.debug('NLTK stopword is already Downloaded.')
except LookupError:
    nltk.download("stopwords")
    logger.info('NLTK stipwords is downloaded.')

lemmatizer= WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# data import Code


def data_import(train_path : str, test_path : str) -> tuple[pd.DataFrame,pd.DataFrame]:# import data from the path

    try :
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logger.debug(f"Train shape: {train_data.shape}")
        logger.debug(f"Test shape: {test_data.shape}")

    except FileNotFoundError:
        logger.error('File is not found please check your input')
        raise
    except Exception as e:
        logger.error(f'Some error :{e}')
        raise
    else:
        return train_data, test_data

# transformation Code

def lemmatization(text :str) -> str: #reduces higer order word to a single “base” form (the lemma)
    try :
        text = text.split()
        text=[lemmatizer.lemmatize(y) for y in text]
    except Exception as e:
        logger.error(f'Some error:{e}')
        raise
    else:
        return " " .join(text)

def remove_stop_words(text :str) ->str: # removes stop words from text

    try:
        Text=[i for i in str(text).split() if i not in stop_words]
    except Exception as e:
        logger.error(f'Some error :{e}')
        raise
    return " ".join(Text)

def removing_numbers(text :str) ->str: # removes numbers from the text
    try :
        text=''.join([i for i in text if not i.isdigit()])
    except TypeError:
        logger.error("Check you input, it don't match with string")
        raise
    except Exception as e:
        logger.error(f'Some error :{e}')
    else:
        return text

def lower_case(text :str) ->str: # transfrom whole text in lower case
    try :
        text = text.split()
        text=[y.lower() for y in text]
    except TypeError:
        logger.error("Check you input, it don't match with string")
        raise
    except Exception as e:
        logger.error(f'Some error :{e}')
        raise
    else:  
        return " " .join(text)

def removing_punctuations(text :str) ->str: # Remove punctuations
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛',"", )

        ## remove extra whitespace
        text = re.sub('\s+', ' ', text)
        text =  " ".join(text.split())
    except TypeError:
        logger.error("Check you input, it don't match with string")
        raise
    else :
        return text.strip()

def removing_urls(text :str) ->str: # removes url from the data
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def normalize_text(df :pd.DataFrame) -> pd.DataFrame:
    try :
        logger.debug('Text preprocessing Started')

        df.content=df.content.apply(lambda content : lower_case(content))
        logger.debug('Converting lower case completed.')

        df.content=df.content.apply(lambda content : remove_stop_words(content))
        logger.debug('Removing Stop words completed.')

        df.content=df.content.apply(lambda content : removing_numbers(content))
        logger.debug('Removing Number from text completed.')

        df.content=df.content.apply(lambda content : removing_punctuations(content))
        logger.debug('Removing Punctuation from text completed.')

        df.content=df.content.apply(lambda content : removing_urls(content))
        logger.debug('Removing URLs from text completed.')

        df.content=df.content.apply(lambda content : lemmatization(content))
        logger.debug('Lemmatization of text completed.')

    except NameError:
        logger.error('Check Your Function name in data preposccing.')
        raise
    else:
        return df


def data_storing(train_data :pd.DataFrame,test_data :pd.DataFrame) ->None:
    try:
        data_path = os.path.join('data','interim')
        os.makedirs(data_path,exist_ok=True)
        logger.debug('Directory created succesfully.')
    except Exception as e:
        logger.error(e)
        raise
    else:
        train_data.to_csv(os.path.join(data_path,'train_pre_processed.csv'),index=False)
        test_data.to_csv(os.path.join(data_path,'test_pre_processed.csv'),index=False)


def complete_data_pre_processing():
    try:
        train, test =data_import(train_path='./data/raw/train.csv', test_path='./data/raw/test.csv')

        train_processed_data = normalize_text(train)
        logger.debug('Trained Data preprocessing completed.')

        test_processed_data = normalize_text(test)
        logger.debug('Test Data preprocessing completed.')
    except NameError:
        logger.error('Check your function calling in main function on data preprocessing.')
        raise
    except Exception as e:
        logger.error(f'Pileline Failure in data preprocesing :{e}')
        raise
    else:
        data_storing(train_data=train_processed_data,
                 test_data=test_processed_data)
        logger.debug("Data preprocessing pipeline finished successfully.")

if __name__=='__main__':
    complete_data_pre_processing()
