import json
import logging
import os
import pandas as pd


def setup_logger(log_file, mode='w'):
    # create log folder if not exists
    log_folder = os.path.dirname(log_file)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder, exist_ok=True)
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def load_data(mode='train', file_path='data/data_process/output/mimic-iii/data4LLM_with_note.csv'):
    '''
    Load EHR data and drug names
    '''
    # load EHR data
    # data_path = './data/'

    data = pd.read_csv(file_path)

    # split data into train, val, test
    split_point = int(len(data) * 2 / 3)
    val_len = int(len(data[split_point:]) / 2)
    data_train = data[:split_point]
    data_val = data[split_point:split_point+val_len]
    data_val.reset_index(inplace=True, drop=True)
    data_test = data[split_point+val_len:]
    data_test.reset_index(inplace=True, drop=True)

    # return data and drug names
    if mode == 'train':
        # print(f'Train data size: {len(data_train)}\n') 
        return data_train
    elif mode == 'val':
        # print(f'Val data size: {len(data_val)}\n') 
        return data_val
    elif mode == 'test':
        print(f'Test data size: {len(data_test)}\n')
        return data_test
    else:
        raise ValueError('Wrong mode!')
