import fire
import json
import numpy as np
import random
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import utils

# fix random seed
np.random.seed(2025)
random.seed(2025)

def add_patients(prompter, data, med_names, num_visits, json_data, mask_text=False):
    pos_num_visit = 0
    neg_num_visit = 0
    if num_visits < 0:
        num_visits = len(data)
    for idx, row in data.iloc[:num_visits].iterrows():
        pos_med_list = eval(row['drug_name'])
        history = None
        if idx > 0 and data.iloc[idx-1]['SUBJECT_ID'] == row['SUBJECT_ID']:
            history, history_diag_id_list, history_proc_id_list, history_drug_id_list = prompter.generate_history(data.iloc[idx-1], mask_text=mask_text)
        for med in med_names:
            prompt, diag_id_list, proc_id_list, drug_id_list = prompter.generate_input(row, drug_candidate=med, mask_text=mask_text)
            
            new_hadm_id = row['HADM_ID'] * 1000 + drug_id_list[-1]
            if med in pos_med_list:
                pos_num_visit += 1
                output = 'Yes.'
            else:
                neg_num_visit += 1
                output = 'No.'
            if history is not None:
                diag_id_list = history_diag_id_list + diag_id_list
                proc_id_list = history_proc_id_list + proc_id_list
                drug_id_list = history_drug_id_list + drug_id_list
                json_item = {"history": history,
                             "input": prompt, "output": output, 
                             "hadm_id": new_hadm_id, "diag_id": diag_id_list, 
                             "pro_id": proc_id_list, "drug_id": drug_id_list}
            else:
                json_item = {"history": None,
                             "input": prompt, "output": output,
                             "hadm_id": new_hadm_id, "diag_id": diag_id_list,
                             "pro_id": proc_id_list, "drug_id": drug_id_list}
            json_data.append(json_item)
    return json_data, pos_num_visit, neg_num_visit
            


def generate_finetune_data(output_dir, med_name2idx='data/data_process/output/mimic-iii/med_name2idx.json', logger=None,
                           num_val_visits=20, file_path='data/data_process/output/mimic-iii/data4LLM_with_note.csv',
                           use_history=True, use_note=True, use_code=True, mask_text=False, generate_test = False, generate_val = False, save_file_name='data'):
    
    med_name2idx = json.load(open(med_name2idx, 'r'))
    prompter = utils.Prompter(med_name2idx, use_note=use_note, use_code=use_code, template_name='cls')

    med_names = list(med_name2idx.keys())

    os.makedirs(output_dir, exist_ok=True)

    if generate_test:
        data_test = utils.load_data(mode='test', file_path=file_path)
        if logger is None:
            print(f'Loaded data from {file_path}')
        else:    
            logger.info(f'Loaded data from {file_path}')
        json_data = []

        pos_num_visit, neg_num_visit = 0, 0
        pos_num_visit_val, neg_num_visit_val = 0, 0
        # test
        json_data, pos_num_visit_, neg_num_visit_ = add_patients(
                prompter, data_test, med_names, -1, json_data, mask_text=mask_text)
        pos_num_visit += pos_num_visit_
        neg_num_visit += neg_num_visit_            
        
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f'{save_file_name}.json')
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        test_length = pos_num_visit + neg_num_visit

        if logger is None:
            print(f'Generate finetune data for {save_file_name} successfully! Test Length: {test_length}={pos_num_visit}+{neg_num_visit}')
        else:
            logger.info(f'Generate finetune data for {save_file_name} successfully! Test Length: {test_length}={pos_num_visit}+{neg_num_visit}')
        return file_path

    elif generate_val:
        data_test = utils.load_data(mode='val', file_path=file_path)
        if logger is None:
            print(f'Loaded data from {file_path}')
        else:    
            logger.info(f'Loaded data from {file_path}')
        json_data = []

        pos_num_visit, neg_num_visit = 0, 0
        pos_num_visit_val, neg_num_visit_val = 0, 0
        # test
        json_data, pos_num_visit_, neg_num_visit_ = add_patients(
                prompter, data_test, med_names, -1, json_data, mask_text=mask_text)
        pos_num_visit += pos_num_visit_
        neg_num_visit += neg_num_visit_            
        
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f'{save_file_name}.json')
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        test_length = pos_num_visit + neg_num_visit

        if logger is None:
            print(f'Generate finetune data for {save_file_name} successfully! Test Length: {test_length}={pos_num_visit}+{neg_num_visit}')
        else:
            logger.info(f'Generate finetune data for {save_file_name} successfully! Test Length: {test_length}={pos_num_visit}+{neg_num_visit}')
        return file_path

    else:
        data_train = utils.load_data(mode='train', file_path=file_path)
        data_val = utils.load_data(mode='val', file_path=file_path)
        if logger is None:
            print(f'Loaded data from {file_path}')
        else:    
            logger.info(f'Loaded data from {file_path}')
        json_data = []

        pos_num_visit, neg_num_visit = 0, 0
        pos_num_visit_val, neg_num_visit_val = 0, 0

        json_data, pos_num_visit_, neg_num_visit_ = add_patients(
                prompter, data_train, med_names, -1, json_data, mask_text=mask_text)   
        pos_num_visit += pos_num_visit_
        neg_num_visit += neg_num_visit_
        
        # val set
        json_data, pos_num_visit_val, neg_num_visit_val = add_patients(
            prompter, data_val, med_names, num_val_visits, json_data, mask_text=mask_text)

        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f'{save_file_name}.json')
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        train_length = pos_num_visit + neg_num_visit
        val_length = pos_num_visit_val + neg_num_visit_val

        if logger is None:
            print(f'Generate finetune data for {save_file_name} successfully! Train Length: {train_length}={pos_num_visit}+{neg_num_visit}, Val Length: {val_length}={pos_num_visit_val}+{neg_num_visit_val}')
        else:
            logger.info(f'Generate finetune data for {save_file_name} successfully! Train Length: {train_length}={pos_num_visit}+{neg_num_visit}, Val Length: {val_length}={pos_num_visit_val}+{neg_num_visit_val}')
        return file_path, pos_num_visit_val, neg_num_visit_val

if __name__ == "__main__":
    fire.Fire(generate_finetune_data)
