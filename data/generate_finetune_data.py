import fire
import json
import numpy as np
import random
import os
import sys
from tqdm import tqdm
import utils

# fix random seed
np.random.seed(2025)
random.seed(2025)


def add_candidate_drug(prompter, data, candidate_drug, json_data,
                       max_pos_num_visit=10e6, max_visit=10e6,
                       mask_text=False):
    pos_num_visit = 0
    neg_num_visit = 0
    for idx, row in data.iterrows():
        pos_med_list = eval(row['drug_name'])
        history = None
        
        if idx > 0 and data.iloc[idx-1]['SUBJECT_ID'] == row['SUBJECT_ID']:
            history, history_diag_id_list, history_proc_id_list, history_drug_id_list = prompter.generate_history(data.iloc[idx-1], mask_text=mask_text)

        prompt, diag_id_list, proc_id_list, drug_id_list = prompter.generate_input(row, drug_candidate=candidate_drug, mask_text=mask_text)

        if candidate_drug in pos_med_list:
            pos_num_visit += 1
            output = 'Yes.'
        else:
            neg_num_visit += 1
            output = 'No.'
        
        new_hadm_id = row['HADM_ID'] * 1000 + drug_id_list[-1]

        if history is not None:
            diag_id_list = history_diag_id_list + diag_id_list
            proc_id_list = history_proc_id_list + proc_id_list
            drug_id_list = history_drug_id_list + drug_id_list
            json_item = {"history": history, "input": prompt, "output": output, "hadm_id": new_hadm_id, "diag_id": diag_id_list, "pro_id": proc_id_list, "drug_id": drug_id_list}
        else:
            json_item = {"history": None, "input": prompt, "output": output, "hadm_id": new_hadm_id, "diag_id": diag_id_list, "pro_id": proc_id_list, "drug_id": drug_id_list}
        json_data.append(json_item)
        if pos_num_visit >= max_pos_num_visit:
            break
        if pos_num_visit + neg_num_visit >= max_visit:
            break
    return json_data, pos_num_visit, neg_num_visit


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
            


def generate_finetune_data(candidate_drugs, med_name2idx, output_dir, logger=None,
                           num_val_examples=100, num_val_visits=20, file_name='data4LLM.csv',
                           use_history=True, use_note=True, use_code=True, mask_text=False, generate_test = False, generate_val = False):
    """
    Generate finetune data for each drug
    :param candidate_drugs: list of candidate drugs
    :param logger: logger
    :param num_val_examples: 
    """
    med_name2idx = json.load(open(med_name2idx, 'r'))
    prompter = utils.Prompter(med_name2idx, use_note=use_note, use_code=use_code)

    med_names = list(med_name2idx.keys())

    # if num_val_examples < 0, use all val data
    if num_val_examples < 0:
        num_val_examples = 10e6
    # if candidate_drugs is str, make it a list
    if isinstance(candidate_drugs, str):
        candidate_drugs = [candidate_drugs]

    if generate_test:
        data_test = utils.load_data(mode='test', file_name=file_name)
        if logger is None:
            print(f'Loaded data from {file_name}')
        else:    
            logger.info(f'Loaded data from {file_name}')
        json_data = []

        if candidate_drugs is None:
            candidate_drugs = med_names

        pos_num_visit, neg_num_visit = 0, 0
        pos_num_visit_val, neg_num_visit_val = 0, 0
        # test
        json_data, pos_num_visit_, neg_num_visit_ = add_patients(
                prompter, data_test, med_names, -1, json_data, mask_text=mask_text)
        pos_num_visit += pos_num_visit_
        neg_num_visit += neg_num_visit_            
        
        if len(candidate_drugs) == len(med_names):
            file_name = f'all_test_no_code' if mask_text==False else f'all_test_mask_text'
        elif len(candidate_drugs) == 1:
            med_idx = med_names.index(candidate_drugs[0])
            file_name = f"{med_idx}-{candidate_drugs[0].replace(' ', '-')}"
        elif len(candidate_drugs) > 4:
            med_idx = med_names.index(candidate_drugs[0])
            file_name = f"{len(candidate_drugs)}_start_{med_idx}"
        else:
            file_name = '-'.join(candidate_drugs).replace(' ', '-')
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f'{file_name}.json')
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        test_length = pos_num_visit + neg_num_visit

        if logger is None:
            print(f'Generate finetune data for {file_name} successfully! Test Length: {test_length}={pos_num_visit}+{neg_num_visit}')
        else:
            logger.info(f'Generate finetune data for {file_name} successfully! Test Length: {test_length}={pos_num_visit}+{neg_num_visit}')
        return file_path

    elif generate_val:
        data_test = utils.load_data(mode='val', file_name=file_name)
        if logger is None:
            print(f'Loaded data from {file_name}')
        else:    
            logger.info(f'Loaded data from {file_name}')
        json_data = []

        if candidate_drugs is None:
            candidate_drugs = med_names

        pos_num_visit, neg_num_visit = 0, 0
        pos_num_visit_val, neg_num_visit_val = 0, 0
        # test
        json_data, pos_num_visit_, neg_num_visit_ = add_patients(
                prompter, data_test, med_names, -1, json_data, mask_text=mask_text)
        pos_num_visit += pos_num_visit_
        neg_num_visit += neg_num_visit_            
        
        if len(candidate_drugs) == len(med_names):
            file_name = f'all_val' if mask_text==False else f'all_val_mask_text'
        elif len(candidate_drugs) == 1:
            med_idx = med_names.index(candidate_drugs[0])
            file_name = f"{med_idx}-{candidate_drugs[0].replace(' ', '-')}"
        elif len(candidate_drugs) > 4:
            med_idx = med_names.index(candidate_drugs[0])
            file_name = f"{len(candidate_drugs)}_start_{med_idx}"
        else:
            file_name = '-'.join(candidate_drugs).replace(' ', '-')
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f'{file_name}.json')
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        test_length = pos_num_visit + neg_num_visit

        if logger is None:
            print(f'Generate finetune data for {file_name} successfully! Test Length: {test_length}={pos_num_visit}+{neg_num_visit}')
        else:
            logger.info(f'Generate finetune data for {file_name} successfully! Test Length: {test_length}={pos_num_visit}+{neg_num_visit}')
        return file_path

    else:
        data_train = utils.load_data(mode='train', file_name=file_name)
        data_val = utils.load_data(mode='val', file_name=file_name)
        if logger is None:
            print(f'Loaded data from {file_name}')
        else:    
            logger.info(f'Loaded data from {file_name}')
        json_data = []

        if candidate_drugs is None:
            candidate_drugs = med_names

        pos_num_visit, neg_num_visit = 0, 0
        pos_num_visit_val, neg_num_visit_val = 0, 0

        json_data, pos_num_visit_, neg_num_visit_ = add_patients(
                prompter, data_train, med_names, -1, json_data, mask_text=mask_text)   
        pos_num_visit += pos_num_visit_
        neg_num_visit += neg_num_visit_
        
        # val set
        if len(candidate_drugs) == 1:
            json_data, pos_num_visit_val, neg_num_visit_val = add_candidate_drug(
                prompter, data_val, candidate_drugs[0], json_data, max_pos_num_visit=num_val_examples, mask_text=mask_text)
        elif len(candidate_drugs) == len(med_names):
            json_data, pos_num_visit_val, neg_num_visit_val = add_patients(
                prompter, data_val, med_names, num_val_visits, json_data, mask_text=mask_text)
        else:
            for drug in candidate_drugs:
                json_data, pos_num_visit_val_, neg_num_visit_val_ = add_candidate_drug(
                    prompter, data_val, drug, json_data, max_visit=num_val_examples, mask_text=mask_text)
                pos_num_visit_val += pos_num_visit_val_
                neg_num_visit_val += neg_num_visit_val_

        if len(candidate_drugs) == len(med_names):
            file_name = f'all_val{num_val_visits}' if mask_text==False else f'all_val{num_val_visits}_mask_text'
        elif len(candidate_drugs) == 1:
            med_idx = med_names.index(candidate_drugs[0])
            file_name = f"{med_idx}-{candidate_drugs[0].replace(' ', '-')}"
        elif len(candidate_drugs) > 4:
            med_idx = med_names.index(candidate_drugs[0])
            file_name = f"{len(candidate_drugs)}_start_{med_idx}"
        else:
            file_name = '-'.join(candidate_drugs).replace(' ', '-')
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f'{file_name}.json')
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        train_length = pos_num_visit + neg_num_visit
        val_length = pos_num_visit_val + neg_num_visit_val

        if logger is None:
            print(f'Generate finetune data for {file_name} successfully! Train Length: {train_length}={pos_num_visit}+{neg_num_visit}, Val Length: {val_length}={pos_num_visit_val}+{neg_num_visit_val}')
        else:
            logger.info(f'Generate finetune data for {file_name} successfully! Train Length: {train_length}={pos_num_visit}+{neg_num_visit}, Val Length: {val_length}={pos_num_visit_val}+{neg_num_visit_val}')
        return file_path, pos_num_visit_val, neg_num_visit_val

if __name__ == "__main__":
    fire.Fire(generate_finetune_data)
