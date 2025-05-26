import fire
import json
import numpy as np
import random
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import utils

def generate_list_sft_data(sample_num = 100000, 
                           output_dir = 'data/list_sft', 
                           med_name2idx = 'data/data_process/output/mimic-iii/med_name2idx.json', 
                           file_path ='data/data_process/output/mimic-iii/data4LLM_with_note.csv', 
                           evaluate_results_path = 'evaluate_results/cls/train'):

    if sample_num > 0:
        print(f"Train Set, Sample Num: {sample_num}")
        output_dir = os.path.join(output_dir, str(sample_num))
        os.makedirs(output_dir, exist_ok=True)
        output_path_remove = os.path.join(output_dir, 'remove.json')
        output_path_add = os.path.join(output_dir, 'add.json')

        hadm_count = defaultdict(lambda: 0)
        med_name2idx = json.load(open(med_name2idx, 'r'))
        med_names = list(med_name2idx.keys())

        prompter = utils.Prompter(med_name2idx, template_name='list')

        data_train = utils.load_data(mode='train', file_path=file_path)
        json_data_remove = []
        json_data_add = []

        error_data = pd.read_csv(os.path.join(evaluate_results_path, 'visit_error.csv'))

        each_gtlen_miss_dict = {}
        each_gtlen_extra_dict = {}

        for idx in range(len(error_data)):
            gt_len = error_data.iloc[idx]['GT Len']
            miss_num = error_data.iloc[idx]['Miss']
            extra_num = error_data.iloc[idx]['Extra']
            if gt_len not in each_gtlen_miss_dict:
                each_gtlen_miss_dict[gt_len] = []
                each_gtlen_extra_dict[gt_len] = []
            each_gtlen_miss_dict[gt_len].append(miss_num)
            each_gtlen_extra_dict[gt_len].append(extra_num)

        def sample_miss_extra(gt_len):
            if gt_len not in each_gtlen_miss_dict:
                raise ValueError(f"GT Len {gt_len} not found in the dictionary.")
            miss_num = random.choice(each_gtlen_miss_dict[gt_len])
            extra_num = random.choice(each_gtlen_extra_dict[gt_len])
            return miss_num, extra_num

        drug_error_data = pd.read_csv(os.path.join(evaluate_results_path, 'drug_error.csv'))
        miss_weights = drug_error_data['Miss'].tolist()
        extra_weights = drug_error_data['Extra'].tolist()


        def generate_input_medids(gt_medids):
            gt_len = len(gt_medids)
            miss_num, extra_num = sample_miss_extra(gt_len)
            candidate_miss_weights = [miss_weights[idx] for idx in gt_medids]
            sample_miss_meds = np.random.choice(gt_medids, size=miss_num, replace=False, p=np.array(candidate_miss_weights) / np.sum(candidate_miss_weights))
            sample_miss_meds = sample_miss_meds.tolist()

            extra_med_candidates = list(set(range(0, 151)) - set(gt_medids))
            candidate_extra_weights = [extra_weights[idx] for idx in extra_med_candidates]
            sample_extra_meds = np.random.choice(extra_med_candidates, size=extra_num, replace=False, p=np.array(candidate_extra_weights) / np.sum(candidate_extra_weights))
            sample_extra_meds = sample_extra_meds.tolist()

            input_medids = list(set(gt_medids) - set(sample_miss_meds)) + sample_extra_meds
            return sorted(input_medids), sorted(sample_miss_meds), sorted(sample_extra_meds)



        for sample_idx in tqdm(range(sample_num)):
            visit_idx = random.randint(0, len(data_train) - 1)

            if visit_idx > 0 and data_train.iloc[visit_idx]['SUBJECT_ID'] == data_train.iloc[visit_idx - 1]['SUBJECT_ID']:
                history, history_diag_id_list, history_proc_id_list, history_drug_id_list = prompter.generate_history(data_train.iloc[visit_idx - 1])

            input, diag_id_list, proc_id_list = prompter.generate_rethink_input(data_train.iloc[visit_idx])

            output_medids = sorted(eval(data_train.iloc[visit_idx]['drug_id']))
            input_medids, miss_medids, extra_medids = generate_input_medids(output_medids)

            input_med = []
            for medid in input_medids:
                input_med.append(f"{med_names[medid]} [DrugEmb]")
            input_med = '\n'.join(input_med)

            miss_med = []
            for medid in miss_medids:
                miss_med.append(med_names[medid])
            miss_med = '\n'.join(miss_med)

            extra_med = []
            for medid in extra_medids:
                extra_med.append(med_names[medid])
            extra_med = '\n'.join(extra_med)


            if visit_idx > 0 and data_train.iloc[visit_idx]['SUBJECT_ID'] == data_train.iloc[visit_idx - 1]['SUBJECT_ID']:
                input_add = prompter.template["prompt_history_add"].format(
                    history=history,
                    input=input,
                    pre_predicted=input_med
                )
                input_remove = prompter.template["prompt_history_remove"].format(
                    history=history,
                    input=input,
                    pre_predicted=input_med
                )
                diag_id_list = history_diag_id_list + diag_id_list
                proc_id_list = history_proc_id_list + proc_id_list
                drug_id_list = history_drug_id_list + input_medids
            else:
                input_add = prompter.template["prompt_no_history_add"].format(
                    input=input,
                    pre_predicted=input_med
                )
                input_remove = prompter.template["prompt_no_history_remove"].format(
                    input=input,
                    pre_predicted=input_med
                )
                drug_id_list = input_medids


            hadm_id = data_train.iloc[visit_idx]['HADM_ID']
            data_point_hadm_id = int(data_train.iloc[visit_idx]['HADM_ID'] * 1000 + hadm_count[hadm_id])
            hadm_count[hadm_id] += 2

            if hadm_count[hadm_id] > 500:
                raise ValueError(f"hadm_count for {hadm_id} exceeds 500.")

            data_point_remove = {
                'input': input_remove,
                'output': extra_med,
                'hadm_id': data_point_hadm_id,
                'diag_id': diag_id_list,
                'pro_id': proc_id_list,
                'drug_id': drug_id_list
            }
            json_data_remove.append(data_point_remove)
            data_point_add = {
                'input': input_add,
                'output': miss_med,
                'hadm_id': data_point_hadm_id + 1,
                'diag_id': diag_id_list,
                'pro_id': proc_id_list,
                'drug_id': drug_id_list
            }
            json_data_add.append(data_point_add)

        with open(output_path_remove, 'w') as f:
            json.dump(json_data_remove, f, indent=4)
        with open(output_path_add, 'w') as f:
            json.dump(json_data_add, f, indent=4)


        with open(output_path_remove, 'r') as f:
            data_1 = json.load(f)
        with open(output_path_add, 'r') as f:
            data_2 = json.load(f)

        # Combine the two lists
        combined_data = data_1 + data_2
        # Shuffle the combined data
        random.shuffle(combined_data)

        mix_data_path = os.path.join(output_dir, 'mix.json')
        with open(mix_data_path, 'w') as f:
            json.dump(combined_data, f, indent=4)
        print(f"Combined data saved to {mix_data_path}, total {len(combined_data)} samples.")

    else:
        print(f"Test Set, Just use the original data.")
        output_dir = os.path.join(output_dir, 'test')
        os.makedirs(output_dir, exist_ok=True)
        output_path_remove = os.path.join(output_dir, 'remove.json')
        output_path_add = os.path.join(output_dir, 'add.json')


        hadm_count = defaultdict(lambda: 0)
        med_name2idx = json.load(open(med_name2idx, 'r'))
        med_names = list(med_name2idx.keys())

        prompter = utils.Prompter(med_name2idx, template_name='list')

        data_test = utils.load_data(mode='test', file_path=file_path)
        # json_data = []
        json_data_remove = []
        json_data_add = []
        
        predict_data = os.path.join(evaluate_results_path, "evaluation_result.csv")
        pred_data = pd.read_csv(predict_data)

        # for visit_idx in range(len(data_test)):
        for visit_idx in tqdm(range(len(data_test))):

            if visit_idx > 0 and data_test.iloc[visit_idx]['SUBJECT_ID'] == data_test.iloc[visit_idx - 1]['SUBJECT_ID']:
                history, history_diag_id_list, history_proc_id_list, history_drug_id_list = prompter.generate_history(data_test.iloc[visit_idx - 1])

            input, diag_id_list, proc_id_list = prompter.generate_rethink_input(data_test.iloc[visit_idx])

            output_medids = sorted(eval(data_test.iloc[visit_idx]['drug_id']))
            input_medids = eval(pred_data.iloc[visit_idx]['preds'])
            gt_medids = eval(pred_data.iloc[visit_idx]['labels'])
            assert gt_medids == output_medids

            output_med = []
            for medid in output_medids:
                output_med.append(med_names[medid])
            output_med = '\n'.join(output_med)

            input_med = []
            for medid in input_medids:
                input_med.append(f"{med_names[medid]} [DrugEmb]")
            input_med = '\n'.join(input_med)

            if visit_idx > 0 and data_test.iloc[visit_idx]['SUBJECT_ID'] == data_test.iloc[visit_idx - 1]['SUBJECT_ID']:
                input_text_add = prompter.template["prompt_history_add"].format(
                    history=history,
                    input=input,
                    pre_predicted=input_med
                )
                input_text_remove = prompter.template["prompt_history_remove"].format(
                    history=history,
                    input=input,
                    pre_predicted=input_med
                )
                diag_id_list = history_diag_id_list + diag_id_list
                proc_id_list = history_proc_id_list + proc_id_list
                drug_id_list = history_drug_id_list + input_medids
            else:
                input_text_add = prompter.template["prompt_no_history_add"].format(
                    input=input,
                    pre_predicted=input_med
                )
                input_text_remove = prompter.template["prompt_no_history_remove"].format(
                    input=input,
                    pre_predicted=input_med
                )
                drug_id_list = input_medids

            hadm_id = data_test.iloc[visit_idx]['HADM_ID']
            data_point_hadm_id = int(data_test.iloc[visit_idx]['HADM_ID'] * 1000 + hadm_count[hadm_id])
            hadm_count[hadm_id] += 2

            data_point_remove = {
                'input': input_text_remove,
                'output': output_med,
                'hadm_id': data_point_hadm_id,
                'diag_id': diag_id_list,
                'pro_id': proc_id_list,
                'drug_id': drug_id_list,
                'input_drug_id': input_medids,
                'gt_id': gt_medids
            }
            json_data_remove.append(data_point_remove)
            data_point_add = {
                'input': input_text_add,
                'output': output_med,
                'hadm_id': data_point_hadm_id + 1,
                'diag_id': diag_id_list,
                'pro_id': proc_id_list,
                'drug_id': drug_id_list,
                'input_drug_id': input_medids,
                'gt_id': gt_medids
            }
            json_data_add.append(data_point_add)

        with open(output_path_remove, 'w') as f:
            json.dump(json_data_remove, f, indent=4)
        with open(output_path_add, 'w') as f:
            json.dump(json_data_add, f, indent=4)

if __name__ == '__main__':
    fire.Fire(generate_list_sft_data)
    