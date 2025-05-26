import fire
import json
import numpy as np
import random
import os
import sys
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

def generate_list_grpo_data(med_name2idx='data/data_process/output/mimic-iii/med_name2idx.json',
                            file_path ='data/data_process/output/mimic-iii/data4LLM_with_note.csv',
                            output_dir = 'data/list_grpo',
                            sample_num = 20000,
                            evaluate_results_path = 'evaluate_results/cls/train'):

    if sample_num > 0:
        print(f"Train Set, Sample Num: {sample_num}")
        os.makedirs(output_dir, exist_ok=True)
        output_path_mix = os.path.join(output_dir, f'mix_{sample_num}.json')

        med_name2idx = json.load(open(med_name2idx, 'r'))
        med_names = list(med_name2idx.keys())

        prompter = utils.Prompter(med_name2idx, template_name='list')

        data_train = utils.load_data(mode='train', file_path=file_path)
        json_data = []

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

        drug_error_data_path = os.path.join(evaluate_results_path, 'drug_error.csv')

        drug_error_data = pd.read_csv(drug_error_data_path)
        miss_weights = drug_error_data['Miss'].tolist()
        extra_weights = drug_error_data['Extra'].tolist()


        def generate_input_medids_t5(gt_medids):
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



        for sample_idx in tqdm(range(sample_num // 2)):
            visit_idx = random.randint(0, len(data_train) - 1)

            if visit_idx > 0 and data_train.iloc[visit_idx]['SUBJECT_ID'] == data_train.iloc[visit_idx - 1]['SUBJECT_ID']:
                history = prompter.generate_history_GRPO(data_train.iloc[visit_idx - 1])

            input = prompter.generate_rethink_input_GRPO(data_train.iloc[visit_idx])

            output_medids = sorted(eval(data_train.iloc[visit_idx]['drug_id']))
            input_medids, miss_medids, extra_medids = generate_input_medids_t5(output_medids)

            input_med = []
            for medid in input_medids:
                input_med.append(f"{med_names[medid]} <DrugEmb-{medid}>")
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
            else:
                input_add = prompter.template["prompt_no_history_add"].format(
                    input=input,
                    pre_predicted=input_med
                )
                input_remove = prompter.template["prompt_no_history_remove"].format(
                    input=input,
                    pre_predicted=input_med
                )


            data_point_remove = {
                'prompt': input_remove,
                'task': 'remove',
                'input_drug_id': input_medids,
                'gt_id': output_medids
            }
            json_data.append(data_point_remove)
            data_point_add = {
                'prompt': input_add,
                'task': 'add',
                'input_drug_id': input_medids,
                'gt_id': output_medids
            }
            json_data.append(data_point_add)

        with open(output_path_mix, 'w') as f:
            json.dump(json_data, f, indent=4)
    else:
        print("Test Set, Just use the original data.")
        output_dir = os.path.join(output_dir, 'test')
        predict_data = os.path.join(evaluate_results_path, 'evaluation_result.csv')

        os.makedirs(output_dir, exist_ok=True)
        output_path_remove = os.path.join(output_dir, 'remove.json')
        output_path_add = os.path.join(output_dir, 'add.json')


        med_name2idx = json.load(open(med_name2idx, 'r'))
        med_names = list(med_name2idx.keys())

        prompter = utils.Prompter(med_name2idx, template_name='list')

        data_test = utils.load_data(mode='test', file_path=file_path)
        # json_data = []
        json_data_remove = []
        json_data_add = []

        pred_data = pd.read_csv(predict_data)

        # for visit_idx in range(len(data_test)):
        for visit_idx in tqdm(range(len(data_test))):

            if visit_idx > 0 and data_test.iloc[visit_idx]['SUBJECT_ID'] == data_test.iloc[visit_idx - 1]['SUBJECT_ID']:
                history = prompter.generate_history_GRPO(data_test.iloc[visit_idx - 1])

            input = prompter.generate_rethink_input_GRPO(data_test.iloc[visit_idx])

            output_medids = sorted(eval(data_test.iloc[visit_idx]['drug_id']))
            input_medids = eval(pred_data.iloc[visit_idx]['preds'])
            gt_medids = eval(pred_data.iloc[visit_idx]['labels'])
            assert gt_medids == output_medids

            input_med = []
            for medid in input_medids:
                input_med.append(f"{med_names[medid]} <DrugEmb-{medid}>")
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
            else:
                input_text_add = prompter.template["prompt_no_history_add"].format(
                    input=input,
                    pre_predicted=input_med
                )
                input_text_remove = prompter.template["prompt_no_history_remove"].format(
                    input=input,
                    pre_predicted=input_med
                )

            data_point_remove = {
                'prompt': input_text_remove,
                'input_drug_id': input_medids,
                'gt_id': gt_medids
            }
            json_data_remove.append(data_point_remove)
            data_point_add = {
                'prompt': input_text_add,
                'input_drug_id': input_medids,
                'gt_id': gt_medids
            }
            json_data_add.append(data_point_add)

        with open(output_path_remove, 'w') as f:
            json.dump(json_data_remove, f, indent=4)
        with open(output_path_add, 'w') as f:
            json.dump(json_data_add, f, indent=4)

if __name__ == '__main__':
    fire.Fire(generate_list_grpo_data)
