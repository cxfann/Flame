import os
import sys
from typing import List
import fire
import json
import numpy as np
import logging
import torch
import transformers
import csv
import dill
from datasets import load_dataset
from tqdm import tqdm

from transformers import LlamaForCausalLM, LlamaTokenizer, PretrainedConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import Prompter, MyConfig, MyModel
import utils

from transformers import EarlyStoppingCallback, TrainerCallback, \
    TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

ddi_path = "./data/ddi_A_final.pkl"
ddi_metric = dill.load(open(ddi_path, "rb"))

def cal_metric(pred, gt):
    # pred: list of predicted labels
    # gt: list of ground truth labels
    # cal jacard, precision, recall, f1
    if len(gt) == 0:
        return 0, 0, 0, 0, 0, 0
    tp = len(set(pred) & set(gt))
    fp = len(set(pred) - set(gt))
    fn = len(set(gt) - set(pred))
    jacard = tp / (tp + fp + fn)    
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    ddi_count = 0
    total_count = 0
    for i in range(len(pred)):
        for j in range(i+1, len(pred)):
            ddi_count += ddi_metric[pred[i]][pred[j]]
            total_count += 1
    ddi = ddi_count / total_count if total_count != 0 else 0

    ddi_count = 0
    total_count = 0
    for i in range(len(gt)):
        for j in range(i+1, len(gt)):
            ddi_count += ddi_metric[gt[i]][gt[j]]
            total_count += 1
    ddi_gt = ddi_count / total_count if total_count != 0 else 0


    return jacard, precision, recall, f1, ddi, ddi_gt

def evaluate_pretrain(
    model_path: str = None,
    data_path: str = None,
    med_name2idx_path: str = "./data/med_name2idx.json",
    main_log_path: str = None,
):
    
    hadm_id_map = {}
    with open(data_path, "r") as f:
        data = json.load(f)
        for i in range(len(data)):
            hadm_id = data[i]["hadm_id"]
            if hadm_id not in hadm_id_map.keys():
                hadm_id_map[hadm_id] = {
                    "dias_ids": data[i]["diag_id"],
                    "pro_ids": data[i]["pro_id"],
                    "drug_ids": data[i]["drug_id"],
                }

    mymodel = MyModel.from_pretrained(model_path, hadm_id_map=hadm_id_map)
    mymodel.to("cuda")
    mymodel.eval()
    tokenizer = mymodel.tokenizer

    data = load_dataset("json", data_files=data_path)
    test_data = data["train"]

    med_name2idx = json.load(open(med_name2idx_path, "r"))

    result_path = os.path.join(model_path, "evaluate.csv")

    with open(result_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["hadm_id", "input_list", "output_list", "gt_list", "error_names"])

    jaccard_list, precision_list, recall_list, f1_list, ddi_list, ddi_gt_list = [], [], [], [], [], []

    for i in tqdm(range(len(test_data))):
        tokenize_input = tokenizer(test_data[i]["input"], return_tensors="pt")
        input_ids = tokenize_input["input_ids"].to("cuda")
        attention_mask = tokenize_input["attention_mask"].to("cuda")
        hadm_ids = torch.tensor([test_data[i]["hadm_id"]]).to("cuda")
        output_text = mymodel.generate(input_ids, attention_mask, hadm_ids, max_new_tokens=1000, pad_token_id=0)[0]

        hadm_id = test_data[i]["hadm_id"] // 1000
        input_list = test_data[i]["drug_id"]
        output_list = []
        gt_list = test_data[i]["gt_id"]
        error_names = []

        output_text_list = output_text.split("\n")
        for line in output_text_list:
            line_clean = line.strip().lower().replace(" ", "").replace("-", "")
            matched = False
            for med_name, idx in med_name2idx.items():
                med_name_clean = med_name.lower().replace(" ", "").replace("-", "")
                if line_clean == med_name_clean:
                    output_list.append(idx)
                    matched = True
                    break
            if not matched:
                error_names.append(line)

        jacard, precision, recall, f1, ddi, ddi_gt = cal_metric(output_list, gt_list)
        jaccard_list.append(jacard)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        ddi_list.append(ddi)
        ddi_gt_list.append(ddi_gt)

        if error_names:
            print(f"***output error***")
            print(f"hadm_id: {hadm_id}: {output_text}")

        with open(result_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([hadm_id, input_list, output_list, gt_list, error_names])
        
    avg_jaccard = np.mean(jaccard_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    avg_ddi = np.mean(ddi_list)
    avg_ddi_gt = np.mean(ddi_gt_list)

    print(f'Finish evaluate model {model_path}')
    print(f'on {len(test_data)} samples, data: {data_path}')
    print(f'jaccard: {avg_jaccard}')
    print(f'precision: {avg_precision}')
    print(f'recall: {avg_recall}')
    print(f'f1: {avg_f1}')
    print(f'ddi: {avg_ddi}')
    print(f'ddi_gt: {avg_ddi_gt}')
    print(f'output result to {result_path}')

    if main_log_path:
        with open(main_log_path, "a") as f:
            f.write(f'*evaluate model {model_path} - jaccard: {avg_jaccard}, precision: {avg_precision}, recall: {avg_recall}, f1: {avg_f1}, ddi: {avg_ddi}, ddi_gt: {avg_ddi_gt}\n')
        


if __name__ == '__main__':
    fire.Fire(evaluate_pretrain)