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


def evaluate_pretrain(
    model_path: str = None,
    data_path: str = None,
    med_name2idx_path: str = "data/data_process/output/mimic-iii/med_name2idx.json",
    evaluate_result_path: str = None,
    task: str = None,
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

    result_path = os.path.join(evaluate_result_path, f"{task}.csv")


    with open(result_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["hadm_id", "input_list", "output_list", "gt_list", "error_names"])

    for i in tqdm(range(len(test_data))):
        tokenize_input = tokenizer(test_data[i]["input"], return_tensors="pt")
        input_ids = tokenize_input["input_ids"].to("cuda")
        attention_mask = tokenize_input["attention_mask"].to("cuda")
        hadm_ids = torch.tensor([test_data[i]["hadm_id"]]).to("cuda")
        output_text = mymodel.generate(input_ids, attention_mask, hadm_ids, max_new_tokens=1000, pad_token_id=0)[0]

        hadm_id = test_data[i]["hadm_id"] // 1000
        input_list = test_data[i]["input_drug_id"]
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

        with open(result_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([hadm_id, input_list, output_list, gt_list, error_names])


if __name__ == '__main__':
    fire.Fire(evaluate_pretrain)