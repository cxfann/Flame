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

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

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
    base_model_path: str = './Models/sft_rethink',
    lora_path: str = None,
    data_path: str = None,
    med_name2idx_path: str = "./data/med_name2idx.json",
    evaluate_result_path: str = None,
    task: str = None,
):
    if not os.path.exists(evaluate_result_path):
        os.makedirs(evaluate_result_path)

    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()
    model = model.to("cuda")

    data = load_dataset("json", data_files=data_path)
    test_data = data["train"]

    med_name2idx = json.load(open(med_name2idx_path, "r"))

    result_path = os.path.join(evaluate_result_path, f"{task}.csv")


    with open(result_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["input_list", "output_list", "gt_list", "error_names"])

    for i in tqdm(range(len(test_data))):
        tokenize_input = tokenizer(test_data[i]["prompt"], return_tensors="pt")
        output_token = model.generate(
            input_ids=tokenize_input["input_ids"].cuda(),
            attention_mask=tokenize_input["attention_mask"].cuda(),
            pad_token_id=0
        )
        output_token = output_token[0][len(tokenize_input["input_ids"][0]):]
        output_text = tokenizer.decode(output_token, skip_special_tokens=True)

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
            writer.writerow([input_list, output_list, gt_list, error_names])



if __name__ == '__main__':
    fire.Fire(evaluate_pretrain)