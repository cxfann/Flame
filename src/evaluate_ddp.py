import os
import sys
from typing import List
import fire
import json
import numpy as np
import logging
import torch
import transformers
import dill
import csv
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, PretrainedConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import Prompter, MyConfig, MyModel
import utils

from transformers import EarlyStoppingCallback, TrainerCallback, \
    TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def evaluate(
    model_path : str = None,
    data_path = None,
    cutoff_len = 5120,
    train_on_inputs = 0,
    med_num = 151,
    ddi_path = "./data/ddi_A_final.pkl",
    save_path = None,
):
    log_path = os.path.join(os.path.dirname(model_path), 'test.log')
    logger = utils.setup_logger(log_path, mode='a')
    train_on_inputs = True if train_on_inputs else False

    prompt_template_name = 'llama-2'
    prompter = Prompter(prompt_template_name)

    devices = 'cuda'
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        devices = f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}'

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        result_path = os.path.join(save_path, 'evaluation_result.csv')
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            with open(result_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['preds', 'labels'])

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

    ddi_matrix = dill.load(open(ddi_path, "rb"))

    mymodel = MyModel.from_pretrained(model_path, hadm_id_map=hadm_id_map)
    mymodel.to(devices)
    mymodel.eval()

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print('\nSuccessfully loaded trained model')
        print(mymodel, '\n')
        print(mymodel.config, '\n')

    tokenizer = mymodel.tokenizer


    if mymodel.config.llm_name == 'llama-2-7b':
        Yes_token_id = 8241
        No_token_id = 3782
    else:
        Yes_token_id = tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
        No_token_id = tokenizer("No", add_special_tokens=False)["input_ids"][0]
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"\nYes_token_id: {Yes_token_id}, No_token_id: {No_token_id}\n")

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=False,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["history"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["history"],
                data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could not speed up
            tokenized_full_prompt["hadm_ids"] = data_point["hadm_id"]
        return tokenized_full_prompt

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)
    
    data = (
        data["train"].map(generate_and_tokenize_prompt)
    )
    
    val_set_size = len(data)
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f'Successfully loaded data from {data_path} with {len(data)} data points'
            f'\nAn example of data point: {data[0]}'
        )

    data = data.select_columns(['input_ids', 'attention_mask', 'labels', 'hadm_ids'])

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"After selecting columns, an example of data point: {data[0]}"
        )

    class SavePeftModelCallback(TrainerCallback):
        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            kwargs["model"].save_pretrained(checkpoint_folder)

            return control    

    class MyEarlyStoppingCallback(EarlyStoppingCallback):
        def __init__(self, early_stopping_patience: int = 3, early_stopping_threshold=0.01):
            super().__init__(early_stopping_patience=early_stopping_patience, 
                                early_stopping_threshold=early_stopping_threshold)

        def on_evaluate(self, args, state, control, metrics, **kwargs):
            metric_to_check = args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics.get(metric_to_check)

            if metric_value is None:
                logger.warning(
                    f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                    " is disabled"
                )
                return

            self.check_metric_value(args, state, control, metric_value)
            if state.global_step > 640 and self.early_stopping_patience_counter >= self.early_stopping_patience:
                control.should_training_stop = True

    def calculate_ddi_rate(preds):
        ddi_count = 0
        total_count = 0
        idx_list = np.nonzero(preds)[0].tolist()
        for i in range(len(idx_list)):
            for j in range(i+1, len(idx_list)):
                ddi_count += ddi_matrix[idx_list[i], idx_list[j]]
                total_count += 1
        ddi_rate = ddi_count / total_count if total_count > 0 else 0
        return ddi_rate

    def metrics_per_visit(preds, labels):
        recall = (preds * labels).sum() / labels.sum() if labels.sum() > 0 else 0
        precision = (preds * labels).sum() / preds.sum() if preds.sum() > 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0
        jaccard = (preds * labels).sum() / ((preds + labels) > 0).sum() if (preds + labels).sum() > 0 else 0
        ddi_rate = calculate_ddi_rate(preds)
        num_drugs = preds.sum()
        num_drugs_gt = labels.sum()
        num_visits = preds.shape[0]
        return recall, precision, f1, jaccard, ddi_rate, num_drugs, num_drugs_gt, num_visits


    def metrics(preds, labels):
        recall_list, precision_list, f1_list, jaccard_list, ddi_list, \
            num_drugs_list, num_drugs_gt_list = [], [], [], [], [], [], []
        val_visit_num = val_set_size // med_num
        for i in range(val_visit_num):
            start = i * med_num
            end = (i + 1) * med_num
            recall, precision, f1, jaccard, ddi, num_drugs, num_drugs_gt, _ = metrics_per_visit(
                preds[start:end], labels[start:end])
            recall_list.append(recall)
            precision_list.append(precision)
            f1_list.append(f1)
            jaccard_list.append(jaccard)
            ddi_list.append(ddi)
            num_drugs_list.append(num_drugs)
            num_drugs_gt_list.append(num_drugs_gt)
            
            if save_path and int(os.environ.get("LOCAL_RANK", 0)) == 0:
                with open(result_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    pred_result = np.nonzero(preds[start:end])[0].tolist()
                    labels_result = np.nonzero(labels[start:end])[0].tolist()
                    writer.writerow([pred_result, labels_result])

        recall = np.mean(recall_list)
        precision = np.mean(precision_list)
        f1 = np.mean(f1_list)
        jaccard = np.mean(jaccard_list)
        ddi_rate = np.mean(ddi_list)
        num_drugs = np.mean(num_drugs_list)
        num_drugs_gt = np.mean(num_drugs_gt_list)

        return recall, precision, f1, jaccard, ddi_rate, num_drugs, num_drugs_gt
    

    def compute_metrics(eval_preds):
        (preds, labels), _ = eval_preds
        recall, precision, f1, jaccard, ddi, num_drugs, num_drugs_gt = metrics(preds, labels)
        return {
            "num_drugs": num_drugs,
            "num_drugs_gt": num_drugs_gt,
            "jaccard": round(jaccard, 4),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "ddi_rate": round(ddi, 4),
        }


    def preprocess_logits_for_metrics(logits, labels):
        """
        This function is used to preprocess logits for the compute_metrics function.
        Output: 
            logits: [batch_size], binary logits, 0 for No_token_id (No), 1 for Yes_token_id (Yes), \
                indicating whether the drug is predicted or not
            labels: [batch_size], binary labels, \
                indicating whether the drug is prescribed or not
        """
        # labels_index = torch.argwhere(torch.bitwise_or(labels == Yes_token_id, labels == No_token_id))

        # get the last index of Yes_token_id or No_token_id in each row

        if isinstance(logits, tuple):
            logits = logits[0]

        condition = torch.bitwise_or(labels == Yes_token_id, labels == No_token_id)
        indices = torch.nonzero(condition)
        result = []

        for i in range(labels.size(0)):
            row_indices = indices[indices[:, 0] == i][:, 1]
            if len(row_indices) > 0:
                last_index = row_indices[-1]
                if labels[i, last_index] == Yes_token_id or labels[i, last_index] == No_token_id:
                    result.append([i, last_index])

        labels_index = torch.tensor(result)

        gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == No_token_id, 0, 1)
        labels_index[: , 1] = labels_index[: , 1] - 1
        logits = logits.softmax(dim=-1)
        logits = torch.softmax(logits[labels_index[:, 0], labels_index[:, 1]][:,[No_token_id, Yes_token_id]], dim = -1)
        logits = torch.argmax(logits, dim=-1)
        return logits, gold

    trainer = transformers.Trainer(
        model=mymodel,
        eval_dataset=data,
        args=transformers.TrainingArguments(
            output_dir=os.path.join(os.path.dirname(model_path), 'evaluation'),
            per_device_eval_batch_size=8,
            bf16=True,
            optim="adamw_torch",
            ddp_find_unused_parameters=False if ddp else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    result = trainer.evaluate()
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f'finished evaluation')
        print(result)
    
    logger.info(f'finished evaluation')
    logger.info(result)

if __name__ == '__main__':
    fire.Fire(evaluate)