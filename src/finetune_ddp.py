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
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, PretrainedConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import Prompter, MyModel, MyConfig
import utils

from transformers import EarlyStoppingCallback, TrainerCallback, \
    TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def finetune_ddp(
    # model & data hyperparams
    llm_name = None,
    checkpoint_path = None, # if not None, resume training from this checkpoint, use from_pretrained method of the class MyModel
    data_path = None,
    output_dir = None,
    load_in_8bit = False,
    eval_epochs = 32,
    ddi_path = "data/data_process/output/mimic-iii/ddi_A_final.pkl",
    # embedding hyperparams
    use_pat_embed = True,
    use_dias_embed = True,
    use_pro_embed = True,
    use_drug_embed = True,
    pat_embed_table_path= None,
    dias_embed_table_path= None,
    pro_embed_table_path= None,
    drug_embed_table_path= None,
    # training hyperparams
    batch_size = 64,
    micro_batch_size = 1,
    num_epochs = 10,
    learning_rate = 5e-4,
    cutoff_len = 5120,
    med_num = 151,
    val_set_size = 2368,
    random_jaccard = 0.0,
    early_stopping_patience = 25,
    # lora hyperparams
    lora_r = 8,
    lora_alpha = 16,
    lora_dropout = 0.05,
    lora_target_modules = ["q_proj", "v_proj"],
    # llm hyperparams
    train_on_inputs = 0,  # if False, masks out inputs in loss
    resume_from_checkpoint = 0,  # either training checkpoint or final adapter
    group_by_length = False,  # faster, but produces an odd training loss curve
    use_flash_attn = True,  # whether to use Flash Attention
    logging_dir = None
):
    if data_path is None:
        raise ValueError("data_path is required")
    if output_dir is None:
        raise ValueError("output_dir is required")
    if use_pat_embed and not pat_embed_table_path:
        raise ValueError("Please specify a --pat_embed_table_path while using patient embeddings")
    if use_dias_embed and not dias_embed_table_path:
        raise ValueError("Please specify a --dias_embed_table_path while using diagnosis embeddings")
    if use_pro_embed and not pro_embed_table_path:
        raise ValueError("Please specify a --pro_embed_table_path while using procedure embeddings")
    if use_drug_embed and not drug_embed_table_path:
        raise ValueError("Please specify a --drug_embed_table_path while using drug embeddings")
    if llm_name is None and checkpoint_path is None:
        raise ValueError("llm_name or checkpoint_path is required")

    log_file = os.path.join(output_dir, "train.log")
    logger = utils.setup_logger(log_file, mode='a')

    train_on_inputs = True if train_on_inputs else False
    resume_from_checkpoint = True if resume_from_checkpoint else False
    gradient_accumulation_steps = batch_size // micro_batch_size
    prompt_template_name = 'llama-2'
    prompter = Prompter(prompt_template_name)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"\nTraining LLM model with params:\n"
            f"llm_name: {llm_name}\n"
            f"checkpoint_path: {checkpoint_path}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"ddi_path: {ddi_path}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"eval_epochs: {eval_epochs}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"early_stopping_patience: {early_stopping_patience}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"use_flash_attn: {use_flash_attn}\n"
            f"\nembedding hyperparams:\n"
            f"use_pat_embed: {use_pat_embed}\n"
            f"use_dias_embed: {use_dias_embed}\n"
            f"use_pro_embed: {use_pro_embed}\n"
            f"use_drug_embed: {use_drug_embed}\n"
            f"pat_embed_table_path: {pat_embed_table_path}\n"
            f"dias_embed_table_path: {dias_embed_table_path}\n"
            f"pro_embed_table_path: {pro_embed_table_path}\n"
            f"drug_embed_table_path: {drug_embed_table_path}\n"  
            f'\nsome other hyperparams:\n'
            f'med_num: {med_num}\n'
            f'random_jaccard: {random_jaccard}\n'
        )
    
    devices = 'cuda'
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        devices = f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}'
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

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

    if checkpoint_path is None:
        myconfig = MyConfig(
                        llm_name=llm_name,
                        use_pat_embed=use_pat_embed,
                        use_dias_embed=use_dias_embed,
                        use_pro_embed=use_pro_embed,
                        use_drug_embed=use_drug_embed,
                        pat_embed_table_path=pat_embed_table_path,
                        dias_embed_table_path=dias_embed_table_path,
                        pro_embed_table_path=pro_embed_table_path,
                        drug_embed_table_path=drug_embed_table_path,
                        lora_r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        lora_target_modules=lora_target_modules,
                        )

        mymodel = MyModel(myconfig, use_flash_attn = use_flash_attn, hadm_id_map = hadm_id_map, devices = devices)
    else:
        mymodel = MyModel.from_pretrained(checkpoint_path, hadm_id_map=hadm_id_map, devices = devices)
        myconfig = mymodel.config
    # mymodel.cuda()
    mymodel.to(devices)
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print('\nSuccessfully loaded specified model and config\n')
        print(mymodel, '\n')
        print(myconfig, '\n')
        mymodel.print_trainable_parameters()

    tokenizer = mymodel.tokenizer
    if myconfig.llm_name == "llama-2-7b":
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
        
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=False
        )
        train_data = (
            train_val["train"].map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].map(generate_and_tokenize_prompt)
        val_data = None

    # shuffle the training and validation data
    train_data = train_data.shuffle(seed=42)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training on {len(train_data)} samples, validating on {len(val_data)} samples\n"
            f"an example of the training data: {train_data[0]}"
        )

    # After map, select only the needed columns
    train_data = train_data.select_columns(['input_ids', 'attention_mask', 'labels', 'hadm_ids'])

    if val_data:
        val_data = val_data.select_columns(['input_ids', 'attention_mask', 'labels', 'hadm_ids'])
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"After selecting columns, an example of the training data: {train_data[0]}\n"
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
            if state.global_step > 640 and self.early_stopping_patience_counter >= self.early_stopping_patience and state.best_metric > random_jaccard:
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
        recall = np.mean(recall_list)
        precision = np.mean(precision_list)
        f1 = np.mean(f1_list)
        jaccard = np.mean(jaccard_list)
        ddi = np.mean(ddi_list)
        num_drugs = np.mean(num_drugs_list)
        num_drugs_gt = np.mean(num_drugs_gt_list)

        return recall, precision, f1, jaccard, ddi, num_drugs, num_drugs_gt

    # Metric
    if med_num == 1:
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print("*****med_num == 1, using special compute_metrics function*****")
        def compute_metrics(eval_preds):
            (preds, labels), _ = eval_preds
            recall, precision, f1, jaccard, num_drugs, num_drugs_gt, num_visits = metrics_per_visit(preds, labels)
            return {
                "num_pred": num_drugs,
                "num_gt": num_drugs_gt,
                "num_total_visits": num_visits,
                "jaccard": round(jaccard, 4),
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "f1": round(f1, 4),
            }        

    else:
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
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_strategy="steps",
            # warmup_ratio=0.1,
            warmup_steps=10,
            logging_steps=eval_epochs,
            save_steps=eval_epochs,
            eval_steps=eval_epochs if val_set_size > 0 else None,
            save_total_limit=1,
            metric_for_best_model='jaccard',
            # fp16=True,
            bf16=True,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            optim="adamw_torch",
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="tensorboard",
            log_level="info",
            ignore_data_skip=True,
            logging_dir=logging_dir,
            lr_scheduler_type='inverse_sqrt',
            log_on_each_node=False,
            logging_first_step=True,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[SavePeftModelCallback, 
                    MyEarlyStoppingCallback(
                        early_stopping_patience=early_stopping_patience,
                        early_stopping_threshold=0.0),
                #    EvaluateFirstStepCallback()
                    ],
    )

    mymodel.llm.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"Training on {data_path} complete, best metric: {trainer.state.best_metric}, saved in: {trainer.state.best_model_checkpoint}")
    logger.info(f"Training on {data_path} complete, best metric: {trainer.state.best_metric}, saved in: {trainer.state.best_model_checkpoint}")

if __name__ == "__main__":
    fire.Fire(finetune_ddp)
