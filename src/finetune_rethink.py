import os
import sys
from typing import List
import fire
import json
import numpy as np
import logging
import torch
import transformers
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

def finetune_rethink(
    # model & data hyperparams
    llm_name = None,
    checkpoint_path = None,
    data_path = None,
    output_dir = None,
    save_steps = 256,
    fix_mlp = False,
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
    val_set_size = 0,
    limit_checkpoints = False,
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


    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"\nTraining LLM model with params:\n"
            f"llm_name: {llm_name}\n"
            f"checkpoint_path: {checkpoint_path}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"save_steps: {save_steps}\n"
            f"fix_mlp: {fix_mlp}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"use_flash_attn: {use_flash_attn}\n"
            f"limit_checkpoints: {limit_checkpoints}\n"
            f"\nembedding hyperparams:\n"
            f"use_pat_embed: {use_pat_embed}\n"
            f"use_dias_embed: {use_dias_embed}\n"
            f"use_pro_embed: {use_pro_embed}\n"
            f"use_drug_embed: {use_drug_embed}\n"
            f"pat_embed_table_path: {pat_embed_table_path}\n"
            f"dias_embed_table_path: {dias_embed_table_path}\n"
            f"pro_embed_table_path: {pro_embed_table_path}\n"
            f"drug_embed_table_path: {drug_embed_table_path}\n"  
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
    mymodel.to(devices)

    if fix_mlp:
        for name, param in mymodel.named_parameters():
            if "projector" in name:
                param.requires_grad = False
                if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    print(f"Freezing {name}")
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print('\nSuccessfully loaded specified model and config\n')
        print(mymodel, '\n')
        print(myconfig, '\n')
        mymodel.print_trainable_parameters()

    tokenizer = mymodel.tokenizer

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
        full_prompt = f'{data_point["input"]}{data_point["output"]}'
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            tokenized_input = tokenize(data_point["input"], add_eos_token=False)
            input_len = len(tokenized_input["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * input_len + tokenized_full_prompt["labels"][
                input_len:
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
            f"Training on {len(train_data)} samples\n"
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
            logging_steps=10,
            eval_steps=None,
            save_steps=save_steps,
            # fp16=True,
            bf16=True,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            save_total_limit=1 if limit_checkpoints else 200,
            optim="adamw_torch",
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=True if ddp else None,
            group_by_length=group_by_length,
            report_to="tensorboard",
            log_level="info",
            ignore_data_skip=True,
            logging_dir=logging_dir,
            lr_scheduler_type='inverse_sqrt',
            log_on_each_node=False,
            logging_first_step=True,
            deepspeed="./deepspeed/ds_z2_config.json"
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
    )

    mymodel.llm.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

if __name__ == "__main__":
    fire.Fire(finetune_rethink)
