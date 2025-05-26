import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from utils import Prompter, MyConfig, MyModel
import utils


def model_process(original_model_path="outputs/list_sft/checkpoint-XXXX",
                    output_model_path="Models/list_sft",):
    sft_model = MyModel.from_pretrained(original_model_path, hadm_id_map = {})
    grpo_model = sft_model.llm
    grpo_tokenizer = sft_model.tokenizer
    special_tokens = []

    for dias_id in sft_model.dias_embed_table.keys():
        special_tokens.append(f"<DiasEmb-{dias_id}>")
    for pro_id in sft_model.pro_embed_table.keys():
        special_tokens.append(f"<ProEmb-{pro_id}>")
    for drug_id in sft_model.drug_embed_table.keys():
        special_tokens.append(f"<DrugEmb-{drug_id}>")
    for pat_id in sft_model.pat_embed_table.keys():
        special_tokens.append(f"<PatEmb-{pat_id}>")

    grpo_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    grpo_model.resize_token_embeddings(len(grpo_tokenizer))
        
    grpo_model.to("cuda")
    pat_projector, dias_projector, pro_projector, drug_projector = sft_model.pat_projector, sft_model.dias_projector, sft_model.pro_projector, sft_model.drug_projector

    pat_projector.to("cuda")
    dias_projector.to("cuda")
    pro_projector.to("cuda")
    drug_projector.to("cuda")

    for pat_id, pat_embed in sft_model.pat_embed_table.items():
        pat_embed = torch.tensor(pat_embed).to("cuda")
        pat_embed = pat_projector(pat_embed.to(pat_projector[0].weight.dtype))
        pat_tokenizer_id = grpo_tokenizer.convert_tokens_to_ids(f"<PatEmb-{pat_id}>")
        pat_embed.to(grpo_model.base_model.model.model.embed_tokens.weight.dtype)
        grpo_model.base_model.model.model.embed_tokens.weight[pat_tokenizer_id] = pat_embed

    for dias_id, dias_embed in sft_model.dias_embed_table.items():
        dias_embed = torch.tensor(dias_embed).to("cuda")
        dias_embed = dias_projector(dias_embed.to(dias_projector[0].weight.dtype))
        dias_tokenizer_id = grpo_tokenizer.convert_tokens_to_ids(f"<DiasEmb-{dias_id}>")
        dias_embed.to(grpo_model.base_model.model.model.embed_tokens.weight.dtype)
        grpo_model.base_model.model.model.embed_tokens.weight[dias_tokenizer_id] = dias_embed

    for pro_id, pro_embed in sft_model.pro_embed_table.items():
        pro_embed = torch.tensor(pro_embed).to("cuda")
        pro_embed = pro_projector(pro_embed.to(pro_projector[0].weight.dtype))
        pro_tokenizer_id = grpo_tokenizer.convert_tokens_to_ids(f"<ProEmb-{pro_id}>")
        pro_embed.to(grpo_model.base_model.model.model.embed_tokens.weight.dtype)
        grpo_model.base_model.model.model.embed_tokens.weight[pro_tokenizer_id] = pro_embed

    for drug_id, drug_embed in sft_model.drug_embed_table.items():
        drug_embed = torch.tensor(drug_embed).to("cuda")
        drug_embed = drug_projector(drug_embed.to(drug_projector[0].weight.dtype))
        drug_tokenizer_id = grpo_tokenizer.convert_tokens_to_ids(f"<DrugEmb-{drug_id}>")
        drug_embed.to(grpo_model.base_model.model.model.embed_tokens.weight.dtype)
        grpo_model.base_model.model.model.embed_tokens.weight[drug_tokenizer_id] = drug_embed
    grpo_model.merge_and_unload()

    grpo_model.base_model.model.save_pretrained(output_model_path)
    grpo_tokenizer.save_pretrained(output_model_path)