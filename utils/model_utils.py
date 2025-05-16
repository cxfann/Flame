import os
import sys
from typing import List
import fire
import numpy as np
import logging
import torch
import transformers
import json
import dill
from datasets import load_dataset
sys.path.append('./')


from transformers import PretrainedConfig, PreTrainedModel, AutoTokenizer, AutoModelForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

class MyConfig(PretrainedConfig):
    def __init__(self,
                llm_name="llama-2-7b",
                use_pat_embed=True,
                use_dias_embed=True,
                use_pro_embed=True,
                use_drug_embed=True,
                pat_emb_dim=512,
                pat_mlp_dim=1024,
                dias_emb_dim=64,
                dias_mlp_dim=512,
                pro_emb_dim=64,
                pro_mlp_dim=512,
                drug_emb_dim=300,
                drug_mlp_dim=1024,
                llm_emb_dim=4096,
                lora_r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                lora_target_modules=["q_proj", "v_proj"],
                pat_embed_table_path=None,
                dias_embed_table_path=None,
                pro_embed_table_path=None,
                drug_embed_table_path=None,
                **kwargs):

        super().__init__(**kwargs)
        self.llm_name = llm_name
        self.use_pat_embed = use_pat_embed
        self.use_dias_embed = use_dias_embed
        self.use_pro_embed = use_pro_embed
        self.use_drug_embed = use_drug_embed
        self.pat_emb_dim = pat_emb_dim
        self.pat_mlp_dim = pat_mlp_dim
        self.dias_emb_dim = dias_emb_dim
        self.dias_mlp_dim = dias_mlp_dim
        self.pro_emb_dim = pro_emb_dim
        self.pro_mlp_dim = pro_mlp_dim
        self.drug_emb_dim = drug_emb_dim
        self.drug_mlp_dim = drug_mlp_dim
        self.llm_emb_dim = llm_emb_dim
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.pat_embed_table_path = pat_embed_table_path
        self.dias_embed_table_path = dias_embed_table_path
        self.pro_embed_table_path = pro_embed_table_path
        self.drug_embed_table_path = drug_embed_table_path


class MyModel(PreTrainedModel):
    config_class = MyConfig
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
        self.devices = kwargs.get('devices', 'cuda')
        self.llm = AutoModelForCausalLM.from_pretrained(os.path.join("./Models", config.llm_name),
                                                        torch_dtype=torch.bfloat16,
                                                        use_flash_attention_2=kwargs.get("use_flash_attn", False),
                                                        )
        self.init_special_tokenizer()
        self.init_peft_model()
        self.load_projector()
        self.load_embedding_tables()
        if 'hadm_id_map' not in kwargs:
            raise ValueError('hadm_id_map must be provided in kwargs')
        self.hadm_id_map = kwargs.get('hadm_id_map')
        
    def print_trainable_parameters(self):
        print('*****trainable parameters:', sum(p.numel() for p in self.parameters() if p.requires_grad), '| total parameters:', sum(p.numel() for p in self.parameters()), '| trainable ratio: {:.5f}%'.format(sum(p.numel() for p in self.parameters() if p.requires_grad) / sum(p.numel() for p in self.parameters()) * 100))
        print('trainable para list: ')
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f'{param.numel()} -- {name}')

    def init_special_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join("./Models", self.config.llm_name))
        self.tokenizer.pad_token_id = (0)
        self.tokenizer.padding_side = "left"
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[PatEmb]','[DiasEmb]','[ProEmb]','[DrugEmb]']})
        self.llm.resize_token_embeddings(len(self.tokenizer))
        if self.devices == 'cuda' or self.devices == 'cuda:0':
            print('#######Successfully initialized special tokens')
            print('*****tokenizer:')
            print(self.tokenizer)
            print('*****resize llm model:')
            print(self.llm)

    def init_peft_model(self):
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, lora_config)

    def load_projector(self):
        if self.config.use_pat_embed:
            self.pat_projector = torch.nn.Sequential(
                torch.nn.Linear(self.config.pat_emb_dim, self.config.pat_mlp_dim),
                torch.nn.GELU(),
                torch.nn.Linear(self.config.pat_mlp_dim, self.config.llm_emb_dim),
            )
            self.pat_projector.requires_grad = True

        if self.config.use_dias_embed:
            self.dias_projector = torch.nn.Sequential(
                torch.nn.Linear(self.config.dias_emb_dim, self.config.dias_mlp_dim),
                torch.nn.GELU(),
                torch.nn.Linear(self.config.dias_mlp_dim, self.config.llm_emb_dim),
            )
            self.dias_projector.requires_grad = True

        if self.config.use_pro_embed:
            self.pro_projector = torch.nn.Sequential(
                torch.nn.Linear(self.config.pro_emb_dim, self.config.pro_mlp_dim),
                torch.nn.GELU(),
                torch.nn.Linear(self.config.pro_mlp_dim, self.config.llm_emb_dim),
            )
            self.pro_projector.requires_grad = True

        if self.config.use_drug_embed:
            self.drug_projector = torch.nn.Sequential(
                torch.nn.Linear(self.config.drug_emb_dim, self.config.drug_mlp_dim),
                torch.nn.GELU(),
                torch.nn.Linear(self.config.drug_mlp_dim, self.config.llm_emb_dim),
            )
            self.drug_projector.requires_grad = True

    def load_embedding_tables(self):
        if self.config.use_pat_embed:
            self.pat_embed_table = dill.load(open(self.config.pat_embed_table_path, "rb"))
        if self.config.use_dias_embed:
            self.dias_embed_table = dill.load(open(self.config.dias_embed_table_path, "rb"))
        if self.config.use_pro_embed:
            self.pro_embed_table = dill.load(open(self.config.pro_embed_table_path, "rb"))
        if self.config.use_drug_embed:
            self.drug_embed_table = dill.load(open(self.config.drug_embed_table_path, "rb"))

    def get_embedding(self, hadm_id):
        if self.config.use_pat_embed and (hadm_id // 1000) in self.pat_embed_table: 
            pat_emb = self.pat_embed_table[hadm_id // 1000]
            llm_pat_emb = self.pat_projector(pat_emb.to(self.pat_projector[0].weight.dtype).to(self.devices)).unsqueeze(0)
        else:
            llm_pat_emb = None

        if self.config.use_dias_embed and self.hadm_id_map[hadm_id]['dias_ids']:
            dias_ids = self.hadm_id_map[hadm_id]['dias_ids']
            dias_emb = torch.stack([self.dias_embed_table[id] for id in dias_ids], dim=0)
            # llm_dias_emb = self.dias_projector(dias_emb.cuda())
            llm_dias_emb = self.dias_projector(dias_emb.to(self.dias_projector[0].weight.dtype).to(self.devices))
        else:
            llm_dias_emb = None

        if self.config.use_pro_embed and self.hadm_id_map[hadm_id]['pro_ids']:
            pro_ids = self.hadm_id_map[hadm_id]['pro_ids']
            pro_emb = torch.stack([self.pro_embed_table[id] for id in pro_ids], dim=0)
            llm_pro_emb = self.pro_projector(pro_emb.to(self.pro_projector[0].weight.dtype).to(self.devices))
        else:
            llm_pro_emb = None

        if self.config.use_drug_embed and self.hadm_id_map[hadm_id]['drug_ids']:
            drug_ids = self.hadm_id_map[hadm_id]['drug_ids']
            drug_emb = torch.stack([self.drug_embed_table[id] for id in drug_ids], dim=0)
            llm_drug_emb = self.drug_projector(drug_emb.to(self.drug_projector[0].weight.dtype).to(self.devices))
        else:
            llm_drug_emb = None

        return llm_pat_emb, llm_dias_emb, llm_pro_emb, llm_drug_emb
    

    def forward(self, input_ids, attention_mask, labels, hadm_ids):
        input_embeds = self.llm.get_input_embeddings()(input_ids)  # (bs, seq_len, emb_dim)
        
        pat_token_id = self.tokenizer('[PatEmb]', return_tensors="pt",add_special_tokens=False).input_ids.item()
        dias_token_id = self.tokenizer('[DiasEmb]', return_tensors="pt",add_special_tokens=False).input_ids.item()
        pro_token_id = self.tokenizer('[ProEmb]', return_tensors="pt",add_special_tokens=False).input_ids.item()
        drug_token_id = self.tokenizer('[DrugEmb]', return_tensors="pt",add_special_tokens=False).input_ids.item()

        for idx, hadm_id in enumerate(hadm_ids):
            hadm_id_int = hadm_id.item()
            llm_pat_emb, llm_dias_emb, llm_pro_emb, llm_drug_emb = self.get_embedding(hadm_id_int)

            if self.config.use_pat_embed and (input_ids[idx] == pat_token_id).nonzero().shape[0] > 0:
                pat_positions = (input_ids[idx] == pat_token_id).nonzero().view(-1)
                for change_idx, pat_embed in zip(pat_positions, llm_pat_emb):
                    input_embeds[idx, change_idx] = pat_embed

            if self.config.use_dias_embed and (input_ids[idx] == dias_token_id).nonzero().shape[0] > 0:
                dias_positions = (input_ids[idx] == dias_token_id).nonzero().view(-1)
                for change_idx, dias_embed in zip(dias_positions, llm_dias_emb):
                    input_embeds[idx, change_idx] = dias_embed

            if self.config.use_pro_embed and (input_ids[idx] == pro_token_id).nonzero().shape[0] > 0:
                pro_positions = (input_ids[idx] == pro_token_id).nonzero().view(-1)
                for change_idx, pro_embed in zip(pro_positions, llm_pro_emb):
                    input_embeds[idx, change_idx] = pro_embed

            if self.config.use_drug_embed and (input_ids[idx] == drug_token_id).nonzero().shape[0] > 0:
                drug_positions = (input_ids[idx] == drug_token_id).nonzero().view(-1)
                for change_idx, drug_embed in zip(drug_positions, llm_drug_emb):
                    input_embeds[idx, change_idx] = drug_embed

        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs

    def generate(self, input_ids, attention_mask, hadm_ids, max_new_tokens, **kwargs):
        input_embeds = self.llm.get_input_embeddings()(input_ids)

        pat_token_id = self.tokenizer('[PatEmb]', return_tensors="pt",add_special_tokens=False).input_ids.item()
        dias_token_id = self.tokenizer('[DiasEmb]', return_tensors="pt",add_special_tokens=False).input_ids.item()
        pro_token_id = self.tokenizer('[ProEmb]', return_tensors="pt",add_special_tokens=False).input_ids.item()
        drug_token_id = self.tokenizer('[DrugEmb]', return_tensors="pt",add_special_tokens=False).input_ids.item()

        for idx, hadm_id in enumerate(hadm_ids):
            hadm_id_int = hadm_id.item()
            llm_pat_emb, llm_dias_emb, llm_pro_emb, llm_drug_emb = self.get_embedding(hadm_id_int)

            if self.config.use_pat_embed and (input_ids[idx] == pat_token_id).nonzero().shape[0] > 0:
                pat_positions = (input_ids[idx] == pat_token_id).nonzero().view(-1)
                for change_idx, pat_embed in zip(pat_positions, llm_pat_emb):
                    input_embeds[idx, change_idx] = pat_embed

            if self.config.use_dias_embed and (input_ids[idx] == dias_token_id).nonzero().shape[0] > 0:
                dias_positions = (input_ids[idx] == dias_token_id).nonzero().view(-1)
                for change_idx, dias_embed in zip(dias_positions, llm_dias_emb):
                    input_embeds[idx, change_idx] = dias_embed

            if self.config.use_pro_embed and (input_ids[idx] == pro_token_id).nonzero().shape[0] > 0:
                pro_positions = (input_ids[idx] == pro_token_id).nonzero().view(-1)
                for change_idx, pro_embed in zip(pro_positions, llm_pro_emb):
                    input_embeds[idx, change_idx] = pro_embed

            if self.config.use_drug_embed and (input_ids[idx] == drug_token_id).nonzero().shape[0] > 0:
                drug_positions = (input_ids[idx] == drug_token_id).nonzero().view(-1)
                for change_idx, drug_embed in zip(drug_positions, llm_drug_emb):
                    input_embeds[idx, change_idx] = drug_embed

        output_ids = self.llm.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens = True)

        return output_text

    @classmethod
    def from_pretrained(self, pretrained_model_name_or_path, *args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        model.llm = model.llm.merge_and_unload()
        model.init_peft_model()

        print(f'Successfully loaded model from {pretrained_model_name_or_path}')
        print('The lora module has been merged and unloaded')
        print('Created a new peft model with the lora module for continued training')
        print('*****model:')
        print(model)

        return model

        
        
        