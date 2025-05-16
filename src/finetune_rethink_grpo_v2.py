import os
import sys
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from typing import List
import fire
import json
import numpy as np
import dill
import torch
from datasets import load_dataset
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def finetune_rethink_grpo(
    # model & data hyperparams
    grpo_model_path = None,
    data_path = None,
    output_dir = None,
    # training hyperparams
    batch_size = 1,
    gradient_accumulation_steps = 1,
    num_epochs = 1,
    num_generations = 8,
    learning_rate = 5e-6,
    val_set_size = 0,
    save_steps = 128,
    use_vllm = False,
    # lora hyperparams
    use_lora = True,
    lora_r = 8,
    lora_alpha = 16,
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # GRPO hyperparams
    alpha = 0,
    beta = 0.5,
    step_reward_weight = 0.5
):
    if data_path is None:
        raise ValueError("data_path is required")
    if output_dir is None:
        raise ValueError("output_dir is required")


    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"\nTraining LLM model with params:\n"
            f"  grpo_model_path: {grpo_model_path}\n"
            f"  data_path: {data_path}\n"
            f"  output_dir: {output_dir}\n"
            f"  batch_size: {batch_size}\n"
            f"  gradient_accumulation_steps: {gradient_accumulation_steps}\n"
            f"  num_epochs: {num_epochs}\n"
            f"  num_generations: {num_generations}\n"
            f"  learning_rate: {learning_rate}\n"
            f"  val_set_size: {val_set_size}\n"
            f"  save_steps: {save_steps}\n"
            f"  use_vllm: {use_vllm}\n"
            f"  use_lora: {use_lora}\n"
            f"  lora_r: {lora_r}\n"
            f"  lora_alpha: {lora_alpha}\n"
            f"  lora_target_modules: {lora_target_modules}\n"
            f"  alpha: {alpha}\n"
            f"  beta: {beta}\n"
            f"  step_reward_weight: {step_reward_weight}\n"
        )
    
    if use_lora:
        grpo_model, grpo_tokenizer = FastLanguageModel.from_pretrained(
            model_name=grpo_model_path,
            max_seq_length=5120,
            fast_inference=True if use_vllm else False,
            max_lora_rank=lora_r
        )

        grpo_model = FastLanguageModel.get_peft_model(
            grpo_model,
            r = lora_r,
            target_modules = lora_target_modules,
            lora_alpha = lora_alpha,
            use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        )
    else:
        grpo_model, grpo_tokenizer = FastLanguageModel.from_pretrained(
            model_name=grpo_model_path,
            max_seq_length=5120,
            fast_inference=True if use_vllm else False
        )

    data = load_dataset("json", data_files=data_path)
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"]
        val_data = train_val["test"]
    else:
        train_data = data["train"]
        val_data = None

    train_data = train_data.shuffle(seed=42)

    print(
        f"Training on {len(train_data)} samples\n"
        f"an example of the training data: {train_data[0]}"
    )

    ddi_path = "./data/ddi_A_final.pkl"
    med_name2idx_path = "./data/med_name2idx.json"

    ddi_metric = dill.load(open(ddi_path, "rb"))
    med_name2idx = json.load(open(med_name2idx_path, "r"))

    def get_jaccard(pred, gt):

        if len(gt) == 0 or len(pred) == 0:
            return 0
        
        return len(set(pred) & set(gt)) / len(set(pred) | set(gt))

    def get_ddi(pred):

        ddi_count = 0
        total_count = 0
        for i in range(len(pred)):
            for j in range(i+1, len(pred)):
                ddi_count += ddi_metric[pred[i]][pred[j]]
                total_count += 1
        ddi = ddi_count / total_count if total_count != 0 else 0

        return ddi

    def get_completion_id_and_refuses(completion):

        if len(completion) == 0:
            return [], [0]
        
        pred_names = completion.split("\n")
        pred_ids = []
        refuses = []
        for pred_name in pred_names:
            pred_name_clean = pred_name.strip().lower().replace(" ", "").replace("-", "")
            matched = False
            for med_name, idx in med_name2idx.items():
                med_name_clean = med_name.lower().replace(" ", "").replace("-", "")
                if pred_name_clean == med_name_clean:
                    pred_ids.append(idx)
                    refuses.append(0)
                    matched = True
                    break
            if not matched:
                pred_ids.append(-1)
                refuses.append(1)

        return pred_ids, refuses


    # def jaccard_reward(prompts, completions, task, input_drug_id, gt_id):
    #     rewards = []
    #     for completion, t, in_drugs, gt_drugs in zip(completions, task, input_drug_id, gt_id):
    #         pred_ids, refuse_rate = get_completion_id_and_refuse_rate(completion)
    #         original_jaccard = get_jaccard(in_drugs, gt_drugs)
    #         if t == 'add':
    #             modidied_drugs = sorted(list(set(in_drugs) | set(pred_ids)))
    #         elif t == 'remove':
    #             modidied_drugs = sorted(list(set(in_drugs) - set(pred_ids)))
    #         else:
    #             raise ValueError(f"Invalid task type. Expected 'add' or 'remove', got {t}.")
    #         modified_jaccard = get_jaccard(modidied_drugs, gt_drugs)
            
    #         reward = modified_jaccard - original_jaccard
    #         rewards.append(reward)

    #     return rewards

    def jaccard_reward(prompts, completions, task, input_drug_id, gt_id):
        total_reward = []
        step_rewards = []
        for completion, t, in_drugs, gt_drugs in zip(completions, task, input_drug_id, gt_id):
            pred_ids, refuses = get_completion_id_and_refuses(completion)
            part_modified_drugs = in_drugs.copy()
            step_potential = []
            step_potential.append(get_jaccard(part_modified_drugs, gt_drugs))
            for end_idx, pred_id in enumerate(pred_ids):
                if pred_id != -1:
                    if t == 'add':
                        part_modified_drugs = sorted(list(set(part_modified_drugs) | set([pred_id])))
                    elif t == 'remove':
                        part_modified_drugs = sorted(list(set(part_modified_drugs) - set([pred_id])))
                    else:
                        raise ValueError(f"Invalid task type. Expected 'add' or 'remove', got {t}.")
                step_potential.append(get_jaccard(part_modified_drugs, gt_drugs))
            step_reward = []
            for i in range(len(step_potential) - 1):
                step_reward.append(step_potential[i + 1] - step_potential[i])
            step_rewards.append(step_reward)

            total_reward.append(step_potential[-1] - step_potential[0])
        return total_reward, step_rewards
                
                

    # def ddi_reward(prompts, completions, task, input_drug_id, gt_id):
    #     rewards = []
    #     for completion, t, in_drugs, gt_drugs in zip(completions, task, input_drug_id, gt_id):
    #         pred_ids, refuse_rate = get_completion_id_and_refuse_rate(completion)
    #         original_ddi = get_ddi(in_drugs)
    #         if t == 'add':
    #             modidied_drugs = sorted(list(set(in_drugs) | set(pred_ids)))
    #         elif t == 'remove':
    #             modidied_drugs = sorted(list(set(in_drugs) - set(pred_ids)))
    #         else:
    #             raise ValueError(f"Invalid task type. Expected 'add' or 'remove', got {t}.")
    #         modified_ddi = get_ddi(modidied_drugs)
            
    #         reward = - modified_ddi + original_ddi
    #         rewards.append(reward)

    #     return rewards

    def ddi_reward(prompts, completions, task, input_drug_id, gt_id):
        total_reward = []
        step_rewards = []
        for completion, t, in_drugs, gt_drugs in zip(completions, task, input_drug_id, gt_id):
            pred_ids, refuses = get_completion_id_and_refuses(completion)
            part_modified_drugs = in_drugs.copy()
            step_potential = []
            step_potential.append(get_ddi(part_modified_drugs))
            for end_idx, pred_id in enumerate(pred_ids):
                if pred_id != -1:
                    if t == 'add':
                        part_modified_drugs = sorted(list(set(part_modified_drugs) | set([pred_id])))
                    elif t == 'remove':
                        part_modified_drugs = sorted(list(set(part_modified_drugs) - set([pred_id])))
                    else:
                        raise ValueError(f"Invalid task type. Expected 'add' or 'remove', got {t}.")
                step_potential.append(get_ddi(part_modified_drugs))
            step_reward = []
            for i in range(len(step_potential) - 1):
                step_reward.append(-step_potential[i + 1] + step_potential[i])
            step_rewards.append(step_reward)

            total_reward.append(-step_potential[-1] + step_potential[0])
        return total_reward, step_rewards

    # def refuse_rate_reward(prompts, completions, task, input_drug_id, gt_id):
    #     rewards = []
    #     for completion, t, in_drugs, gt_drugs in zip(completions, task, input_drug_id, gt_id):
    #         pred_ids, refuse_rate = get_completion_id_and_refuse_rate(completion)
    #         reward = -refuse_rate
    #         rewards.append(reward)

    #     return rewards

    def refuse_rate_reward(prompts, completions, task, input_drug_id, gt_id):
        total_reward = []
        step_rewards = []
        for completion, t, in_drugs, gt_drugs in zip(completions, task, input_drug_id, gt_id):
            pred_ids, refuses = get_completion_id_and_refuses(completion)
            step_reward = []
            refuse_num = 0
            for i in range(len(refuses)):
                refuse_num += refuses[i]
                step_reward.append(-refuse_num / (i + 1))
            step_rewards.append(step_reward)
            if len(refuses) == 0:
                total_reward.append(0)
            else:
                total_reward.append(-refuse_num / len(refuses))
        return total_reward, step_rewards
            


    from trl import GRPOConfig, GRPOTrainer

    trainer = GRPOTrainer(
        model=grpo_model,
        processing_class=grpo_tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        reward_funcs=[
            jaccard_reward,
            ddi_reward,
            refuse_rate_reward
        ],
        step_reward_weight=step_reward_weight,
        args=GRPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_strategy="steps",
            num_generations=num_generations,
            max_prompt_length=2048,
            max_completion_length=512,
            warmup_steps=200,
            logging_steps=10,
            save_steps=save_steps,
            bf16=True,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            report_to="tensorboard",
            optim="adamw_torch",
            log_level="info",
            ignore_data_skip=True,
            lr_scheduler_type='cosine',
            reward_weights=[1.0, alpha, beta]
        )
    )

    grpo_model.config.use_cache = False
    trainer.train()

    print(f"\nTraining finished. Saving model to {output_dir}")


if __name__ == "__main__":
    fire.Fire(finetune_rethink_grpo)
