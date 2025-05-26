# Fine-grained List-wise Alignment for Generative Medication Recommendation

This repository provides the official PyTorch implementation and reproduction for our paper titled **"Fine-grained List-wise Alignment for Generative Medication Recommendation"**. 

## Installation

### 1. Clone this git repository and change directory to this repository

### 2. A new [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) is suggested. 

```bash
conda create -n FLAME python=3.10
```

### 3. Activate the newly created environment.

```bash
conda activate FLAME
```

### 4. Install the required modules.

```bash
sh install.sh
```

## Data Preparation

### 1. MIMIC-III Dataset

- You must have obtained access to [MIMIC-III](https://physionet.org/content/mimiciii/).

- Download the following CSV files and place them under: `data/data_process/input/mimic-iii` directory
   - `ADMISSIONS.csv`  
   - `DIAGNOSES_ICD.csv`  
   - `PROCEDURES_ICD.csv`  
   - `PATIENTS.csv`  
   - `PRESCRIPTIONS.csv`  
   - `NOTEEVENTS.csv`

### 2. DrugBank Data
- Download the `drugbank_drugs_info.csv` file, and and place it under: `data/data_process/input`

## Preprocess the Data

### Step 1: Filter & process raw MIMIC-III data

```bash
python data/process.py
```
- Output: `data/data_process/output/mimic-iii/*`

### Step 2: Process unstructured clinical notes

- Requires GPT API access.
- Run the notebook:

```
data/generate_note.ipynb
```

- Output: `data/data_process/output/mimic-iii/data4LLM_with_note.csv`

> This step will add a `NOTE` column to the records in `data/data_process/output/mimic-iii/data4LLM.csv`, resulting in a new file:  `data/data_process/output/mimic-iii/data4LLM_with_note.csv`.

### Step 3: Generate external embeddings

- Run the following notebooks:

```
data/saved_embedding/get_embed/get_embed_diagpro.ipynb  
data/saved_embedding/get_embed/get_embed_med.ipynb  
data/saved_embedding/get_embed/get_embed_pat.ipynb
```

- Output files:

```
data/save_embedding/pat_embed_raremed.pkl
data/save_embedding/diag_embed_micron.pkl  
data/save_embedding/pro_embed_micron.pkl  
data/save_embedding/med_embed_molebert.pkl
```

> These scripts require the vocabulary file `data/data_process/output/mimic-iii/voc_final.pkl` generated during the initial data processing step, as well as pretrained models from [MICRON](https://github.com/ycq091044/MICRON), [RAREMed](https://github.com/zzhUSTC2016/RAREMed), and [Mole-BERT](https://github.com/junxia97/Mole-BERT).

> We plan to release these four embedding `.pkl` files soon to further reduce the reproduction cost for the community.

## Running the Code

### Step 1: Drug-level Classifier ($\pi_{\text{cls}}$)

#### 1. Generate training prompts
```bash
python ./data/generate_cls_data.py  --output_dir ./data/cls --num_val_visits 100 --save_file_name cls_train_val100
```
- Output: `data/cls/cls_train_val100.json`

#### 2.Train the $\pi_{\text{cls}}$
- Download [Llama3.1-Aloe-Beta-8B](https://huggingface.co/HPAI-BSC/Llama3.1-Aloe-Beta-8B) from HuggingFace
and place it under `Models/.`

```bash
torchrun --nproc_per_node=X --master_port=XXX ./src/finetune_ddp.py \
    --llm_name "Llama3-Aloe-8B-Alpha" \
    --data_path data/cls/cls_train_val100.json \
    --output_dir outputs/cls \
    --use_pat_embed True \
    --use_dias_embed True \
    --use_pro_embed True \
    --use_drug_embed True \
    --pat_embed_table_path data/saved_embedding/pat_embed_raremed.pkl \
    --dias_embed_table_path data/saved_embedding/diag_embed_micron.pkl \
    --pro_embed_table_path data/saved_embedding/pro_embed_micron.pkl \
    --drug_embed_table_path data/saved_embedding/med_embed_molebert.pkl \
    --eval_epochs 1000 \
    --batch_size 128 \
    --micro_batch_size 1 \
    --num_epochs 1 \
    --learning_rate 5e-4 \
    --early_stopping_patience 20 \
    --med_num 151 \
    --val_set_size 15100
```

#### 3.Evaluate on test set
- Generate the test set data required for evaluation：
```bash
python ./data/generate_cls_data.py  --output_dir ./data/cls --save_file_name cls_test --generate_test True
```
- Output: `data/cls/cls_test.json`
- Evaluate:

```bash
torchrun --nproc_per_node=X --master_port=XXX ./src/evaluate_ddp.py \
                            --model_path outputs/cls/checkpoint-XXXX \
                            --data_path data/cls/cls_test.json \
                            --save_path evaluate_results/cls/test
```
- Output: `evaluate_results/cls/test/evaluation_result.csv`

### Stage 2: List-wise Policy ($\pi_{\text{list}}$)
$\pi_{\text{list}}$ aims to **refine** drug combinations by learning from $\pi_{\text{cls}}$’s mistakes via **add/remove actions**.

#### 1. Generate $\pi_{\text{cls}}$ predictions on train set

```bash
python ./data/generate_cls_data.py  --output_dir ./data/cls --num_val_visits 0 --save_file_name cls_train

torchrun --nproc_per_node=X --master_port=XXX ./src/evaluate_ddp.py \
                            --model_path outputs/cls/checkpoint-XXXX \
                            --data_path data/cls/cls_train.json \
                            --save_path evaluate_results/cls/train
```
- Output: `evaluate_results/cls/train/*`

#### 2. Construct training data for $\pi_{\text{list}}$
```bash
python ./data/generate_list_sft_data.py --evaluate_results_path evaluate_results/cls/train \
                                        --sample_num 100000 \
                                        --output_dir data/list_sft
python ./data/generate_list_sft_data.py --evaluate_results_path evaluate_results/cls/test \
                                        --sample_num 0 \
                                        --output_dir data/list_sft
python ./data/generate_list_grpo_data.py --evaluate_results_path evaluate_results/cls/train \
                                         --sample_num 100000 \
                                         --output_dir data/list_grpo
python ./data/generate_list_grpo_data.py --evaluate_results_path evaluate_results/cls/test \
                                         --sample_num 0 \
                                         --output_dir data/list_grpo
```
#### 3. Train $\pi_{\text{list}}$ via SFT
```bash
torchrun --nproc_per_node=X --master_port=XXX ./src/finetune_rethink.py \
    --checkpoint_path outputs/cls/checkpoint-XXXX \
    --data_path data/list_sft/100000/mix.json \
    --output_dir outputs/list_sft \
    --save_steps 256 \
    --batch_size 64 \
    --micro_batch_size 1 \
    --num_epochs 1 \
    --learning_rate 5e-4
```
#### 4. (Optional) Evaluate $\pi_{\text{list}}$ (SFT version)

```
python src/evaluate_rethink.py --model_path outputs/list_sft/checkpoint-XXXX --data_path data/list_sft/test/add.json --evaluate_result_path evaluate_results/list_sft --task add

python src/evaluate_rethink.py --model_path outputs/list_sft/checkpoint-XXXX --data_path data/list_sft/test/remove.json --evaluate_result_path evaluate_results/list_sft --task remove

python src/cal_metric.py --results_path evaluate_results/list_sft
```

### Stage 3: Fine-tune and Evaluate $\pi_{\text{list}}$ with Step-wise GRPO (Final Model)
#### 1. Continue from SFT Checkpoint:
```bash
python src/model_process.py --original_model_path outputs/list_sft/checkpoint-XXXX --output_model_path Models/list_sft
python src/finetune_rethink_step_grpo.py \
    --grpo_model_path Models/list_sft \
    --data_path data/list_grpo/mix_100000.json \
    --output_dir outputs/list_grpo \
    --batch_size 32 \
    --gradient_accumulation_steps 2 \
    --num_epochs 1 \
    --num_generations 8 \
    --learning_rate 1e-5 \
    --val_set_size 0 \
    --save_steps 100 \
    --use_vllm True \
    --use_lora True \
    --lora_r 32 \
    --lora_alpha 32 \
    --lora_target_modules "['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']" \
    --alpha 0 \
    --beta 0.2 \
    --step_reward_weight 0.5
```
#### 2. Evaluate Final Model:
```bash
python src/evaluate_rethink_grpo.py --base_model_path Models/list_sft \
                                    --lora_path outputs/list_grpo/checkpoint-XXXX \
                                    --data_path data/list_grpo/test/add.json \
                                    --evaluate_result_path evaluate_results/list_GRPO \
                                    --task add
python src/evaluate_rethink_grpo.py --base_model_path Models/list_sft \
                                    --lora_path outputs/list_grpo/checkpoint-XXXX \
                                    --data_path data/list_grpo/test/remove.json \
                                    --evaluate_result_path evaluate_results/list_GRPO \
                                    --task remove
python src/cal_metric.py --results_path evaluate_results/list_GRPO
```
