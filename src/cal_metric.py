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
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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


    return jacard, precision, recall, f1, ddi, ddi_gt, len(pred), len(gt)

def get_metric(results_path, ckpt):
    add_path = os.path.join(results_path, f'add-{ckpt}.csv')
    remove_path = os.path.join(results_path, f'remove-{ckpt}.csv')

    if not os.path.exists(remove_path):

        df_add = pd.read_csv(add_path)

        jaccard_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        ddi_list = []
        ddi_gt_list = []
        pred_num_list = []
        gt_num_list = []

        for i in range(len(df_add)):

            rethink_list = eval(df_add.iloc[i]['input_list'])
            gt_list = eval(df_add.iloc[i]['gt_list'])
            add_list = eval(df_add.iloc[i]['output_list'])


            rethink_list = list(set(rethink_list) | set(add_list))
            rethink_list = sorted(rethink_list)

            jaccard, precision, recall, f1, ddi, ddi_gt, pred_num, gt_num = cal_metric(rethink_list, gt_list)
            jaccard_list.append(jaccard)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            ddi_list.append(ddi)
            ddi_gt_list.append(ddi_gt)
            pred_num_list.append(pred_num)
            gt_num_list.append(gt_num)

    else:

        df_add = pd.read_csv(add_path)
        df_remove = pd.read_csv(remove_path)

        jaccard_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        ddi_list = []
        ddi_gt_list = []
        pred_num_list = []
        gt_num_list = []

        for i in range(len(df_add)):
            assert eval(df_add.iloc[i]['gt_list']) == eval(df_remove.iloc[i]['gt_list']), f"gt_list not match: {df_add.iloc[i]['gt_list']} {df_remove.iloc[i]['gt_list']}"
            assert eval(df_add.iloc[i]['input_list']) == eval(df_remove.iloc[i]['input_list']), f"input_list not match: {df_add.iloc[i]['input_list']} {df_remove.iloc[i]['input_list']}"

            rethink_list = eval(df_add.iloc[i]['input_list'])
            gt_list = eval(df_add.iloc[i]['gt_list'])
            add_list = eval(df_add.iloc[i]['output_list'])
            remove_list = eval(df_remove.iloc[i]['output_list'])


            rethink_list = list(set(rethink_list) | set(add_list))
            rethink_list = list(set(rethink_list) - set(remove_list))
            rethink_list = sorted(rethink_list)

            jaccard, precision, recall, f1, ddi, ddi_gt, pred_num, gt_num = cal_metric(rethink_list, gt_list)
            jaccard_list.append(jaccard)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            ddi_list.append(ddi)
            ddi_gt_list.append(ddi_gt)
            pred_num_list.append(pred_num)
            gt_num_list.append(gt_num)

    print(f"Evaluate {ckpt} results:")
    print(f"Jaccard: {np.mean(jaccard_list):.4f}, Precision: {np.mean(precision_list):.4f}, Recall: {np.mean(recall_list):.4f}, F1: {np.mean(f1_list):.4f}, DDI: {np.mean(ddi_list):.4f}, DDI_gt: {np.mean(ddi_gt_list):.4f}, Pred_num: {np.mean(pred_num_list):.4f}, GT_num: {np.mean(gt_num_list):.4f}")


if __name__ == '__main__':
    fire.Fire(get_metric)