import dill
import logging
import pandas as pd


class Evaluator(object):
    def __init__(self):
        ddi_path = './data/ddi_A_final.pkl'
        voc_path = './data/voc_final.pkl'
        drugbank2name_file = './data/drugbank2name.csv'
        self.ddi_A = dill.load(open(ddi_path, 'rb'))
        self.voc = dill.load(open(voc_path, 'rb'))
        self.drugbank2name = pd.read_csv(drugbank2name_file)

    def convert_to_drugbank_ids(self, drug_names):
        df = self.drugbank2name
        drugbank_ids = []
        for drug_name in drug_names:
            drugbank_id = df[df['drug_name'] ==
                             drug_name]['drugbank_id'].values
            if len(drugbank_id) > 0:
                drugbank_ids.append(drugbank_id[0])
        return drugbank_ids

    def convert_to_idx(self, drugbank_ids):
        idx_dict = self.voc['med_voc'].word2idx
        idx_list = []
        for drugbank_id in drugbank_ids:
            idx = idx_dict.get(drugbank_id)
            if idx is not None:
                idx_list.append(idx)
        return idx_list

    def calculate_ddi_rate(self, idx_list):
        ddi_matrix = self.ddi_A
        ddi_count = 0
        total_count = 0
        for i in range(len(idx_list)):
            for j in range(i+1, len(idx_list)):
                ddi_count += ddi_matrix[idx_list[i], idx_list[j]]
                total_count += 1
        ddi_rate = ddi_count / total_count if total_count > 0 else 0
        return ddi_rate

    def get_ddi_rate(self, drug_names):
        if len(drug_names) == 0:
            return 0
        if isinstance(drug_names[0], str):
            drugbank_ids = self.convert_to_drugbank_ids(drug_names)
            idx_list = self.convert_to_idx(drugbank_ids)
            ddi_rate = self.calculate_ddi_rate(idx_list)
        elif isinstance(drug_names[0], int):
            ddi_rate = self.calculate_ddi_rate(drug_names)
        else:
            raise ValueError('Each element in pred_list must be str or int, but got {}'.format(
                type(drug_names[0])))
        return ddi_rate

    def get_recall(self, pred_list, gt_list):
        pred_set = set(pred_list)
        gt_set = set(gt_list)
        recall = len(pred_set.intersection(gt_set)) / len(gt_set)
        return recall

    def get_precision(self, pred_list, gt_list):
        pred_set = set(pred_list)
        gt_set = set(gt_list)
        precision = len(pred_set.intersection(gt_set)) / len(pred_set) if len(pred_set) > 0 else 0
        return precision

    def get_f1(self, recall, precision):
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
        return f1

    def get_jaccard(self, pred_list, gt_list):
        pred_set = set(pred_list)
        gt_set = set(gt_list)
        jaccard = len(pred_set.intersection(gt_set)) / \
            len(pred_set.union(gt_set))
        return jaccard

    def get_drug_num(self, drug_names):
        drugbank_ids = self.convert_to_drugbank_ids(drug_names)
        return len(drugbank_ids)

    def get_metrics(self, pred_list, gt_list):
        # :param pred_list: list of drug names
        # :param gt_list: list of drug names
        # :return: recall, precision, f1, jaccard, ddi_rate, drug_num
        recall = self.get_recall(pred_list, gt_list)
        precision = self.get_precision(pred_list, gt_list)
        f1 = self.get_f1(recall, precision)
        jaccard = self.get_jaccard(pred_list, gt_list)
        ddi_rate = self.get_ddi_rate(pred_list)
        if len(pred_list) > 0 and isinstance(pred_list[0], str):
            drug_num = self.get_drug_num(pred_list)
        else:
            drug_num = len(pred_list)
        if len(gt_list) > 0 and isinstance(gt_list[0], str):
            drug_num_gt = self.get_drug_num(gt_list)
        else:
            drug_num_gt = len(gt_list)
        return recall, precision, f1, jaccard, ddi_rate, drug_num, drug_num_gt
    
    def get_metrics_each_drug(self, pred_list, gt_list):
        recall = self.get_recall(pred_list, gt_list)
        precision = self.get_precision(pred_list, gt_list)
        f1 = self.get_f1(recall, precision)
        jaccard = self.get_jaccard(pred_list, gt_list)
        return recall, precision, f1, jaccard        
        