import pandas as pd
import os
import dill
import numpy as np
import json
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import BRICS
import matplotlib.pyplot as plt

##### process medications #####
# load med data
def med_process(med_file):
    med_pd = pd.read_csv(med_file, header=0, dtype={"NDC": "category", "GSN": "category"},
                         names=[column.upper() for column in pd.read_csv(med_file, nrows=0).columns])
    med_pd = med_pd[med_pd['DRUG_TYPE'] == 'MAIN']
    med_pd = med_pd[['SUBJECT_ID', 'HADM_ID', 'DRUG', 'NDC']]

    med_pd.drop(index=med_pd[med_pd["NDC"] == "0"].index, axis=0, inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd.sort_values(by=["SUBJECT_ID", "HADM_ID"], inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    return med_pd


# drugbank ID to SMILES
def db2SMILES_process(drugbankinfo_file, med_voc=None):
    drug_info = pd.read_csv(drugbankinfo_file, usecols=['drugbank_id', 'moldb_smiles'])
    drug_info.drop_duplicates(inplace=True)
    drug_info= drug_info.dropna(axis=0)
    drug_info.moldb_smiles = drug_info.moldb_smiles.map(lambda x: [x])
    if med_voc is not None:
        valid_drugbank_ids = list(med_voc.idx2word.values())
        mask = drug_info['drugbank_id'].isin(valid_drugbank_ids)
        drug_info = drug_info[mask]
    db2smiles = dict(zip(drug_info['drugbank_id'], drug_info['moldb_smiles']))
    return db2smiles


# medication mapping
def codeMapping2db(med_pd):
    ndc2db = pd.read_csv(ndc2db_file, dtype={'NDC': 'category'})
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: x[:-2])
    med_pd = med_pd.merge(ndc2db, on=['NDC'])

    med_pd.drop(columns=['NDC'], inplace=True)
    med_pd = med_pd.drop_duplicates().reset_index(drop=True)
    return med_pd


# visit >= 2
def process_visit_lg2(med_pd):
    a = (
        med_pd[["SUBJECT_ID", "HADM_ID"]]
        .groupby(by="SUBJECT_ID")["HADM_ID"]
        .unique()
        .reset_index()
    )
    a["HADM_ID_Len"] = a["HADM_ID"].map(lambda x: len(x))
    a = a[a["HADM_ID_Len"] > 1]
    return a


# most common medications
def filter_K_most_med(med_pd, K=200):
    med_count = med_pd.groupby(by=['drugbank_id']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    med_pd = med_pd[med_pd['drugbank_id'].isin(med_count.loc[:K, 'drugbank_id'])]

    return med_pd.reset_index(drop=True)


##### process diagnosis #####
def diag_process(diag_file):
    diag_pd = pd.read_csv(diag_file, header=0, names=[column.upper() for column in pd.read_csv(diag_file, nrows=1).columns])
    if "ICD9_CODE" in diag_pd.columns:
        diag_pd.rename(columns={'ICD9_CODE':'ICD_CODE'}, inplace=True)
    diag_pd.dropna(inplace=True)
    diag_pd = diag_pd[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD_CODE']]
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=["SUBJECT_ID", "HADM_ID"], inplace=True)
    diag_pd = diag_pd.reset_index(drop=True)

    if dataset == "mimic-iv":
        # first, get the number of unique subject_id
        num_subject_id = len(diag_pd["SUBJECT_ID"].unique())
        # second, select the first 10% of the patients
        diag_pd = diag_pd[diag_pd["SUBJECT_ID"].isin(diag_pd["SUBJECT_ID"].unique()[: int(num_subject_id * 0.1)])]

    return diag_pd


##### process procedure #####
def procedure_process(procedure_file):
    pro_pd = pd.read_csv(procedure_file, dtype={"ICD9_CODE": "category"}, header=0,
                         names=[column.upper() for column in pd.read_csv(procedure_file, nrows=1).columns])
    if "ICD9_CODE" in pro_pd.columns:
        pro_pd.rename(columns={'ICD9_CODE':'ICD_CODE'}, inplace=True)
    pro_pd = pro_pd[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD_CODE']]
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], inplace=True)
    pro_pd.drop(columns=["SEQ_NUM"], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    return pro_pd


def filter_K_diag(diag_pd, K=5):
    # filter diagnosis with less than K occurrences
    # record length of diag_pd
    origin_len = len(diag_pd)

    diag_count = diag_pd.value_counts('ICD_CODE')
    diag_pd = diag_pd[diag_pd['ICD_CODE'].isin(diag_count[diag_count>K].index)]
    diag_pd = diag_pd.reset_index(drop=True)

    # record length of diag_pd
    new_len = len(diag_pd)
    print('filter diagnosis with less than {} occurrences: {} -> {}'.format(K, origin_len, new_len))
    filter_flag = (origin_len != new_len)
    return diag_pd, filter_flag


def filter_K_pro(pro_pd, K=5):
    # filter procedure with less than 10 occurrences
    # record length of pro_pd
    origin_len = len(pro_pd)

    pro_count = pro_pd.value_counts('ICD_CODE')
    pro_pd = pro_pd[pro_pd['ICD_CODE'].isin(pro_count[pro_count>K].index)]
    pro_pd = pro_pd.reset_index(drop=True)

    # record length of pro_pd
    new_len = len(pro_pd)
    print('filter procedure with less than {} occurrences: {} -> {}'.format(K, origin_len, new_len))
    filter_flag = (origin_len != new_len)
    return pro_pd, filter_flag

def get_patient_info(admmisions_file, patients_file):
    '''
    get patient info about age and gender, etc.
    :param admmisions_file: path to admmisions.csv
    :param patients_file: path to patients.csv
    :return: patient_info: dataframe
    '''
    admmisions_pd = pd.read_csv(admmisions_file, header=0, dtype={"HADM_ID": "int"},
                                names=[column.upper() for column in pd.read_csv(admmisions_file, nrows=1).columns])
    patients_pd = pd.read_csv(patients_file, header=0, dtype={"SUBJECT_ID": "int", "HADM_ID": "int"},
                                names=[column.upper() for column in pd.read_csv(patients_file, nrows=1).columns])
    patient_info = admmisions_pd.merge(patients_pd, on=['SUBJECT_ID'], how='inner')
    # get age and gender
    patient_info['ADMITTIME'] = pd.to_datetime(patient_info['ADMITTIME'])
    patient_info['DOB'] = pd.to_datetime(patient_info['DOB'])
    patient_info['AGE'] = patient_info['ADMITTIME'].dt.year - patient_info['DOB'].dt.year
    patient_info.loc[patient_info['AGE'] > 90, 'AGE'] = 90

    patient_info["GENDER"] = patient_info["GENDER"].replace({"F": "female", "M": "male"})

    patient_info = patient_info[['SUBJECT_ID', 'HADM_ID', 'GENDER', 'AGE', 'ADMITTIME']]
    patient_info.drop_duplicates(inplace=True)
    patient_info.sort_values(by=["SUBJECT_ID", "ADMITTIME"], inplace=True)
    patient_info = patient_info.reset_index(drop=True)
    return patient_info


###### combine three tables #####
def process_data4LLM(data, diag_dict_file, procedure_dict_file, drugbank2name_file, vocabulary_file):
    patient_info = get_patient_info(admissions_file, patients_file)
    data = data.merge(patient_info, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data.sort_values(by=['SUBJECT_ID', 'ADMITTIME'], inplace=True)

    diag_dict = pd.read_csv(diag_dict_file, header=0)
    pro_dict = pd.read_csv(procedure_dict_file, header=0)
    med_dict = pd.read_csv(drugbank2name_file, header=0)
    diag_icd2name = dict(zip(diag_dict['ICD9_CODE'], diag_dict['CONCISE_TITLE']))
    pro_icd2name = dict(zip(pro_dict['ICD9_CODE'], pro_dict['CONCISE_TITLE']))
    med_db2name = dict(zip(med_dict['drugbank_id'], med_dict['drug_name']))

    voc = dill.load(open(vocabulary_file, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    diag_icd2id, pro_icd2id, med_db2id = diag_voc.word2idx, pro_voc.word2idx, med_voc.word2idx

    data4LLM = []

    for _, row in data.iterrows():
        subject_id = row['SUBJECT_ID']
        hadm_id = row['HADM_ID']
        gender = row['GENDER']
        age = row['AGE']
        admittime = row['ADMITTIME']

        diag_codes = row['ICD_CODE']
        pro_codes = row['PRO_CODE']
        drug_dbs = row['drugbank_id']

        diag_names = [diag_icd2name.get(code, '') for code in diag_codes]
        pro_names = [pro_icd2name.get(int(code), '') for code in pro_codes]
        drug_names = [med_db2name.get(code, '') for code in drug_dbs]

        full_diag_ids_and_names = [(diag_icd2id.get(code), diag_name) for code, diag_name in zip(diag_codes, diag_names) if diag_name != '']
        full_pro_ids_and_names = [(pro_icd2id.get(code), pro_name) for code, pro_name in zip(pro_codes, pro_names) if pro_name != '']
        full_drug_ids_and_names = [(med_db2id.get(code), drug_name) for code, drug_name in zip(drug_dbs, drug_names) if drug_name != '']

        diag_ids, diag_names = (list(t) for t in zip(*full_diag_ids_and_names)) if full_diag_ids_and_names else ([], [])
        pro_ids, pro_names = (list(t) for t in zip(*full_pro_ids_and_names)) if full_pro_ids_and_names else ([], [])
        drug_ids, drug_names = (list(t) for t in zip(*full_drug_ids_and_names)) if full_drug_ids_and_names else ([], [])

        data4LLM.append([
            subject_id, hadm_id, gender, age, admittime,
            diag_names, pro_names, drug_names,
            diag_ids, pro_ids, drug_ids
        ])

    columns = ['SUBJECT_ID', 'HADM_ID', 'GENDER', 'AGE', 'ADMITTIME',
               'diagnose', 'procedure', 'drug_name',
               'diag_id', 'pro_id', 'drug_id']
    return pd.DataFrame(data4LLM, columns=columns)

###### combine three tables #####
def combine_process(med_pd, diag_pd, pro_pd):
    # filter out the clinical codes with few occurrences, filter out patients with no clinical codes
    filter_flag = True
    while filter_flag:
        med_pd_key = med_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
        diag_pd_key = diag_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
        pro_pd_key = pro_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()

        combined_key = med_pd_key.merge(
            diag_pd_key, on=["SUBJECT_ID", "HADM_ID"], how="inner"
        )
        combined_key = combined_key.merge(
            pro_pd_key, on=["SUBJECT_ID", "HADM_ID"], how="inner"
        )
        diag_pd = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
        med_pd = med_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
        pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
        diag_pd, filter_flag_diag = filter_K_diag(diag_pd)
        pro_pd, filter_flag_pro = filter_K_pro(pro_pd)
        filter_flag = filter_flag_diag or filter_flag_pro

    # flatten and merge
    diag_pd = diag_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD_CODE'].unique().reset_index()  
    med_pd = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['drugbank_id'].unique().reset_index()
    pro_pd = pro_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD_CODE'].unique().reset_index().rename(columns={'ICD_CODE':'PRO_CODE'})  
    med_pd['drugbank_id'] = med_pd['drugbank_id'].map(lambda x: list(x))
    pro_pd['PRO_CODE'] = pro_pd['PRO_CODE'].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(pro_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data['drugbank_id_num'] = data['drugbank_id'].map(lambda x: len(x))

    return data

def statistics(data):
    print("#patients ", data["SUBJECT_ID"].unique().shape[0])
    print("#clinical events ", len(data))

    diag = data["ICD_CODE"].values
    med = data['drugbank_id'].values
    pro = data["PRO_CODE"].values

    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    unique_pro = set([j for i in pro for j in list(i)])

    print("#diagnosis ", len(unique_diag))
    print("#med ", len(unique_med))
    print("#procedure", len(unique_pro))

    (
        avg_diag,
        avg_med,
        avg_pro,
        max_diag,
        max_med,
        max_pro,
        cnt,
        max_visit,
        avg_visit,
    ) = [0 for i in range(9)]

    for subject_id in data["SUBJECT_ID"].unique():
        item_data = data[data["SUBJECT_ID"] == subject_id]
        visit_cnt = 0
        for index, row in item_data.iterrows():
            x, y, z = [], [], []
            visit_cnt += 1
            cnt += 1
            x.extend(list(row["ICD_CODE"]))
            y.extend(list(row['drugbank_id']))
            z.extend(list(row["PRO_CODE"]))
            avg_diag += len(x)
            avg_med += len(y)
            avg_pro += len(z)
            avg_visit += visit_cnt
            if len(x) > max_diag:
                max_diag = len(x)
            if len(y) > max_med:
                max_med = len(y)
            if len(z) > max_pro:
                max_pro = len(z)
        if visit_cnt > max_visit:
            max_visit = visit_cnt

    print(f'#avg of visits {avg_visit/ len(data["SUBJECT_ID"].unique()):.2f}')
    print(f'#avg of diagnoses {avg_diag/ cnt:.2f}')
    print(f'#avg of medicines {avg_med/ cnt:.2f}')
    print(f'#avg of procedures {avg_pro/ cnt:.2f}')

    print("#max of diagnoses ", max_diag)
    print("#max of medicines ", max_med)
    print("#max of procedures ", max_pro)
    print("#max of visit ", max_visit)


##### indexing file and final record
class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


# create voc set
def create_str_token_mapping(df):
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()

    for index, row in df.iterrows():
        diag_voc.add_sentence(row["ICD_CODE"])
        med_voc.add_sentence(row['drugbank_id'])
        pro_voc.add_sentence(row["PRO_CODE"])

    dill.dump(
        obj={"diag_voc": diag_voc, "med_voc": med_voc, "pro_voc": pro_voc},
        file=open(vocabulary_file, "wb"),
    )

    ### genera
    voc = dill.load(open(vocabulary_file, "rb"))
    med_v = voc['med_voc'].word2idx

    med_word2name = pd.read_csv(drugbank2name_file)
    med_map = dict(zip(med_word2name['drugbank_id'], med_word2name['drug_name']))
    med_name2idx = {med_map[db_id]: idx for db_id, idx in med_v.items()}

    with open(med_name2idx_file, 'w') as f:
        json.dump(med_name2idx, f, indent=4)
    return diag_voc, med_voc, pro_voc

class PatientWeight():
    def __init__(self, data):
        icd_counts_diag = data['ICD_CODE'].apply(pd.Series).stack().value_counts()
        weight_diag = 1/icd_counts_diag

        norm_effi =  (icd_counts_diag * weight_diag).values.sum() / icd_counts_diag.values.sum()
        weight_diag = weight_diag / norm_effi

        self.weight_diag_df = pd.DataFrame({'Count': weight_diag.values}, index=weight_diag.index)

        icd_counts_pro = data['PRO_CODE'].apply(pd.Series).stack().value_counts()
        weight_pro = 1/icd_counts_pro

        norm_effi =  (icd_counts_pro * weight_pro).values.sum() / icd_counts_pro.values.sum()
        weight_pro = weight_pro / norm_effi

        self.weight_pro_df = pd.DataFrame({'Count': weight_pro.values}, index=weight_pro.index)

    def get(self, visit):
        weight_diag = self.weight_diag_df.loc[visit['ICD_CODE']].values
        weight_pro = self.weight_pro_df.loc[visit['PRO_CODE']].values
        # get average weight
        weight = np.concatenate((weight_diag, weight_pro), axis=0)
        return np.mean(weight)

# create patient record
def create_patient_record(data, diag_voc, med_voc, pro_voc):
    get_weight = PatientWeight(data)
    records = [] # (patient, code_kind:3, codes)  code_kind:diag, proc, med
    visit_weights = []
    for subject_id in data['SUBJECT_ID'].unique():
        item_df = data[data['SUBJECT_ID'] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([diag_voc.word2idx[i] for i in row['ICD_CODE']])
            admission.append([pro_voc.word2idx[i] for i in row['PRO_CODE']])
            admission.append([med_voc.word2idx[i] for i in row['drugbank_id']])
            visit_weight = get_weight.get(row)
            visit_weights.append(visit_weight)
            admission.append(visit_weight)
            patient.append(admission)
        records.append(patient) 
    dill.dump(obj=records, file=open(ehr_sequence_file, 'wb'))
    return records, visit_weights


# get ddi matrix
def get_ddi_matrix(records, med_voc, ddi_file):
    med_voc_size = len(med_voc.idx2word)
    
    # weighted ehr adj 
    ehr_adj = np.zeros((med_voc_size, med_voc_size))
    for patient in records:
        for adm in patient:
            med_set = adm[2]
            for i, med_i in enumerate(med_set):
                for j, med_j in enumerate(med_set):
                    if j<=i:
                        continue
                    ehr_adj[med_i, med_j] = 1
                    ehr_adj[med_j, med_i] = 1
    dill.dump(ehr_adj, open(ehr_adjacency_file, 'wb'))  
    
    # ddi adj
    # ddi_df = pd.read_csv(ddi_file, sep='\t')
    ddi_df = pd.read_csv(ddi_file)
    
    ddi_adj = np.zeros((med_voc_size,med_voc_size))
    ddi_adj_syn = np.zeros((med_voc_size,med_voc_size))
    for index, row in ddi_df.iterrows():
        # ddi
        drug1 = row['drug1']
        drug2 = row['drug2']
        type = row['type']
        # TODO 增加类别
        if type in [112, 73, 41, 29, 23, 63, 98, 95, 111]:
            if drug1 in med_voc.word2idx.keys() and drug2 in med_voc.word2idx.keys():
                ddi_adj[med_voc.word2idx[drug1], med_voc.word2idx[drug2]] = 1
                ddi_adj[med_voc.word2idx[drug2], med_voc.word2idx[drug1]] = 1
        elif type == 9:
            if drug1 in med_voc.word2idx.keys() and drug2 in med_voc.word2idx.keys():
                ddi_adj_syn[med_voc.word2idx[drug1], med_voc.word2idx[drug2]] = 1
                ddi_adj_syn[med_voc.word2idx[drug2], med_voc.word2idx[drug1]] = 1
    print('#ddi_pairs:', ddi_adj.sum(), '#synergistic_pairs:', ddi_adj_syn.sum())
    dill.dump(ddi_adj, open(ddi_adverse_file, 'wb')) 
    dill.dump(ddi_adj_syn, open(ddi_synergistic_file, 'wb'))

    return ddi_adj, ddi_adj_syn

def cal_ddi_rate_score(records, ddi_adj):
    # ddi rate
    all_cnt = 0
    dd_cnt = 0
    # dd_syn_cnt = 0
    for patient in records:
        for adm in patient:
            med_set = adm[2]
            for i, med_i in enumerate(med_set):
                for j, med_j in enumerate(med_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_adj[med_i, med_j] == 1 or ddi_adj[med_j, med_i] == 1:
                        dd_cnt += 1
                    # if ddi_adj_syn[med_i, med_j] == 1 or ddi_adj_syn[med_j, med_i] == 1:
                    #     dd_syn_cnt += 1
    ddi_rate_score = dd_cnt / all_cnt if all_cnt > 0 else 0 
    # ddi_rate_score_syn = dd_syn_cnt / all_cnt if all_cnt > 0 else 0 
    print(f'{ddi_rate_score=}')
    return ddi_rate_score

def get_ddi_mask(db2SMILES, med_voc):
    fraction = []
    for k, v in med_voc.idx2word.items():
        tempF = set()
        SMILES = db2SMILES[v][0]
        try:
            m = BRICS.BRICSDecompose(Chem.MolFromSmiles(SMILES))
            for frac in m:
                tempF.add(frac)
        except:
            pass
        fraction.append(tempF)
    fracSet = []
    for i in fraction:
        fracSet += i
    fracSet = list(set(fracSet)) # set of all segments

    ddi_matrix = np.zeros((len(med_voc.idx2word), len(fracSet)))
    for i, fracList in enumerate(fraction):
        for frac in fracList:
            ddi_matrix[i, fracSet.index(frac)] = 1
    return ddi_matrix, fracSet

def get_medicine_popularity(records, medicine_pop_file):
    def get_EHR_pd(ehr_data):
        EHR_pd = pd.DataFrame(columns=['patient', 'visit','disease','procedure','medicine'])
        p_id = v_id = 0
        for patient in ehr_data:
            for visit in patient:
                EHR_pd.loc[v_id] = [p_id, v_id, visit[0], visit[1], visit[2]]
                v_id += 1
            p_id += 1
        return EHR_pd
    EHR_pd = get_EHR_pd(records)
    all_medicine = [medicine for medicines in EHR_pd['medicine'] for medicine in medicines]
    medicine_pop = pd.value_counts(all_medicine)
    medicine_pop.to_csv(medicine_pop_file, header = False)
    return medicine_pop

dataset = 'mimic-iii'
data_path = './data/data_process/input/'
output_dir = os.path.join('./data/data_process/output/', dataset)

print("-" * 10, "processing dataset: ", dataset, "-" * 10)
# files can be downloaded from https://mimic.physionet.org/gettingstarted/dbsetup/
# please change into your own MIMIC folder
if dataset == 'mimic-iii':
    med_file = os.path.join(data_path, dataset, "PRESCRIPTIONS.csv")
    diag_file = os.path.join(data_path, dataset, "DIAGNOSES_ICD.csv")
    procedure_file = os.path.join(data_path, dataset, "PROCEDURES_ICD.csv")
    diag_dict_file = os.path.join(data_path, dataset, "D_ICD_DIAGNOSES.csv")
    procedure_dict_file = os.path.join(data_path, dataset, "D_ICD_PROCEDURES.csv")
    admissions_file = os.path.join(data_path, dataset, "ADMISSIONS.csv")
    patients_file = os.path.join(data_path, dataset, "PATIENTS.csv")
elif dataset == 'mimic-iv':
    raise NotImplementedError("MIMIC-IV is not supported yet.")

# input auxiliary files
ddi_file = os.path.join(data_path, 'ddi_data_all.csv')
ndc2db_file = os.path.join(data_path, 'ndc2db_all.csv')
drugbankinfo_file = os.path.join(data_path, 'drugbank_drugs_info.csv')
drugbank2name_file = os.path.join(data_path, 'drugbank2name.csv')

# output files
os.makedirs(output_dir, exist_ok=True)
ddi_adverse_file = os.path.join(output_dir, 'ddi_A_final.pkl')
ddi_synergistic_file = os.path.join(output_dir, 'ddi_synergistic_A_final.pkl')
ehr_adjacency_file = os.path.join(output_dir, 'ehr_adj_final.pkl')
ehr_sequence_file = os.path.join(output_dir, 'records_final.pkl')
vocabulary_file = os.path.join(output_dir, 'voc_final.pkl')
med_name2idx_file = os.path.join(output_dir, 'med_name2idx.json')
ddi_mask_H_file = os.path.join(output_dir, 'ddi_mask_H.pkl')
db2SMILES_file = os.path.join(output_dir, 'db2SMILES.pkl')
medicine_pop_file = os.path.join(output_dir, 'medicine_pop.csv')
data4LLM_file = os.path.join(output_dir, 'data4LLM.csv')
substructure_smiles_file = os.path.join(output_dir, "substructure_smiles.pkl")
data_file = os.path.join(output_dir, 'data.csv')

# for med
med_pd = med_process(med_file)
med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)
med_pd = med_pd.merge(
    med_pd_lg2[["SUBJECT_ID"]], on="SUBJECT_ID", how="inner"
).reset_index(drop=True)

med_pd = codeMapping2db(med_pd)
med_pd = filter_K_most_med(med_pd)

# filter out drugs without smiles
db2SMILES = db2SMILES_process(drugbankinfo_file)
med_pd = med_pd[med_pd.drugbank_id.isin(db2SMILES.keys())]
med_pd.sort_values(by=['SUBJECT_ID','HADM_ID'], inplace=True)
print ('complete medication processing')

# for diagnosis
diag_pd = diag_process(diag_file)
print("complete diagnosis processing")

# for procedure
pro_pd = procedure_process(procedure_file)
print("complete procedure processing")

# combine
data = combine_process(med_pd, diag_pd, pro_pd)
data.to_csv(data_file, index=None)
statistics(data)

print("complete combining")

# create vocab
diag_voc, med_voc, pro_voc = create_str_token_mapping(data)
print("obtain voc")

# med to SMILES mapping
db2SMILES = db2SMILES_process(drugbankinfo_file, med_voc)
dill.dump(db2SMILES, open(db2SMILES_file,'wb'))

# create ehr sequence data
records, visit_weights = create_patient_record(data, diag_voc, med_voc, pro_voc)
print("obtain ehr sequence data")

data4LLM = process_data4LLM(data, diag_dict_file, procedure_dict_file, drugbank2name_file, vocabulary_file)
data4LLM.to_csv(data4LLM_file, index=None)
print("complete data4LLM processing")

# create ddi adj matrix
ddi_adj, _ = get_ddi_matrix(records, med_voc, ddi_file)
print("obtain ddi adj matrix")

# calculate ddi rate in EHR
cal_ddi_rate_score(records, ddi_adj)

# get ddi_mask_H
ddi_mask_H, fracSet = get_ddi_mask(db2SMILES, med_voc)
dill.dump(ddi_mask_H, open(ddi_mask_H_file, "wb"))
dill.dump(fracSet, open(substructure_smiles_file, 'wb'))

# get medicine popularity
medicine_pop = get_medicine_popularity(records, medicine_pop_file)

plt.hist(visit_weights, bins=100, log=True)
plt.ylabel('Number of visits')
plt.xlabel('Weight of visits')
plt.title('Distribution of visit weights')
plt.show()