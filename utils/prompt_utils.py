import json
import os.path as osp
from typing import Union


class Prompter(object):

    def __init__(self, med_name2idx_dict: dict, use_note: bool = True, use_code: bool = True, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        self.med_name2idx = med_name2idx_dict
        self.use_note = use_note
        self.use_code = use_code
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "cls"
        file_name = osp.join("./utils", "templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name, encoding='utf-8') as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        history: Union[None, str] = None,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.

        if history is not None:
            res = self.template["prompt_history"].format(
                history=history, input=input)
        else:
            res = self.template["prompt_no_history"].format(input=input)
        
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
    
    def list_to_string(self, lst):
        return '[' + ', '.join(lst) + ']' if lst else '[]'

    def generate_history(self, patient_info, mask_text = False):
        age = patient_info['AGE']
        gender = patient_info['GENDER']
        diagnose = eval(patient_info['diagnose'])
        procedure = eval(patient_info['procedure'])
        drug_name = eval(patient_info['drug_name'])

        diag_id_list = eval(patient_info['diag_id'])
        proc_id_list = eval(patient_info['pro_id'])
        drug_id_list = eval(patient_info['drug_id'])
        
        if mask_text:
            diagnose = ['Diagnose [DiasEmb]' for d in diagnose]
            procedure = ['Procedure [ProEmb]' for p in procedure]
            drug_name = ['Drug [DrugEmb]' for d in drug_name]

        else:
            diagnose = [f'{d} [DiasEmb]' for d in diagnose]
            procedure = [f'{p} [ProEmb]' for p in procedure]
            drug_name = [f'{d} [DrugEmb]' for d in drug_name]

        if self.use_code:
            history = f'Age: {age}, ' \
                    f'Gender: {gender}, '\
                    f'Diagnose: {self.list_to_string(diagnose)}, ' \
                    f'Procedure: {self.list_to_string(procedure)}, ' \
                    f'Drug names: {self.list_to_string(drug_name)}.'
        else:
            history = f'Drug names: {self.list_to_string(drug_name)}.'
                
        return history, diag_id_list, proc_id_list, drug_id_list

    def generate_history_GRPO(self, patient_info, mask_text = False):
        age = patient_info['AGE']
        gender = patient_info['GENDER']
        diagnose = eval(patient_info['diagnose'])
        procedure = eval(patient_info['procedure'])
        drug_name = eval(patient_info['drug_name'])

        diag_id_list = eval(patient_info['diag_id'])
        proc_id_list = eval(patient_info['pro_id'])
        drug_id_list = eval(patient_info['drug_id'])
        
        if mask_text:
            diagnose = ['Diagnose [DiasEmb]' for d in diagnose]
            procedure = ['Procedure [ProEmb]' for p in procedure]
            drug_name = ['Drug [DrugEmb]' for d in drug_name]

        else:
            # diagnose = [f'{d} [DiasEmb]' for d in diagnose]
            # procedure = [f'{p} [ProEmb]' for p in procedure]
            # drug_name = [f'{d} [DrugEmb]' for d in drug_name]
            diagnose = [f'{diagnose[i]} <DiasEmb-{diag_id_list[i]}>' for i in range(len(diagnose))]
            procedure = [f'{procedure[i]} <ProEmb-{proc_id_list[i]}>' for i in range(len(procedure))]
            drug_name = [f'{drug_name[i]} <DrugEmb-{drug_id_list[i]}>' for i in range(len(drug_name))]

        if self.use_code:
            history = f'Age: {age}, ' \
                    f'Gender: {gender}, '\
                    f'Diagnose: {self.list_to_string(diagnose)}, ' \
                    f'Procedure: {self.list_to_string(procedure)}, ' \
                    f'Drug names: {self.list_to_string(drug_name)}.'
        else:
            history = f'Drug names: {self.list_to_string(drug_name)}.'
                
        return history

    def generate_input(
        self,
        patient_info,
        drug_candidate,
        mask_text = False
    ):
        age = patient_info['AGE']
        gender = patient_info['GENDER']
        diagnose = eval(patient_info['diagnose'])
        procedure = eval(patient_info['procedure'])
        diag_id_list = eval(patient_info['diag_id'])
        proc_id_list = eval(patient_info['pro_id'])
        drug_id = self.med_name2idx[drug_candidate]
        
        if mask_text:
            diagnose = ['Diagnose [DiasEmb]' for d in diagnose]
            procedure = ['Procedure [ProEmb]' for p in procedure]
            drug_candidate = 'Drug'
        else:
            diagnose = [f'{d} [DiasEmb]' for d in diagnose]
            procedure = [f'{p} [ProEmb]' for p in procedure]

        if 'NOTE' in patient_info.keys():
            note = patient_info['NOTE'].strip('\n ')
        else:
            note = None

        if note is not None and self.use_note:
            if self.use_code:
                input_text = 'Patient representation: [PatEmb], ' \
                        f'{note} ' \
                        f'Diagnose: {self.list_to_string(diagnose)}, ' \
                        f'Procedure: {self.list_to_string(procedure)}, ' \
                        f'Candidate drug: {drug_candidate} [DrugEmb].'
            else:
                input_text = 'Patient representation: [PatEmb], ' \
                        f'{note} ' \
                        f'Candidate drug: {drug_candidate} [DrugEmb].'

        else:
            if self.use_code:
                input_text = 'Patient representation: [PatEmb], ' \
                            f'Age: {age}, ' \
                            f'Gender: {gender}, ' \
                            f'Diagnose: {self.list_to_string(diagnose)}, ' \
                            f'Procedure: {self.list_to_string(procedure)}, ' \
                            f'Candidate drug: {drug_candidate} [DrugEmb].'
            else:
                input_text = 'Patient representation: [PatEmb], ' \
                            f'Candidate drug: {drug_candidate} [DrugEmb].'
                      
        return input_text, diag_id_list, proc_id_list, [drug_id]
    
    def generate_input_GRPO(
        self,
        patient_info,
        drug_candidate,
        mask_text = False
    ):
        hadm_id = patient_info['HADM_ID']
        age = patient_info['AGE']
        gender = patient_info['GENDER']
        diagnose = eval(patient_info['diagnose'])
        procedure = eval(patient_info['procedure'])
        diag_id_list = eval(patient_info['diag_id'])
        proc_id_list = eval(patient_info['pro_id'])
        drug_id = self.med_name2idx[drug_candidate]
        
        if mask_text:
            diagnose = ['Diagnose [DiasEmb]' for d in diagnose]
            procedure = ['Procedure [ProEmb]' for p in procedure]
            drug_candidate = 'Drug'
        else:
            # diagnose = [f'{d} [DiasEmb]' for d in diagnose]
            # procedure = [f'{p} [ProEmb]' for p in procedure]
            diagnose = [f'{diagnose[i]} <DiasEmb-{diag_id_list[i]}>' for i in range(len(diagnose))]
            procedure = [f'{procedure[i]} <ProEmb-{proc_id_list[i]}>' for i in range(len(procedure))]

        if 'NOTE' in patient_info.keys():
            note = patient_info['NOTE'].strip('\n ')
        else:
            note = None

        if note is not None and self.use_note:
            if self.use_code:
                input_text = f'Patient representation: <PatEmb-{hadm_id}>, ' \
                        f'{note} ' \
                        f'Diagnose: {self.list_to_string(diagnose)}, ' \
                        f'Procedure: {self.list_to_string(procedure)}, ' \
                        f'Candidate drug: {drug_candidate} <DrugEmb-{drug_id}>.'
            else:
                input_text = f'Patient representation: <PatEmb-{hadm_id}>, ' \
                        f'{note} ' \
                        f'Candidate drug: {drug_candidate} <DrugEmb-{drug_id}>.'

        else:
            if self.use_code:
                input_text = f'Patient representation: <PatEmb-{hadm_id}>, ' \
                            f'Age: {age}, ' \
                            f'Gender: {gender}, ' \
                            f'Diagnose: {self.list_to_string(diagnose)}, ' \
                            f'Procedure: {self.list_to_string(procedure)}, ' \
                            f'Candidate drug: {drug_candidate} <DrugEmb-{drug_id}>.'
            else:
                input_text = f'Patient representation: <PatEmb-{hadm_id}>, ' \
                            f'Candidate drug: {drug_candidate} <DrugEmb-{drug_id}>.'
                      
        return input_text

    def generate_rethink_input(
        self,
        patient_info
    ):
        age = patient_info['AGE']
        gender = patient_info['GENDER']
        diagnose = eval(patient_info['diagnose'])
        procedure = eval(patient_info['procedure'])
        diag_id_list = eval(patient_info['diag_id'])
        proc_id_list = eval(patient_info['pro_id'])

        diagnose = [f'{d} [DiasEmb]' for d in diagnose]
        procedure = [f'{p} [ProEmb]' for p in procedure]

        if 'NOTE' in patient_info.keys():
            note = patient_info['NOTE'].strip('\n ')
        else:
            note = None
        if note is not None and self.use_note:
            input_text = 'Patient representation: [PatEmb], ' \
                    f'{note} ' \
                    f'Diagnose: {self.list_to_string(diagnose)}, ' \
                    f'Procedure: {self.list_to_string(procedure)}.'
        else:
            input_text = 'Patient representation: [PatEmb], ' \
                        f'Age: {age}, ' \
                        f'Gender: {gender}, ' \
                        f'Diagnose: {self.list_to_string(diagnose)}, ' \
                        f'Procedure: {self.list_to_string(procedure)}, '
            
        return input_text, diag_id_list, proc_id_list
    
    def generate_rethink_input_GRPO(
        self,
        patient_info
    ):
        hadm_id = patient_info['HADM_ID']
        age = patient_info['AGE']
        gender = patient_info['GENDER']
        diagnose = eval(patient_info['diagnose'])
        procedure = eval(patient_info['procedure'])
        diag_id_list = eval(patient_info['diag_id'])
        proc_id_list = eval(patient_info['pro_id'])

        # diagnose = [f'{d} [DiasEmb]' for d in diagnose]
        # procedure = [f'{p} [ProEmb]' for p in procedure]
        diagnose = [f'{diagnose[i]} <DiasEmb-{diag_id_list[i]}>' for i in range(len(diagnose))]
        procedure = [f'{procedure[i]} <ProEmb-{proc_id_list[i]}>' for i in range(len(procedure))]

        if 'NOTE' in patient_info.keys():
            note = patient_info['NOTE'].strip('\n ')
        else:
            note = None
        if note is not None and self.use_note:
            if self.use_code:
                input_text = f'Patient representation: <PatEmb-{hadm_id}>, ' \
                        f'{note} ' \
                        f'Diagnose: {self.list_to_string(diagnose)}, ' \
                        f'Procedure: {self.list_to_string(procedure)}.'
            else:
                input_text = f'Patient representation: <PatEmb-{hadm_id}>, ' \
                        f'{note} '
        else:
            if self.use_code:
                input_text = f'Patient representation: <PatEmb-{hadm_id}>, ' \
                            f'Age: {age}, ' \
                            f'Gender: {gender}, ' \
                            f'Diagnose: {self.list_to_string(diagnose)}, ' \
                            f'Procedure: {self.list_to_string(procedure)}, '
            else:
                input_text = f'Patient representation: <PatEmb-{hadm_id}>, '
            
        return input_text
    
                        