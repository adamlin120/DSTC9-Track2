# -*- coding: utf-8 -*-
import copy
import json
import os
import re
import shutil
import urllib.request
from collections import OrderedDict
from io import BytesIO
from zipfile import ZipFile
import difflib
import numpy as np
import argparse
from shutil import copyfile
from tqdm import tqdm 
def get_act(dialog_act):
    acts = []
    for act in dialog_act:
        if act[0] == "Inform":
            acts.append([act[2], act[3]])
        elif act[0] == "Request":
            acts.append(act[2])
    return acts

def get_diff_dict(dict1, dict2):
    diff = {}
    # Sometimes, metadata is an empty dictionary, bug?
    if not dict1 or not dict2:
        return diff

    for ((k1, v1), (k2, v2)) in zip(dict1.items(), dict2.items()):
        assert k1 == k2
        if v1 != v2: # updated
            diff[k2] = v2
    return diff

def get_Domain(idx, log, domains, last_domain):
    act_domain = []
    for act in log[idx]["dialog_act"]:
        if act[2] != "none":
            act_domain.append(act[1])
    domain = act_domain[0] if len(act_domain) != 0 else (domains[0] if idx == 0 or idx == 1 else last_domain[0])
    return domain

def bulid_data(data):
    
    dial_data = []

    for log_idx, dialogue_name in tqdm(enumerate(data)):
        dialogue_content = data[dialogue_name]
        dialogue = {"dialogue_idx":dialogue_name, "dialogue":[]}
        domains = []
        #統計對話中出現的domain
        for goal in dialogue_content["goal"]:
            domain = goal[1]
            if domain not in domains:
                domains.append(domain)
        dialogue["domains"] = domains
        turns = []
        last_domain = []
        total_belief_state = []
        #initialize an empty system_transcript
        turn = {"system_transcript": "", "belief_state":[], "ture_label":[], "system_acts": [], "domain":get_Domain(0, dialogue_content["messages"], domains, [])}
        for m_idx, message in enumerate(dialogue_content["messages"]):
            if m_idx % 2 == 0: #user utterance
                turn["transcript"] = message["content"]
                turn["turn_idx"] = m_idx / 2
                for act in message["dialog_act"]:
                    if act[0] == "Inform":
                        turn["ture_label"].append([f'{act[1]}-{act[2]}', act[3]])
                for s in turn["ture_label"]:        
                    total_belief_state.append({"slots": [s], "act": "inform"})
                turn["belief_state"] = list(total_belief_state)
                turns.append(turn)
                #initialize a new turn dictionary
                turn = {"belief_state":[], "ture_label":[]}
            else: #system utterance
                turn["system_transcript"] = message["content"]
                turn["system_acts"] = get_act(message["dialog_act"])
                turn["domain"] = get_Domain(m_idx, dialogue_content["messages"], domains ,last_domain)
                last_domain = [turn["domain"]]
            dialogue["dialogue"] = list(turns)
        dial_data.append(dialogue)
    return dial_data


def main(args):
    #載入train、val、test的資料
    with open(os.path.join(args.main_dir, 'train.json'), 'r', encoding='utf8') as train_f:
        train_data = json.load(train_f)

    with open(os.path.join(args.main_dir, 'val.json'), 'r', encoding='utf8') as val_f:
        val_data = json.load(val_f)

    with open(os.path.join(args.main_dir, 'test.json'), 'r', encoding='utf8') as test_f:
        test_data = json.load(test_f)

    train_dial = bulid_data(train_data)
    print("finish train data")
    val_dial = bulid_data(val_data)
    print("finish val data")
    test_dial = bulid_data(test_data)
    print("finish test data")
    
    #輸出train_dial、val_dial、test_dial檔案
    with open(os.path.join(args.target_path, 'train_dial.json'), 'w', encoding='utf8') as train_f:
        json.dump(train_dial, train_f, indent = 4, ensure_ascii=False)

    with open(os.path.join(args.target_path, 'val_dial.json'), 'w', encoding='utf8') as val_f:
        json.dump(val_dial, val_f, indent = 4, ensure_ascii=False)

    with open(os.path.join(args.target_path, 'test_dial.json'), 'w', encoding='utf8') as test_f:
        json.dump(test_dial, test_f, indent = 4, ensure_ascii=False)  
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dir", type=str, default='../crosswoz')
    parser.add_argument("--target_path", type=str, default='../crosswoz')
    args = parser.parse_args()
    main(args)

