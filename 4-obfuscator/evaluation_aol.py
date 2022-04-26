'''
Parsing crawling dataset to get bids
'''
import os
import re
import h5py
import json
import html2text
from copy import deepcopy
import random
random.seed(0)
import numpy as np
import argparse
from tqdm import tqdm
import spacy
spacy.load('en')
from spacy.lang.en import English
import nltk
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import gensim.models as g
import logging
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import argparse
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from heapq import heappop, heappush

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', help="path for storing models", default="/SSD/model", type=str)
parser.add_argument('--dataset_path', help="path for storing dataset", default="/SSD/dataset", type=str)
args = parser.parse_args()

hb_domain = [
    "www.speedtest.net",
    "www.kompas.com",
    "cnn.com"
]
parser = English()
try:
    en_stop = set(nltk.corpus.stopwords.words('english'))
    en_words = set(nltk.corpus.words.words())
except:
    nltk.download('words')
    nltk.download('wordnet')
    nltk.download('stopwords')
    en_stop = set(nltk.corpus.stopwords.words('english'))
    en_words = set(nltk.corpus.words.words())

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
def tokenize(text):
    filter_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.like_url:
            continue
        elif token.orth_.startswith('@'):
            continue
        else:
            token = token.lower_
            filter_tokens.append(token)
    return filter_tokens

def prepare_text(text):
    if len(text) > 1000000:
        text = text[0:1000000]
    tokens = tokenize(text)
    tokens = [get_lemma(token) for token in tokens]
    tokens = [token for token in tokens if token in en_words]
    return tokens

alpha = 0.1
result_filename = [
    "aol_3"
]


node_list = [path[10:-4] for path in os.listdir("/SSD/baseline/oracle")]
bids_list = [path[10:-4] for path in os.listdir("/SSD/baseline/bids")]

with open("../DATA/analysis_result/bids_result.json", "r") as file:
    bids_th = json.load(file)
hb_domain = [
    "www.speedtest.net",
    "www.kompas.com",
    "cnn.com"
]

print(node_list, bids_list)
pre_data = []

count = 0
bids_data = []

url_content = {}
Data = {}
get_content = False
count = 0
dataset = {}
all_bids = {}
all_bids_th = {}
for idx in range(4):
    with open("../DATA/crawl_result/{}.json".format(result_filename[0]), "r") as file:
        result_dataset = json.load(file)

    for j in range(len(result_dataset.keys())):
        # if int(key.split("_")[-1]) >= 10:
        #     continue
        if j % 4 != idx:
            continue
        key = list(result_dataset.keys())[j]
        # print(idx, key)
        persona = result_dataset[key][0]
        # print(idx, key)
        # break
        try:
            for domain in hb_domain:
                collect_bids_data = persona["base"]["bids"]['bids_by_ad_bidder'][domain]

                for ad in collect_bids_data.keys():
                    tmp = {}
                    for item in collect_bids_data[ad]:
                        bids = item[0]
                        bidder = item[1]
                        if bidder not in tmp.keys():
                            tmp[bidder] = []
                        tmp[bidder].append(bids)
                    for bidder in tmp.keys():
                        if ad.startswith("/"):
                            ad = ad[1:]
                        tag = "base_bids_{}_{}".format(re.sub('[^0-9a-zA-Z]+', '_', ad), bidder)
                        if tag in bids_list:
                            if tag not in all_bids.keys():
                                all_bids[tag] = []
                            all_bids[tag].append(np.mean(tmp[bidder]))             
        except:
            print(key, persona["base"]["bids"])
            continue
for tag in all_bids.keys():
    sorted_list = sorted(all_bids[tag])
    all_bids_th[tag] = sorted_list[int(0.8 * len(sorted_list))]
print(all_bids_th)

for idx in range(4):
    dataset[idx] = {}
    with open("../DATA/crawl_result/{}.json".format(result_filename[0]), "r") as file:
        result_dataset = json.load(file)

    # print("[INFO] Data points in {}: {}".format("../DATA/crawl_result/{}.json".format(i), len(result_dataset)))
    count = 0
    tmp = []
    err = [0,0]
    dataset[idx]["bids_cpm_avg"] = 0
    dataset[idx]["bids_avg"] = 0

    for j in range(len(result_dataset.keys())):
        # if int(key.split("_")[-1]) >= 10:
        #     continue
        if j % 4 != idx:
            continue
        key = list(result_dataset.keys())[j]

        # if int(key.split("_")[-1]) >= 10:
        #     continue
        key_id = str(count)
        count += 1
        persona = result_dataset[key][0]
        
        dataset[idx][key_id] = {}
        
        try:
            collect_data = persona["collect"]
            seg_data_raw = collect_data["content"]["seg_info"].split("\"")
            seg_data = []
            for i in range(len(seg_data_raw)):
                if seg_data_raw[i] == "," or  seg_data_raw[i] == ", " or "[" in seg_data_raw[i] or \
                "]" in seg_data_raw[i] or "A/B Test Groups" in seg_data_raw[i] or \
                "null" in seg_data_raw[i]:
                    continue
                seg_data.append(seg_data_raw[i].replace("\n", " "))
            
            seg_data_clean = dict(zip(node_list,[0 for _ in range(len(node_list))]))  
            for i in range(len(seg_data)):
                seg = seg_data[i]
                seg = re.sub('[^0-9a-zA-Z]+', '_', seg)
                if seg not in node_list:
                    continue
                seg_data_clean[seg] = 1

            dataset[idx][key_id]["segs"] = deepcopy(seg_data_clean)
        except:
            err[0] += 1
            dataset[idx][key_id]["segs"] = dict(zip(node_list,[0 for _ in range(len(node_list))]))  

        try:
            dataset[idx][key_id]["bids"] = dict(zip(bids_list,[0 for _ in range(len(bids_list))]))
            dataset[idx][key_id]["bids_cpm"] = dict(zip(bids_list,[0 for _ in range(len(bids_list))]))
            
            for domain in hb_domain:
                collect_bids_data = persona["base"]["bids"]['bids_by_ad_bidder'][domain]

                for ad in collect_bids_data.keys():
                    tmp = {}
                    for item in collect_bids_data[ad]:
                        bids = item[0]
                        bidder = item[1]
                        if bidder not in tmp.keys():
                            tmp[bidder] = []
                        tmp[bidder].append(bids)
                    for bidder in tmp.keys():
                        if ad.startswith("/"):
                            ad = ad[1:]
                        tag = "base_bids_{}_{}".format(re.sub('[^0-9a-zA-Z]+', '_', ad), bidder)
                        if tag in bids_list:
                            # th = bids_th[tag]["th"]
                            th = all_bids_th[tag]
                            # if i == "Intent_0_bids_0.1":
                            #     print(th, np.mean(tmp[bidder]), np.mean(tmp[bidder]) >= th, int(np.mean(tmp[bidder]) >= th))
                            dataset[idx][key_id]["bids"][tag] = int(np.mean(tmp[bidder]) >= th)
                            dataset[idx][key_id]["bids_cpm"][tag] = np.mean(tmp[bidder])
                            
        except:
            err[1] += 1
            dataset[idx][key_id]["bids"] = dict(zip(bids_list,[0 for _ in range(len(bids_list))]))
            dataset[idx][key_id]["bids_cpm"] = dict(zip(bids_list,[0 for _ in range(len(bids_list))]))
        dataset[idx]["bids_cpm_avg"] += np.mean(list(dataset[idx][key_id]["bids_cpm"].values()))
        dataset[idx]["bids_avg"] += np.mean(list(dataset[idx][key_id]["bids"].values()))
    dataset[idx]["bids_cpm_avg"] = dataset[idx]["bids_cpm_avg"] / count
    dataset[idx]["bids_avg"] = dataset[idx]["bids_avg"] / count
    print(count, err, key, key_id, i, dataset[idx]["bids_cpm_avg"], dataset[idx]["bids_avg"])

for idx in range(3):
    M1_list = []
    M2_list = []
    M3_list = []
    M4_list = []
    base_seg_list = []
    cur_seg_list = []
    for i in range(25):
        base_seg = list(dataset[0][str(i)]["segs"].values())
        cur_seg = list(dataset[idx+1][str(i)]["segs"].values())
        base_bid = list(dataset[0][str(i)]["bids"].values())
        cur_bid = list(dataset[idx+1][str(i)]["bids"].values())
        dataset[idx+1][str(i)]["M1"] = sum([(base_seg[j] - cur_seg[j]) ** 2 for j in range(len(base_seg))]) / len(base_seg)
        dataset[idx+1][str(i)]["M2"] = sum([int(base_seg[j] == 0 and cur_seg[j] == 1) for j in range(len(base_seg))]) / (1e-6 + sum(cur_seg))
        dataset[idx+1][str(i)]["M3"] = (sum(cur_bid) - sum(base_bid)) / len(base_bid)
        dataset[idx+1][str(i)]["M4"] = sum([int(base_seg[j] == 0 and cur_seg[j] == 1) for j in range(len(base_seg))])
        M1_list.append(dataset[idx+1][str(i)]["M1"])
        M2_list.append(dataset[idx+1][str(i)]["M2"])
        M3_list.append(dataset[idx+1][str(i)]["M3"])
        M4_list.append(dataset[idx+1][str(i)]["M4"])
        base_seg_list.append(sum(base_seg))
        cur_seg_list.append(sum(cur_seg))
    dataset[idx+1]["M1_avg"] = np.mean(M1_list)
    dataset[idx+1]["M2_avg"] = np.mean(M2_list)
    dataset[idx+1]["M3_avg"] = np.mean(M3_list)
    dataset[idx+1]["M4_avg"] = np.mean(M4_list)
    print(idx+1, np.mean(M1_list), np.mean(M2_list), np.mean(M3_list), np.mean(M4_list))
    # print(np.mean(base_seg_list), np.mean(cur_seg_list))