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
import scipy.io as sio

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', help="path for storing models", default="/SSD/model", type=str)
parser.add_argument('--dataset_path', help="path for storing dataset", default="/SSD/dataset", type=str)
args = parser.parse_args()

# model = Doc2Vec.load("../DATA/model/model-final-dbow-300.bin") 
model = Doc2Vec.load("../DATA/model/model_new_100.bin") 

hb_domain = [
    "www.speedtest.net",
    "www.kompas.com",
    "cnn.com"
]
result_filename = ["crawl_new_0_test", "crawl_new_1_test"]
profile_filename = ["crawl_new_0_test", "crawl_new_1_test"]
pre_data = []

count = 0
bids_data = []
all_bidder_ad = {}
all_bidder_ad_ = {}

url_content = {}
Data = {}
get_content = False
segment_tree = {"root":{}}
for i in range(len(result_filename)):
    avg_bids_1 = []
    avg_bids_2 = []
    with open("../DATA/crawl_result/{}.json".format(result_filename[i]), "r") as file:
        result_dataset = json.load(file)
    with open("../DATA/profiles/{}.json".format(profile_filename[i]), "r") as file:
        profile_dataset = json.load(file)

    print("[INFO] Data points in {}: {}".format("../DATA/crawl_result/{}.json".format(result_filename[i]), len(profile_dataset)))

    for key in result_dataset.keys():
        persona = result_dataset[key][0]
        count += 1
        visit_url = profile_dataset[key]

        try:
            base_data = persona["base"]
            persona_content_data = base_data["content"]
        except:
            continue
        parse_result = {"persona_id": key, "visit_url":visit_url, "all_seg": []}
        try:
            collect_data = persona["collect"]
            seg_data_raw = collect_data["content"]["seg_info"].split("\"")
            seg_data = []
            for i in range(len(seg_data_raw)):
                if ", " in seg_data_raw[i] or "[" in seg_data_raw[i] or \
                "]" in seg_data_raw[i] or "A/B Test Groups" in seg_data_raw[i] or "null" in seg_data_raw[i]:
                    continue
                seg_data.append(seg_data_raw[i].replace("\n", " "))
            for i in range(len(seg_data)):
                sub_segs = seg_data[i].split(" > ")
                for j in range(len(sub_segs)-1):
                    sub_segs[j+1] = sub_segs[j] + " > " + sub_segs[j+1]
                parent = segment_tree["root"]
                for j in range(len(sub_segs)):
                    if sub_segs[j] not in parent.keys():
                        parent[sub_segs[j]] = {}
                    parent = parent[sub_segs[j]]
        except:
            continue

leaf_seg = []
# print(segment_tree)
def get_leaf_node(tree, node_list):
    for child in tree.keys():
        if len(tree[child]) == 0:
            node_list.append(child)
        else:
            node_list = get_leaf_node(tree[child], node_list)
    return node_list
node_list = get_leaf_node(segment_tree, [])

seg_prob_rank = []
pre_data = []
count = 0
Segment_data = []
All_segment_data = {}
URL_content = {}
Data = {}
get_content = False
segment_tree = {"root":{}}
for i in range(len(result_filename)):
    with open("../DATA/crawl_result/{}.json".format(result_filename[i]), "r") as file:
        result_dataset = json.load(file)
    with open("../DATA/profiles/{}.json".format(profile_filename[i]), "r") as file:
        profile_dataset = json.load(file)

    print("[INFO] Data points in {}: {}".format("../DATA/crawl_result/{}.json".format(result_filename[i]), len(profile_dataset)))

    for key in result_dataset.keys():
        persona = result_dataset[key][0]
        count += 1
        visit_url = profile_dataset[key]
        try:
            base_data = persona["base"]
            persona_content_data = base_data["content"]
        except:
            continue
        parse_result = {"persona_id": key, "visit_url":visit_url, "all_seg": []}

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
            
            seg_data_clean = []                   
            for i in range(len(seg_data)):
                seg = seg_data[i]
                if seg not in node_list:
                    continue
                seg_data_clean.append(seg)
                if seg not in All_segment_data:
                    All_segment_data[seg] = 0
                All_segment_data[seg] += 1
            parse_result["all_seg"] = deepcopy(seg_data_clean)
        except:
            parse_result["all_seg"] = None
            continue
        Segment_data.append(parse_result)

for key in All_segment_data.keys():
    heappush(seg_prob_rank, (All_segment_data[key], key))
print("[INFO] Total number of segments: {}".format(len(All_segment_data)))


Selected_segment_data = {}
pre_prob = 0
grou_segment_data = {"0":[]}
for i in range(len(seg_prob_rank)):
    item = heappop(seg_prob_rank)
    if item[1] == ', ' or item[1] == '_':
        continue
    pre_prob = item[0]
    grou_segment_data[pre_prob] = [item[1]]
    Selected_segment_data[item[1]] = item[0]

seg_mat = []
seg_key = {"*":0}
for name in Selected_segment_data.keys():
    if name not in seg_key.keys():
        seg_key[name] = seg_key["*"]
        seg_key["*"] += 1

for i in range(len(Segment_data)):
    flag = 0
    base_seg = []
    segments = Segment_data[i]["all_seg"]
    segments_dict = {}
    for name in Selected_segment_data.keys():
        segments_dict[name] = 0
    for name in segments:
        if name == ', ' or name == '_':
            continue
        segments_dict[name] = 1
    seg_mat.append(list(segments_dict.values()))
    assert(len(list(segments_dict.values())) == len(seg_key)-1)

sio.savemat("../DATA/oracle.mat", {"mat":np.array(seg_mat)})
with open("../DATA/oracle_segs.json", "w") as f:
    json.dump(seg_key, f)

