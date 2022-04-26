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

VERSION = "_sep"

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', help="path for storing models", default=f"/SSD/model{VERSION}", type=str)
parser.add_argument('--dataset_path', help="path for storing dataset", default=f"/SSD/dataset{VERSION}", type=str)
args = parser.parse_args()


# model = Doc2Vec.load("../DATA/model/model-final-dbow-300.bin") 
model = Doc2Vec.load("../DATA/model/model_new_100.bin") 

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

# result_filename = ["crawl_new_1", "crawl_new_2"]
# profile_filename = ["crawl_new_1", "crawl_new_2"]
# result_filename = ["crawl_new_0_test", "crawl_new_1_test"]
# profile_filename = ["crawl_new_0_test", "crawl_new_1_test"]
result_filename = ["crawl_new_1_sep"]
profile_filename = ["crawl_new_1"]

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
        # try:
        #     assert str(visit_url[0:20]) == str(persona["base"]["bids"]["visit_url"][0:20])
        # except:
        #     print(visit_url)
        #     print(persona["base"]["bids"]["visit_url"])
        # if count > 2:
        #     break
        try:
            base_data = persona["base"]
            persona_content_data = base_data["content"]
        except:
            continue
        parse_result = {"persona_id": key, "visit_url":visit_url, "all_seg": []}
        if get_content:
            for url in persona_content_data.keys():
                if url not in url_content.keys():
                    url_content[url] = prepare_text(persona_content_data[url])
        try:
            collect_data = persona["collect"]
            seg_data_raw = collect_data["content"]["seg_info"].split("\"")
            seg_data = []
            for i in range(len(seg_data_raw)):
                if ", " in seg_data_raw[i] or "[" in seg_data_raw[i] or \
                "]" in seg_data_raw[i] or "A/B Test Groups" in seg_data_raw[i] or "null" in seg_data_raw[i]:
                    continue
                seg_data.append(seg_data_raw[i].replace("\n", " "))
            # print(key, seg_data)
            for i in range(len(seg_data)):
                sub_segs = seg_data[i].split(" > ")
                for j in range(len(sub_segs)-1):
                    sub_segs[j+1] = sub_segs[j] + " > " + sub_segs[j+1]
                parent = segment_tree["root"]
#                 print(sub_segs)
                for j in range(len(sub_segs)):
                    if sub_segs[j] not in parent.keys():
                        parent[sub_segs[j]] = {}
                    parent = parent[sub_segs[j]]
        except:
            continue
        
with open(f"../DATA/analysis_result/segment_tree{VERSION}.json", "w") as file:
    json.dump(segment_tree, file)

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
with open(f"../DATA/analysis_result/segment_tree_leaf{VERSION}.json", "w") as json_file:
    json.dump(node_list, json_file)

if get_content:
    with open("../DATA/analysis_result/url_content.json", "w") as file:
        json.dump(url_content, file)


with open(f"../DATA/analysis_result/segment_tree_leaf{VERSION}.json", "r") as json_file:
    node_list = json.load(json_file)

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
with open(f"../DATA/analysis_result/all_segment_data{VERSION}.json", "w") as file:
    json.dump(Segment_data, file)

Selected_segment_data = {}
pre_prob = 0
grou_segment_data = {"0":[]}
for i in range(len(seg_prob_rank)):
    item = heappop(seg_prob_rank)
    if item[1] == ', ':
        continue
    if item[0] == pre_prob:
        grou_segment_data[pre_prob].append(item[1])
        continue
    pre_prob = item[0]
    grou_segment_data[pre_prob] = [item[1]]
    Selected_segment_data[item[1]] = item[0]
    
with open(f"../DATA/analysis_result/all_segment_filter{VERSION}.json", "w") as file:
    json.dump(Selected_segment_data, file)
with open(f"../DATA/analysis_result/group_segment_filter{VERSION}.json", "w") as file:
    json.dump(grou_segment_data, file)
print("[INFO] Total number of segments after filtering: {}".format(len(Selected_segment_data)))

with open(f"../DATA/analysis_result/all_segment_filter{VERSION}.json", "r") as file:
    Selected_segment_data = list(json.load(file).keys())
with open(f"../DATA/analysis_result/all_segment_data{VERSION}.json", "r") as file:
    Segment_data = json.load(file)

os.system(f"rm -rf run_oracle{VERSION}.sh")
seg_list = {}
for name in Selected_segment_data:
    rd_index = np.arange(len(Segment_data))
    random.shuffle(rd_index)
    content = []
    label = []
    count = 0
    count_1 = 0
    count_0 = 0
    URL = {}
    for index in range(len(rd_index)):
        i = rd_index[index]
        url_list = Segment_data[i]["visit_url"][0:20]
        flag = 0
        base_seg = []
        segments = Segment_data[i]["all_seg"]
        if name in segments:
            reward = 1
            count_1 += 1
        else:
            reward = 0
            count_0 += 1
        tmp = []
        if str(url_list) not in URL:
            URL[str(url_list)] = 0
            for url in url_list:
                try:
                    tmp.append(model.docvecs[url])
                except:
                    tmp.append(np.zeros(300) / 300)
            label.append(np.array([reward]))
            content.append(np.stack(np.reshape(tmp,(1,20,300)), axis=0))
        else:
            print(str(url_list))

    print("[INFO] Segment: {}, # of data points: {}, shape of content: {}.".format(name, len(rd_index), np.shape(content[0])))
    ratio = int(count_0 * 10/count_1) / 10
    if ratio > 25 or ratio < 0.05:
        continue
    print("[INFO] Ratio: {}, count_0: {}, count_1: {}.".format(ratio, count_0, count_1))
    
    name = re.sub('[^0-9a-zA-Z]+', '_', name)

    hf = h5py.File(f"{args.dataset_path}/dataset_{name}.h5", 'w')
    hf.create_dataset('input', data=content)
    hf.create_dataset('label', data=label)
    hf.close()    
    seg_list[name] = {"0": count_0, "1": count_1}

    cmd = f"python ../attack/train_bp.py --type train_bp "\
          f"--data {args.dataset_path}/dataset_{name}.h5 "\
          f"--param {args.model_path}/simulator_{name}.pkl --retrain 0 --lr 0.01 --th 0.1 --ratio {ratio} --seg {name}"

    with open(f"run_oracle{VERSION}.sh", "a") as f:
        f.write(cmd)
        f.write("\n")

with open(f"../DATA/analysis_result/oracle_result{VERSION}.json", "w") as f:
    json.dump(seg_list, f)



