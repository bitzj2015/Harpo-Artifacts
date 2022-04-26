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
parser.add_argument('--model_path', help="path for storing models", default="/SSD/model_sep", type=str)
parser.add_argument('--dataset_path', help="path for storing dataset", default="/SSD/dataset_sep", type=str)
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
VERSION = "_sep"
result_filename = [
    "base_{}".format(alpha),
    "AdNauseam_{}".format(alpha),
    "TrackThis_{}".format(alpha),
    # "Intent_{}".format(alpha),
    "Intent_{}_new".format(alpha),
    # "Intent_0_bids_{}".format(alpha),
    "Intent_0_bids_{}_new".format(alpha),
    # "Intent_0_segs_{}_M1".format(alpha),
    # "Intent_0_segs_{}_M2".format(alpha),
    "Intent_0_segs_{}_M1_new".format(alpha),
    "Intent_0_segs_{}_M2_new".format(alpha),
    "Intent_2_bids_{}".format(alpha),
    "Intent_2_segs_{}_M1".format(alpha),
    "Intent_2_segs_{}_M2".format(alpha)
]

result_filename = [
    "aol_BASE",
    "aol_adnauseam",
    "aol_trackthis",
    "aol_bids_1",
    "aol_bids_0_new",
    "aol_M1_0",
    "aol_M2_0",
    "aol_bids_2_new",
    "aol_M1_2",
    "aol_M2_2",
    "base_0_0.1_update",
    "Intent_0_0.1_update"
]

result_filename = [
    "base_sep",
    "Intent_1_0.1_sep",
    "Intent_2_0.1_sep",
    "Intent_0_0.1_sep"
]

result_filename = [
    "aol_base_sep",
    "aol_adnauseam_sep",
    "aol_trackthis_sep",
    "aol_bids_1_sep",
    "aol_bids_0_sep",
    "aol_bids_2_sep",
    "aol_M1_0_sep",
    "aol_M2_0_sep",
    "aol_M1_2_sep",
    "aol_M2_2_sep"
]
result_filename = [
    "base_sep",
    "AdNauseam_0.1_sep",
    "TrackThis_0.1_sep",
    "Intent_1_0.1_sep",
    "Intent_0_0.1_sep",
    "Intent_2_0.1_sep",
    "Intent_M1_0_sep",
    "Intent_M2_0_sep",
    "Intent_M1_2_sep",
    "Intent_M2_2_sep"
]


node_list = [path[10:-4] for path in os.listdir(f"/SSD/baseline{VERSION}/oracle")]
bids_list = [path[10:-4] for path in os.listdir(f"/SSD/baseline{VERSION}/bids")]

with open(f"../DATA/analysis_result/bids_result{VERSION}.json", "r") as file:
    bids_th = json.load(file)
hb_domain = [
    "www.speedtest.net",
    "www.kompas.com",
    "cnn.com"
]

# print(node_list, bids_list, bids_th)
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
for index in range(len(result_filename)):
    with open("../DATA/crawl_result/{}.json".format(result_filename[index]), "r") as file:
        result_dataset = json.load(file)

    for key in result_dataset.keys():
        # if int(key.split("_")[-1]) >= 10:
        #     continue
        persona = result_dataset[key][0]
        # print(key, persona["base"]["bids"]["visit_url"])
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
                        tag = "base_bids_{}_{}{}".format(re.sub('[^0-9a-zA-Z]+', '_', ad), bidder, VERSION)
                        if tag in bids_list:
                            if tag not in all_bids.keys():
                                all_bids[tag] = []
                            all_bids[tag].append(np.mean(tmp[bidder]))             
        except:
            continue
c = 0
for tag in all_bids.keys():
    sorted_list = sorted(all_bids[tag])
    # c += 1
    # if c != 10:
    #     continue
    # if len(sorted_list) < 100:
    #     continue
    sorted_list[:0] += [0 for _ in range(1000-len(sorted_list))]
    # all_bids_th[tag] = sorted_list[int(0.8 * len(sorted_list))]
    all_bids_th[tag] = np.mean(sorted_list) + np.std(sorted_list)
    # print(sum(sorted_list>=all_bids_th[tag]))
    print(len(sorted_list), tag)
    # plt.figure()
    # plt.plot(sorted_list)
    # plt.savefig("./fig/{}.jpg".format(tag))
    

print(all_bids_th)
# all_bids_th = {
#     'base_bids_6692_speedtest_net_stnext_top_rectangle_rubicon': 0.980615, 
#     'base_bids_6692_speedtest_net_stnext_bottom_rectangle_rubicon': 0.966847, 
#     'base_bids_div_gpt_ad_Skyscraper_openx': 0.5735, 
#     'base_bids_div_gpt_ad_Skyscraper_criteorn': 0.9833948612213135, 
#     'base_bids_div_gpt_ad_Giant_criteorn': 0.5842216610908508, 
#     'base_bids_div_gpt_ad_Giant_openx': 0.8394999999999999, 
#     'base_bids_ad_bnr_atf_01_criteo': 1.7576372623443604, 
#     'base_bids_ad_bnr_atf_01_rubicon': 2.203982, 
#     'base_bids_div_gpt_ad_Top_1_1_openx': 1.6355, 
#     'base_bids_div_gpt_ad_Top_1_1_pubmatic': 1.3716666666666668
# }

for index in range(len(result_filename)):
    IO_DATA = {}
    dataset[result_filename[index]] = {}
    with open("../DATA/crawl_result/{}.json".format(result_filename[index]), "r") as file:
        result_dataset = json.load(file)

    # print("[INFO] Data points in {}: {}".format("../DATA/crawl_result/{}.json".format(result_filename[index]), len(result_dataset)))
    count = 0
    tmp = []
    err = [0,0]
    dataset[result_filename[index]]["bids_cpm_avg"] = 0
    dataset[result_filename[index]]["bids_avg"] = 0

    for key in result_dataset.keys():
        
        # if int(key.split("_")[-1]) >= 10:
        #     continue
        key_id = str(count)
        count += 1
        persona = result_dataset[key][0]
        IO_DATA[key] = {}
        try:
            IO_DATA[key]["persona"] = persona["base"]["bids"]["visit_url"]
        except:
            IO_DATA[key]["persona"] = []
        
        dataset[result_filename[index]][key_id] = {}
        
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

            dataset[result_filename[index]][key_id]["segs"] = deepcopy(seg_data_clean)
        except:
            err[0] += 1
            dataset[result_filename[index]][key_id]["segs"] = dict(zip(node_list,[0 for _ in range(len(node_list))]))  

        IO_DATA[key]["segs"] = dataset[result_filename[index]][key_id]["segs"]
        try:
            dataset[result_filename[index]][key_id]["bids"] = dict(zip(list(all_bids_th.keys()),[0 for _ in range(len(list(all_bids_th.keys())))]))
            dataset[result_filename[index]][key_id]["bids_cpm"] = dict(zip(list(all_bids_th.keys()),[0 for _ in range(len(list(all_bids_th.keys())))]))
            
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
                        tag = "base_bids_{}_{}{}".format(re.sub('[^0-9a-zA-Z]+', '_', ad), bidder, VERSION)
                        if tag in list(all_bids_th.keys()):
                            # th = bids_th[tag]["th"]
                            th = all_bids_th[tag]
                            # if result_filename[index] == "Intent_0_bids_0.1":
                            #     print(th, np.mean(tmp[bidder]), np.mean(tmp[bidder]) >= th, int(np.mean(tmp[bidder]) >= th))
                            dataset[result_filename[index]][key_id]["bids"][tag] = int(np.mean(tmp[bidder]) >= th)
                            dataset[result_filename[index]][key_id]["bids_cpm"][tag] = np.mean(tmp[bidder])
                            
        except:
            err[1] += 1
            dataset[result_filename[index]][key_id]["bids"] = dict(zip(list(all_bids_th.keys()),[0 for _ in range(len(list(all_bids_th.keys())))]))
            dataset[result_filename[index]][key_id]["bids_cpm"] = dict(zip(list(all_bids_th.keys()),[0 for _ in range(len(list(all_bids_th.keys())))]))
        dataset[result_filename[index]]["bids_cpm_avg"] += np.mean(list(dataset[result_filename[index]][key_id]["bids_cpm"].values()))
        dataset[result_filename[index]]["bids_avg"] += np.mean(list(dataset[result_filename[index]][key_id]["bids"].values()))
    dataset[result_filename[index]]["bids_cpm_avg"] = dataset[result_filename[index]]["bids_cpm_avg"] / count
    dataset[result_filename[index]]["bids_avg"] = dataset[result_filename[index]]["bids_avg"] / count
    print(count, err, key, key_id, result_filename[index], dataset[result_filename[index]]["bids_cpm_avg"], dataset[result_filename[index]]["bids_avg"])

    with open(f"../DATA/analysis_result/oracle_{result_filename[index]}.json", "w") as json_file:
        json.dump(IO_DATA, json_file)

for index in range(len(result_filename)-1):
    M1_list = []
    M2_list = []
    M3_list = []
    M4_list = []
    base_seg_list = []
    cur_seg_list = []
    for i in range(100):
        base_seg = list(dataset[result_filename[0]][str(i)]["segs"].values())
        cur_seg = list(dataset[result_filename[index+1]][str(i)]["segs"].values())
        base_bid = list(dataset[result_filename[0]][str(i)]["bids"].values())
        cur_bid = list(dataset[result_filename[index+1]][str(i)]["bids"].values())
        dataset[result_filename[index+1]][str(i)]["M1"] = sum([(base_seg[j] - cur_seg[j]) ** 2 for j in range(len(base_seg))]) / len(base_seg)
        dataset[result_filename[index+1]][str(i)]["M2"] = sum([int(base_seg[j] == 0 and cur_seg[j] == 1) for j in range(len(base_seg))]) / (1e-6 + sum(cur_seg))
        dataset[result_filename[index+1]][str(i)]["M3"] = (sum(cur_bid) - sum(base_bid)) / len(base_bid)
        dataset[result_filename[index+1]][str(i)]["M4"] = sum([int(base_seg[j] == 0 and cur_seg[j] == 1) for j in range(len(base_seg))])
        M1_list.append(dataset[result_filename[index+1]][str(i)]["M1"])
        M2_list.append(dataset[result_filename[index+1]][str(i)]["M2"])
        M3_list.append(dataset[result_filename[index+1]][str(i)]["M3"])
        M4_list.append(dataset[result_filename[index+1]][str(i)]["M4"])
        base_seg_list.append(sum(base_seg))
        cur_seg_list.append(sum(cur_seg))
    dataset[result_filename[index+1]]["M1_avg"] = np.mean(M1_list)
    dataset[result_filename[index+1]]["M2_avg"] = np.mean(M2_list)
    dataset[result_filename[index+1]]["M3_avg"] = np.mean(M3_list)
    dataset[result_filename[index+1]]["M4_avg"] = np.mean(M4_list)
    print(result_filename[index+1], np.mean(M1_list) * 20, np.mean(M2_list), np.mean(M3_list), np.mean(M4_list))
    # print(np.mean(base_seg_list), np.mean(cur_seg_list))