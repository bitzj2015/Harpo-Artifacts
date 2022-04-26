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


VERSION = "_sep_test"

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', help="path for storing models", default=f"/SSD/model{VERSION}", type=str)
parser.add_argument('--dataset_path', help="path for storing dataset", default=f"/SSD/dataset{VERSION}", type=str)
args = parser.parse_args()

# model = Doc2Vec.load("../DATA/model/model_user_1.bin") 
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

result_filename = ["crawl_new_1_test_sep"]
profile_filename = ["crawl_new_1_test"]
pre_data = []

count = 0
bids_data = []
all_bidder_ad = {}
all_bidder_ad_ = {}

url_content = {}
Data = {}
get_content = False
for i in range(len(result_filename)):
    avg_bids_1 = {"0.0":[], "0.1":[], "0.3":[], "0.5":[]}
    avg_bids_2 = {"0.0":[], "0.1":[], "0.3":[], "0.5":[]}
    with open("../DATA/crawl_result/{}.json".format(result_filename[i]), "r") as file:
        result_dataset = json.load(file)
    with open("../DATA/profiles/{}.json".format(profile_filename[i]), "r") as file:
        profile_dataset = json.load(file)

    print("[INFO] Data points in {}: {}".format("../DATA/crawl_result/{}.json".format(result_filename[i]), len(profile_dataset)))

    for key in result_dataset.keys():
        persona = result_dataset[key][0]
        count += 1
        visit_url = profile_dataset[key]
        if "0.0" in key:
            alpha = "0.0"
            # continue
        elif "0.1" in key:
            alpha = "0.1"
            # continue
        elif "0.3" in key:
            alpha = "0.3"
            # continue
        elif "0.5" in key:
            alpha = "0.5"
            # continue
        else:
            print(key)
        try:
            base_data = persona["base"]
            persona_content_data = base_data["content"]
        except:
            continue
        parse_result = {"persona_id":key, "visit_url":visit_url, "base_bids": {}, "collect_bids": {}}
        if get_content:
            for url in persona_content_data.keys():
                if url not in url_content.keys():
                    url_content[url] = prepare_text(persona_content_data[url])

        try:
            for domain in hb_domain:
                base_bids_data = persona["base"]["bids"]['bids_by_ad_bidder'][domain]
                for ad in base_bids_data.keys():
                    parse_result["base_bids"][ad] = {}
                    if ad not in all_bidder_ad.keys():
                        all_bidder_ad[ad] = {}
                        
                    tmp = {}
                    for item in base_bids_data[ad]:
                        bids = item[0]
                        bidder = item[1]
                        avg_bids_1[alpha].append(bids)
                        if bidder not in tmp.keys():
                            tmp[bidder] = []
                        tmp[bidder].append(bids)
                    for bidder in tmp.keys():
                        if bidder not in all_bidder_ad[ad].keys():
                            all_bidder_ad[ad][bidder] = 0
                        all_bidder_ad[ad][bidder] += 1
                        parse_result["base_bids"][ad][bidder] = np.mean(tmp[bidder])
        except:
            parse_result["base_bids"] = None
        try:
            for domain in hb_domain:
                collect_bids_data = persona["collect"]["bids"]['bids_by_ad_bidder'][domain]
                for ad in collect_bids_data.keys():
                    parse_result["collect_bids"][ad] = {}
                    if ad not in all_bidder_ad_.keys():
                        all_bidder_ad_[ad] = {}
                    tmp = {}
                    for item in collect_bids_data[ad]:
                        bids = item[0]
                        bidder = item[1]
                        avg_bids_2[alpha].append(bids)
                        if bidder not in tmp.keys():
                            tmp[bidder] = []
                        tmp[bidder].append(bids)
                    for bidder in tmp.keys():
                        if bidder not in all_bidder_ad_[ad].keys():
                            all_bidder_ad_[ad][bidder] = 0
                        all_bidder_ad_[ad][bidder] += 1
                        parse_result["collect_bids"][ad][bidder] = np.mean(tmp[bidder]) 
        except:
            parse_result["collect_bids"] = None
        bids_data.append(deepcopy(parse_result))
        
    for alpha in avg_bids_1.keys():
        print(len(avg_bids_1[alpha]), alpha)
        print(result_filename[i], np.mean(avg_bids_1[alpha]), np.mean(avg_bids_2[alpha]))
        print(result_filename[i], np.std(avg_bids_1[alpha]), np.std(avg_bids_2[alpha]))

with open(f"../DATA/analysis_result/all_bids_data{VERSION}.json", "w") as file:
    json.dump(bids_data, file)
with open(f"../DATA/analysis_result/all_bidder_ad_base{VERSION}.json", "w") as file:
    json.dump(all_bidder_ad, file)
with open(f"../DATA/analysis_result/all_bidder_ad_collect{VERSION}.json", "w") as file:
    json.dump(all_bidder_ad_, file)

if get_content:
    with open("../DATA/analysis_result/url_content.json", "w") as file:
        json.dump(url_content, file)

with open(f"../DATA/analysis_result/all_bids_data{VERSION}.json", "r") as file:
    bids_data = json.load(file)


tags = ["base_bids", "collect_bids"]
doc2vec_emb_size = 300
name_list = {}
url_none = {}
for tag in tags:
    os.system(f"rm -rf train_surrogate_model_bids_{tag}{VERSION}.sh")
    if tag.startswith("base"):
        all_bidder_ad = json.load(open(f"../DATA/analysis_result/all_bidder_ad_base{VERSION}.json", "r"))
    else:
        all_bidder_ad = json.load(open(f"../DATA/analysis_result/all_bidder_ad_collect{VERSION}.json", "r"))
    for ad in all_bidder_ad.keys():
        for bidder in all_bidder_ad[ad]:
            bids_list = []
            rd_index = np.arange(len(bids_data))
            random.shuffle(rd_index)
            content = []
            label = []
            count = 0
            count_1 = 0
            count_0 = 0
            t = 0
            URL = {}
            name = "{}_{}_{}".format(tag, ad, bidder)
            for index in range(len(rd_index)):
                i = rd_index[index]
                try:
                    reward = bids_data[i][tag][ad][bidder]
                except:
                    reward = 0
                bids_list.append(reward)
                url_list = bids_data[i]["visit_url"][0:20]
                tmp = []
                if str(url_list) not in URL:
                    URL[str(url_list)] = 0
                    for url in url_list:
                        try:
                            tmp.append(model.docvecs[url])
                        except:
                            print(url)
                            tmp.append(np.zeros(doc2vec_emb_size) / doc2vec_emb_size)
                    label.append(np.array([reward]))
                    content.append(np.stack(np.reshape(tmp,(1,20,doc2vec_emb_size)), axis=0))
                else:
                    print(str(url_list))
                    continue

            print("[INFO] Ad: {}, bidder: {}, # of points: {}, shape of content: {}".format(
                ad, bidder, len(content), np.shape(content[0])))
            if len(content) < 500:
                continue
            name = re.sub('[^0-9a-zA-Z]+', '_', name)
            num_of_points = len(bids_list)
            th = np.mean(bids_list) + np.std(bids_list) 
            # kmeans = KMeans(n_clusters=2, random_state=0).fit(label)
            # print(sum(kmeans.labels_==0), sum(kmeans.labels_==1), sum(kmeans.labels_==2), kmeans.cluster_centers_)
            # th = sorted(bids_list)[int(0.8 * num_of_points)]

            for i in range(len(label)):
                # if kmeans.cluster_centers_[0] < kmeans.cluster_centers_[1]:
                #     label[i][0] = kmeans.labels_[i]
                # else:
                #     label[i][0] = int(1 - kmeans.labels_[i])
                # if label[i][0] == 1:
                #     count_1 += 1
                # else:
                #     count_0 += 1
                if label[i][0] > th:
                    label[i] = np.array([1])
                    count_1 += 1
                else:
                    label[i] = np.array([0])
                    count_0 += 1  
            ratio = int(count_0 * 10 / (count_1 + 1e-6)) / 10
            if ratio > 25:
                continue
            # if name != "base_bids_6692_speedtest_net_stnext_leaderboard_rubicon":
            #     continue
            # else:
            #     print("name")
            #     plt.plot(sorted(bids_list)[0:6000])
            #     plt.savefig("test.jpg")
            # name = "{}_{}".format(name, "km")
            name = name + VERSION
            print("[INFO] Ratio: {}, count_0: {}, count_1: {}, th:{}".format(ratio, count_0, count_1, th))
            hf = h5py.File(f"{args.dataset_path}/dataset_{name}.h5", 'w')
            hf.create_dataset('input', data=content)
            hf.create_dataset('label', data=label)
            hf.close()
            cmd = f"python ../attack/train_bp.py --type train_bp "\
                f"--data {args.dataset_path}/dataset_{name}.h5 "\
                f"--param {args.model_path}/simulator_{name}.pkl --retrain 0 --lr 0.01 --th 0.1 --ratio {ratio} --seg {name}"
            name_list[name] = {"0": count_0, "1": count_1, "th": th}
            with open(f"train_surrogate_model_bids_{tag}{VERSION}.sh", "a") as f:
                f.write(cmd)
                f.write("\n")
                # break
with open(f"../DATA/analysis_result/bids_result{VERSION}.json", "w") as f:
    json.dump(name_list, f)
print(content[0])
print(content[10])
