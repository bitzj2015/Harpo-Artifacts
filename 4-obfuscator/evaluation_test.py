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
import collections

result_filename = [
    "data_poison_test",
    "data_poison_test_new"
]

result_filename = [
    "data_poison_test2",
    "data_poison_test_intent2",
    "data_poison_test_new3"
]

# result_filename = [
#     "data_poison_3",
#     "data_poison_intent_3",
#     "data_poison_3_new"
# ]

hb_domain = [
    "www.speedtest.net",
    "www.kompas.com",
    "cnn.com"
]

bidders =  {
    'nobid': 0.5299388010363619, 
    'rubicon': 2.3082578949395742, 
    'medianet': 1.6566254691810247, 
    'openx': 0.8005624352326812, 
    'aol': 2.1036876320055833, 
    'teads': 0.11397959183673467, 
    'criteorn': 2.2691571739554344, 
    'criteo': 2.025538697363844, 
    'pubmatic': 3.251172958783691, 
    'trustx': 0.728658572373792, 
    'appier': 0.009150797497354498, 
    'smartadserver': 2.4313018540411275, 
    'ix': 0.5466666666666667, 
    'triplelift': 0.034125
}


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
    bids_data = {}
    print(result_filename[index])
    with open("../DATA/crawl_result/{}.json".format(result_filename[index]), "r") as file:
        result_dataset = json.load(file)

    for key in result_dataset.keys():
        cate = key.split("_")[1]
        if cate not in bids_data.keys():
            bids_data[cate] = {}
        persona = result_dataset[key][0]
        bids_list = {}
        try:
            for domain in hb_domain:
                collect_bids_data = persona["base"]["bids"]['bids_by_ad_bidder'][domain]
                for ad in collect_bids_data.keys():
                    for item in collect_bids_data[ad]:
                        bids = item[0]
                        bidder = item[1]
                        if bidder not in bids_list.keys():
                            bids_list[bidder] = [bids]
                        else:
                            bids_list[bidder].append(bids)
        except:
            continue
        for bidder in bids_list.keys():
            if bidder not in bids_data[cate].keys():
                bids_data[cate][bidder] = [np.mean(bids_list[bidder])]
            else:
                bids_data[cate][bidder].append(np.mean(bids_list[bidder]))
    for cate in bids_data.keys():
        print(cate)
        tmp = {}
        for bidder in bids_data[cate].keys():
            tmp[bidder] = np.mean(bids_data[cate][bidder])
        print(np.mean(list(tmp.values())), np.sum(list(tmp.values())))
        tmp_ = {}
        for bidder in bidders.keys():
            if bidder not in tmp.keys():
                tmp_[bidder] = 0
            else:
                tmp_[bidder] = tmp[bidder]
        print(list(collections.OrderedDict(sorted(tmp_.items())).values()))


