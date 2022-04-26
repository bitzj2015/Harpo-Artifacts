import statistics
import os
import sqlite3
import json
from copy import deepcopy
import argparse
import bs4
import time
import pandas as pd
import irlutils
import html2text
import ray
ray.init()

parser = argparse.ArgumentParser()
parser.add_argument('--dir_path', help="/XXX/bp_crawl_1/train_bp", type=str)
parser.add_argument('--dir_id', help="e.g. bp_crawl_1", type=str)
parser.add_argument('--num_workers', help="e.g. number of workers working in parallel", default=1, type=int)
parser.add_argument('--debug', help="whether in debug mode", action='store_true')
args = parser.parse_args()
print(args.debug)
hb_domain = [
    "www.speedtest.net",
    "www.kompas.com",
    "cnn.com"
]

def parse_content_and_bids(hb_domain, bids_db):
    connection = sqlite3.connect(bids_db)
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT top_url, link FROM page_links")
        rows = cursor.fetchall()
        ads = {}
        ads_by_url = {}
        if len(rows) > 0:
            for i in range(len(rows)):
                top_url = rows[i][0]
                ad_url = rows[i][1]
                ads[ad_url] = 1
                if top_url not in ads_by_url.keys():
                    ads_by_url[top_url] = []
                ads_by_url[top_url].append(ad_url)

            return {"ads":ads, "ads_by_url": ads_by_url}
    except:
        pass
    return {"ads":None, "ads_by_url": None}

@ray.remote
def bp_crawl_analysis(dir_path, hb_domain, remove_zero_bids=True):
    bids_repo = []
    for browser_dir in os.listdir(dir_path):
        base_bids_db = dir_path + "/" + browser_dir + "/openwpm/crawl-data.sqlite"
        base_result = parse_content_and_bids(hb_domain, base_bids_db)
        bids_repo.append({dir_path:base_result}) 
    return {dir_path.split("/")[-1]:bids_repo}

bids_repo_all = {}
count = 0
sub_dir_path_list = []
for sub_dir in sorted(os.listdir(args.dir_path)):
    sub_dir_path = args.dir_path + "/" + sub_dir
    sub_dir_path_list.append(sub_dir_path)
    count += 1
    if count == args.num_workers:
        result = ray.get([bp_crawl_analysis.remote(sub_dir_path, hb_domain) for sub_dir_path in sub_dir_path_list])
        count = 0
        sub_dir_path_list = []
        for item in result:
            bids_repo_all.update(item)
        if args.debug:
            break
if args.debug:
    with open("../DATA/crawl_result/ads_{}_debug.json".format(args.dir_id), "w") as f:
        json.dump(bids_repo_all, f)
else:
    with open("../DATA/crawl_result/ads_{}.json".format(args.dir_id), "w") as f:
        json.dump(bids_repo_all, f)
    
            
