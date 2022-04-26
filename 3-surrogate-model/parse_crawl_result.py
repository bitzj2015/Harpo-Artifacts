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

def parse_content_and_bids(hb_domain, content_dir, bids_db, tag="base", remove_zero_bids=True):
    content = {}
    if tag == "base":
        for content_file in os.listdir(content_dir):
            try:
                content_file_path = content_dir + "/" + content_file
                with open(content_file_path, "r") as json_file:
                    content_data = json.load(json_file)
                    url = content_data["url"]
                    content_url = html2text.html2text(content_data["text"])
                content["{}_{}".format(url,content_file[:-5].split("_")[-1])] = content_url
            except:
                content["{}_{}".format(url,content_file[:-5].split("_")[-1])] = None
    elif tag == "collect":
        content_file_path = content_dir + "/url_1.json"
        try:
            with open(content_file_path, "r") as json_file:
                content_data = json.load(json_file)
                content_url = html2text.html2text(content_data["text"])
            content["seg_info"] = content_url
        except:
            # print(content_file_path)
            content["seg_info"] = ""
    connection = sqlite3.connect(bids_db)
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT cpm, bidder, domain, adUnitCode, visit_url FROM pbjs_bids")
        rows = cursor.fetchall()
        if len(rows) > 0:
            bids_repo_tmp = {}
            bids_repo_tmp["logpath"] = bids_db
            url_list = rows[0][4].split("'")
            url_list_tmp = []
            for url in url_list:
                if url.startswith("http"):
                    url_list_tmp.append(url)
            bids_repo_tmp["visit_url"] = deepcopy(url_list_tmp[0:-1])
            bids_repo_tmp["bids_by_bidder"] = {}
            bids_repo_tmp["bids_by_ad"] = {}
            bids_repo_tmp["bids_by_ad_bidder"] = {}
            for domain in hb_domain:
                bids_repo_tmp["bids_by_bidder"][domain] = {}
                bids_repo_tmp["bids_by_ad"][domain] = {} 
                bids_repo_tmp["bids_by_ad_bidder"][domain] = {}

            for i in range(len(rows)):
                domain = rows[i][2]
                if domain in hb_domain:
                    if rows[i][1] not in bids_repo_tmp["bids_by_bidder"][domain].keys():
                        bids_repo_tmp["bids_by_bidder"][domain][rows[i][1]] = []
                    if rows[i][3] not in bids_repo_tmp["bids_by_ad"][domain].keys():
                        bids_repo_tmp["bids_by_ad"][domain][rows[i][3]] = []
                    if rows[i][3] not in bids_repo_tmp["bids_by_ad_bidder"][domain].keys():
                        bids_repo_tmp["bids_by_ad_bidder"][domain][rows[i][3]] = []
                    bids_value = float(rows[i][0])
                    if remove_zero_bids == True:
                        if bids_value > 1e-4:
                            bids_repo_tmp["bids_by_bidder"][domain][rows[i][1]].append(bids_value)
                            bids_repo_tmp["bids_by_ad"][domain][rows[i][3]].append(bids_value)
                            bids_repo_tmp["bids_by_ad_bidder"][domain][rows[i][3]].append((bids_value,rows[i][1]))

            for domain in hb_domain:
                med_bids = 0
                med_tmp = []
                for key in bids_repo_tmp["bids_by_bidder"][domain].keys():
                    if len(bids_repo_tmp["bids_by_bidder"][domain][key]) > 0:
                        med_bids += statistics.median(bids_repo_tmp["bids_by_bidder"][domain][key])
                        med_tmp += bids_repo_tmp["bids_by_bidder"][domain][key]

            return {"content":content, "bids": bids_repo_tmp}
        if tag == "base":
            print(tag, content_dir)
    except:
        pass
    return {"content":content, "bids": None}

@ray.remote
def bp_crawl_analysis(dir_path, hb_domain, remove_zero_bids=True):
    bids_repo = []
    for browser_dir in os.listdir(dir_path):
        browser_dir_path = dir_path + "/" + browser_dir
        print("browser:", browser_dir_path)
        filename = browser_dir
        file_list = sorted(os.listdir(browser_dir_path + "/bids"))
        is_collect = 0
        for i in range(len(file_list)):
            content_dir = browser_dir_path + "/bids/" + file_list[i] + "/content"
            url_list = sorted(os.listdir(content_dir))
            if len(url_list) > 10:
                is_collect = 1
                break
        if is_collect == 1:
            base_bids_dir = file_list[i]
        else:
            base_bids_dir = file_list[-1]
        # try:
        #     base_bids_dir = file_list[-2]
        # except:
        #     base_bids_dir = file_list[-1]
            # print(base_bids_dir)
        # base_bids_dir = file_list[-1]
        collect_bids_dir = file_list[-1]
        base_content_dir= browser_dir_path + "/bids/" + base_bids_dir + "/content"
        collect_content_dir= browser_dir_path + "/bids/" + collect_bids_dir + "/content"
        print("base_content_dir", base_content_dir, "collect_bids_dir", collect_content_dir)
        base_bids_db = browser_dir_path + "/bids/" + base_bids_dir + "/hb_bids.sqlite"
        collect_bids_db = browser_dir_path + "/bids/" + collect_bids_dir + "/hb_bids.sqlite"
        base_result = parse_content_and_bids(hb_domain, base_content_dir, base_bids_db, "base", remove_zero_bids)
        collect_result = parse_content_and_bids(hb_domain, collect_content_dir, collect_bids_db, "collect", remove_zero_bids)
        bids_repo.append({"base":base_result, "collect": collect_result}) 
    return {dir_path.split("/")[-1]:bids_repo}
bids_repo_all = {}
count = 0
sub_dir_path_list = []
for sub_dir in sorted(os.listdir(args.dir_path)):
    # print(sub_dir)
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
    # bids_repo = bp_crawl_analysis(sub_dir_path, hb_domain)
    # bids_repo_all[sub_dir] = deepcopy(bids_repo)
if args.debug:
    with open("../DATA/crawl_result/{}_debug.json".format(args.dir_id), "w") as f:
        json.dump(bids_repo_all, f)
else:
    with open("../DATA/crawl_result/{}.json".format(args.dir_id), "w") as f:
        json.dump(bids_repo_all, f)
    
            
