import os
import json
import time
import argparse
from docker_api import *
import pandas as pd
import lcdk.lcdk  as LeslieChow
from time import gmtime, strftime
import irlutils.file.file_utils as fu
import ray
from copy import deepcopy
"""
TODO: add document

"""


parser = argparse.ArgumentParser()
parser.add_argument('--sites_path', help="The path of base crawl site list", type=str)
parser.add_argument('--num_process', help="Number of process for base crawl", type=int)
parser.add_argument('--dump_root', help="The path in docker for storing files. E.g. /home/user/Desktop/base_crawls", type=str)
parser.add_argument('--recollect', help="recollect data, 0 or 1", type=int, default=0)
parser.add_argument('--timeout', help="how many minutes", type=str)
parser.add_argument('--s', help="shift", type=int, default=0)


args = parser.parse_args()

def run_docker(number_browsers, sites, sites_cat, sites_click, sites_index, run_b=True, load_p=True, save_p=True, dump_root="/home/user/Desktop/crawls"):
    DBG = LeslieChow.lcdk()
    DOCKER_VOLUME = "docker-volume"
    DB = os.path.join(DOCKER_VOLUME, "logs", "manifest_drl.sqlite")
    path = os.path.join(DOCKER_VOLUME, "logs")
    fu.mkdir(path, exist_ok=True)
    fu.touch(DB)
    fu.chmod(DOCKER_VOLUME, mode=777, recursive=True)

    d_api = []
    # root = os.getcwd() + "/docker-volume"
    root = "/SSD/docker-volume"
    fu.mkdir(os.path.join(root, dump_root.split("/")[-1]))
    for index in range(number_browsers):
        cur_sites = deepcopy(sites[index])
        cur_sites_click = deepcopy(sites_click[index])

        cfg = {"browser_id":'',
            "start_time":'',
            "crawl_type":'',
            "dump_root":dump_root,
            "cmd":'',
            "run_bids_server": run_b,
            "load_profile_dir": None,
            "save_profile_dir": None,
            "sites": cur_sites,
            "sites_getads": cur_sites_click
            }
        Time = time.time()
        cfg["browser_id"]= "browser_{}".format(int(sites_index[index]))
        cfg["start_time"] = "{}".format(strftime("%Y%m%d%H%M%S0000", gmtime(Time)))
        cfg["crawl_type"] = sites_cat[index]
        if load_p:
            cfg["load_profile_dir"] = os.path.join(cfg['dump_root'], cfg['crawl_type'], cfg['browser_id'])
        if save_p:
            cfg["save_profile_dir"] = os.path.join(cfg['dump_root'], cfg['crawl_type'], cfg['browser_id'])
        cmd = json.dumps(cfg)
        cfg["cmd"] = "sudo timeout "+ args.timeout + "m /usr/bin/python3 /opt/OpenWPM/run_ml_crawl.py --cfg '{}'".format(json.dumps(cfg))
        DBG.lt_green(cfg['cmd'])
        d_api.append(docker_api.remote(cmd=cfg["cmd"], cfg=cfg, stand_alone=False))

    process = [c.run_container.remote() for c in d_api]
    ray.get(process)

with open(args.sites_path, "r") as json_file:
    data = json.load(json_file)

# base_url_seq = data
if isinstance(data,list):
    base_url_seq = {} # data
    for item in data:
        base_url_seq.update(item)
else:
    base_url_seq = data

if args.recollect == 0:
    count = 0
    batch_url_seq = []
    batch_url_seq_cat = []
    batch_url_seq_ad = []
    base_url_seq_index = []
    index = 0
    number_browsers = args.num_process
    for crawl_id in base_url_seq.keys():
        index += 1
        if "aol" in args.dump_root:
            if (index-1) % 100 >= 100 or index <= len(base_url_seq.keys()) - 1000:
                continue
        else:
            if (index-1) % 100 >= 10 or index <= len(base_url_seq.keys()) - 1000:
                continue
        flag = 0
        dir_path = "./docker-volume/{}/{}".format(args.dump_root.split("/")[-1],crawl_id)
        try:
            for browser_dir in os.listdir(dir_path):
                browser_dir_path = dir_path + "/" + browser_dir
                if os.path.isfile(browser_dir_path + "/profile.tar.gz"):
                    flag = 1
        except:
            flag = 0
        if flag == 1:
            continue
        count += 1
        base_url_seq[crawl_id] = base_url_seq[crawl_id] + ["http://wwww.speedtest.net","http://www.kompas.com", "http://www.cnn.com"]
        print(len(base_url_seq[crawl_id]))
        batch_url_seq.append(base_url_seq[crawl_id])
        batch_url_seq_cat.append(crawl_id)
        batch_url_seq_ad.append([0] * len(base_url_seq[crawl_id]))
        base_url_seq_index.append((index-1)%number_browsers)
        if count == number_browsers:
            print(batch_url_seq_cat)
            ray.init()
            run_docker(number_browsers, batch_url_seq, batch_url_seq_cat, batch_url_seq_ad, base_url_seq_index, run_b=True, load_p=False, save_p=True, dump_root=args.dump_root)
            batch_url_seq = []
            batch_url_seq_cat = []
            batch_url_seq_ad = []
            base_url_seq_index = []
            count = 0
            time.sleep(10)
            os.system("sudo systemctl restart docker")
            ray.shutdown()
            # break
    print(batch_url_seq_cat)
    ray.init()
    run_docker(len(batch_url_seq), batch_url_seq, batch_url_seq_cat, batch_url_seq_ad, base_url_seq_index, run_b=True, load_p=False, save_p=True, dump_root=args.dump_root)
    ray.shutdown()
else:
    count = 0
    index = 0
    batch_url_seq = []
    batch_url_seq_cat = []
    batch_url_seq_ad = []
    base_url_seq_index = []
    number_browsers = args.num_process
    for crawl_id in base_url_seq.keys():
        index += 1
        if "aol" in args.dump_root:
            if (index-1) % 100 >= 100 or index <= len(base_url_seq.keys()) - 1000:
                continue
        else:
            if (index-1) % 100 >= 10 or index <= len(base_url_seq.keys()) - 1000:
                continue
            
        flag = 0
        dir_path = "/SSD/docker-volume/{}/{}".format(args.dump_root.split("/")[-1],crawl_id)
        count += 1

        base_url_seq[crawl_id] = [
            "https://registry.bluekai.com/get_categories"
            # "https://registry.bluekai.com/get_categories",
            # "http://wwww.speedtest.net",
            # "http://www.kompas.com",
            # "http://www.cnn.com"
        ]
        batch_url_seq.append(base_url_seq[crawl_id])
        batch_url_seq_cat.append(crawl_id)
        batch_url_seq_ad.append([0] * len(base_url_seq[crawl_id]))
        base_url_seq_index.append((index-1)%number_browsers)
        if count == number_browsers:
            print(batch_url_seq_cat, base_url_seq_index)
            ray.init()
            run_docker(number_browsers, batch_url_seq, batch_url_seq_cat, batch_url_seq_ad, base_url_seq_index, run_b=True, load_p=True, save_p=True, dump_root=args.dump_root)
            batch_url_seq = []
            batch_url_seq_cat = []
            batch_url_seq_ad = []
            base_url_seq_index = []
            count = 0
            time.sleep(10)
            os.system("sudo systemctl restart docker")
            ray.shutdown()
            # break
    print(batch_url_seq_cat, base_url_seq_index)
    ray.init()
    run_docker(len(batch_url_seq), batch_url_seq, batch_url_seq_cat, batch_url_seq_ad, base_url_seq_index, run_b=True, load_p=True, save_p=True, dump_root=args.dump_root)
    ray.shutdown()
