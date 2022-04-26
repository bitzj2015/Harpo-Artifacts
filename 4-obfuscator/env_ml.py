import numpy as np
import torch
from utils import get_features
import os
import json
import time
import sqlite3
import uuid
import random
import argparse
import threading
import sys
import docker_api
import pandas as pd
import lcdk.lcdk  as LeslieChow
from time import gmtime, strftime
import irlutils.file.file_utils as fu
import irlutils.url.crawl.database_utils as dbu
from termcolor import colored
import scipy.io as sio
import pickle
from copy import deepcopy
import ray

def run_docker(number_browsers, sites, sites_cat, sites_click, sites_index, run_b=True, load_p=True, save_p=True, dump_root="/home/user/Desktop/crawls"):
    DBG = LeslieChow.lcdk()
    DOCKER_VOLUME = "docker-volume"
    DB = os.path.join(DOCKER_VOLUME, "logs", "manifest_drl.sqlite")
    path = os.path.join(DOCKER_VOLUME, "logs")
    fu.mkdir(path, exist_ok=True)
    fu.touch(DB)
    fu.chmod(DOCKER_VOLUME, mode=777, recursive=True)
    rewards_set = []
    avg_bids_set = []
    browser_id_set = []
    d_api = []
    # root = os.getcwd() + "/docker-volume"
    root = "/SSD/docker-volume"
    fu.mkdir(os.path.join(root, dump_root.split("/")[-1]))
    for index in range(number_browsers):
        cur_sites = deepcopy(sites[index])
        cur_sites_click = deepcopy(sites_click[index])
        cur_sites_getads = {}
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

class EnvArgs(object):
    def __init__(self, num_words=50, embedding_dim=300, his_len=3, bp_len=10, max_words=30, max_len=10, \
                 realdata_filename='real_data.json', datalist_filename='datalist',num_browsers=2, rho=0.5, \
                 diff_reward=False, cate_comb=None, user_url=None, obfuscation_url=None, \
                 user_url_index=None, obfuscation_url_index=None, P_MC=None, num_state=1):
        self.num_words = num_words
        self.embedding_dim = embedding_dim
        self.his_len = his_len
        self.bp_len = bp_len
        self.max_words = max_words
        self.use_simu = False
        self.max_len = max_len
        self.realdata_filename = realdata_filename
        self.datalist_filename = datalist_filename
        self.num_browsers = num_browsers
        self.rho = rho
        self.diff_reward = diff_reward
        self.cate_comb = cate_comb
        self.user_url = user_url
        self.user_url_index = user_url_index
        self.obfuscation_url = obfuscation_url
        self.obfuscation_url_index = obfuscation_url_index
        self.P_MC = P_MC
        self.num_state = num_state

class Env(object):
    def __init__(self, env_args, emb_model, simu, bidding_sites, ID, r_dim):
        self.args = env_args
        self.global_round = 0
        self.reward = [0 for _ in range(self.args.num_browsers)]
        self.avg_bids = []
        self.bids_data = []
        self.browsing_history = None
        self.browsing_history_base = None
        self.browsing_history_type = None
        self.browsing_history_cate = None
        self.emb_model = emb_model
        self.done = False
        self.url_seq = []
        self.use_simu = False
        self.simu = simu
        self.datalist_size = 0
        self.url_list = []
        self.url2vec_list = []
        self.url2bids_list = []
        self.bids_set = []
        self.url_seq_set = []
        self.bid_flag = 0
        self.root=os.getcwd() + "/docker-volume/crawls"
        self.crawl_type = "drl_" + ID
        self.bid_path = [os.path.join(self.root, self.crawl_type, \
            "browser_{}".format(int(index)), 'bid.json') for index in range(self.args.num_browsers)]
        self.r_dim = r_dim
        self.bidding_sites = [bidding_sites for _ in range(self.args.num_browsers)]

        self.reward_vec = [[0 for _ in range(r_dim)] for _ in range(self.args.num_browsers)]
        self.reward_vec_base = [[0 for _ in range(r_dim)] for _ in range(self.args.num_browsers)]

    def start_env(self, initial_profile, initial_profile_type, use_simu=False):
        self.global_round = 0
        self.reward = [0 for _ in range(self.args.num_browsers)]
        self.reward_bias = [0 for _ in range(self.args.num_browsers)]
        self.use_simu = use_simu
        self.browsing_history = [[] for j in range(self.args.num_browsers)]
        self.browsing_history_base = [[] for j in range(self.args.num_browsers)]
        self.browsing_history_type = initial_profile_type
        self.browsing_history_cate = [[] for j in range(self.args.num_browsers)]
        if (self.use_simu == False):
            print("Initialize docker container ... ")
            for index in range(len(initial_profile)):
                initial_profile_set = []
                for i in range(len(initial_profile[index])):
                    cur_url = initial_profile[index][i]
                    initial_profile_set.append(cur_url)
                    self.browsing_history[i].append(cur_url)
                    self.browsing_history_base[i].append(cur_url)
                    self.browsing_history_cate[i].append(-1)
            self.run_crawl_online(cur_url_set=[self.browsing_history[i] for i in range(self.args.num_browsers)], \
                                  run_b=False, load_p=False, save_p=True, collect_bids=False)
        else:
            for index in range(len(initial_profile)):
                for i in range(len(initial_profile[index])):
                    cur_url = initial_profile[index][i]
                    self.browsing_history[i].append(cur_url)
                    self.browsing_history_base[i].append(cur_url)
                    self.browsing_history_cate[i].append(-1)
        return self.get_cur_state()

    def run_crawl_online(self, cur_url_set, run_b, load_p, save_p, cur_url_click=None, collect_bids=True):
        reward_set = []
        avg_bids_set = []
        bids_data_set = []
        data_json = {}
        sites_cate = [self.crawl_type for _ in cur_url_set]
        if cur_url_click == None:
            if collect_bids == False:
                cur_url_click = [[0 for i in range(len(cur_url_set[0]))] for j in range(len(cur_url_set))]
            else:
                cur_url_click = [[0 for i in range(len(cur_url_set[0])+4)] for j in range(len(cur_url_set))]
        if collect_bids == False:
            cur_url_set = [cur_url_set[i] for i in range(len(cur_url_set))]
        else:
            cur_url_set = [cur_url_set[i] + self.bidding_sites[i] for i in range(len(cur_url_set))]
        url_seq_index = [i for i in range(len(cur_url_set))]
        run_docker(self.args.num_browsers, cur_url_set, sites_cate, cur_url_click, url_seq_index, run_b, load_p, save_p, dump_root="/home/user/Desktop/crawls")
        if run_b == True:
            for index in range(self.args.num_browsers):
                data_json = {}
                try:
                    with open(self.bid_path[index], "r") as fp:
                        data_json = json.load(fp)
                        fp.close()
                    reward_set.append(data_json["reward"])
                    avg_bids_set.append(data_json["avg_bids"])
                    bids_data_set.append(data_json["bids_data"])
                except:
                    reward_set.append(0)
                    avg_bids_set.append(0.0)
                    bids_data_set.append({})
        return avg_bids_set, avg_bids_set, bids_data_set

    def run_crawl_offline(self):
        self.reward, self.reward_vec, self.reward_vec_base = self.simu.predict_batch(
            self.browsing_history,\
            self.browsing_history_base, \
            self.reward_vec,\
            self.reward_vec_base
        )

    def step(self, cur_url, cur_url_type, cur_url_cate, crawling, collect_bids=True):
        if crawling == True:
            if self.use_simu == False:
                self.reward, self.avg_bids, self.bids_data = self.run_crawl_online(cur_url_set=self.browsing_history, \
                    run_b=True, load_p=True, save_p=True, collect_bids=collect_bids)
            else:
                self.run_crawl_offline()
            self.reward = [self.reward[i] + self.reward_bias[i] for i in range(len(self.reward))]
            self.global_round += 1
        else:
            self.browsing_history_type.append(cur_url_type)
            
            for index in range(len(cur_url)):
                if cur_url_type == 1:
                    # stealthiness penalty
                    if cur_url_cate[index] in self.browsing_history_cate[index]:
                        self.reward_bias[index] -= 0.00
                    # if cur_url_cate[index] in cur_url_cate:
                        # self.reward_bias[index] -= 0.01 * ((cur_url_cate == cur_url_cate[index]).sum() - 1)
            
                self.browsing_history[index].append(cur_url[index])
                self.browsing_history_cate[index].append(cur_url_cate[index])
                if cur_url_type == 0:
                    self.browsing_history_base[index].append(cur_url[index])
                    self.browsing_history_cate[index].append(-1)
            self.reward = [0 for _ in range(self.args.num_browsers)]
            
            if self.use_simu == True:
                self.run_crawl_offline()
            self.reward = [self.reward[i] + self.reward_bias[i] for i in range(len(self.reward))]
            # print(self.reward[0:10], self.reward_bias[0:10])
        return self.get_cur_state(), self.get_cur_reward(), self.reward_vec, self.reward_vec_base

    def get_cur_state(self, index=-1):
        if index >=0:
            return get_features(
                self.browsing_history[index][-self.args.his_len:],
                self.emb_model, 
                self.args
            )
        else:
            return [get_features(
                self.browsing_history[index][-self.args.his_len:],
                self.emb_model,
                self.args) for index in range(self.args.num_browsers)
            ]

    def get_cur_reward(self, index=-1):
        if index >= 0:
            return self.reward[index]
        else:
            return self.reward
