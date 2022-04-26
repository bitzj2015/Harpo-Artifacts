import os
import sys
import json
import time
import uuid
import random
import gensim
import sqlite3
import argparse
import threading
import numpy as np
from copy import deepcopy
from lcdk import lcdk as LeslieChow
from agent import A3Clstm, Agent
from network import CNNClassifier
from env_ml import Env,EnvArgs
from simu_bids_exp import Simu,SimuArgs
from utils import WebDataset, ToTensor, sample_url, get_next_state, doc2vec_model
import torch
import torch.optim as optim
import csv
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import gensim.models as g
import logging
import h5py
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


#Debug
DBG = LeslieChow.lcdk(print_output=True)


parser = argparse.ArgumentParser()
parser.add_argument('--type', help="train_bp, test_bp, train_rl, test_rl", type=str)
parser.add_argument('--use_simu', help="whether running in the wild", type=int)
parser.add_argument('--keyword', help="intent/trackthis/ad", type=str)
parser.add_argument('--iid', help="whether running iid approach", type=int)
parser.add_argument('--seed', help="random seed", type=int)
parser.add_argument('--diff_reward', help="whether to use diff_reward", type=int, default=0)
parser.add_argument('--attacker_path', help="attacker path", type=str)
parser.add_argument('--test_ep', help="testing ep", type=int, default=5)
parser.add_argument('--alpha', help="alpha", type=float, default=0.1)
parser.add_argument('--model_path', help="/SSD/baseline/bids", type=str, default="/SSD/baseline/bids")

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
ALPHA = args.alpha
NUM_STATE = 4
NUM_PROCESS = 100
TAG = "_new052_"
# augment doc2vec with adnauseam and trackthis
model = Doc2Vec.load("../DATA/model/model_new_100.bin")
model = doc2vec_model(model)

with open("../content_extraction/generator/user_url_100.json", "r") as json_file:
    user_url = json.load(json_file)
with open("../content_extraction/generator/product_url_100.json", "r") as json_file:
    obfuscation_url = json.load(json_file)
with open("../DATA/aol/MC_4.json", "r") as json_file:
    cate_comb = json.load(json_file)
ACTION_DIM = len(obfuscation_url.keys())

user_url_index = {}
index = 0
for key in user_url.keys():
    if key == "product":
        index = 16
        user_url_index[index] = key
    else:
        index = cate_comb[key]
        user_url_index[index] = key

if args.keyword == "AdNauseam":
    obfuscation_url = model.adnauseam_url
elif args.keyword == "TrackThis":
    obfuscation_url = model.trackthis_url

obfuscation_url_index = {}
index = 0
for key in obfuscation_url.keys():
    obfuscation_url_index[index] = key
    index += 1
print(obfuscation_url_index[60])
print(obfuscation_url_index[62])
print(obfuscation_url_index[98])
print(obfuscation_url_index[115])
hf = h5py.File("../DATA/aol/MC_4_acv.h5", "r")
P1 = np.array(hf["P1"])
print(P1)

# define simulator parameters
simu_path = ["{}/{}".format(args.model_path, path) for path in os.listdir(args.model_path)]
print("[INFO] Surrogate models: {}".format(simu_path))

simu_args = SimuArgs(max_words=30,
                     max_len=20,
                     batch_size=NUM_PROCESS,
                     max_epoch=10,
                     lr=0.003,
                     kernel_dim=150,
                     embedding_dim=300,
                     kernel_size=[3,4,5],
                     output_size=2, 
                     dropout=0.0,
                     use_cuda=False,
                     simu_path=simu_path,
                     agent_path="./param/{}".format(args.attacker_path),
                     param_path="./param",
                     action_dim=ACTION_DIM,
                     num_browsers=NUM_PROCESS,
                     num_real_url=None,
                     diff_reward=args.diff_reward)

# define environment parameters
env_args = EnvArgs( num_words=30,
                    embedding_dim=300,
                    his_len=20,
                    max_words=30,
                    max_len=20,
                    realdata_filename="../DATA/dataset/dataset.mat",
                    datalist_filename="../DATA/dataset/datalist",
                    num_browsers=NUM_PROCESS,
                    rho=0.5,
                    diff_reward=args.diff_reward,
                    cate_comb=cate_comb,
                    user_url=user_url,
                    user_url_index=user_url_index,
                    obfuscation_url=obfuscation_url,
                    obfuscation_url_index=obfuscation_url_index,
                    P_MC=P1,
                    num_state=NUM_STATE)

# define bidspredictor network
if simu_args.use_cuda:
    BidsPredictor = CNNClassifier(simu_args).cuda()
else:
    BidsPredictor = CNNClassifier(simu_args)

# define simulator (bids predictor)
simu = Simu(BidsPredictor, simu_args, model, W=[0.1,0.4])
simu.load_model()
# define environment
bidding_sites = ["https://registry.bluekai.com/get_categories", "https://www.speedtest.net", "https://www.kompas.com", "http://www.cnn.com"]
main_env = Env(env_args, model, simu, bidding_sites, str(args.seed), len(simu_path))

# define attacker (url generator)
attacker = Agent(None, None, main_env, env_args, simu_args, None)
attacker.model = A3Clstm(simu_args)
if simu_args.use_cuda:
    attacker.model = attacker.model.cuda()
attacker.optimizer = optim.Adam(attacker.model.parameters(), lr=simu_args.lr)

def train_url_generator(
    attacker,
    env_args,
    simu_args,
    USE_SIMU=False,
    GAMMA=0.99,
    T=1,
    MAX_EP=5,
    MAX_STEP=20,
    RETRAIN=True
):
    # Generate initial profile randomly
    def initialize_profile():
        initial_profile = []
        initial_profile_state = []
        user_url_map_list = []
        for i in range(simu_args.num_browsers):
            index = random.randint(0,99)
            top3_cate_comb = env_args.cate_comb["*"][index].split("_")
            top3_cate_comb = [int(item) for item in top3_cate_comb]
            user_url_index_list = top3_cate_comb
            for j in range(len(env_args.user_url.keys())):
                if j not in top3_cate_comb:
                    user_url_index_list.append(j)

            user_url_map = {}
            for j in range(len(user_url_index_list)):
                if j <= env_args.num_state - 2:
                    user_url_map[j] = [top3_cate_comb[j]]
                elif j == env_args.num_state - 1:
                    user_url_map[env_args.num_state - 1] = [user_url_index_list[j]]
                else:
                    user_url_map[env_args.num_state - 1].append(user_url_index_list[j])
            user_url_map[env_args.num_state - 1].append(16)
            user_url_map_list.append(deepcopy(user_url_map))

        for l in range(simu_args.max_len):
            initial_profile.append([])
            initial_profile_state.append([])
            for i in range(simu_args.num_browsers):
                user_url_map = user_url_map_list[i]
                if l == 0:
                    cur_state = random.randint(0, env_args.num_state-2)
                    initial_profile_state[-1].append(cur_state)
                else:
                    prev_state = initial_profile_state[-2][i]
                    P1_t = env_args.P_MC[prev_state]
                    # cur_state = np.argmax(np.random.multinomial(1, P1_t, size=1))
                    cur_state = get_next_state(P1_t)
                    initial_profile_state[-1].append(cur_state)
                if cur_state == env_args.num_state - 1:
                    url = sample_url(
                        url_set=env_args.user_url, 
                        url_set_index=env_args.user_url_index, 
                        cate_list=user_url_map[cur_state]
                    )
                else:
                    url = sample_url(
                        url_set=env_args.user_url, 
                        url_set_index=env_args.user_url_index, 
                        cate_list=user_url_map[cur_state]
                    )
                initial_profile[-1].append(url)
        initial_profile_type = [0 for _ in range(simu_args.max_len)]
        return initial_profile, initial_profile_type, initial_profile_state, user_url_map_list
    
    random.seed(args.seed)
    if RETRAIN:
        if simu_args.use_cuda:
            attacker.model.load_state_dict(torch.load(simu_args.agent_path)) #,map_location=torch.device('cpu')))
        else:
            attacker.model.load_state_dict(torch.load(simu_args.agent_path, map_location=torch.device('cpu')))
    avg_reward = 0
    count = 0
    f_reward = 0
    avg_loss = 0
    reward = 0
    loss = 0
    state = 0
    count = 0
    max_r = 0
    r_list = []
    # Training loop, each epoch contains 20 + MAX_STEP urls, as one persona
    for ep in range(MAX_EP):
        # Initialize profile
        initial_profile, initial_profile_type, initial_profile_state, user_url_map_list = initialize_profile()
        state = attacker.env.start_env(initial_profile, initial_profile_type, use_simu=USE_SIMU)
        attacker.state = torch.from_numpy(np.stack(state, axis=0).reshape(simu_args.num_browsers,1,-1,simu_args.embedding_dim))
        if simu_args.use_cuda:
            attacker.state = attacker.state.cuda()
        for _ in range(MAX_STEP):
            # t = time.time()
            flag = random.random()
            if flag > ALPHA:
                initial_profile_state.append([])
                action_list = []
                action_id = []
                for i in range(simu_args.num_browsers):
                    user_url_map = user_url_map_list[i]
                    prev_state = initial_profile_state[-2][i]
                    P1_t = env_args.P_MC[prev_state]
                    # cur_state = np.argmax(np.random.multinomial(1, P1_t, size=1))
                    cur_state = get_next_state(P1_t)
                    initial_profile_state[-1].append(cur_state)
                    if cur_state == env_args.num_state - 1:
                        url = sample_url(
                            url_set=env_args.user_url, 
                            url_set_index=env_args.user_url_index, 
                            cate_list=user_url_map[cur_state]
                        )
                    else:
                        url = sample_url(
                            url_set=env_args.user_url, 
                            url_set_index=env_args.user_url_index, 
                            cate_list=user_url_map[cur_state]
                        )
                    action_list.append(url)
                    action_id.append(-1)
                state, reward, _, _ = attacker.env.step(action_list, cur_url_type=0, cur_url_cate=action_id, crawling=False)
                attacker.state = torch.from_numpy(np.stack(state, axis=0).reshape(attacker.simu_args.num_browsers,1,-1,attacker.simu_args.embedding_dim))
            else:
                attacker.action_train(Terminate=False)
            # print("step:", time.time() - t)
        attacker.action_train(Terminate=True)
        if len(attacker.rewards) > 0:
            loss, reward = attacker.update(GAMMA, T, retrain=RETRAIN)
            f_reward = reward * len(attacker.rewards)
        print(attacker.env.reward_vec[0:10], attacker.env.reward_vec_base[0:10])
        attacker.clear_actions()
        attacker.env.reward_vec = [[0 for _ in range(len(simu_path))] for _ in range(env_args.num_browsers)]
        attacker.env.reward_vec_base = [[0 for _ in range(len(simu_path))] for _ in range(env_args.num_browsers)]
        avg_reward += reward
        avg_loss += loss

        # if (ep + 1) % 10 == 0:
        #     DBG.lt_cyan("URL: {}".format( attacker.env.browsing_history))
        DBG.lt_green("Epoch: {}".format(ep))
        DBG.lt_green("Average reward: {}".format(avg_reward))
        DBG.lt_green("Final reward: {}".format(f_reward))
        DBG.lt_green("Loss: {}".format(avg_loss))
        r_list.append(avg_reward)
        # attacker.save_param("{}_{:0.4f}".format(ep, f_reward))
        if np.mean(r_list[-5:]) > max_r:
            attacker.save_param("stealth")
            max_r = np.mean(r_list[-5:])
        # attacker.save_param("{}".format("stealth"))
        csvFile = open("../DATA/rl_result/attacker_train_bids_{}{}.csv".format(args.alpha,TAG), 'a', newline='')
        writer = csv.writer(csvFile)
        writer.writerow([avg_reward, f_reward, avg_loss, attacker.env.url_seq])
        csvFile.close()
        avg_reward = 0
        avg_loss = 0
        count = 0


def test_url_generator( 
    attacker,
    env_args,
    simu_args,
    USE_SIMU=False,
    GAMMA=0.99,
    T=1,
    MAX_EP=5,
    MAX_STEP=20,
    RETRAIN=True
):
    # Generate initial profile randomly
    def initialize_profile():
        initial_profile = []
        initial_profile_state = []
        user_url_map_list = []
        for i in range(simu_args.num_browsers):
            index = random.randint(0,99)
            top3_cate_comb = env_args.cate_comb["*"][index].split("_")
            top3_cate_comb = [int(item) for item in top3_cate_comb]
            user_url_index_list = top3_cate_comb
            for j in range(len(env_args.user_url.keys())):
                if j not in top3_cate_comb:
                    user_url_index_list.append(j)

            user_url_map = {}
            for j in range(len(user_url_index_list)):
                if j <= env_args.num_state - 2:
                    user_url_map[j] = [top3_cate_comb[j]]
                elif j == env_args.num_state - 1:
                    user_url_map[env_args.num_state - 1] = [user_url_index_list[j]]
                else:
                    user_url_map[env_args.num_state - 1].append(user_url_index_list[j])
            user_url_map[env_args.num_state - 1].append(16)
            user_url_map_list.append(deepcopy(user_url_map))

        for l in range(simu_args.max_len):
            initial_profile.append([])
            initial_profile_state.append([])
            for i in range(simu_args.num_browsers):
                user_url_map = user_url_map_list[i]
                if l == 0:
                    cur_state = random.randint(0, env_args.num_state-2)
                    initial_profile_state[-1].append(cur_state)
                else:
                    prev_state = initial_profile_state[-2][i]
                    P1_t = env_args.P_MC[prev_state]
                    cur_state = get_next_state(P1_t)
                    # cur_state = np.argmax(np.random.multinomial(1, P1_t, size=1))
                    initial_profile_state[-1].append(cur_state)
                if cur_state == env_args.num_state - 1:
                    url = sample_url(
                        url_set=env_args.user_url, 
                        url_set_index=env_args.user_url_index, 
                        cate_list=user_url_map[cur_state]
                    )
                else:
                    url = sample_url(
                        url_set=env_args.user_url, 
                        url_set_index=env_args.user_url_index, 
                        cate_list=user_url_map[cur_state]
                    )
                initial_profile[-1].append(url)
        initial_profile_type = [0 for _ in range(simu_args.max_len)]
        return initial_profile, initial_profile_type, initial_profile_state, user_url_map_list
    
    random.seed(args.seed)

    if simu_args.use_cuda:
        attacker.model.load_state_dict(torch.load(simu_args.agent_path)) #,map_location=torch.device('cpu')))
    else:
        attacker.model.load_state_dict(torch.load(simu_args.agent_path, map_location=torch.device('cpu')))

    avg_reward = 0
    avg_loss = 0
    state = None
    reward = 0
    state = 0
    count = 0
    
    random_weights = [1e-2 for i in range(simu_args.action_dim)]
    random.seed(0)
    final_avg_reward = 0
    final_avg_M2 = 0
    # Add loggers to record runtime data
    csvFile = open("../DATA/rl_result/attacker_test_bids_{}_{}_{}{}.csv".format(args.keyword, args.iid, args.alpha, TAG), 'w', newline='')
    writer = csv.writer(csvFile)
    # with open("../DATA/rl_result/bias_weight_bids_{}_{}_{}.json".format(args.keyword, args.iid, args.alpha), "r") as file:
    #     random_weights_norm_ = np.array(json.load(file)["weight"])
    if simu_args.use_cuda:
        attacker.state = attacker.state.cuda()
    test_persona = []
    test_persona_base = []
    for ep in range(MAX_EP):
        if ep == MAX_EP-10:
            random.seed(args.seed)
        pre_reward = [0 for i in range(simu_args.num_browsers)]

        initial_profile, initial_profile_type, initial_profile_state, user_url_map_list = initialize_profile()
        state = attacker.env.start_env(initial_profile, initial_profile_type, use_simu=USE_SIMU)
        attacker.state = torch.from_numpy(np.stack(state, axis=0).reshape(simu_args.num_browsers,1,-1,simu_args.embedding_dim))
        # DBG.lt_cyan("Initial profile: {}".format(initial_profile))
        ep_persona = []
        for i in range(simu_args.num_browsers):
            persona_id = "user_{}_{}".format(ep,i)
            initial_persona = []
            for j in range(len(initial_profile)):
                initial_persona.append(initial_profile[j][i])
            test_persona.append({persona_id: deepcopy(initial_persona)})
            test_persona_base.append({persona_id: deepcopy(initial_persona)})
            ep_persona.append(persona_id)
        # print(ep_persona)

        for step in range(MAX_STEP):
            flag = random.random()
            if ALPHA == 0.05:
                flag = ((step + 1) % 20) + 0.05
            if flag > ALPHA:
                initial_profile_state.append([])
                action_list = []
                action_id = []
                for i in range(simu_args.num_browsers):
                    user_url_map = user_url_map_list[i]
                    prev_state = initial_profile_state[-2][i]
                    P1_t = env_args.P_MC[prev_state]
                    cur_state = get_next_state(P1_t)
                    # cur_state = np.argmax(np.random.multinomial(1, P1_t, size=1))
                    initial_profile_state[-1].append(cur_state)
                    if cur_state == env_args.num_state - 1:
                        url = sample_url(
                            url_set=env_args.user_url, 
                            url_set_index=env_args.user_url_index, 
                            cate_list=user_url_map[cur_state]
                        )
                    else:
                        url = sample_url(
                            url_set=env_args.user_url, 
                            url_set_index=env_args.user_url_index, 
                            cate_list=user_url_map[cur_state]
                        )
                    action_list.append(url)
                    action_id.append(-1)
                    persona_id = ep_persona[i]
                    test_persona[-simu_args.num_browsers+i][persona_id].append(url)
                    test_persona_base[-simu_args.num_browsers+i][persona_id].append(url)
                state, reward, _, _ = attacker.env.step(action_list, cur_url_type=0, cur_url_cate=action_id, crawling=False)
                # attacker.state = torch.from_numpy(np.stack(state, axis=0).reshape(attacker.simu_args.num_browsers,1,-1,attacker.simu_args.embedding_dim))
            else:
                count += 1
                if args.iid == 2:
                    action_list = []
                    action_id = []
                    random_weights_norm = [item / sum(random_weights) for item in random_weights]
                    for i in range(simu_args.num_browsers):
                        if ep < 90:
                            url_id = np.random.choice(range(0, simu_args.action_dim), 1)[0]
                        else:
                            url_id = list(np.random.multinomial(1, random_weights_norm)).index(1)
                        action_url = sample_url(
                            url_set=env_args.obfuscation_url, 
                            url_set_index=env_args.obfuscation_url_index, 
                            cate_list=[url_id]
                        )   
                        action_list.append(action_url)
                        action_id.append(url_id)
                        persona_id = ep_persona[i]
                        test_persona[-simu_args.num_browsers+i][persona_id].append(action_url)
                    state, reward, _, _ = attacker.env.step(action_list, cur_url_type=1, cur_url_cate=action_id, crawling=False)
                    if ep < 90:
                        diff_reward = [abs(reward[i] - pre_reward[i]) for i in range(len(reward))]
                        if sum(diff_reward) > 0:
                            diff_reward = [item / sum(diff_reward) for item in diff_reward]
                        for i in range(simu_args.num_browsers):
                            random_weights[action_id[i]] += diff_reward[i]
                        pre_reward = reward
                    print(action_id)
                    
                elif args.iid == 1:
                    action_list = []
                    action_id = []
                    for i in range(simu_args.num_browsers):
                        url_id = np.random.choice(range(0, simu_args.action_dim), 1)[0]
                        action_url = sample_url(
                            url_set=env_args.obfuscation_url, 
                            url_set_index=env_args.obfuscation_url_index, 
                            cate_list=[url_id]
                        )   
                        action_list.append(action_url)
                        action_id.append(url_id)
                        persona_id = ep_persona[i]
                        test_persona[-simu_args.num_browsers+i][persona_id].append(action_url)
                    state, reward, _, _ = attacker.env.step(action_list, cur_url_type=1, cur_url_cate=action_id, crawling=False)
                    
                else:
                    attacker.action_test(Terminate=False)
                    for i in range(simu_args.num_browsers):
                        persona_id = ep_persona[i]
                        test_persona[-simu_args.num_browsers+i][persona_id].append(attacker.env.browsing_history[i][-1])
                
        if args.iid == 0:
            avg_reward += np.mean(attacker.rewards[-1].numpy())
            attacker.action_test(Terminate=True)
            
            
        else:
            avg_reward += np.mean(reward)
            
        M2 = []
        bid_reward = [{"base_bid":0, "obfu_bid":0} for _ in range(len(attacker.env.reward_vec))]
        for l in range(len(attacker.env.reward_vec)):
            for k in range(len(attacker.env.reward_vec[l])):
                bid_reward[l]["base_bid"] += attacker.env.reward_vec_base[l][k]
                bid_reward[l]["obfu_bid"] += attacker.env.reward_vec[l][k]
            print(attacker.env.reward_vec[l], attacker.env.reward_vec_base[l])
            print(attacker.env.reward[l])
            M2.append((bid_reward[l]["obfu_bid"]-bid_reward[l]["base_bid"])/10)
        
        attacker.clear_actions()
        attacker.env.reward_vec = [[0 for _ in range(len(simu_path))] for _ in range(env_args.num_browsers)]
        attacker.env.reward_vec_base = [[0 for _ in range(len(simu_path))] for _ in range(env_args.num_browsers)]

        # if (ep + 1) % 10 == 0:
        #     DBG.lt_green("URL: {}".format(attacker.env.browsing_history))
        DBG.lt_green("Epoch: {}".format(ep))
        if args.diff_reward:
            DBG.lt_green("Average reward: {}".format(avg_reward))
        else:
            DBG.lt_green("Average reward: {}".format(avg_reward / count))
        DBG.lt_green("Final reward: {}".format(np.mean(M2)))
        writer.writerow([np.mean(M2), avg_reward, avg_loss, attacker.env.url_seq])
        final_avg_M2 += np.mean(M2)
        final_avg_reward += avg_reward
        avg_reward = 0
        avg_loss = 0
        count = 0


    writer.writerow([final_avg_M2/MAX_EP, final_avg_reward/MAX_EP])
    csvFile.close()
    with open("../DATA/rl_result/attacker_persona_bids_{}_{}_{}{}.json".format(args.keyword, args.iid, args.alpha, TAG), "w") as file:
        json.dump(test_persona, file)
    with open("../DATA/rl_result/attacker_base_persona_bids_{}_{}_{}{}.json".format(args.keyword, args.iid, args.alpha, TAG), "w") as file:
        json.dump(test_persona_base, file)
    if args.iid == 2:
        with open("../DATA/rl_result/bias_weight_bids_{}_{}_{}.json".format(args.keyword, args.iid, args.alpha), "w") as file:
            json.dump({"weight":list(random_weights_norm)}, file)

USE_SIMU = bool(args.use_simu)
print(USE_SIMU,args.use_simu)
GAMMA = 0.99
T = 1
MAX_EP = 3
MAX_STEP = 40
RETRAIN = False
st = time.time()

if args.type == "train_rl":
    train_url_generator(
        attacker,
        env_args,
        simu_args,
        USE_SIMU=USE_SIMU,
        GAMMA=0.99,
        T=1,
        MAX_EP=150,
        MAX_STEP=80,
        RETRAIN=RETRAIN
    )

elif args.type == "test_rl":
    for i in range(1):
        test_url_generator(
            attacker,
            env_args,
            simu_args,
            USE_SIMU=USE_SIMU,
            GAMMA=0.99,
            T=1,
            MAX_EP=args.test_ep,
            MAX_STEP=80,
            RETRAIN=True
        )
else:
    DBG.lt_cyan("No such function!")
DBG.lt_cyan("{}".format(time.time() - st))
