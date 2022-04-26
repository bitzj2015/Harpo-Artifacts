import os
import sys
import glob
import time
import uuid
import json
import numpy
import torch
import torch.optim as optim
import struct
import html2text
from collections import OrderedDict
from gensim.models.doc2vec import Doc2Vec
from simu_segs_exp import SimuArgs
from agent import A3Clstm, Agent

# constants
T = 1
GAMMA = 0.99

# number of interest segment (url, reward) tuples to collect before updating model
NUM_REWARDS = 3

simu_args = SimuArgs(
    max_words=30,
    max_len=20,
    batch_size=1,
    max_epoch=10,
    lr=0.003,   
    kernel_dim=150,
    embedding_dim=300,
    kernel_size=[3,4,5],
    output_size=2, dropout=0.0,
    use_cuda=False,
	simu_path='./simu',
	agent_path="./rl_agent.pkl",
    param_path="./param",
    action_dim=184,
    num_browsers=1,
    num_real_url=None,
    diff_reward=0,
    metric='extension'
)

# initalize model

model = A3Clstm(simu_args)

# what does this line do?
model.load_state_dict(torch.load(simu_args.agent_path))
model.train()
# initalize training pipeline
optimizer = optim.Adam(model.parameters(), lr=simu_args.lr)
agent = Agent(model, optimizer, simu_args)

buffer=[]

# interest segment history related
prev_interest_segments = []
model_rewards = OrderedDict()

def latest_doc2vec():
    print("latest_doc2vec reached")
    files = glob.glob("*.bin")
    if(len(files)==1):
        print("latest_doc2vec finished")
        return Doc2Vec.load("./model_new_100.bin")
    else:
        files.remove("model_new_100.bin")
        files.sort()
        print("latest_doc2vec finished")
        return Doc2Vec.load("./"+files[-1])

# def harpo_api_old(history):
#     print("harpo_api reached")
#     disabled=load_disabled("./category_data.txt")
#     doc2vec = latest_doc2vec()
#     tensor = torch.tensor([doc2vec.infer_vector(each["html"].split(" ")) for each in history])
#     tensor = torch.reshape(tensor,(1,1,20,300))
#     # only need to call the model one time
#     # hx and cx should passed to the model() call, from the previous call
#     # initially hx and cx (global variables) should be set using torch.zeros(1, 256)

#     for count in range(0,20):
#         temp=tensor[:,:,count:count+1,0:256]
#         temp=torch.reshape(temp, (1,256))
#         res = model((tensor, (temp, temp)))

#     # doc2vec.save("./{}.bin".format(time.time()))
#     # idx = res[1].argmax().item() # gives the largest vector

#     # capture hx and cx from the model() response and store them for the next call

#     max_list=torch.topk(res[1],len(disabled)+1).indices.tolist()
#     # len(disabled) + 1 in case all of the k selected are disabled categories
#     print("harpo_api finished")
#     return choose_max_on_constraint(max_list[0],disabled)

def harpo_api(history, Terminate=False):
    print("harpo_api reached")
    doc2vec = latest_doc2vec()
    tensor = torch.tensor([doc2vec.infer_vector(each["html"].split(" ")) for each in history])
    tensor = torch.reshape(tensor,(1,1,20,300))
    
    # get model response + train model
    if not Terminate:
        model_resp = agent.action_train(tensor)
        return model_resp
    else:
        agent.action_train(tensor, Terminate=True)

def maintain_twenty():
    print("maintain_twenty reached")
    storage=[]
    list_of_files = glob.glob("../history/*.json")
    if(len(list_of_files)<20):
        latest_file = max(list_of_files, key=os.path.getctime)
        while(len(storage)<20):
            with open(latest_file, "r") as json_file:
                storage.append(json.loads(json_file.read()))
    else:
        while(len(storage)<20):
            latest_file = max(list_of_files, key=os.path.getctime)
            with open(latest_file, "r") as json_file:
                storage.append(json.loads(json_file.read()))
            list_of_files.remove(latest_file)
    print("maintain_twenty finished")
    return storage

def obfuscation_url():
    print("obfuscation_url reached")
    history = maintain_twenty()
    url_cat, obfuscation_url=harpo_api(history, False)
    ret=[]
    ret.append(obfuscation_url)
    url_uuid = uuid.uuid4().hex
    model_rewards[url_uuid] = None
    print("obfuscation_url finished")
    return url_uuid, url_cat, ret

def save_html2text(data):
    print("save_html reached")
    text = html2text.html2text(data['html'])
    page={
        "url":data["url"],
        "html":text
    }
    with open("../history/{}.json".format(time.time()), "w") as json_file:
        json.dump(page, json_file)
    print("save_html finished")

def calc_reward(new_interest_segments):
    reward = 0

    # find interest segments that are new
    for segment in new_interest_segments:
        if segment not in prev_interest_segments:
            reward += 1
    
    # find interest segments that were removed
    for segment in prev_interest_segments:
        if segment not in new_interest_segments:
            reward += 1

    return reward

def maintain_int_seg_history(data):
    uuid = data['uuid']
    interest_segments = data['data']

    reward = calc_reward(interest_segments)
    prev_interest_segments = interest_segments

    model_rewards[uuid] = reward

    if len(model_rewards) >= NUM_REWARDS:
        top_rewards_comp = True

        index = 0

        for uuid, reward in model_rewards.items():
            if index >= NUM_REWARDS:
                break

            if reward == None:
                top_rewards_comp = False
                break
            else:
                pass

            index += 1

        if top_rewards_comp:
            rewards = [reward for uuid, reward in model_rewards]

            history = maintain_twenty()
            harpo_api(history, True)

            agent.update(rewards, GAMMA, T)
            agent.clear_actions()

            index = 0

            for uuid, reward in model_rewards.items():
                if index >= NUM_REWARDS:
                    break

                model_rewards.pop(uuid)

                index += 1

    return {'status': 'interest segment update recorded successfully'}

# code below is only for testing purposes

def maintain_int_seg_history_test(uuid, reward):
    model_rewards[uuid] = reward

    if len(model_rewards) >= NUM_REWARDS:
        print("Threshold reached. Retraining model!")

        top_rewards_comp = True

        index = 0

        for uuid, reward in model_rewards.items():
            if index >= NUM_REWARDS:
                break

            if reward == None:
                top_rewards_comp = False
                break
            else:
                pass

            index += 1

        if top_rewards_comp:
            print("GOT TO RETRAINING STEP")
            rewards = [reward for uuid, reward in model_rewards.items()]

            history = maintain_twenty()
            harpo_api(history, True)

            agent.update(rewards, GAMMA, T)
            agent.clear_actions()

            index = 0

            for uuid, reward in model_rewards.items():
                if index >= NUM_REWARDS:
                    break

                model_rewards.pop(uuid)

                index += 1

    return {'status': 'interest segment update recorded successfully'}