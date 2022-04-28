import os
import sys
import json
import glob
import time
import numpy
import torch
import struct
import html2text
from gensim.models.doc2vec import Doc2Vec
from simu_segs_exp import SimuArgs
from agent import A3Clstm
from random import choice

hx_prev = torch.zeros(1, 256)
cx_prev = torch.zeros(1, 256)

simu_args = SimuArgs(max_words=30,
    max_len=20,
    batch_size=1,
    max_epoch=10,
    lr=0.003,   
    kernel_dim=150,
    embedding_dim=300,
    kernel_size=[3,4,5],
    output_size=2, dropout=0.0,
    use_cuda=False,
	simu_path='../simu',
	agent_path="../param/rl_agent.pkl",
    param_path="../param",
    action_dim=184,
    num_browsers=1,
    num_real_url=None,
    diff_reward=0,
    metric='extension'
)

model = A3Clstm(simu_args)
model.load_state_dict(torch.load(simu_args.agent_path))
with open("../param/product_url_100.json") as f:
    obfuscation_url_set = json.load(f)
    obfuscation_url_cats = list(obfuscation_url_set.keys())
buffer=[]

def latest_doc2vec():
    print("latest_doc2vec reached")
    files = glob.glob("../param/*.bin")
    if(len(files)==1):
        print("latest_doc2vec finished")
        return Doc2Vec.load("../param/model_new_100.bin")
    else:
        files.remove("model_new_100.bin")
        files.sort()
        print("latest_doc2vec finished")
        return Doc2Vec.load("../param/"+files[-1])

def harpo_api_old(history):
    print("harpo_api reached")
    disabled=load_disabled("../param/category_data.txt")
    doc2vec = latest_doc2vec()
    tensor = torch.tensor([doc2vec.infer_vector(each["html"].split(" ")) for each in history])
    tensor = torch.reshape(tensor,(1,1,20,300))
    # only need to call the model one time
    # hx and cx should passed to the model() call, from the previous call
    # initially hx and cx (global variables) should be set using torch.zeros(1, 256)

    for count in range(0,20):
        temp=tensor[:,:,count:count+1,0:256]
        temp=torch.reshape(temp, (1,256))
        res = model((tensor, (temp, temp)))

    # doc2vec.save("./{}.bin".format(time.time()))
    # idx = res[1].argmax().item() # gives the largest vector

    # capture hx and cx from the model() response and store them for the next call

    max_list=torch.topk(res[1],len(disabled)+1).indices.tolist()
    # len(disabled) + 1 in case all of the k selected are disabled categories
    print("harpo_api finished")
    return choose_max_on_constraint(max_list[0],disabled)

def harpo_api(history):
    print("harpo_api reached")
    disabled=load_disabled("../param/category_data.txt")
    doc2vec = latest_doc2vec()
    tensor = torch.tensor([doc2vec.infer_vector(each["html"].split(" ")) for each in history])
    tensor = torch.reshape(tensor,(1,1,20,300))
   
    global hx_prev
    global cx_prev

    # get model response
    res = model((tensor, (hx_prev, cx_prev)))

    # capture hx and cx from res and store them for the next call
    hx_prev = res[2]
    cx_prev = res[3]

    # get the top vectors (the number of disabled categories + 1)
    # in case all of the top vectors are disabled
    max_list=torch.topk(res[1],len(disabled)+1).indices.tolist()

    print("model returned, category selected")

    # should I use max_list or max_list[0]
    print(max_list)

    return choose_max_on_constraint(max_list[0], disabled)

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
    obfuscation_url=harpo_api(maintain_twenty())
    ret=[]
    ret.append(obfuscation_url)
    print("obfuscation_url finished")
    return {"obfuscation url": ret}

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

def load_disabled(file):
    print("load_disabled reached")
    storage=[]
    with open(file, "r") as json_file:
        pref=json.loads(json_file.read())
    for i in pref:
        if pref[i]["checked"]==False:
            storage.append(pref[i]["name"])
    print("load_disabled finished")
    return storage

def choose_max_on_constraint(max_list, disabled):
    print("choose_max_on_constraint reached")
    for i in max_list:
        cat = obfuscation_url_cats[i]
        if cat in disabled:
            pass
        else:
            # return a randomly chosen obfuscation url from the most relevant category (that is not disabled)
            return choice(list(obfuscation_url_set[cat].keys()))
    print("choose_max_on_constraint finished")
