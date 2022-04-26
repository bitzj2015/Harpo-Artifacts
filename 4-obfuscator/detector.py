import json
import argparse
import random
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils import *
import h5py

random.seed(0)

model = Doc2Vec.load("../DATA/model/model_new_100.bin")
model = doc2vec_model(model)


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', help="/SSD/baseline/bids", type=str, default="/opt/src/data/baseline/bid")
args = parser.parse_args()
<<<<<<< HEAD

with open("../DATA/rl_result/attacker_persona_segs_Intent_0_0.1_M1.json", "r") as file:
    obfuscated_data = json.load(file)

=======
# with open("../DATA/rl_result/attacker_persona_segs_Intent_0_0.2_M2_new052_.json", "r") as file:
with open("../DATA/rl_result/attacker_persona_bids_Intent_0_0.1_new052_.json", "r") as file:
# with open("../DATA/rl_result/attacker_persona_segs_AdNauseam_1_0.25_M2_new052_.json", "r") as file:
    obfuscated_data = json.load(file)[-1000:]
    print(len(obfuscated_data))
# with open("../DATA/rl_result/attacker_persona_bids_Intent_1_0.0.json", "r") as file:
#     unobfuscated_data = json.load(file)
>>>>>>> 88cc585190d1f4cf605c726149752cb69753c058
with open("../DATA/rl_result/attacker_persona_segs_Intent_1_0.0_M1.json", "r") as file:
    unobfuscated_data = json.load(file)

# with open("../DATA/rl_result/attacker_persona_segs_AdNauseam_1_0.05_M2_new052_.json", "r") as file:
#     unobfuscated_data = json.load(file)

doc2vec_emb_size = 300
content = []
label = []
count = 0
for data in obfuscated_data:
    key = list(data.keys())[0]
    tmp = []
    count += 1
<<<<<<< HEAD
    if count <= 0:
        continue
    elif count > 500:
        break
    for url in data[key][20:50]:
        tmp.append(np.array(model.docvecs[url]))
    content.append(np.reshape(np.array(tmp),(1,20,doc2vec_emb_size)))
=======
    if count % 100 >= 10 * (count // 100 + 1) or count % 100 < 10 * (count // 100) or count <= 9000 * 0:
        continue
    # if count % 50 >= 10 * (count // 100 + 1) or count % 50 < 10 * (count // 100) :
    #     continue
    print(count, count // 100)
    for url in data[key]:
        tmp.append(np.array(model.docvecs[url]))
    content.append(np.reshape(np.array(tmp),(1,100,doc2vec_emb_size)))
>>>>>>> 88cc585190d1f4cf605c726149752cb69753c058
    label.append(np.array([1]))


count = 0
for data in unobfuscated_data:
    key = list(data.keys())[0]
    tmp = []
    for url in data[key][20:50]:
        tmp.append(np.array(model.docvecs[url]))
    content.append(np.reshape(np.array(tmp),(1,20,doc2vec_emb_size)))
    label.append(np.array([0]))
    count += 1
    if count == 100:
        break
rd_index = np.arange(len(content))
random.shuffle(rd_index)
rd_content = []
rd_label = []
for index in rd_index:
    rd_content.append(content[index])
    rd_label.append(label[index])

hf = h5py.File("./param/stealthiness.h5", 'w')
hf.create_dataset('input', data=rd_content[0:200])
hf.create_dataset('label', data=rd_label[0:200])
hf.close()
