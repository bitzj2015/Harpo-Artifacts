import json
import copy
import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
#import ray
import time
import random
import pickle

class doc2vec_model(object):
    def __init__(self, model):
        self.docvecs = {}
#        self.adnauseam_url = {}
#        self.trackthis_url = {}
        for url in model.docvecs.index2entity:
            self.docvecs[url] = model.docvecs[url]


def get_next_state(P):
    flag = random.random()
    th = 0
    for i in range(len(P)):
        if flag <= th+P[i]:
            return i
        else:
            th += P[i]
            continue


def sample_url(url_set, url_set_index, cate_list):
    num_cate = len(cate_list)
    # sample_cate = cate_list[
    #     np.argmax(
    #         np.random.multinomial(1, 
    #                               [1/num_cate]*num_cate, 
    #                               size=1)
    #     )
    # ]
    sample_cate = cate_list[get_next_state([1/num_cate]*num_cate)]
    sample_cate = url_set_index[sample_cate]
    url_list = list(url_set[sample_cate].keys())
    url = url_list[get_next_state([1/len(url_list)]*len(url_list))]
    # url = url_list[
    #     np.argmax(
    #         np.random.multinomial(1,
    #                               [1/len(url_list)]*len(url_list),
    #                               size=1)
    #     )
    # ]
    return url

# @ray.remote
def get_inference(inputs, model):
    model = model.eval()
    y = model(inputs, is_training=False)
    _, upred = torch.max(y.data, 1)
    return upred.int().tolist()

@ray.remote
class infer(object):
    def __init__(self, model):
        self.model = model.eval()

    def run(self, inputs):
        y = self.model(inputs, is_training=False)
        _, upred = torch.max(y.data, 1)
        return upred.int().tolist()

def get_features(url_seq, doc2vec_model, args):
    """ get_features

    Args:
        simu: Simulator Object - default: simu
        env_args: Envioronmental arguments object - default: env_args
        TRAIN_SPLIT:  [TODO]: FILL IN DESCRIPTION - default: 0.7
        BATCH_SIZE:  [TODO]: FILL IN DESCRIPTION - default: 32
        RETRAIN:  [TODO]: FILL IN DESCRIPTION - default: True

    Returns:  [TODO]: FILL IN DESCRIPTION
        np.stack(cur_url2vec_seq, axis=0)

    Typical usage example:

        import utils

        vector = get_features(url_seq,
                     url_type,
                     url2content_dict,
                     word2vec_model,
                     args)


    """
    content = [doc2vec_model.docvecs[url_seq[i]] for i in range(len(url_seq))]
    content = np.reshape(np.array(content),(1,len(url_seq),300))
    return content

class WebDataset(Dataset):
    """ class WebDataset

    Atributes:
        input_data: [TODO]: FILL IN DESCRIPTION
        label_data: [TODO]: FILL IN DESCRIPTION
        threshold: [TODO]: FILL IN DESCRIPTION - default: 2.0
        transform: [TODO]: FILL IN DESCRIPTION - default: None

    Typical usage example:

    import utils

    WebDataset( input_data,
                label_data,
                threshold=2.0,
                transform=None)

    """
    def __init__(self,
                input_data,
                label_data,
                threshold=2.0,
                transform=None):
        self.input = input_data
        self.label = label_data
        tmp1 = 0
        tmp2 = 0
        for i in range(np.shape(self.label)[0]):
            if self.label[i][0] <= threshold:
                self.label[i][0] = 0
                tmp1 += 1
            else:
                self.label[i][0] = 1
                tmp2 += 1
        self.transform = transform
        # self.input = (self.input - self.input.mean(0)) / np.sqrt(self.input.var(0))
        print("low:", tmp1, "; high:", tmp2)
    def __len__(self):
        return np.shape(self.label)[0]

    def __getitem__(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()
        data = self.input[idx]
#         print(np.shape(data))
#         d1,d2,d3=np.shape(data)
        data = data.astype('float32')#.reshape(-1,d1,d2,d3)
        user = self.label[idx]
        user = user.astype('int').reshape(-1,1)
        sample = {'x':data, 'u':user}
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    """ class ToTensor

    Atributes:
        sample: [TODO]: FILL IN DESCRIPTION
    Returns:
        {'x':torch.from_numpy(data), 'u':torch.from_numpy(user)}

    Typical usage example:
    import utils
    data = ToTensor(sample)
    """
    def __call__(self, sample):
        data, user = sample['x'], sample['u']
        return {'x':torch.from_numpy(data), 'u':torch.from_numpy(user)}
