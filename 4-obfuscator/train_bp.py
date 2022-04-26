import os
import sys
import h5py
import torch
import csv
import json
import time
import uuid
import random
random.seed(0)
import gensim
import sqlite3
import logging
import argparse
import threading
import numpy as np
np.random.seed(0)
import scipy.io as sio
from copy import deepcopy
import torch.optim as optim
from termcolor import colored
from env_ml import Env,EnvArgs
from optimizer import SharedAdam
from network import CNNClassifier
from lcdk import lcdk as LeslieChow
from simu_bids_exp import Simu,SimuArgs
from utils import WebDataset, ToTensor
from torch.multiprocessing import Process
from torch.utils.data import Dataset, DataLoader
# torch.manual_seed(1)
# Template
# ----------------
#
# References: [1] google.github.io/styleguide/pyguide.html#382-modules
#             [2] https://www.python.org/dev/peps/pep-0257/
# """A one line summary of the module or program, terminated by a period.
#
# Leave one blank line.  The rest of this docstring should contain an
# overall description of the module or program.  Optionally, it may also
# contain a brief description of exported classes and functions and/or usage
# examples.
#
#  Typical usage example:
#
#  foo = ClassFoo()
#  bar = foo.FunctionBar()
#
#"""

#start time
st = time.time()

#Debug
DBG = LeslieChow.lcdk(print_output=False)


parser = argparse.ArgumentParser()
parser.add_argument('--type', help="train_bp, test_bp", type=str)
parser.add_argument('--data', help="name of dataset for training bp, under -- (Dataset-f.h5)", type=str)
parser.add_argument('--param', help="param/simulator-alpha.pkl", type=str)
parser.add_argument('--retrain', help="whether to retrain", type=int)
parser.add_argument('--lr', help="learning rate", type=float)
parser.add_argument('--th', help="threshold", type=float)
parser.add_argument('--ratio', help="threshold", type=float)
parser.add_argument('--seg', help="segment", type=str)
parser.add_argument('--ep', help="# of epochs", type=int, default=30)

args = parser.parse_args()
BATCH_SIZE=20

# define simulator parameters
args.lr = 0.01
simu_args = SimuArgs(max_words=30,
                    max_len=20,
                    batch_size=BATCH_SIZE,
                    max_epoch=args.ep,
                    lr=args.lr,
                    kernel_dim=150,
                    embedding_dim=300,
                    kernel_size=[3,4,5],
                    output_size=2, 
                    dropout=0.5,
                    use_cuda=False,
                    simu_path=args.param,
                    agent_path=None,
                    param_path="../DATA/csv",
                    action_dim=None,
                    num_browsers=16,
                    num_real_url=None)


# define environment parameters
env_args = EnvArgs( num_words=30,
                    embedding_dim=300,
                    his_len=3,
                    max_words=100,
                    max_len=20,
                    realdata_filename=args.data,
                    datalist_filename=None,
                    num_browsers=16,
                    rho=0.5)

# define bids predictor network
if simu_args.use_cuda:
    BidsPredictor = CNNClassifier(simu_args).cuda()
else:
    BidsPredictor = CNNClassifier(simu_args)

# define simulator (bids predictor)
simu = Simu(BidsPredictor, simu_args, None, W=[0.1,args.ratio/10], seg=args.seg)



def train_bids_predictor(simu,
                         env_args,
                         TRAIN_SPLIT=0.7,
                         BATCH_SIZE=32,
                         RETRAIN=True):

    """ train_bids_predictor

        Train Bids Predictor

        Args:
            simu: Simulator Object
            env_args: Envioronmental arguments object
            TRAIN_SPLIT: [TODO]: FILL IN DESCRIPTION - default: 0.7
            BATCH_SIZE: [TODO]: FILL IN DESCRIPTION - default: 32
            RETRAIN: default: True


        Returns:
            nothing

        Typical usage example:
        train_bp = train_bids_predictor(simu,
                                        env_args,
                                        TRAIN_SPLIT=0.7,
                                        BATCH_SIZE=32,
                                        RETRAIN=True)


    """
    train_split = TRAIN_SPLIT
    batch_size = BATCH_SIZE
    retrain = RETRAIN
    data_path = env_args.realdata_filename
#     dataset = sio.loadmat(data_path)
    dataset = h5py.File(data_path, 'r')
    dataset_size = len(dataset["input"])
    train_size = int(train_split*dataset_size)
    test_size = dataset_size-train_size
    train_dataset = WebDataset(dataset["input"][:train_size],dataset["label"][:train_size],threshold=args.th,transform=ToTensor())
    test_dataset = WebDataset(dataset["input"][train_size:dataset_size],dataset["label"][train_size:dataset_size],threshold=args.th,transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    DBG.lt_cyan("training size: {}".format(train_size))
    simu.train(train_loader, test_loader, retrain)


def test_bids_predictor(simu,
                        env_args,
                        TRAIN_SPLIT=0.7,
                        BATCH_SIZE=32,
                        RETRAIN=True):
    """ Test Bids Predictor

    Args:
        simu (obj): Simulator Object
        env_args (dtype): Envioronmental arguments object
        TRAIN_SPLIT (float) : [TODO]: FILL IN DESCRIPTION - default: 0.7
        BATCH_SIZE (int): [TODO]: FILL IN DESCRIPTION - default: 32
        RETRAIN (bool): default: True

    Returns:

    Typical Usage:

        test_bids_predictor(simu,
                            env_args,
                            TRAIN_SPLIT=0.7,
                            BATCH_SIZE=32,
                            RETRAIN=True):


    """
    train_split = TRAIN_SPLIT
    batch_size = BATCH_SIZE
    retrain = RETRAIN
    data_path = env_args.realdata_filename
    dataset = h5py.File(data_path, 'r')
    dataset_size = len(dataset["input"])
    train_size = int(train_split*dataset_size)
    test_size = dataset_size-train_size
    train_dataset = WebDataset(dataset["input"][:train_size],dataset["label"][:train_size],threshold=args.th,transform=ToTensor())
    test_dataset = WebDataset(dataset["input"][train_size:dataset_size],dataset["label"][train_size:dataset_size],threshold=args.th,transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(train_size)
    simu.test(test_loader,load_param=True)

if __name__ == "__main__":
    """
    Command line arguments passed in to bid predictor

    Args:
        type: Designates the type of training peformed
              (str) train_bp - train bid predictor
                                - or -
                          test_bp  - test bid predictor
        data:path to .h5 training file - default: Dataset-f.h5
        param:      Path of .pkl file. default: param/simulator-alpha.pkl
        retrain:Indicates whether to retrain. default: 0 (don't retrain), 1  (retrain)
        lr:Designates the learning rate.range: 0.001-0.0001
        th:Designates the threshhold for

    Returns:

    Typical usage example:
    python train_bp.py --type train_bp
                       --data ../Dataset/Dataset-f.h5
                       --param ../Dataset/simulator-alpha.pkl
                       --retain 0
                       --lr 0.001
                       --th

    """
    if args.retrain == 0:
        RETRAIN = False
    else:
        RETRAIN = True
    if args.type == "train_bp":
        train_bids_predictor(simu, env_args, TRAIN_SPLIT=0.8, BATCH_SIZE=BATCH_SIZE, RETRAIN=RETRAIN)
    elif args.type == "test_bp":
        test_bids_predictor(simu, env_args, TRAIN_SPLIT=0.8, BATCH_SIZE=BATCH_SIZE, RETRAIN=RETRAIN)
    else:
        print("No such function!")
    print(time.time() - st)
