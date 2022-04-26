import re
import os
import csv
import nltk
import time
import torch
import random
import numpy as np
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim
from utils import *
import torch.nn.functional as F
from torch.autograd import Variable
from lcdk import lcdk as LeslieChow
from collections import Counter, OrderedDict
import ray

DBG = LeslieChow.lcdk(print_output=True)

flatten = lambda l: [item for sublist in l for item in sublist]
random.seed(1024)


class SimuArgs(object):

    """ class SimuArgs - Simulator Parameters

        [TODO]: FILL IN DESCRIPTION

        Attributes:
            max_words: [TODO]: FILL IN DESCRIPTION - default: 30
            max_len: [TODO]: FILL IN DESCRIPTION - default: 10
            batch_size: [TODO]: FILL IN DESCRIPTION - default:32
            max_epoch: Training epoch - default: 50
            lr:  Learning Rate
            kernel_dim:  The width of hte kern - default: 150
            embedding_dim: [TODO]: FILL IN DESCRIPTION - default: 300
            kernel_size:  The size of the kernel - default: [3, 4, 5]
            output_size:  Number of labels - default: 2
            dropout: [TODO]: FILL IN DESCRIPTION - default: 0.2
            use_cuda: CUDA GPU [TODO]: FILL IN DESCRIPTION - default: False
            simu_path:  Path to .pkl file
            agent_path: [TODO]: FILL IN DESCRIPTION - default: None
            param_path: [TODO]: FILL IN DESCRIPTION - default: "param"

            action_dim: [TODO]: FILL IN DESCRIPTION -default: None
            num_browsers:  Number of browsers to launch during crawling - default: 16
            num_real_url: [TODO]: FILL IN DESCRIPTION - default: None

        Typical usage example:

        simu_args = SimuArgs(max_words=30,
                            max_len=10,
                            batch_size=32,
                            max_epoch=50,
                            lr=args.lr,
                            kernel_dim=150,
                            embedding_dim=300,
                            kernel_size=[3,4,5],
                            output_size=2,
                            dropout=0.2,
                            use_cuda=False,
                            simu_path=args.param,
                            agent_path=None,
                            param_path="./param",
                            action_dim=None,
                            num_browsers=16,
                            num_real_url=None)

    """
    def __init__(self, 
                 max_words=30,
                 max_len=10,
                 batch_size=32,
                 max_epoch=50,
                 lr=0.001,
                 kernel_dim=150,
                 embedding_dim=300,
                 kernel_size=[3,4,5],
                 output_size=2,
                 dropout=0.2,
                 use_cuda=False,
                 simu_path=None,
                 agent_path=None,
                 param_path="param",
                 action_dim=None,
                 num_browsers=16,
                 num_real_url=None,
                 diff_reward=True):

        self.max_words = max_words
        self.max_len = max_len
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.lr = lr
        self.kernel_dim = kernel_dim
        self.embedding_dim = embedding_dim
        self.kernel_size = kernel_size
        self.output_size = output_size
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.simu_path = simu_path
        self.agent_path = agent_path
        self.param_path = param_path
        self.action_dim = action_dim
        self.num_browsers = num_browsers
        self.num_real_url = num_real_url
        self.diff_reward = diff_reward


class Simu(object):
    """ class Simu - Simulator

    Args:
        BidsPredictor: [TODO]: FILL IN DESCRIPTION - default: 30
        simu_args: [TODO]: FILL IN DESCRIPTION - default: 10
        batch_size: [TODO]: FILL IN DESCRIPTION - default:32
        max_epoch: Training epoch - default: 50
        lr:  Learning Rate
        kernel_dim: [TODO]: FILL IN DESCRIPTION - default: 150

    Typical usage example:

        simu =  Simu(self,
                     BidsPredictor,
                     simu_args,
                     word2vec_model,
                     W):

    """
    def __init__(self, BidsPredictor=30, simu_args=SimuArgs(), doc2vec_model=None, W=None, seg=None):
        self.model = BidsPredictor
        self.args = simu_args
        if self.args.use_cuda:
            self.model = self.model.cuda()
            self.loss_function = nn.CrossEntropyLoss(weight=torch.Tensor(W).cuda())
        else:
            self.loss_function = nn.CrossEntropyLoss(weight=torch.Tensor(W))
        print(torch.Tensor(W))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.loss_list = []
        self.doc2vec_model = doc2vec_model
        self.seg = seg
        try:
            os.makedirs(self.args.param_path)
        except:
            pass

    def save_param(self):

        """ save_param - Save torch parameters

            Saves the model state and simu path

        Args: nothing

            Returns: nothing

            Typical usage example:
                simu.save_param

        """
        torch.save(self.model.state_dict(),
                   self.args.simu_path)


    def train(self, train_loader, test_loader, retrain):
        """ train - Simulator Training

        Args:
            train_loader: [TODO]: FILL IN DESCRIPTION
            test_loader: [TODO]: FILL IN DESCRIPTION
            retrain: [TODO]: FILL IN DESCRIPTION

        Returns: nothing

        Typical usage example:
            simu.train(self, train_loader, test_loader, retrain)

        """

        DBG.lt_green("Training bids predictor.")
        if retrain == True:
            self.model.load_state_dict(torch.load(self.args.simu_path))
            DBG.lt_green("Load model parameters.")
        else:
            DBG.lt_green("Train from scratch.")
        csvFile = open(self.args.param_path + "/loss_new_"+self.seg+".csv", 'w', newline='')
        writer = csv.writer(csvFile)
        writer.writerow(["epoch", "loss", "train_acc", "test_acc"])
        train_acc = 0
        test_acc = 0
        lr = self.args.lr
        best_acc = 0
        self.model.train()
        for _ in range(1):
            self.model.reset_param()
            for epoch in range(self.args.max_epoch):
                DBG.lt_green("Start epoch: {}".format(epoch))
                self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
                # begin simulator training round
                losses = []
                count = 0
                accuracy = 0
                acc0 = 0
                acc1 = 0
                acc2 = 0
                c0 = 1e-6
                c1 = 1e-6
                c2 = 1e-6
                for i, batch in enumerate(train_loader):
                    x, u = batch['x'], batch['u'].squeeze()
                    if x.shape[0] != self.args.batch_size:
                        break
                    if self.args.use_cuda:
                        x = x.cuda()
                        u = u.cuda()
                    self.optimizer.zero_grad()
                    y = self.model(x)
                    _, upred = torch.max(y.data,1)
                    accuracy += (upred==u).sum().item() / self.args.batch_size
                    for j in range(len(u)):
                        if (upred[j] == u[j] and u[j] == 0):
                            acc0 += 1
                        if u[j] == 0:
                            c0 += 1
                        if (upred[j] == u[j] and u[j] == 1):
                            acc1 += 1
                        if u[j] == 1:
                            c1 += 1
                    count += 1
                    l1_reg = 0
                    l2_reg = 0
                    for W in self.model.parameters():
                        l2_reg += W.norm(2)
                        l1_reg += W.norm(1)
                    loss = self.loss_function(y, u) + 1e-3 * l2_reg + 1e-3 * l1_reg
                    losses.append(loss.data.tolist())
                    loss.backward()
                    self.optimizer.step()
    #                 for param in self.model.parameters():
    #                     print(param.name, param.grad)
                    if i % 10 == 0:
                        DBG.lt_cyan("[OUTS] ep: {}, step: {}, mean_loss: {:0.4f}, mean_acc: {:0.4f} ({:0.4f} {:0.4f})".format(epoch,
                                                                                                                        i,
                                                                                                                        np.mean(losses),
                                                                                                                        accuracy / count,
                                                                                                                        acc0 / c0,
                                                                                                                        acc1 / c1))
                        self.loss_list.append(np.mean(losses))
                DBG.lt_green("Saving training results.")

                train_acc = accuracy / count
                test_acc, test_acc_0, test_acc_1 = self.test(test_loader)
                DBG.lt_green("({}/{}) train_acc: {:0.4f} ({:0.4f} , {:0.4f} ), test_acc: {:0.4f}".format(epoch,
                                                                                                        self.args.max_epoch,
                                                                                                        train_acc,
                                                                                                        acc0 / c0,
                                                                                                        acc1 / c1,
                                                                                                        test_acc))
                writer.writerow([epoch, np.mean(losses), train_acc, test_acc, acc0 / c0, acc1 / c1, test_acc_0, test_acc_1])
                print((test_acc_0+test_acc_1) / 2)
                if (test_acc_0+test_acc_1) / 2 > best_acc:
                    best_acc = (test_acc_0+test_acc_1) / 2
                    self.save_param()
                DBG.lt_green("End epoch: {}.".format(epoch))
                # if (epoch + 1) % 10 == 0:
                #     lr = lr * 0.9
        csvFile.close()
        
        test_acc,_,_ = self.test(test_loader, load_param=True)
        DBG.lt_green("(Final) test_acc: {:0.4f} ".format(test_acc))


    def test(self, test_loader, load_param=False):

        """ test - Simulator Testing

        [TODO]: FILL IN DESCRIPTION

        Args:

            test_loader: [TODO]: FILL IN DESCRIPTION

        Returns: accuracy/count

        Typical usage example:
            simu.train(self, train_loader, test_loader, retrain)

        """
        DBG.lt_green("Testing bids predictor ......")
        accuracy = 0
        acc0 = 0
        acc1 = 0
        acc2 = 0
        count = 0
        c0 = 1e-6
        c1 = 1e-6
        c2 = 1e-6
        # load model parameters
        DBG.lt_green("Load model parameters.")
        if load_param:
            try:
                self.model.load_state_dict(torch.load(self.args.simu_path))
            except:
                DBG.error("Load model parameters failed!")
        # begin simulator testing round
        self.model.eval()
        for i, batch in enumerate(test_loader):
            x, u = batch['x'], batch['u'].reshape(-1)
            # if x.shape[0] != self.args.batch_size:
            #     break
            if self.args.use_cuda:
                x = x.cuda()
                u = u.cuda()
            y = self.model(x, is_training=False)
            _, upred = torch.max(y.data,1)
            accuracy += (upred==u).sum().item() # / self.args.batch_size

            for j in range(len(u)):
                if (upred[j] == u[j] and u[j] == 0):
                    acc0 += 1
                if u[j] == 0:
                    c0 += 1
                if (upred[j] == u[j] and u[j] == 1):
                    acc1 += 1
                if u[j] == 1:
                    c1 += 1
            count += x.shape[0]
        DBG.lt_green("Simulator testing accuracy: {:0.4f}--- ({:0.4f} {:0.4f} .".format(accuracy / count,
                                                                                        acc0/c0,
                                                                                        acc1/c1))
        return accuracy/count, acc0/c0, acc1/c1

    def load_model(self):
        self.models = {}
        for path in self.args.simu_path:
            try:
                self.model.load_state_dict(torch.load(path))
                self.models[path] = deepcopy(self.model.eval())
            except:
                DBG.error("Load model parameters failed!")
        # self.infer = [infer.remote(self.models[path]) for path in self.args.simu_path]
        

    def predict_batch(self, url_seqs, base_url_seqs, pre_reward_vec, pre_reward_vec_base):
        """ predict - Simulator Predictor

        [TODO]: FILL IN DESCRIPTION

        Args:

            test_loader: [TODO]: FILL IN DESCRIPTION

        Returns:
            upred - predicted label - [TODO]: FILL IN DESCRIPTION

        Typical usage example:
            simu.train(self,
                    train_loader,
                    test_loader,
                    retrain)
        """
        decay = 1 / len(self.args.simu_path)
        # get urls features and input them into the simulator
        # t = time.time()
        batch_size = len(url_seqs)       
        x_batch_base = [get_features(base_url_seqs[i][-self.args.max_len:], self.doc2vec_model, self.args) for i in range(batch_size)]
        x_batch_base = np.stack(x_batch_base, axis=0)
        if self.args.use_cuda:
            x_batch_base = torch.Tensor(x_batch_base).reshape(batch_size,1,-1,self.args.embedding_dim).cuda()
        else:
            x_batch_base = torch.Tensor(x_batch_base).reshape(batch_size,1,-1,self.args.embedding_dim)
        
        x_batch = [get_features(url_seqs[i][-self.args.max_len:], self.doc2vec_model, self.args) for i in range(batch_size)]
        x_batch = np.stack(x_batch, axis=0)
        if self.args.use_cuda:
            x_batch = torch.Tensor(x_batch).reshape(batch_size,1,-1,self.args.embedding_dim).cuda()
        else:
            x_batch = torch.Tensor(x_batch).reshape(batch_size,1,-1,self.args.embedding_dim)

        # load model parameters
        # print("load data:", time.time() - t)
        pred_vec_base = [get_inference(x_batch_base, self.models[path]) for path in self.args.simu_path]
        # print("run infer:", time.time() - t)
        # x_batch_base_id = ray.put(x_batch_base)
        # pred_vec_base = [c.run.remote(x_batch_base_id) for c in self.infer]
        # pred_vec_base = ray.get(pred_vec_base)
        # ray.get(self.infer[0].run.remote(x_batch_base_id))
        pred_vec = [get_inference(x_batch, self.models[path]) for path in self.args.simu_path]
        cur_reward_vec = [
            [int(int(pred_vec[j][i]) | int(pre_reward_vec[i][j])) for j in range(len(pre_reward_vec[0]))] \
            for i in range(batch_size)
        ]
        cur_reward_vec_base = [
            [int(int(pred_vec_base[j][i]) | int(pre_reward_vec_base[i][j])) for j in range(len(pre_reward_vec_base[0]))] \
            for i in range(batch_size)
        ]
        if self.args.diff_reward:
            reward = [(sum(cur_reward_vec[i]) - sum(cur_reward_vec_base[i])) * decay for i in range(batch_size)]
        else: 
            reward = [sum([pred_vec[j][i] - pred_vec_base[j][i] for j in range(len(pred_vec))]) * decay for i in range(batch_size)]
        return reward, cur_reward_vec, cur_reward_vec_base
