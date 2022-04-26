import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from obf_url_utils import load_disabled, choose_max_on_constraint

torch.autograd.set_detect_anomaly(True)

BIAS = True

def norm_col_init(weights,std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1,keepdim=True))
    return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        # m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        # m.bias.data.fill_(0)


class A3Clstm(torch.nn.Module):
    def __init__(self, args):
        super(A3Clstm,self).__init__()
        self.simu_args = args
        self.convs = nn.ModuleList([nn.Conv2d(1, self.simu_args.kernel_dim, (self.simu_args.K, self.simu_args.embedding_dim), bias=BIAS)
                                    for self.simu_args.K in self.simu_args.kernel_size])
        self.dropout = nn.Dropout(self.simu_args.dropout)
        self.fc = nn.Linear(len(self.simu_args.kernel_size) * self.simu_args.kernel_dim, 256, bias=BIAS)

        self.lstm = nn.LSTMCell(256, 256)

        self.critic_linear = nn.Linear(256, 1, bias=BIAS)
        self.actor_linear = nn.Linear(256, self.simu_args.action_dim, bias=BIAS)
        self.actor_linear_ad = nn.Linear(256, self.simu_args.action_dim + 1, bias=BIAS)

        self.apply(weights_init)
        self.actor_linear.weight_data = norm_col_init(self.actor_linear.weight.data, 0.01)
        self.critic_linear.weight_data = norm_col_init(self.critic_linear.weight.data, 0.01)

        self.train()

    def forward(self, inputs, is_training=False):
        inputs, (hx, cx) = inputs
        inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.convs]
        inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs]
        concated = torch.cat(inputs, 1)
        if is_training:
            concated = self.dropout(concated) # (N,len(Ks)*Co)

        x = self.fc(concated)
        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), hx, cx

# what optimizer should I use?
class Agent(object):
    def __init__(self, model, optimizer, simu_args):
        self.model = model
        self.simu_args = simu_args
        self.state = torch.zeros(1,1,20,300)
        if self.simu_args.use_cuda:
            self.hx = torch.zeros(self.simu_args.num_browsers, 256).cuda()
            self.cx = torch.zeros(self.simu_args.num_browsers, 256).cuda()
        else:
            self.hx = torch.zeros(self.simu_args.num_browsers, 256)
            self.cx = torch.zeros(self.simu_args.num_browsers, 256)
        self.step_count = 0
        self.global_step_count = 0
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.terminate = False
        self.info = None
        self.reward = 0
        self.reward_vec = None
        self.reward_vec_base = None
        self.optimizer = optimizer
        self.mask = [[1 for _ in range(self.simu_args.action_dim)] for _ in range(self.simu_args.num_browsers)]
        self.count = {}
        try:
            os.makedirs(self.simu_args.param_path)
        except:
            pass

    def action_train(self, html_doc_vecs, Terminate=False):
        if not Terminate:
            self.terminate = Terminate

            # make prediction using model
            self.state = html_doc_vecs
            value, logit, self.hx, self.cx = self.model((self.state, (self.hx, self.cx)), True)

            self.values.append(value.squeeze(1))
            prob = F.softmax(logit, 1)
            log_prob = F.log_softmax(logit, 1)
            entropy = -(log_prob * prob).sum(1)
            self.entropies.append(entropy)
            action = prob.multinomial(num_samples=1).data
            print(action.view(-1))
            log_prob = log_prob.gather(1, action)
            action_list = list(action.squeeze(1).cpu().numpy().astype("int"))
            self.log_probs.append(log_prob.squeeze(1))

            ### choose URLs here
            disabled=load_disabled("./category_data.txt")

            # get the top vectors (the number of disabled categories + 1)
            # in case all of the top vectors are disabled
            max_list=torch.topk(logit,len(disabled)+1).indices.tolist()

            url_cat, action_url = choose_max_on_constraint(max_list[0], disabled)

            self.step_count += 1
            self.global_step_count += 1

            return url_cat, action_url
        else:
            if self.simu_args.diff_reward:
                tmp = deepcopy(self.rewards)
                for i in range(len(self.rewards)-1):
                    self.rewards[i+1] = tmp[i+1] - tmp[i]
            f_value, _, _, _ = self.model((self.state, (self.hx, self.cx)), True)
            self.hx = self.hx.detach()
            self.cx = self.cx.detach()
            self.values.append(f_value.squeeze(1))
            self.step_count = 0

            print("n+1 call made to agent train action")

    def update(self, rewards, GAMMA, T, retrain=True):
        # for testing so the code below doesn't have to be rewritten right now
        self.rewards = rewards

        if retrain:
            self.model.load_state_dict(torch.load(self.simu_args.agent_path))
        policy_loss = 0
        value_loss = 0
        loss = 0
        gae = 0
        avg_R = 0
        if self.simu_args.use_cuda:
            self.rewards = [self.rewards[i].cuda() for i in range(len(self.rewards))]
            self.values = [self.values[i].cuda() for i in range(len(self.values))]
        else:
            self.rewards = [self.rewards[i] for i in range(len(self.rewards))]
            self.values = [self.values[i] for i in range(len(self.values))]
        R = self.values[-1]

        for i in reversed(range(len(self.rewards))):
            avg_R += self.rewards[i]
            R = GAMMA * R + self.rewards[i]
            advantage = R - self.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            # Generalized Advantage Estimataion

            # ask Jiang about this
            delta_t = self.rewards[i] + GAMMA * self.values[i + 1].data - \
                self.values[i].data
            gae = gae * GAMMA * T + delta_t
            policy_loss = policy_loss - self.log_probs[i] * gae -\
                    0.01 * self.entropies[i]
        self.optimizer.zero_grad()

        loss = policy_loss.sum() + 0.5 * value_loss.sum(0)
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 10)
        self.optimizer.step()
        return loss.detach().cpu().numpy()/len(self.rewards)/self.simu_args.num_browsers, \
        avg_R.sum(0).cpu().numpy()/self.simu_args.num_browsers / len(self.rewards)

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.step_count = 0
        self.mask = [[1 for _ in range(self.simu_args.action_dim)] for _ in range(self.simu_args.num_browsers)]
        self.count = {}
        if self.simu_args.use_cuda:
            self.hx = torch.zeros(self.simu_args.num_browsers, 256).cuda()
            self.cx = torch.zeros(self.simu_args.num_browsers, 256).cuda()
        else:
            self.hx = torch.zeros(self.simu_args.num_browsers, 256)
            self.cx = torch.zeros(self.simu_args.num_browsers, 256)