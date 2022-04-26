import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from lcdk import lcdk as LeslieChow

#Debugger
DBG = LeslieChow.lcdk(print_output=False)



def train_url_generator(attacker,
                        simu_args,
                        USE_SIMU=False,
                        GAMMA=0.99,
                        T=1,
                        MAX_EP=5,
                        MAX_STEP=20,
                        RETRAIN=True):

    """ train_url_generator - [TODO]: FILL IN DESCRIPTION

    [TODO]: FILL IN DESCRIPTION

    Args:
        attacker: [TODO]: FILL IN DESCRIPTION
        simu_args: [TODO]: FILL IN DESCRIPTION
        USE_SIMU:[TODO]: FILL IN DESCRIPTION - default: False
        GAMMA: [TODO]: FILL IN DESCRIPTION -  default: 0.99
        T: [TODO]: FILL IN DESCRIPTION -default: 1
        MAX_EP: [TODO]: FILL IN DESCRIPTION - default: 5
        MAX_EP: [TODO]: FILL IN DESCRIPTION - default: 20
        RETRAIN: [TODO]: FILL IN DESCRIPTION - default: True

    Returns: nothing

    Typical usage example:
    train_url_generator(attacker,
                        simu_args,
                        USE_SIMU=False,
                        GAMMA=0.99,
                        T=1,
                        MAX_EP=5,
                        MAX_STEP=20,
                        RETRAIN=True)
    """
    if RETRAIN:
        attacker.model.load_state_dict(torch.load(simu_args.agent_path)) #,map_location=torch.device('cpu')))
    avg_reward = 0
    avg_loss = 0
    reward = 0
    loss = 0
    state = 0

    # Generate initial profile randomly
    initial_profile = [random.sample(range(0, simu_args.num_real_url), simu_args.num_browsers) \
        for _ in range(simu_args.max_len)]
    DBG.lt_cyan("{}".format(initial_profile))
    initial_profile_type = [0 for _ in range(simu_args.max_len)]
    state = attacker.env.start_env(initial_profile, initial_profile_type, use_simu=USE_SIMU)
    attacker.state = torch.from_numpy(np.stack(state, axis=0).reshape(simu_args.num_browsers,1,-1,simu_args.embedding_dim))
    if simu_args.use_cuda:
        attacker.state = attacker.state.cuda()
    for ep in range(MAX_EP):
        for step in range(MAX_STEP):
            flag = random.randint(0,1)
            if flag == 0:
                action_list = random.sample(range(0, simu_args.num_real_url), simu_args.num_browsers)
                state, reward = attacker.env.step(action_list, cur_url_type=0, crawling=False)
            else:
                attacker.action_train(Terminate=False)
        attacker.action_train(Terminate=True)
        if len(attacker.rewards) > 0:
            loss, reward = attacker.update(GAMMA, T, retrain=RETRAIN)
        attacker.clear_actions()
        avg_reward += reward
        avg_loss += loss
        if (ep + 1) % 1 == 0:
            DBG.lt_green("Epoch: {}".format(ep))
            DBG.lt_green("Average reward: {}".format(avg_reward / 1))
            DBG.lt_green("Loss:  {}".format(avg_loss / 1))
            if (ep + 1) % 10 == 0:
                DBG.lt_cyan("URL: {}".format( attacker.env.browsing_history))
            DBG.lt_green("Epoch: {}".format(ep))
            DBG.lt_green("Average reward: {}".format(avg_reward))
            DBG.lt_green("Loss: {}".format(avg_loss))
            DBG.lt_green("URL: {}".format(attacker.env.browsing_history))




            attacker.save_param()
            csvFile = open("./Result/attacker-alpha.csv", 'a', newline='')
            writer = csv.writer(csvFile)
            writer.writerow([avg_reward, avg_loss, attacker.env.url_seq])
            csvFile.close()
            avg_reward = 0
            avg_loss = 0

        if not USE_SIMU:
            attacker.env.store_real_bids()


def norm_col_init(weights,std=1.0):
    """ norm_col_init

    Description: [TODO]: FILL IN DESCRIPTION

    Args:
        weights: [TODO]: FILL IN DESCRIPTION
        std: [TODO]: FILL IN DESCRIPTION - default:1.0

    Returns:
        x: [TODO]: FILL IN DESCRIPTION


    Typical usage example:
        import Network
        norm_col_init(weights,std=1.0)
    """
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1,keepdim=True))
    return x


def weights_init(m):

    """ weights_init - [TODO]: FILL IN DESCRIPTION

        [TODO]: FILL IN DESCRIPTION

        Args:
            m: [TODO]: FILL IN DESCRIPTION

        Returns: nothing

        Typical usage exmaple:
            import Network
            weights_init(m)
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class  CNNClassifier(nn.Module):
    """ class CNNClassifier

        [TODO]: FILL IN DESCRIPTION

        Attributes:
            nn.Module: [TODO]: FILL IN DESCRIPTION

        Returns: [TODO]: FILL IN DESCRIPTION
            torch.cat([F.sigmoid(out), 1-F.sigmoid(out)], axis=-1)

        Typical usage example:
            import Network
            cnn = CNNClassifier(nn.module)
    """

    def __init__(self, predictor_args):

        super(CNNClassifier,self).__init__()
        self.args = predictor_args
        # self.emb = nn.Embedding(75, self.args.embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, self.args.kernel_dim, (self.args.K, self.args.embedding_dim)) for self.args.K in self.args.kernel_size])

        self.dropout = nn.Dropout(self.args.dropout)
        self.fc = nn.Linear(6000, 50)
        self.bn = nn.BatchNorm1d(6000)
        self.fc1 = nn.Linear(len(self.args.kernel_size) * self.args.kernel_dim, self.args.output_size)
        self.fc2 = nn.Linear(50, 2)
        self.bn1 = nn.BatchNorm1d(len(self.args.kernel_size) * self.args.kernel_dim)
        self.bn2 = nn.BatchNorm1d(50)
        self.conv_bn = nn.ModuleList([nn.BatchNorm2d(self.args.kernel_dim) for self.args.K in self.args.kernel_size])
        self.apply(weights_init)

    def reset_param(self):
        self.apply(weights_init)

    def forward(self, inputs, is_training=True):
#         emb = self.emb(inputs)
#         print(emb.size())
#         emb = emb.unsqueeze(1)

        inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.convs]
        # inputs = [self.conv_bn[i](inputs[i]).squeeze(3) for i in range(len(inputs))]
        inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs] #[(N,Co), ...]*len(Ks)

        concated = torch.cat(inputs, 1)
        # concated = inputs.view(inputs.size(0), -1)
        # concated = self.bn(concated)
        # concated = F.relu(self.fc(concated))
        
        if is_training:
        # concated = self.bn1(concated)
            concated = self.dropout(concated) # (N,len(Ks)*Co)
        # concated = self.bn1(concated)
        out = self.fc1(concated)
        # out = self.bn2(out)
        # out = self.fc2(out)
        # return torch.cat([F.sigmoid(out), 1-F.sigmoid(out)], axis=-1)
        return F.log_softmax(out,1)