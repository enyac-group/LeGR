import os
import copy
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import random
from collections import deque
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from utils.drivers import get_dataloader
from pruner.fp_mbnetv2 import FilterPrunerMBNetV2
from pruner.fp_resnet import FilterPrunerResNet

def measure_model(model, pruner, img_size):
    pruner.reset() 
    model.eval()
    pruner.forward(torch.zeros((1,3,img_size,img_size), device='cuda'))
    cur_flops = pruner.cur_flops
    cur_size = pruner.cur_size
    return cur_flops, cur_size

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        '''     
        batch_size specifies the number of experiences to add 
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least 
        batch_size elements before beginning to sample from it.
        '''
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = torch.Tensor([_[0].numpy() for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = torch.Tensor([_[4].numpy() for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.LayerNorm(400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 300),
            nn.LayerNorm(300),
            nn.ReLU(inplace=True),
            nn.Linear(300, action_dim),
            nn.Sigmoid()
        )
        self.net[6].weight.data.mul_(0.1)
        self.net[6].bias.data.mul_(0.1)
    def forward(self, state):
        return self.net(state)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.embed_state = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.LayerNorm(400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 300),
        )
        self.embed_action = nn.Sequential(
            nn.Linear(action_dim, 300),
        )
        self.joint = nn.Sequential(
            nn.LayerNorm(300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 1)
        )
        self.joint[2].weight.data.mul_(0.1)
        self.joint[2].bias.data.mul_(0.1)

    def forward(self, state, action):
        state = self.embed_state(state)
        action = self.embed_action(action)
        value = self.joint(state+action)
        return value


class Actor(object):
    def __init__(self, state_dim, action_dim, learning_rate, tau):
        self.net = ActorNetwork(state_dim, action_dim)
        self.target_net = ActorNetwork(state_dim, action_dim)
        self.tau = tau
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.update_target_network(1)

    def train_step(self, policy_loss):
        self.net.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def predict(self, state):
        return self.net(state)

    def predict_target(self, state):
        return self.target_net(state)

    def update_target_network(self, custom_tau=-1):
        if custom_tau >= 0:
            tau = custom_tau
        else:
            tau = self.tau
        target_params = self.target_net.named_parameters()
        params = self.net.named_parameters()

        dict_target_params = dict(target_params)

        for name, param in params:
            if name in dict_target_params:
                dict_target_params[name].data.copy_(tau*param.data + (1-tau)*dict_target_params[name].data)

class Critic(object):
    def __init__(self, state_dim, action_dim, learning_rate, tau):
        self.net = CriticNetwork(state_dim, action_dim)
        self.target_net = CriticNetwork(state_dim, action_dim)
        self.tau = tau
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_network(1)

    def train_step(self, state, action, target):
        self.net.zero_grad()
        pred = self.net(state, action)
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
        return pred

    def predict(self, state, action):
        return self.net(state, action)

    def predict_target(self, state, action):
        return self.target_net(state, action)

    def update_target_network(self, custom_tau=-1):
        if custom_tau >= 0:
            tau = custom_tau
        else:
            tau = self.tau
        target_params = self.target_net.named_parameters()
        params = self.net.named_parameters()

        dict_target_params = dict(target_params)

        for name, param in params:
            if name in dict_target_params:
                dict_target_params[name].data.copy_(tau*param.data + (1-tau)*dict_target_params[name].data)

class AMCEnv(object):
    def __init__(self, datapath, dataset, filter_pruner, model, max_sparsity, metric, steps, large_input):

        self.image_size = 32 if 'CIFAR' in dataset else 224
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(self.image_size, dataset, datapath, 128, True)

        if 'ImageNet' in dataset:
            num_classes = 1000
        elif 'CIFAR100' in dataset:
            num_classes = 100
        elif 'CIFAR10' in dataset:
            num_classes = 10
        else:
            num_classes = train_set.num_classes

        self.model_name = model
        self.pruner = filter_pruner
        self.num_cls = num_classes
        self.max_sparsity = max_sparsity
        self.metric = metric
        self.steps = steps
        self.orig_model = torch.load(self.model_name)
        while isinstance(self.orig_model, nn.DataParallel):
            self.orig_model = self.orig_model.module
        self.orig_model = self.orig_model.cuda()

    def reset(self):
        self.model = copy.deepcopy(self.orig_model)
        self.filter_pruner = eval(self.pruner)(self.model, self.metric, num_cls=self.num_cls)
        self.filter_pruner.reset()
        self.model.eval()
        self.filter_pruner.forward(torch.zeros((1,3,self.image_size,self.image_size), device='cuda'))
        self.full_size = self.filter_pruner.cur_size
        self.full_flops = self.filter_pruner.cur_flops
        self.checked = []
        self.layer_counter = 0
        self.reduced = 0
        self.rest = self.full_flops * self.max_sparsity
        self.last_act = 0
        self.max_oc = 0
        for key in self.filter_pruner.conv_out_channels:
            if self.max_oc < self.filter_pruner.conv_out_channels[key]:
                self.max_oc = self.filter_pruner.conv_out_channels[key]
        self.max_ic = 0
        for key in self.filter_pruner.conv_in_channels:
            if self.max_ic < self.filter_pruner.conv_in_channels[key]:
                self.max_ic = self.filter_pruner.conv_in_channels[key]
        allh = [self.filter_pruner.omap_size[t][0] for t in range(len(self.filter_pruner.activation_to_conv))]
        allw = [self.filter_pruner.omap_size[t][1] for t in range(len(self.filter_pruner.activation_to_conv))]
        self.max_fh = np.max(allh)
        self.max_fw = np.max(allw)
        self.max_stride = 0
        self.max_k = 0
        for key in self.filter_pruner.activation_to_conv:
            if self.max_stride < self.filter_pruner.activation_to_conv[key].stride[0]:
                self.max_stride = self.filter_pruner.activation_to_conv[key].stride[0]
            if self.max_k < self.filter_pruner.activation_to_conv[key].weight.size(2):
                self.max_k = self.filter_pruner.activation_to_conv[key].weight.size(2)
        conv = self.filter_pruner.activation_to_conv[self.layer_counter]
        h = self.filter_pruner.omap_size[self.layer_counter][0]
        w = self.filter_pruner.omap_size[self.layer_counter][1]

        flops = 0
        k = self.layer_counter
        while k in self.filter_pruner.chains:
            ratio = float(self.filter_pruner.filter_ranks[self.layer_counter].size(0)) / self.filter_pruner.filter_ranks[k].size(0)
            flops += ratio * self.filter_pruner.cost_map[k]*self.filter_pruner.conv_in_channels[k]*self.filter_pruner.conv_out_channels[k]
            k = self.filter_pruner.chains[k]
        ratio = float(self.filter_pruner.filter_ranks[self.layer_counter].size(0)) / self.filter_pruner.filter_ranks[k].size(0)
        flops += ratio * self.filter_pruner.cost_map[k]*self.filter_pruner.conv_in_channels[k]*self.filter_pruner.conv_out_channels[k]

        state = torch.Tensor([float(self.layer_counter)/len(self.filter_pruner.activation_to_conv),
                float(self.filter_pruner.conv_out_channels[self.layer_counter])/self.max_oc,
                float(self.filter_pruner.conv_in_channels[self.layer_counter])/self.max_ic,
                float(h)/self.max_fh,
                float(w)/self.max_fw,
                float(conv.stride[0])/self.max_stride,
                float(conv.weight.size(2))/self.max_k,
                float(flops) /self.full_flops,
                float(self.reduced)/self.full_flops,
                float(self.rest)/self.full_flops,
                self.last_act])

        return state, [self.full_flops, self.rest, self.reduced, flops]

    def train_epoch(self, optim, criterion):
        self.model.train()
        total = 0
        top1 = 0

        data_t = 0
        train_t = 0
        total_loss = 0
        s = time.time()
        for i, (batch, label) in enumerate(self.train_loader):
            data_t += time.time()-s
            s = time.time()
            optim.zero_grad()
            batch, label = batch.to('cuda'), label.to('cuda')
            total += batch.size(0)

            out = self.model(batch)
            loss = criterion(out, label)
            loss.backward()
            total_loss += loss.item()
            optim.step()
            train_t += time.time()-s

            if (i % 100 == 0) or (i == len(self.train_loader)-1):
                print('Batch ({}/{}) | Loss: {:.3f} | (PerBatch) Data: {:.3f}s,  Network: {:.3f}s'.format(
                    i+1, len(self.train_loader), total_loss/(i+1), data_t/(i+1), train_t/(i+1)))
            s = time.time()

    def train_steps(self, steps):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=4.5e-3, momentum=0.9, weight_decay=4e-5, nesterov=True)
        criterion = torch.nn.CrossEntropyLoss()

        s = 0
        avg_loss = []
        iterator = iter(self.train_loader)
        while s < steps:
            try:
                batch, label = next(iterator)
            except StopIteration:
                iterator = iter(self.train_loader)
                batch, label = next(iterator)
            batch, label = batch.to('cuda'), label.to('cuda')
            optimizer.zero_grad()
            out = self.model(batch)
            loss = criterion(out, label)
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            s += 1
        print('Avg Loss: {:.3f}'.format(np.mean(avg_loss)))

    def train(self, model, epochs, name):
        self.model = model
        optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(epochs*0.3), int(epochs*0.6), int(epochs*0.8)], gamma=0.2)
        criterion = torch.nn.CrossEntropyLoss()

        for e in range(epochs):
            print('Epoch {}...'.format(e))
            print('Train')
            self.train_epoch(optimizer, criterion)

            top1, _ = self.test(self.test_loader)
            print('Test | Top-1: {:.2f}'.format(top1))
            scheduler.step()
        top1, _ = self.test(self.test_loader)
        torch.save(model, './ckpt/{}_final.t7'.format(name))
        return top1

    def test(self, data_loader, n_img=-1):
        self.model.eval()
        correct = 0
        total = 0
        total_len = len(data_loader)
        criterion = torch.nn.CrossEntropyLoss()
        if n_img > 0 and total_len > int(np.ceil(float(n_img) / data_loader.batch_size)):
            total_len = int(np.ceil(float(n_img) / data_loader.batch_size))
        for i, (batch, label) in enumerate(data_loader):
            if i >= total_len:
                break
            batch, label = batch.to('cuda'), label.to('cuda')
            output = self.model(batch)
            loss = criterion(output, label)
            pred = output.data.max(1)[1]
            correct += pred.eq(label).sum()
            total += label.size(0)
            if (i % 100 == 0) or (i == total_len-1):
                print('Testing | Batch ({}/{}) | Top-1: {:.2f} ({}/{})'.format(i+1, total_len,\
                                                                        float(correct)/total*100, correct, total))
        self.model.train()
        return float(correct)/total*100, loss.item()

    def save_network(self, name):
        torch.save(self.model, '{}.t7'.format(name))

    def step(self, action):
        self.last_act = action

        self.layer_counter, self.cost, self.rest, rest_max_flops = self.filter_pruner.amc_compress(self.layer_counter, action, self.max_sparsity)

        self.reduced = self.full_flops - self.cost

        m_flop, m_size = 0, 0

        if self.layer_counter >= len(self.filter_pruner.activation_to_conv):
            # Just finish, evaluate reward
            state = torch.zeros(1)

            filters_to_prune_per_layer = self.filter_pruner.get_pruning_plan_from_layer_budget(self.filter_pruner.conv_out_channels)
            prune_targets = self.filter_pruner.pack_pruning_target(filters_to_prune_per_layer, get_segment=True, progressive=True)
            layers_pruned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_pruned:
                    layers_pruned[layer_index] = 0
                layers_pruned[layer_index] = layers_pruned[layer_index] + (filter_index[1]-filter_index[0]+1)

            filters_left = {}
            for k in sorted(self.filter_pruner.filter_ranks.keys()):
                if k not in layers_pruned:
                    layers_pruned[k] = 0
                filters_left[k] = len(self.filter_pruner.filter_ranks[k]) - layers_pruned[k]
            print('Filters left: {}'.format(sorted(filters_left.items())))
            print('Prunning filters..')
            for layer_index, filter_index in prune_targets:
                self.filter_pruner.prune_conv_layer_segment(layer_index, filter_index)
            m_flop, m_size = measure_model(self.model, self.filter_pruner, self.image_size)
            flops = 0
            print('FLOPs: {:.3f}M | #Params: {:.3f}M'.format(m_flop/1e6, m_size/1e6))
            self.train_steps(self.steps)
            top1, loss = self.test(self.val_loader)
            reward = top1
            terminal = 1
        else:

            flops = self.filter_pruner.get_unit_flops_for_layer(self.layer_counter)

            conv = self.filter_pruner.activation_to_conv[self.layer_counter]
            h = self.filter_pruner.omap_size[self.layer_counter][0]
            w = self.filter_pruner.omap_size[self.layer_counter][1]

            state = torch.Tensor([float(self.layer_counter)/len(self.filter_pruner.activation_to_conv),
                    float(self.filter_pruner.conv_out_channels[self.layer_counter])/self.max_oc,
                    float(self.filter_pruner.conv_in_channels[self.layer_counter])/self.max_ic,
                    float(h)/self.max_fh,
                    float(w)/self.max_fw,
                    float(conv.stride[0])/self.max_stride,
                    float(conv.weight.size(2))/self.max_k,
                    float(flops) /self.full_flops,
                    float(self.reduced)/self.full_flops,
                    float(self.rest)/self.full_flops,
                    self.last_act])

            reward = 0
            terminal = 0

        return state, reward, terminal, [self.full_flops, rest_max_flops, self.reduced, flops, m_flop, m_size]

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return torch.Tensor(x)

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='pruned_mbnetv2', help='Name for the experiments, the resulting model and logs will use this')
    parser.add_argument("--datapath", type=str, default='../pytorch-cifar/imagenet_data', help='Path toward the dataset that is used for this experiment')
    parser.add_argument("--dataset", type=str, default='ImageNetData', help='The class name of the dataset that is used, please find available classes under the dataset folder')
    parser.add_argument("--model", type=str, default='./ckpt/mobilenetv2_718.pth', help='The pre-trained model that pruning starts from')
    parser.add_argument("--pruner", type=str, default='FilterPrunnerMBNetV2', help='Different network require differnt pruner implementation')
    parser.add_argument("--metric", type=str, default='l2_weight', help='Metric for ranking filters')
    parser.add_argument("--steps", type=float, default=200, help='How many steps of training before getting the reward signal')
    parser.add_argument("--long_ft", type=int, default=60, help='How many epochs to train after the architecture is found')
    parser.add_argument("--max_sparse", type=float, default=80, help='Maximum sparsity for each layer')
    parser.add_argument("--prune_away", type=float, default=50, help='How many percentage of constraints should be pruned away. E.g., 50 means 50% of FLOPs will be pruned away if --csize is not specified, else it will prune away 50% of size')
    parser.add_argument("--large_input", action='store_true', default=False, help='Used for dataset that has 224x224 inputs, otherwise, it will use 32x32 (CIFAR10)')
    args = parser.parse_args()

    TARGET = args.prune_away / 100.
    TAU = 1e-2
    EPISODES = 400
    SIGMA = 0.5
    MINI_BATCH_SIZE = 64
    MAX_SPARSITY = args.max_sparse / 100.
    EXPLORE = 100
    b = None
    np.random.seed(98)

    actor = Actor(11, 1, 1e-4, TAU)
    critic = Critic(11, 1, 1e-3, TAU)

    env= AMCEnv(args.datapath, args.dataset, args.pruner, args.model, MAX_SPARSITY, args.metric, args.steps, args.large_input)
    replay_buffer = ReplayBuffer(6000)
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(1))
    best_reward = 0

    start = time.time()
    e_start = start
    for episode in range(EPISODES):
        print('Episode {}'.format(episode))
        s, info = env.reset()
        t = False
        if episode > EXPLORE:
            SIGMA = SIGMA * 0.95
        rewards = []
        states_actions = []
        while not t:
            a = torch.clamp(actor.predict(s.view(1, -1)) + np.random.normal(0, SIGMA), 0, MAX_SPARSITY).detach().numpy()
            W_duty = TARGET * info[0] - info[1] - info[2]
            a = np.maximum(float(W_duty)/info[3], a)
            a = np.minimum(a, MAX_SPARSITY)
            s2, r, t, info = env.step(a)
            states_actions.append((s, a))
            r = r / 10.
            rewards.append(r)

            if episode > EXPLORE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINI_BATCH_SIZE)
            
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = torch.Tensor(r_batch.reshape(-1,1) + (1-t_batch.reshape(-1,1)) * target_q.detach().numpy().reshape(-1,1) - b)

                # Update the critic given the targets
                predicted_q_value = critic.train_step(s_batch, torch.Tensor(a_batch).view(-1, 1), y_i)

                # Update the actor policy using the sampled gradient
                policy_loss = -critic.predict(s_batch, actor.predict(s_batch))
                policy_loss = policy_loss.mean()
                actor.train_step(policy_loss)

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            if t:
                if best_reward < r:
                    best_reward = r
                    best_flops = info[4]
                    best_size = info[5]
                    torch.save(env.model, './ckpt/{}_best_init.t7'.format(args.name))

            s = s2.clone()

        rewards = np.max(rewards) * np.ones_like(rewards)
        for idx, (state, action) in enumerate(states_actions):
            if idx != len(states_actions) - 1:
                t = 0
                next_state = states_actions[idx+1][0]
            else:
                t = 1
                next_state = torch.zeros_like(state)
            replay_buffer.add(state, action, rewards[idx], t, next_state)

        if not b:
            b = np.mean(rewards)
        else:
            b = 0.95 * b + (1-0.95) * np.mean(rewards)

        print('Time for this episode: {:.2f}s'.format(time.time()-e_start))
        e_start = time.time()
    end = time.time()
    print('Finished. Total search time: {:.2f}h'.format((end-start)/3600.))
    print('Train the best found network')
    model = torch.load('./ckpt/{}_best_init.t7'.format(args.name))
    top1 = env.train(model, args.long_ft, args.name)
    print('Final Test | Top-1 {:.2f}'.format(top1))
    np.savetxt(os.path.join('./log', '{}_test_acc.txt'.format(args.name)), np.array([0, top1, float(best_size)*1e-6, float(best_flops)*1e-6]))
