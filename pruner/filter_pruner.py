import torch
import numpy as np
import torch.nn as nn

from random import shuffle
from model.MobileNetV2 import MobileNetV2_CIFAR10
from model.resnet_cifar10 import DownsampleA

class FilterPruner(object):
    def __init__(self, model, rank_type='l2_weight', num_cls=100, safeguard=0, random=False, device='cuda', resource='FLOPs'):
        self.model = model
        self.rank_type = rank_type
        # Chainning convolutions
        # (use activation index to represent a conv)
        self.chains = {}
        self.y = None
        self.num_cls = num_cls
        self.safeguard = safeguard
        self.random = random
        self.device = device
        self.resource_type = resource
        self.reset()

    def num_params(self):
        all_p = 0
        conv_p = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                conv_p += np.prod(m.weight.shape)
                all_p += np.prod(m.weight.shape)
            if isinstance(m, nn.Linear):
                all_p += np.prod(m.weight.shape) 
        return all_p, conv_p
    
    def reset(self):
        self.amc_checked = []
        self.cur_flops = 0
        self.base_flops = 0
        self.cur_size, conv_size = self.num_params()
        self.base_size = self.cur_size - conv_size
        self.quota = None
        self.filter_ranks = {}
        self.rates = {}
        self.cost_map = {}
        self.in_params = {}
        self.omap_size = {}
        self.conv_in_channels = {}
        self.conv_out_channels = {}

    def flop_regularize(self, l):
        for key in self.filter_ranks:
#            print(self.filter_ranks[key], l*self.cost_map[key])
            self.filter_ranks[key] -= l*self.rates[key]

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index] 
        if self.rank_type == 'analysis':
            if activation_index not in self.filter_ranks:
                self.filter_ranks[activation_index] = activation*grad
            else:
                self.filter_ranks[activation_index] = torch.cat((self.filter_ranks[activation_index], activation*grad), 0)


        else:
            # This is NVIDIA's approach
            if self.rank_type == 'meanAbsMeanImpact':
                values = torch.abs((activation*grad).sum(2).sum(2).data / (activation.size(2)*activation.size(3)))

            # This is equivalent to NVIDIA's approach when E[mean_impact] = 0 
            elif self.rank_type == 'madMeanImpact':
                mean_impact = (activation*grad).sum(2).sum(2).data / (activation.size(2)*activation.size(3))
                values = torch.abs(mean_impact - mean_impact.mean(dim=0))

            elif self.rank_type == 'varMeanImpact':
                mean_impact = (activation*grad).sum(2).sum(2).data / (activation.size(2)*activation.size(3))
                values = torch.pow(mean_impact - mean_impact.mean(dim=0), 2)

            elif self.rank_type == 'MAIVarMAI':
                mean_impact = torch.abs(activation*grad).sum(2).sum(2).data / (activation.size(2)*activation.size(3))
                values = mean_impact.mean(dim=0) * torch.pow(mean_impact - mean_impact.mean(dim=0), 2)

            elif self.rank_type == 'MSIVarMSI':
                mean_impact = torch.pow(activation*grad, 2).sum(2).sum(2).data / (activation.size(2)*activation.size(3))
                values = mean_impact.mean(dim=0) * torch.pow(mean_impact - mean_impact.mean(dim=0), 2)

            elif self.rank_type == 'meanL1Impact':
                values = torch.abs(activation*grad).sum(2).sum(2).data / (activation.size(2) * activation.size(3))

            elif self.rank_type == 'meanL1ImpactRaw':
                values = torch.abs(activation*grad).sum(2).sum(2).data

            elif self.rank_type == 'meanL1Act':
                values = torch.abs(activation).sum(2).sum(2).data / (activation.size(2) * activation.size(3))

            elif self.rank_type == 'meanL1Grad':
                values = torch.abs(grad).sum(2).sum(2).data / (activation.size(2) * activation.size(3))

            elif self.rank_type == 'meanGrad':
                values = grad.sum(2).sum(2).data / (activation.size(2) * activation.size(3))

            elif self.rank_type == 'meanL2Impact':
                values = torch.pow(activation*grad, 2).sum(2).sum(2).data / (activation.size(2) * activation.size(3))

            elif self.rank_type == 'madL2Impact':
                l2_impact = torch.pow(activation*grad, 2).sum(2).sum(2).data / (activation.size(2) * activation.size(3))
                values = torch.abs(l2_impact - l2_impact.mean(dim=0))

            elif self.rank_type == 'varL2Impact':
                l2_impact = torch.pow(activation*grad, 2).sum(2).sum(2).data / (activation.size(2) * activation.size(3))
                values = torch.pow(l2_impact - l2_impact.mean(dim=0), 2)

            elif self.rank_type == 'varMSImpact':
                ms_impact = torch.pow(activation*grad, 2).sum(2).sum(2).data / (activation.size(2) * activation.size(3))
                values = torch.pow(ms_impact - ms_impact.mean(dim=0), 2)

            elif self.rank_type == 'L2IVarL2I':
                l2_impact = torch.sqrt(torch.pow(activation*grad, 2).sum(2).sum(2).data)
                values = l2_impact.mean(dim=0) * torch.pow(l2_impact - l2_impact.mean(dim=0), 2)

            elif self.rank_type == 'meanSquaredImpact':
                impact = torch.pow(activation * grad, 2)
                values = impact.sum(2).sum(2) / (activation.size(2) * activation.size(3))

            elif self.rank_type == 'meanMadImpact':
                impact = activation * grad
                impact = impact.reshape((impact.size(0), impact.size(1), -1))
                mean = impact.mean(dim=2)
                values = torch.abs(impact - mean.reshape((impact.size(0), impact.size(1), 1))).sum(2) / impact.size(2)

            elif self.rank_type == 'meanVarImpact':
                impact = activation * grad
                impact = impact.reshape((impact.size(0), impact.size(1), -1))
                mean = impact.mean(dim=2)
                values = torch.pow(impact - mean.reshape((impact.size(0), impact.size(1), 1)), 2).sum(2) / impact.size(2)

            elif self.rank_type == 'meanVarAct':
                std = activation.reshape((activation.size(0), activation.size(1), -1))
                values = torch.pow(std - std.mean(dim=2).reshape((std.size(0), std.size(1), 1)), 2).sum(2) / std.size(2)

            elif self.rank_type == 'meanAct':
                values = activation.sum(2).sum(2).data / (activation.size(2)*activation.size(3))

            elif self.rank_type == 'varF2Act':
                f2 = torch.sqrt(torch.pow(activation, 2).sum(2).sum(2).data) / (activation.size(2)*activation.size(3))
                values = torch.pow(f2 - f2.mean(dim=0), 2)

            elif self.rank_type == '2-taylor':
                values1 = (activation*grad).sum(2).sum(2).data
                values2 = (torch.pow(activation*grad, 2)*0.5).sum(2).sum(2).data
                values = torch.abs(values1 + values2) / (activation.size(2)*activation.size(3))

            values = values.sum(0) / activation.size(0)
            
            if activation_index not in self.filter_ranks:
                self.filter_ranks[activation_index] = torch.zeros(activation.size(1), device=self.device)

            self.filter_ranks[activation_index] += values

        self.grad_index += 1

    def calculate_cost(self, encoding):
        tmp_in_channels = dict(self.conv_in_channels)
        tmp_out_channels = dict(self.conv_out_channels)
        for i in encoding:
            tmp_out_channels[i] = int(tmp_out_channels[i] * encoding[i])
            next_conv_idx = self.next_conv[i] if i in self.next_conv else None
            #if not i in self.downsample_conv and next_conv_idx:
            if next_conv_idx:
                for j in next_conv_idx:
                    tmp_in_channels[j] = tmp_out_channels[i]
        cost = 0
        for key in self.cost_map:
            cost += self.cost_map[key]*tmp_in_channels[key]*tmp_out_channels[key]
        cost += tmp_out_channels[key]*self.num_cls
        return cost

    def get_unit_flops_for_layer(self, layer_id):
        flops = 0
        k = layer_id
        while k in self.chains:
            flops += self.cost_map[k]*self.conv_in_channels[k]*self.conv_out_channels[k]
            next_conv_idx = self.next_conv[k] if k in self.next_conv else None
            if next_conv_idx:
                for next_conv_i in next_conv_idx:
                    next_conv = self.activation_to_conv[next_conv_i]
                    if (next_conv.groups != next_conv.out_channels or next_conv.groups != next_conv.in_channels):
                        flops += self.cost_map[next_conv_i]*self.conv_in_channels[next_conv_i]*self.conv_out_channels[next_conv_i]
            k = self.chains[k]

        flops += self.cost_map[k]*self.conv_in_channels[k]*self.conv_out_channels[k]
        next_conv_idx = self.next_conv[k] if k in self.next_conv else None
        if next_conv_idx:
            for next_conv_i in next_conv_idx:
                next_conv = self.activation_to_conv[next_conv_i]
                if (next_conv.groups != next_conv.out_channels or next_conv.groups != next_conv.in_channels):
                    flops += self.cost_map[next_conv_i]*self.conv_in_channels[next_conv_i]*self.conv_out_channels[next_conv_i]
        return flops


    def get_unit_filters_for_layer(self, layer_id):
        filters = 0
        k = layer_id
        while k in self.chains:
            filters += self.conv_out_channels[k]
            k = self.chains[k]
        filters += self.conv_out_channels[k]
        return filters

    def one_shot_lowest_ranking_filters(self, target):
        # Consolidation of chained channels
        # Use the maximum rank among the chained channels for the criteria for those channels
        # Greedily pick from the lowest rank.
        # 
        # This return list of [layers act_index, filter_index, rank]
        data = []
        chained = []

        # keys of filter_ranks are activation index
        checked = []
        og_filter_size = {}
        new_filter_size = {}
        for i in sorted(self.filter_ranks.keys()):
            og_filter_size[i] = int(self.filter_ranks[i].size(0))
            if i in checked:
                continue
            current_chain = []
            k = i
            while k in self.chains:
               current_chain.append(k) 
               checked.append(k)
               k = self.chains[k]
            current_chain.append(k) 
            checked.append(k)

            sizes = np.array([self.filter_ranks[j].size(0) for j in current_chain])
            max_size = np.max(sizes)
            for k in current_chain:
                new_filter_size[k] = max_size
            ranks = [self.filter_ranks[j].to(self.device) for j in current_chain]
            cnt = torch.zeros(int(max_size), device=self.device)
            for idx in range(len(ranks)):
                # The padding residual
                rank = ranks[idx]
                if rank.size(0) < max_size:
                    cnt += torch.cat((torch.ones(int(rank.size(0)), device=self.device), torch.zeros(int(max_size-rank.size(0)), device=self.device)))
                    ranks[idx] = torch.cat((ranks[idx], torch.zeros(int(max_size-rank.size(0)), device=self.device)))
                else:
                    cnt += torch.ones(int(max_size), device=self.device)

            ranks = torch.stack(ranks, dim=1)
            sum_ranks = ranks.sum(dim=1) #/ cnt
            weight = len(current_chain)
            layers_index = current_chain

            for j in range(sum_ranks.size(0)):
                # layers_index, filter_index, rank, #chain
                rank = sum_ranks[j].cpu().numpy()
                data.append((layers_index, j, rank, weight))

        if self.random:
            s = list(data)
            shuffle(s)
        else:
            s = sorted(data, key=lambda x: x[2])
        selected = []
        idx = 0

        while idx < len(s):
            # make each layer index an instance to prune
            for lj, l in enumerate(s[idx][0]):
                index = s[idx][1] 
                if self.quota[s[idx][0][lj]] > 0 and index < og_filter_size[l]:
                    selected.append((l, index, s[idx][2]))
                    self.quota[l] -= 1

            # MobileNetV2 just has too many filters, this approximation makes it faster
            if not isinstance(self.model, MobileNetV2_CIFAR10) or idx % 10 == 0:
                tmp = sorted(selected, key=lambda x: x[0])
                tmp_in_channels = dict(self.conv_in_channels)
                tmp_out_channels = dict(self.conv_out_channels)
                for f in tmp:
                    tmp_out_channels[f[0]] -= 1
                    next_conv_idx = self.next_conv[f[0]] if f[0] in self.next_conv else None
                    #if not f[0] in self.downsample_conv and next_conv_idx:
                    if next_conv_idx:
                        for i in next_conv_idx:
                            next_conv = self.activation_to_conv[i]
                            if (next_conv.groups != next_conv.out_channels or next_conv.groups != next_conv.in_channels):
                                tmp_in_channels[i] -= 1

                cost = 0
                for key in self.cost_map:
                    cost += self.cost_map[key]*tmp_in_channels[key]*tmp_out_channels[key]
                cost += tmp_out_channels[key]*self.num_cls

                if cost < target:
                    break

            left = 0
            for k in self.quota:
                left += self.quota[k]
            if left <= 0:
                return selected

            idx += 1
        return selected

    def one_shot_lowest_ranking_filters_multi_targets(self, targets):
        output = [[] for _ in targets]
        data = []
        chained = []
        # keys of filter_ranks are activation index
        checked = []
        og_filter_size = {}
        new_filter_size = {}
        for i in sorted(self.filter_ranks.keys()):
            og_filter_size[i] = int(self.filter_ranks[i].size(0))
            if i in checked:
                continue
            current_chain = []
            k = i
            while k in self.chains:
               current_chain.append(k) 
               checked.append(k)
               k = self.chains[k]
            current_chain.append(k) 
            checked.append(k)

            sizes = np.array([self.filter_ranks[j].size(0) for j in current_chain])
            max_size = np.max(sizes)
            for k in current_chain:
                new_filter_size[k] = max_size
            ranks = [self.filter_ranks[j].to(self.device) for j in current_chain]
            cnt = torch.zeros(int(max_size), device=self.device)
            for idx in range(len(ranks)):
                # The padding residual
                rank = ranks[idx]
                if rank.size(0) < max_size:
                    cnt += torch.cat((torch.ones(int(rank.size(0)), device=self.device), torch.zeros(int(max_size-rank.size(0)), device=self.device)))
                    ranks[idx] = torch.cat((ranks[idx], torch.zeros(int(max_size-rank.size(0)), device=self.device)))
                else:
                    cnt += torch.ones(int(max_size), device=self.device)

            ranks = torch.stack(ranks, dim=1)
            sum_ranks = ranks.sum(dim=1) / cnt
            weight = len(current_chain)
            layers_index = current_chain

            for j in range(sum_ranks.size(0)):
                # layers_index, filter_index, rank, #chain
                rank = sum_ranks[j].cpu().numpy()
                data.append((layers_index, j, rank, weight))

        if self.random:
            s = list(data)
            shuffle(s)
        else:
            s = sorted(data, key=lambda x: x[2])
        selected = []
        idx = 0

        target_idx = 0
        while idx < len(s):
            # make each layer index an instance to prune
            for lj, l in enumerate(s[idx][0]):
                index = s[idx][1] 
                if self.quota[s[idx][0][lj]] > 0 and index < og_filter_size[l]:
                    selected.append((l, index, s[idx][2]))
                    self.quota[l] -= 1

            # MobileNetV2 just has too many filters, this approximation makes it faster
            if not isinstance(self.model, MobileNetV2_CIFAR10) or idx % 10 == 0:
                tmp = sorted(selected, key=lambda x: x[0])
                tmp_in_channels = dict(self.conv_in_channels)
                tmp_out_channels = dict(self.conv_out_channels)
                for f in tmp:
                    tmp_out_channels[f[0]] -= 1
                    next_conv_idx = self.next_conv[f[0]] if f[0] in self.next_conv else None
                    #if not f[0] in self.downsample_conv and next_conv_idx:
                    if next_conv_idx:
                        for i in next_conv_idx:
                            next_conv = self.activation_to_conv[i]
                            if (next_conv.groups != next_conv.out_channels or next_conv.groups != next_conv.in_channels):
                                tmp_in_channels[i] -= 1

                cost = 0
                for key in self.cost_map:
                    cost += self.cost_map[key]*tmp_in_channels[key]*tmp_out_channels[key]
                cost += tmp_out_channels[key]*self.num_cls

                if target_idx < len(targets) and cost < targets[target_idx]:
                    output[target_idx] = selected.copy()
                    target_idx += 1
                
                if target_idx == len(targets):
                    break

            idx += 1
        return output


    def pruning_with_transformations(self, original_dist, perturbation, target, masking=False):
        target = target * self.resource_usage
        print('Targeting resource usage: {:.2f}MFLOPs'.format(target/1e6))
        for k in sorted(self.filter_ranks.keys()):
            self.filter_ranks[k] = original_dist[k] * perturbation[k][0] + perturbation[k][1]
        prune_targets = self.get_pruning_plan(target, progressive=(not masking), get_segment=True)
        layers_pruned = {}
        for layer_index, filter_index in prune_targets:
            if layer_index not in layers_pruned:
                layers_pruned[layer_index] = 0
            layers_pruned[layer_index] = layers_pruned[layer_index] + (filter_index[1]-filter_index[0]+1)
        filters_left = {}
        for k in sorted(self.filter_ranks.keys()):
            if k not in layers_pruned:
                layers_pruned[k] = 0
            filters_left[k] = len(self.filter_ranks[k]) - layers_pruned[k]
        print('Filters left: {}'.format(sorted(filters_left.items())))
        print('Prunning filters..')
        for layer_index, filter_index in prune_targets:
            if masking:
                self.mask_conv_layer_segment(layer_index, filter_index)
            else:
                self.prune_conv_layer_segment(layer_index, filter_index)

    def pruning_with_transformations_multi_target(self, original_dist, perturbation, targets):
        full_idx = -1
        if 1 in targets:
            full_idx = targets.index(1)
        targets = np.array(targets) * self.resource_usage
        for k in sorted(self.filter_ranks.keys()):
            self.filter_ranks[k] = original_dist[k] * perturbation[k][0] + perturbation[k][1]
        prune_targets = self.get_pruning_plan_multi_target(targets)
        output = []
        for i, t in enumerate(targets):
            if i == full_idx:
                in_channels, out_channels = self.og_conv_in_channels.copy(), self.og_conv_out_channels.copy()
            else:
                in_channels, out_channels = self.get_channels_after_pruning(prune_targets[i])
            print('Target: {} | Network: {}'.format(t, sorted(out_channels.items())))
            output.append((in_channels, out_channels))
        return output

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            if self.filter_ranks[i].size(0) > 1:
                v = self.filter_ranks[i]
                v = v / torch.sum(v * v).sqrt()
                self.filter_ranks[i] = v.to(self.device)

    def get_pruning_plan_from_layer_budget(self, layer_budget):
        filters_to_prune_per_layer = {}
        last_residual = 0
        for layer in sorted(self.filter_ranks.keys()):
            current_chain = []
            k = layer
            while k in self.chains:
               current_chain.append(k) 
               k = self.chains[k]
            current_chain.append(k) 

            sizes = np.array([self.filter_ranks[j].size(0) for j in current_chain])
            max_size = np.max(sizes)
            ranks = [self.filter_ranks[j].to(self.device) for j in current_chain]
            cnt = torch.zeros(int(max_size), device=self.device)
            for idx in range(len(ranks)):
                # The padding residual
                rank = ranks[idx]
                if rank.size(0) < max_size:
                    cnt += torch.cat((torch.ones(int(rank.size(0)), device=self.device), torch.zeros(int(max_size-rank.size(0)), device=self.device)))
                    ranks[idx] = torch.cat((ranks[idx], torch.zeros(int(max_size-rank.size(0)), device=self.device)))
                else:
                    cnt += torch.ones(int(max_size), device=self.device)

            ranks = torch.stack(ranks, dim=1)
            sum_ranks = ranks.sum(dim=1) #/ cnt

            rank = sum_ranks.cpu().numpy()

            tbp = np.argsort(rank)

            for k in current_chain:
                if not k in filters_to_prune_per_layer:
                    cur_layer_size = self.filter_ranks[k].size(0)
                    valid_ind = tbp[tbp < cur_layer_size][:(cur_layer_size-layer_budget[k])]
                    filters_to_prune_per_layer[k] = valid_ind
        return filters_to_prune_per_layer
            
    def sort_weights(self):
        # Sort by filters and also sort the subsequent channels
        checked = []
        for layer_index in sorted(self.activation_to_conv.keys()):
            if layer_index in checked:
                continue
            current_chain = []
            next_convs = []
            k = layer_index
            while k in self.chains:
               current_chain.append(k) 
               checked.append(k)
               if k in self.next_conv:
                    next_convs = next_convs + self.next_conv[k]
               k = self.chains[k]
            current_chain.append(k) 
            if k in self.next_conv:
                next_convs = next_convs + self.next_conv[k]
            checked.append(k)

            ranks = [self.filter_ranks[j].to(self.device) for j in current_chain]
            ranks = torch.stack(ranks, dim=1)
            sum_ranks = ranks.sum(dim=1)
            sorted_indices = sum_ranks.cpu().numpy().argsort()[::-1]

            last_conv = False
            for k in current_chain:
                if k == len(self.activation_to_conv)-1:
                    last_conv = True
                conv = self.activation_to_conv[k]
                next_bn = self.bn_for_conv[k]

                conv.weight.data = torch.from_numpy(conv.weight.data.cpu().numpy()[sorted_indices,:,:,:]).to(self.device)
                #conv.weight = conv.weight[sorted_indices,:,:,:]

                next_bn.weight.data = torch.from_numpy(next_bn.weight.data.cpu().numpy()[sorted_indices]).to(self.device)
                next_bn.bias.data = torch.from_numpy(next_bn.bias.data.cpu().numpy()[sorted_indices]).to(self.device)
                next_bn.running_mean.data = torch.from_numpy(next_bn.running_mean.data.cpu().numpy()[sorted_indices]).to(self.device)
                next_bn.running_var.data = torch.from_numpy(next_bn.running_var.data.cpu().numpy()[sorted_indices]).to(self.device)

            if next_convs:
                for next_conv_i in next_convs:
                    next_conv = self.activation_to_conv[next_conv_i]
                    if (next_conv.groups != next_conv.out_channels or next_conv.groups != next_conv.in_channels):
                        tmp_weight = next_conv.weight.data.cpu().numpy().transpose([1,0,2,3])
                        tmp_weight = tmp_weight[sorted_indices,:,:,:]
                        tmp_weight = tmp_weight.transpose([1,0,2,3])
                        next_conv.weight.data = torch.from_numpy(tmp_weight).to(self.device)

            if last_conv:
                tmp_weight = self.linear.weight.data.cpu().numpy().transpose([1,0])
                tmp_weight = tmp_weight[sorted_indices,:]
                tmp_weight = tmp_weight.transpose([1,0])
                self.linear.weight.data = torch.from_numpy(tmp_weight).to(self.device)

    def get_pruning_plan_from_importance(self, target, importance):
        cost = 0

        for key in self.cost_map:
            cost += self.cost_map[key] * self.conv_in_channels[key] * self.conv_out_channels[key] * importance[key]
        alpha = float(target)/cost

        reject = False
        theta = {}
        for key in importance:
            theta[key] = alpha * importance[key]

        new_in_channels = self.conv_in_channels.copy()
        new_out_channels = self.conv_out_channels.copy()

        for key in importance:
            new_out_channels[key] = int((theta[key] / (float(new_in_channels[key])/self.conv_in_channels[key])) * new_out_channels[key])
            print(new_out_channels[key])

            next_conv_idx = self.next_conv[key] if key in self.next_conv else None
            if not key in self.downsample_conv and next_conv_idx:
                for i in next_conv_idx:
                    next_conv = self.activation_to_conv[i]
                    if (next_conv.groups != next_conv.out_channels or next_conv.groups != next_conv.in_channels):
                        new_in_channels[i] = new_out_channels[key]

        filters_to_prune_per_layer = self.get_pruning_plan_from_layer_budget(new_out_channels)
        return filters_to_prune_per_layer 

    def pack_pruning_target(self, filters_to_prune_per_layer, get_segment=True, progressive=True):
        if get_segment:
            filters_to_prune = []
            for l in filters_to_prune_per_layer:
                if len(filters_to_prune_per_layer[l]) > 0:
                    filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
                    prev_len = 0
                    first_ptr = 0
                    j = first_ptr+1
                    while j < len(filters_to_prune_per_layer[l]):
                        if filters_to_prune_per_layer[l][j] != filters_to_prune_per_layer[l][j-1]+1:
                            if progressive:
                                begin = filters_to_prune_per_layer[l][first_ptr] - prev_len
                                end = filters_to_prune_per_layer[l][j-1] - prev_len
                            else:
                                begin = filters_to_prune_per_layer[l][first_ptr]
                                end = filters_to_prune_per_layer[l][j-1]
                            filters_to_prune.append((l, (begin, end)))
                            prev_len += (end-begin+1)
                            first_ptr = j
                        j += 1
                    if progressive:
                        begin = filters_to_prune_per_layer[l][first_ptr] - prev_len
                        end = filters_to_prune_per_layer[l][j-1] - prev_len
                    else:
                        begin = filters_to_prune_per_layer[l][first_ptr]
                        end = filters_to_prune_per_layer[l][j-1]
                    filters_to_prune.append((l, (begin, end)))
        else:
            for l in filters_to_prune_per_layer:
                if len(filters_to_prune_per_layer[l]) > 0:
                    filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
                    if progressive:
                        for i in range(len(filters_to_prune_per_layer[l])):
                            # Progressively pruning starts from the lower filters
                            filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

                    filters_to_prune = []
                    for l in filters_to_prune_per_layer:
                        for i in filters_to_prune_per_layer[l]:
                            filters_to_prune.append((l, i))
        return filters_to_prune

    def get_pruning_plan(self, num_filters_to_prune, progressive=True, get_segment=False):
        if not self.quota:
            self.quota = {}
            if self.safeguard == 0:
                for k in self.filter_ranks:
                    self.quota[k] = int(self.filter_ranks[k].size(0)) - 1
            else:
                for k in self.filter_ranks:
                    self.quota[k] = np.minimum(int(np.floor(self.filter_ranks[k].size(0) * (1-self.safeguard))), int(self.filter_ranks[k].size(0)) - 2)
        filters_to_prune = self.one_shot_lowest_ranking_filters(num_filters_to_prune)

        filters_to_prune_per_layer = {}
        for (l, f, r) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        filters_to_prune = self.pack_pruning_target(filters_to_prune_per_layer, get_segment=get_segment, progressive=progressive)

        return filters_to_prune

    def get_uniform_ratio(self, target):
        first = 0
        second = 0
        for conv_idx in self.activation_to_conv:
            conv = self.activation_to_conv[conv_idx]
            layer_cost = self.cost_map[conv_idx]*conv.weight.size(0)*conv.weight.size(1)
            if conv_idx == 0:
                first += layer_cost
            else:
                second += layer_cost

        # TODO: this is wrong if there are multiple linear layers
        first += self.base_flops
        ratio = (np.sqrt(first**2+4*second*target)-first) / (2.*second)
        return ratio

    def uniform_grow(self, growth_rate):
        assert growth_rate > 1

        first = self.activation_to_conv[0]
        for m in self.model.modules():
            if isinstance(m, DownsampleA):
                m.out_channels = int(np.round(m.out_channels*growth_rate))

            elif isinstance(m, nn.Conv2d):
                conv = m
                if conv.groups == conv.out_channels and conv.groups == conv.in_channels:
                    new_conv = torch.nn.Conv2d(in_channels = int(np.round(conv.out_channels*growth_rate)), \
                                    out_channels = int(np.round(conv.out_channels*growth_rate)),
                                    kernel_size = conv.kernel_size, \
                                    stride = conv.stride,
                                    padding = conv.padding,
                                    dilation = conv.dilation,
                                    groups = int(np.round(conv.out_channels*growth_rate)),
                                    bias = conv.bias)
                    conv.in_channels = int(np.round(conv.out_channels*growth_rate))
                    conv.groups = int(np.round(conv.out_channels*growth_rate))
                    conv.out_channels = int(np.round(conv.out_channels*growth_rate))
                else:
                    in_grown = int(np.round(conv.in_channels*growth_rate)) if conv != first else 3
                    new_conv = torch.nn.Conv2d(in_channels = in_grown, \
                                    out_channels = int(np.round(conv.out_channels*growth_rate)),
                                    kernel_size = conv.kernel_size, \
                                    stride = conv.stride,
                                    padding = conv.padding,
                                    dilation = conv.dilation,
                                    groups = conv.groups,
                                    bias = conv.bias)
                    conv.in_channels = in_grown
                    conv.out_channels = int(np.round(conv.out_channels*growth_rate))

                old_weights = conv.weight.data.cpu().numpy()
                new_weights = new_conv.weight.data.cpu().numpy()

                new_out_channels = new_weights.shape[0]
                new_in_channels = new_weights.shape[1]
                old_out_channels = old_weights.shape[0]
                old_in_channels = old_weights.shape[1]
                
                if old_out_channels < new_out_channels and old_in_channels < new_in_channels:
                    new_weights[:old_out_channels, :old_in_channels, :, :] = old_weights
                elif old_out_channels < new_out_channels:
                    new_weights[:old_out_channels, :, :, :] = old_weights
                    new_weights[old_out_channels:, :, :, :] = 0
                else:
                    new_weights[old_out_channels:, old_in_channels:, :, :] = 0

                conv.weight.data = torch.from_numpy(new_weights).to(self.device)
                conv.weight.grad = None

            elif isinstance(m, nn.BatchNorm2d):
                next_bn = m
                # Surgery on next batchnorm layer
                next_new_bn = \
                    torch.nn.BatchNorm2d(num_features = int(np.round(next_bn.num_features*growth_rate)),\
                            eps =  next_bn.eps, \
                            momentum = next_bn.momentum, \
                            affine = next_bn.affine,
                            track_running_stats = next_bn.track_running_stats)
                next_bn.num_features = int(np.round(next_bn.num_features*growth_rate))

                old_weights = next_bn.weight.data.cpu().numpy()
                new_weights = next_new_bn.weight.data.cpu().numpy()
                old_bias = next_bn.bias.data.cpu().numpy()
                new_bias = next_new_bn.bias.data.cpu().numpy()
                old_running_mean = next_bn.running_mean.data.cpu().numpy()
                new_running_mean = next_new_bn.running_mean.data.cpu().numpy()
                old_running_var = next_bn.running_var.data.cpu().numpy()
                new_running_var = next_new_bn.running_var.data.cpu().numpy()

                new_weights[: old_weights.shape[0]] = old_weights
                next_bn.weight.data = torch.from_numpy(new_weights).to(self.device)
                next_bn.weight.grad = None

                new_bias[: old_bias.shape[0]] = old_bias
                next_bn.bias.data = torch.from_numpy(new_bias).to(self.device)
                next_bn.bias.grad = None

                new_running_mean[: old_running_mean.shape[0]] = old_running_mean
                next_bn.running_mean.data = torch.from_numpy(new_running_mean).to(self.device)
                next_bn.running_mean.grad = None

                new_running_var[: old_running_var.shape[0]] = old_running_var
                next_bn.running_var.data = torch.from_numpy(new_running_var).to(self.device)
                next_bn.running_var.grad = None

            elif isinstance(m, nn.Linear):
                new_linear_layer = torch.nn.Linear(int(np.round(m.in_features*growth_rate)), m.out_features)
                m.in_features = int(np.round(m.in_features*growth_rate))
        
                old_weights = m.weight.data.cpu().numpy()
                new_weights = new_linear_layer.weight.data.cpu().numpy()	 	

                new_weights[:, : old_weights.shape[1]] = old_weights
                
                m.weight.data = torch.from_numpy(new_weights).to(self.device)
                m.weight.grad = None

    def get_pruning_plan_multi_target(self, targets):
        if not self.quota:
            self.quota = {}
            if self.safeguard == 0:
                for k in self.filter_ranks:
                    self.quota[k] = int(self.filter_ranks[k].size(0)) - 1
            else:
                for k in self.filter_ranks:
                    self.quota[k] = np.minimum(int(np.floor(self.filter_ranks[k].size(0) * (1-self.safeguard))), int(self.filter_ranks[k].size(0)) - 2)
        multi_filters_to_prune = self.one_shot_lowest_ranking_filters_multi_targets(targets)

        filters_to_prune_per_layer = [{} for _ in range(len(multi_filters_to_prune))]
        for idx, filters_to_prune in enumerate(multi_filters_to_prune):
            for (l, f, r) in filters_to_prune:
                if l not in filters_to_prune_per_layer[idx]:
                    filters_to_prune_per_layer[idx][l] = 0
                filters_to_prune_per_layer[idx][l] += 1

        return filters_to_prune_per_layer
