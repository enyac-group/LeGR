import torch
import numpy as np
import torch.nn as nn

from pruner.filter_pruner import FilterPruner
from model.resnet_cifar10 import BasicBlock, DownsampleA
from torchvision.models.resnet import Bottleneck

def get_num_gen(gen):
    return sum(1 for x in gen)

def is_leaf(model):
    return get_num_gen(model.children()) == 0

class FilterPrunerResNet(FilterPruner):
    def trace_layer(self, layer, x):
        y = layer.old_forward(x)
        if isinstance(layer, nn.Conv2d):
            self.conv_in_channels[self.activation_index] = layer.weight.size(1)
            self.conv_out_channels[self.activation_index] = layer.weight.size(0)
            h = y.shape[2]
            w = y.shape[3]
            self.omap_size[self.activation_index] = (h, w)
            self.cost_map[self.activation_index] = h * w * layer.weight.size(2) * layer.weight.size(3) / layer.groups

            self.in_params[self.activation_index] = layer.weight.size(1) * layer.weight.size(2) * layer.weight.size(3)
            self.cur_flops += h * w * layer.weight.size(0) * layer.weight.size(1) * layer.weight.size(2) * layer.weight.size(3)

            if self.rank_type == 'l1_weight':
                if self.activation_index not in self.filter_ranks:
                    self.filter_ranks[self.activation_index] = torch.zeros(layer.weight.size(0), device=self.device)
                values = (torch.abs(layer.weight.data)).sum(1).sum(1).sum(1)
                # Normalize the rank by the filter dimensions
                #values = values / (layer.weight.size(1) * layer.weight.size(2) * layer.weight.size(3))
                self.filter_ranks[self.activation_index] = values
            elif self.rank_type == 'l2_weight': 
                if self.activation_index not in self.filter_ranks:
                    self.filter_ranks[self.activation_index] = torch.zeros(layer.weight.size(0), device=self.device)
                values = (torch.pow(layer.weight.data, 2)).sum(1).sum(1).sum(1)
                # Normalize the rank by the filter dimensions
                #values = values / (layer.weight.size(1) * layer.weight.size(2) * layer.weight.size(3))
                self.filter_ranks[self.activation_index] = values
            elif self.rank_type == 'l2_bn' or self.rank_type == 'l1_bn' or self.rank_type == 'l2_bn_param': 
                pass
            else:
                y.register_hook(self.compute_rank)
                self.activations.append(y)

            self.rates[self.activation_index] = self.conv_in_channels[self.activation_index] * self.cost_map[self.activation_index]
            self.activation_to_conv[self.activation_index] = layer
            self.conv_to_index[layer] = self.activation_index
            self.activation_index += 1
            
        elif isinstance(layer, nn.BatchNorm2d):
            self.bn_for_conv[self.activation_index-1] = layer
            if self.rank_type == 'l2_bn': 
                if self.activation_index-1 not in self.filter_ranks:
                    self.filter_ranks[self.activation_index-1] = torch.zeros(layer.weight.size(0), device=self.device)
                values = torch.pow(layer.weight.data, 2)
                self.filter_ranks[self.activation_index-1] = values

            elif self.rank_type == 'l2_bn_param': 
                if self.activation_index-1 not in self.filter_ranks:
                    self.filter_ranks[self.activation_index-1] = torch.zeros(layer.weight.size(0), device=self.device)
                values = torch.pow(layer.weight.data, 2)
                self.filter_ranks[self.activation_index-1] = values * self.in_params[self.activation_index-1]

        elif isinstance(layer, nn.Linear):
            self.base_flops += np.prod(layer.weight.shape)
            self.cur_flops += np.prod(layer.weight.shape)

        self.og_conv_in_channels = self.conv_in_channels.copy()
        self.og_conv_out_channels = self.conv_out_channels.copy()

        return y

    def parse_dependency_btnk(self):
        self.downsample_conv = []
        self.pre_padding = {}
        self.next_conv = {}
        prev_conv_idx = 0
        cur_conv_idx = 0
        prev_res = -1
        for m in self.model.modules():
            if isinstance(m, Bottleneck):
                if prev_res > -1:
                    self.next_conv[prev_res] = [self.conv_to_index[m.conv1]]
                self.next_conv[cur_conv_idx] = [self.conv_to_index[m.conv1]]
                self.next_conv[self.conv_to_index[m.conv1]] = [self.conv_to_index[m.conv2]]
                self.next_conv[self.conv_to_index[m.conv2]] = [self.conv_to_index[m.conv3]]
                cur_conv_idx = self.conv_to_index[m.conv3]
                if m.downsample is not None:
                    residual_conv_idx = self.conv_to_index[m.downsample[0]]
                    self.downsample_conv.append(residual_conv_idx)
                    self.next_conv[prev_conv_idx].append(residual_conv_idx)
                    prev_res = residual_conv_idx
                    self.chains[cur_conv_idx] = residual_conv_idx
                else:
                    if (prev_res > -1) and (not prev_res in self.chains):
                        self.chains[prev_res] = cur_conv_idx
                    elif prev_conv_idx not in self.chains:
                        self.chains[prev_conv_idx] = cur_conv_idx
                prev_conv_idx = cur_conv_idx

    def parse_dependency(self):
        self.downsample_conv = []
        self.pre_padding = {}
        self.next_conv = {}
        prev_conv_idx = 0
        prev_res = -1
        for m in self.model.modules():
            if isinstance(m, BasicBlock):
                cur_conv_idx = self.conv_to_index[m.conv[3]]
                # if there is auxiliary 1x1 conv on shortcut
                if isinstance(m.shortcut, DownsampleA):
                    self.pre_padding[cur_conv_idx] = m.shortcut
                self.chains[prev_conv_idx] = cur_conv_idx
                prev_conv_idx = cur_conv_idx

        last_idx = -1
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) and m.weight.size(2) == 3:
                idx = self.conv_to_index[m]
                if (last_idx > -1) and (not last_idx in self.next_conv):
                    self.next_conv[last_idx] = [idx]
                elif (last_idx > -1):
                    self.next_conv[last_idx].append(idx)
                last_idx = idx

    def forward(self, x):

        self.activation_index = 0
        self.grad_index = 0
        self.activations = []
        self.linear = None
        # activation index to the instance of conv layer
        self.activation_to_conv = {}
        self.conv_to_index = {}
        # retrieve next immediate bn layer using activation index of conv
        self.bn_for_conv = {}
        self.cur_flops = 0

        def modify_forward(model):
            for child in model.children():
                if is_leaf(child):
                    def new_forward(m):
                        def lambda_forward(x):
                            return self.trace_layer(m, x)
                        return lambda_forward
                    child.old_forward = child.forward
                    child.forward = new_forward(child)
                else:
                    modify_forward(child)

        def restore_forward(model):
            for child in model.children():
                # leaf node
                if is_leaf(child) and hasattr(child, 'old_forward'):
                    child.forward = child.old_forward
                    child.old_forward = None
                else:
                    restore_forward(child)

        modify_forward(self.model)
        y = self.model.forward(x)
        restore_forward(self.model)

        self.btnk = False
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                self.linear = m 
            if isinstance(m, Bottleneck):
                self.btnk = True
    
        if self.btnk:
            self.parse_dependency_btnk()
        else:
            self.parse_dependency()

        self.resource_usage = self.cur_flops

        return y

    def get_valid_filters(self):
        filters_to_prune_per_layer = {}
        lastConv = None
        chain_max_dim = 0
        for conv_idx in self.chains:
            num_filters = self.activation_to_conv[conv_idx].weight.size(0)
            chain_max_dim = np.maximum(chain_max_dim, num_filters)

        # Deal with the chain first
        mask = np.zeros(chain_max_dim)
        for conv_idx in self.chains:
            bn = self.bn_for_conv[conv_idx]
            cur_mask = (torch.abs(bn.weight) > 0).cpu().numpy()
            cur_mask = np.concatenate((cur_mask, np.zeros(chain_max_dim - len(cur_mask))))
            chained_bn = self.bn_for_conv[self.chains[conv_idx]]
            chained_mask = (torch.abs(chained_bn.weight) > 0).cpu().numpy()
            chained_mask = np.concatenate((cur_mask, np.zeros(chain_max_dim - len(cur_mask))))
            mask = np.logical_or(mask, cur_mask)
            mask = np.logical_or(mask, chained_mask)

        visited = []
        conv_idx = 0
        while conv_idx in self.chains:
            if conv_idx not in visited:
                bn = self.bn_for_conv[conv_idx]
                cur_mask = mask[:bn.weight.size(0)]
                inactive_filter = np.where(cur_mask == 0)[0]
                if len(inactive_filter) > 0:
                    filters_to_prune_per_layer[conv_idx] = list(inactive_filter.astype(int))
                    if len(inactive_filter) == bn.weight.size(0):
                        filters_to_prune_per_layer[conv_idx] = filters_to_prune_per_layer[conv_idx][:-2]
                visited.append(conv_idx)
            if self.chains[conv_idx] not in visited:
                bn = self.bn_for_conv[self.chains[conv_idx]]
                cur_mask = mask[:bn.weight.size(0)]
                inactive_filter = np.where(cur_mask == 0)[0]
                if len(inactive_filter) > 0:
                    filters_to_prune_per_layer[self.chains[conv_idx]] = list(inactive_filter.astype(int))
                    if len(inactive_filter) == bn.weight.size(0):
                        filters_to_prune_per_layer[self.chains[conv_idx]] = filters_to_prune_per_layer[self.chains[conv_idx]][:-2]
                visited.append(self.chains[conv_idx])
            conv_idx = self.chains[conv_idx]
            

        for conv_idx in self.activation_to_conv:
            if conv_idx not in visited:
                bn = self.bn_for_conv[conv_idx]
                cur_mask = (torch.abs(bn.weight) > 0).cpu().numpy()
                inactive_filter = np.where(cur_mask == 0)[0]
                if len(inactive_filter) > 0:
                    filters_to_prune_per_layer[conv_idx] = list(inactive_filter.astype(int))
                    if len(inactive_filter) == bn.weight.size(0):
                        filters_to_prune_per_layer[conv_idx] = filters_to_prune_per_layer[conv_idx][:-2]
            
        return filters_to_prune_per_layer

    def get_valid_flops(self):
        chain_max_dim = 0
        for conv_idx in self.chains:
            num_filters = self.activation_to_conv[conv_idx].weight.size(0)
            chain_max_dim = np.maximum(chain_max_dim, num_filters)

        # Deal with the chain first
        mask = np.zeros(chain_max_dim)
        for conv_idx in self.chains:
            bn = self.bn_for_conv[conv_idx]
            cur_mask = (torch.abs(bn.weight) > 0).cpu().numpy()
            cur_mask = np.concatenate((cur_mask, np.zeros(chain_max_dim - len(cur_mask))))
            chained_bn = self.bn_for_conv[self.chains[conv_idx]]
            chained_mask = (torch.abs(chained_bn.weight) > 0).cpu().numpy()
            chained_mask = np.concatenate((cur_mask, np.zeros(chain_max_dim - len(cur_mask))))
            mask = np.logical_or(mask, cur_mask)
            mask = np.logical_or(mask, chained_mask)

        out_channels = self.conv_out_channels.copy()
        in_channels = self.conv_in_channels.copy()
        visited = []
        conv_idx = 0
        while conv_idx in self.chains:
            if conv_idx not in visited:
                bn = self.bn_for_conv[conv_idx]
                cur_mask = mask[:bn.weight.size(0)]
                inactive_filter = np.where(cur_mask == 0)[0]
                if len(inactive_filter) > 0:
                    out_channels[conv_idx] -= len(inactive_filter)
                    if len(inactive_filter) == bn.weight.size(0):
                        out_channels[conv_idx] = 2
                visited.append(conv_idx)
            if self.chains[conv_idx] not in visited:
                bn = self.bn_for_conv[self.chains[conv_idx]]
                cur_mask = mask[:bn.weight.size(0)]
                inactive_filter = np.where(cur_mask == 0)[0]
                if len(inactive_filter) > 0:
                    out_channels[self.chains[conv_idx]] -= len(inactive_filter)
                    if len(inactive_filter) == bn.weight.size(0):
                        out_channels[self.chains[conv_idx]] = 2
                visited.append(self.chains[conv_idx])
            conv_idx = self.chains[conv_idx]
            

        for conv_idx in self.activation_to_conv:
            if conv_idx not in visited:
                bn = self.bn_for_conv[conv_idx]
                cur_mask = (torch.abs(bn.weight) > 0).cpu().numpy()
                inactive_filter = np.where(cur_mask == 0)[0]
                if len(inactive_filter) > 0:
                    out_channels[conv_idx] -= len(inactive_filter)
                    if len(inactive_filter) == bn.weight.size(0):
                        out_channels[conv_idx] = 2
            
        flops = 0
        for k in self.activation_to_conv:
            flops += self.cost_map[k] * in_channels[k] * out_channels[k]
        flops += out_channels[k] * self.num_cls
                
        return flops

    def mask_conv_layer_segment(self, layer_index, filter_range):
        filters_begin = filter_range[0]
        filters_end = filter_range[1]
        pruned_filters = filters_end - filters_begin + 1
        # Retrive conv based on layer_index
        conv = self.activation_to_conv[layer_index]

        #if layer_index in self.pre_padding:
        #    self.pre_padding[layer_index].out_channels -= pruned_filters
        next_bn = self.bn_for_conv[layer_index]
        next_conv_idx = self.next_conv[layer_index] if layer_index in self.next_conv else None

        # Surgery on the conv layer to be pruned
        # dw-conv, reduce groups as well
        conv.weight.data[filters_begin:filters_end+1,:,:,:] = 0
        conv.weight.grad = None

        if not conv.bias is None:
            conv.bias.data[filters_begin:filters_end+1] = 0
            conv.bias.grad = None
            
        next_bn.weight.data[filters_begin:filters_end+1] = 0
        next_bn.weight.grad = None

        next_bn.bias.data[filters_begin:filters_end+1] = 0
        next_bn.bias.grad = None

        next_bn.running_mean.data[filters_begin:filters_end+1] = 0
        next_bn.running_mean.grad = None

        next_bn.running_var.data[filters_begin:filters_end+1] = 0
        next_bn.running_var.grad = None
        
    def prune_conv_layer_segment(self, layer_index, filter_range):
        filters_begin = filter_range[0]
        filters_end = filter_range[1]
        pruned_filters = int(filters_end - filters_begin + 1)
        # Retrive conv based on layer_index
        conv = self.activation_to_conv[layer_index]

        if layer_index in self.pre_padding:
            self.pre_padding[layer_index].out_channels -= pruned_filters
        next_bn = self.bn_for_conv[layer_index]
        next_conv_idx = self.next_conv[layer_index] if layer_index in self.next_conv else None

        # Surgery on the conv layer to be pruned
        # dw-conv, reduce groups as well
        if conv.groups == conv.out_channels and conv.groups == conv.in_channels:
            new_conv = \
                torch.nn.Conv2d(in_channels = conv.out_channels - pruned_filters, \
                        out_channels = conv.out_channels - pruned_filters,
                        kernel_size = conv.kernel_size, \
                        stride = conv.stride,
                        padding = conv.padding,
                        dilation = conv.dilation,
                        groups = conv.groups - pruned_filters,
                        bias = conv.bias)

            conv.in_channels -= pruned_filters
            conv.out_channels -= pruned_filters
            conv.groups -= pruned_filters
        else:
            new_conv = \
                torch.nn.Conv2d(in_channels = conv.in_channels, \
                        out_channels = conv.out_channels - pruned_filters,
                        kernel_size = conv.kernel_size, \
                        stride = conv.stride,
                        padding = conv.padding,
                        dilation = conv.dilation,
                        groups = conv.groups,
                        bias = conv.bias)

            conv.out_channels -= pruned_filters

        old_weights = conv.weight.data.cpu().numpy()
        new_weights = new_conv.weight.data.cpu().numpy()

        new_weights[: filters_begin, :, :, :] = old_weights[: filters_begin, :, :, :]
        new_weights[filters_begin : , :, :, :] = old_weights[filters_end + 1 :, :, :, :]

        conv.weight.data = torch.from_numpy(new_weights).to(self.device)
        conv.weight.grad = None

        if not conv.bias is None:
            bias_numpy = conv.bias.data.cpu().numpy()

            bias = np.zeros(shape = (bias_numpy.shape[0] - pruned_filters), dtype = np.float32)
            bias[:filters_begin] = bias_numpy[:filters_begin]
            bias[filters_begin : ] = bias_numpy[filters_end + 1 :]
            conv.bias.data = torch.from_numpy(bias).to(self.device)
            conv.bias.grad = None
            
        # Surgery on next batchnorm layer
        next_new_bn = \
            torch.nn.BatchNorm2d(num_features = next_bn.num_features-pruned_filters,\
                    eps =  next_bn.eps, \
                    momentum = next_bn.momentum, \
                    affine = next_bn.affine,
                    track_running_stats = next_bn.track_running_stats)
        next_bn.num_features -= pruned_filters

        old_weights = next_bn.weight.data.cpu().numpy()
        new_weights = next_new_bn.weight.data.cpu().numpy()
        old_bias = next_bn.bias.data.cpu().numpy()
        new_bias = next_new_bn.bias.data.cpu().numpy()
        old_running_mean = next_bn.running_mean.data.cpu().numpy()
        new_running_mean = next_new_bn.running_mean.data.cpu().numpy()
        old_running_var = next_bn.running_var.data.cpu().numpy()
        new_running_var = next_new_bn.running_var.data.cpu().numpy()

        new_weights[: filters_begin] = old_weights[: filters_begin]
        new_weights[filters_begin :] = old_weights[filters_end + 1 :]
        next_bn.weight.data = torch.from_numpy(new_weights).to(self.device)
        next_bn.weight.grad = None

        new_bias[: filters_begin] = old_bias[: filters_begin]
        new_bias[filters_begin :] = old_bias[filters_end + 1 :]
        next_bn.bias.data = torch.from_numpy(new_bias).to(self.device)
        next_bn.bias.grad = None

        new_running_mean[: filters_begin] = old_running_mean[: filters_begin]
        new_running_mean[filters_begin :] = old_running_mean[filters_end + 1 :]
        next_bn.running_mean.data = torch.from_numpy(new_running_mean).to(self.device)
        next_bn.running_mean.grad = None

        new_running_var[: filters_begin] = old_running_var[: filters_begin]
        new_running_var[filters_begin :] = old_running_var[filters_end + 1 :]
        next_bn.running_var.data = torch.from_numpy(new_running_var).to(self.device)
        next_bn.running_var.grad = None
        

        # Found next convolution layer
        if next_conv_idx:
            if not layer_index in self.downsample_conv:
                for next_conv_i in next_conv_idx:
                    next_conv = self.activation_to_conv[next_conv_i]
                    next_new_conv = \
                        torch.nn.Conv2d(in_channels = next_conv.in_channels - pruned_filters,\
                                out_channels =  next_conv.out_channels, \
                                kernel_size = next_conv.kernel_size, \
                                stride = next_conv.stride,
                                padding = next_conv.padding,
                                dilation = next_conv.dilation,
                                groups = next_conv.groups,
                                bias = next_conv.bias)
                    next_conv.in_channels -= pruned_filters

                    old_weights = next_conv.weight.data.cpu().numpy()
                    new_weights = next_new_conv.weight.data.cpu().numpy()

                    new_weights[:, : filters_begin, :, :] = old_weights[:, : filters_begin, :, :]
                    new_weights[:, filters_begin : , :, :] = old_weights[:, filters_end + 1 :, :, :]
                    next_conv.weight.data = torch.from_numpy(new_weights).to(self.device)
                    next_conv.weight.grad = None
        else:
            #Prunning the last conv layer. This affects the first linear layer of the classifier.
            if self.linear is None:
                raise BaseException("No linear laye found in classifier")
            params_per_input_channel = int(self.linear.in_features / (conv.out_channels+pruned_filters))

            new_linear_layer = \
                    torch.nn.Linear(self.linear.in_features - pruned_filters*params_per_input_channel, 
                            self.linear.out_features)

            self.linear.in_features -= pruned_filters*params_per_input_channel
            
            old_weights = self.linear.weight.data.cpu().numpy()
            new_weights = new_linear_layer.weight.data.cpu().numpy()	 	

            new_weights[:, : int(filters_begin * params_per_input_channel)] = \
                    old_weights[:, : int(filters_begin * params_per_input_channel)]
            new_weights[:, int(filters_begin * params_per_input_channel) :] = \
                    old_weights[:, int((filters_end + 1) * params_per_input_channel) :]
            
            self.linear.weight.data = torch.from_numpy(new_weights).to(self.device)
            self.linear.weight.grad = None

    def amc_filter_compress(self, layer_id, action, max_sparsity):
        # Chain residual connections
        t = layer_id
        current_chains = []
        while t in self.chains:
            current_chains.append(t)
            t = self.chains[t]
        current_chains.append(t)
        prune_away = int(action*self.conv_out_channels[layer_id])
        if (not self.btnk) and (len(current_chains) > 1):
            top_pruning = 16 if current_chains[0] == 0 else int(current_chains[0] / 18)*16
            prune_away = np.minimum(prune_away, top_pruning)
        # Used to identify which layer cannot make decision later on
        # If it is chained with same size, it is determined by the first one.
        cur_filter_size = self.conv_out_channels[layer_id] 
        for layer in current_chains:
            if self.conv_out_channels[layer] == cur_filter_size:
                self.amc_checked.append(layer)
            self.conv_out_channels[layer] -= prune_away

        rest = 0
        rest_min_filters = 0
        rest_total_filters = 0

        tmp_out_channels = self.og_conv_out_channels.copy()
        tmp_in_channels = self.conv_in_channels.copy()

        next_layer = layer_id
        while next_layer in self.amc_checked:
            next_layer += 1

        t = next_layer
        next_chains = []
        if t < len(self.activation_to_conv):
            while t in self.chains:
                next_chains.append(t)
                t = self.chains[t]
            next_chains.append(t)

        for i in range(next_layer, len(self.activation_to_conv)):
            if not i in self.amc_checked:
                rest += self.conv_out_channels[i]

                if not i in next_chains:
                    if max_sparsity == 1:
                        tmp_out_channels[i] = 1
                    else:
                        tmp_out_channels[i] = int(np.ceil(tmp_out_channels[i] * (1-max_sparsity)))

                    rest_total_filters += self.conv_out_channels[i]
                    rest_min_filters += tmp_out_channels[i]
        rest_max_filters = rest_total_filters - rest_min_filters

        cost = 0
        for key in self.cost_map:
            cost += self.conv_out_channels[key]

        return next_layer, cost, rest_max_filters 

    def amc_compress(self, layer_id, action, max_sparsity):
        # Chain residual connections
        t = layer_id
        current_chains = []
        while t in self.chains:
            current_chains.append(t)
            t = self.chains[t]
        current_chains.append(t)
        prune_away = int(action*self.conv_out_channels[layer_id])
        if (not self.btnk) and (len(current_chains) > 1):
            top_pruning = 16 if current_chains[0] == 0 else int(current_chains[0] / 18)*16
            prune_away = np.minimum(prune_away, top_pruning)
        # Used to identify which layer cannot make decision later on
        # If it is chained with same size, it is determined by the first one.
        cur_filter_size = self.conv_out_channels[layer_id] 
        for layer in current_chains:
            if self.conv_out_channels[layer] == cur_filter_size:
                self.amc_checked.append(layer)
            self.conv_out_channels[layer] -= prune_away
            next_conv_idx = self.next_conv[layer] if layer in self.next_conv else None
            if not layer in self.downsample_conv and next_conv_idx:
                for i in next_conv_idx:
                    next_conv = self.activation_to_conv[i]
                    if (next_conv.groups != next_conv.out_channels or next_conv.groups != next_conv.in_channels):
                        self.conv_in_channels[i] -= prune_away

        rest = 0
        rest_min_flops = 0
        rest_total_flops = 0

        tmp_out_channels = self.og_conv_out_channels.copy()
        tmp_in_channels = self.conv_in_channels.copy()

        next_layer = layer_id
        while next_layer in self.amc_checked:
            next_layer += 1

        t = next_layer
        next_chains = []
        if t < len(self.activation_to_conv):
            while t in self.chains:
                next_chains.append(t)
                t = self.chains[t]
            next_chains.append(t)

        init_in = {}
        # If filter in next_chains are prune to maximum, modify the following channels
        for t in next_chains:
            next_conv_idx = self.next_conv[t] if t in self.next_conv else None
            if next_conv_idx:
                for next_conv_i in next_conv_idx:
                    next_conv = self.activation_to_conv[next_conv_i]
                    if (next_conv.groups != next_conv.out_channels or next_conv.groups != next_conv.in_channels):
                        tmp_in_channels[next_conv_i] = self.og_conv_in_channels[next_conv_i] * (1-max_sparsity)
                        init_in[next_conv_i] = self.og_conv_in_channels[next_conv_i] * (1-max_sparsity)

        for i in range(next_layer, len(self.activation_to_conv)):
            if not i in self.amc_checked:
                rest += self.cost_map[i]*self.conv_in_channels[i]*self.conv_out_channels[i]

                if not i in next_chains:
                    tmp_out_channels[i] *= (1-max_sparsity)
                    next_conv_idx = self.next_conv[i] if i in self.next_conv else None
                    if next_conv_idx:
                        for next_conv_i in next_conv_idx:
                            next_conv = self.activation_to_conv[next_conv_i]
                            if (next_conv.groups != next_conv.out_channels or next_conv.groups != next_conv.in_channels):
                                tmp_in_channels[next_conv_i] = tmp_out_channels[i]

                    if i in init_in:
                        rest_total_flops += self.cost_map[i]*init_in[i]*self.conv_out_channels[i]
                    else:
                        rest_total_flops += self.cost_map[i]*self.conv_in_channels[i]*self.conv_out_channels[i]
                    rest_min_flops += self.cost_map[i]*tmp_in_channels[i]*tmp_out_channels[i]
        rest_max_flops = rest_total_flops - rest_min_flops

        cost = 0
        for key in self.cost_map:
            cost += self.cost_map[key]*self.conv_in_channels[key]*self.conv_out_channels[key]
        cost += self.conv_out_channels[key]*self.num_cls

        return next_layer, cost, rest, rest_max_flops 
