import torch
import numpy as np
import torch.nn as nn

from pruner.filter_pruner import FilterPruner
from model.MobileNetV2 import InvertedResidual

class FilterPrunerMBNetV2(FilterPruner):
    def parse_dependency(self):
        pass

    def forward(self, x):
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
        self.activations = []
        self.gradients = []
        self.weight_grad = []
        self.grad_index = 0

        self.linear = None
        # activation index to the instance of conv layer
        self.activation_to_conv = {}
        # retrieve next conv using activation index of conv
        self.next_conv = {}
        # retrieve next immediate bn layer using activation index of conv
        self.bn_for_conv = {}
        # Chainning convolutions
        # (use activation index to represent a conv)
        self.chains = {}

        activation_index = 0
        prev_blk_last_conv = -1

        for l1, m1 in enumerate(model.features.children()):
            skipped = False
            if isinstance(m1, InvertedResidual):
                if m1.use_res_connect:
                    skipped = True
                # m1 is nn.Sequential now
                m1 = m1.conv 

            # use for residual
            tmp_x = x 

            # In the beginning of InvertedResidual block, get prev_conv for chaining purpose
            if activation_index-1 >= 0:
                prev_blk_last_conv = activation_index-1

            cnt = 0
            for l2, m2 in enumerate(m1.children()):
                cnt += 1
                x = m2(x)
                h = x.shape[2]
                w = x.shape[3]
                if isinstance(m2, nn.Conv2d):
                    self.conv_in_channels[activation_index] = m2.weight.size(1)
                    self.conv_out_channels[activation_index] = m2.weight.size(0)
                    self.omap_size[activation_index] = (h, w)
                    self.cost_map[activation_index] = h * w * m2.weight.size(2) * m2.weight.size(3)

                    self.in_params[activation_index] = m2.weight.size(1) * m2.weight.size(2) * m2.weight.size(3)
                    self.cur_flops +=  h * w * m2.weight.size(0) * m2.weight.size(1) * m2.weight.size(2) * m2.weight.size(3)

                    # If this is full group_conv it should be bounded with last conv
                    if m2.groups == m2.out_channels and m2.groups == m2.in_channels:
                        assert activation_index-1 not in self.chains, 'Previous conv has already chained to some other convs!'
                        self.chains[activation_index-1] = activation_index
                    if self.rank_type == 'l1_weight':
                        if activation_index not in self.filter_ranks:
                            self.filter_ranks[activation_index] = torch.zeros(m2.weight.size(0), device=self.device)
                        values = (torch.abs(m2.weight.data)).sum(1).sum(1).sum(1)
                        # Normalize the rank by the filter dimensions
                        #values = values / (m2.weight.size(1) * m2.weight.size(2) * m2.weight.size(3))
                        self.filter_ranks[activation_index] = values
                    elif self.rank_type == 'l2_weight': 
                        if activation_index not in self.filter_ranks:
                            self.filter_ranks[activation_index] = torch.zeros(m2.weight.size(0), device=self.device)
                        values = (torch.pow(m2.weight.data, 2)).sum(1).sum(1).sum(1)
                        # Normalize the rank by the filter dimensions
                        # values = values / (m2.weight.size(1) * m2.weight.size(2) * m2.weight.size(3))
                        self.filter_ranks[activation_index] = values
                    elif self.rank_type == 'l2_bn' or self.rank_type == 'l1_bn': 
                        pass
                    else:
                        x.register_hook(self.compute_rank)
                        self.activations.append(x)
                    self.rates[activation_index] = self.conv_in_channels[activation_index] * self.cost_map[activation_index]
                    self.activation_to_conv[activation_index] = m2
                    if activation_index > 0:
                        self.next_conv[activation_index-1] = [activation_index]
                    activation_index += 1
                elif isinstance(m2, nn.BatchNorm2d):
                    # activation-1 since we increased the index right after conv
                    self.bn_for_conv[activation_index-1] = m2
                    if self.rank_type == 'l2_bn': 
                        if activation_index-1 not in self.filter_ranks:
                            self.filter_ranks[activation_index-1] = torch.zeros(m2.weight.size(0), device=self.device)
                        values = torch.pow(m2.weight.data, 2)
                        self.filter_ranks[activation_index-1] = values

                    elif self.rank_type == 'l2_bn_param': 
                        if activation_index-1 not in self.filter_ranks:
                            self.filter_ranks[activation_index-1] = torch.zeros(m2.weight.size(0), device=self.device)
                        values = torch.pow(m2.weight.data, 2)
                        self.filter_ranks[activation_index-1] = values* self.in_params[activation_index-1]

            if cnt == 0:
                x = m1(x)

            # After we parse through the block, if this block is with residual
            if skipped:
                x = tmp_x + x
                if prev_blk_last_conv >= 0:
                    assert prev_blk_last_conv not in self.chains, 'Previous conv has already chained to some other convs!'
                    # activation-1 is the current convolution since we just increased the pointer
                    self.chains[prev_blk_last_conv] = activation_index-1

        for m in model.classifier.modules():
            if isinstance(m, nn.Linear):
                self.linear = m 
                self.base_flops = np.prod(m.weight.shape)
                self.cur_flops += self.base_flops

        self.og_conv_in_channels = self.conv_in_channels.copy()
        self.og_conv_out_channels = self.conv_out_channels.copy()

        self.resource_usage = self.cur_flops

        return model.classifier(x.view(x.size(0), -1))

    def amc_filter_compress(self, layer_id, action, max_sparsity):
        # Chain residual connections
        t = layer_id
        current_chains = []
        while t in self.chains:
            current_chains.append(t)
            t = self.chains[t]
        current_chains.append(t)
        prune_away = int(action*self.conv_out_channels[layer_id])

        for layer in current_chains:
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

        for layer in current_chains:
            self.amc_checked.append(layer)
            self.conv_out_channels[layer] -= prune_away
            next_conv_idx = self.next_conv[layer] if layer in self.next_conv else None

            if next_conv_idx:
                for next_conv_i in next_conv_idx:
                    next_conv = self.activation_to_conv[next_conv_i]
                    if (next_conv.groups != next_conv.out_channels or next_conv.groups != next_conv.in_channels):
                        self.conv_in_channels[next_conv_i] -= prune_away
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

    def mask_conv_layer_segment(self, layer_index, filter_range):
        filters_begin = filter_range[0]
        filters_end = filter_range[1]
        pruned_filters = filters_end - filters_begin + 1
        # Retrive conv based on layer_index
        conv = self.activation_to_conv[layer_index]

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

    def get_valid_filters(self):
        filters_to_prune_per_layer = {}
        visited = []
        for conv_idx in self.activation_to_conv:
            if not conv_idx in visited:
                cur_chain = []
                t = conv_idx
                chain_max_dim = self.activation_to_conv[t].weight.size(0)
                while t in self.chains:
                    num_filters = self.activation_to_conv[t].weight.size(0)
                    chain_max_dim = np.maximum(chain_max_dim, num_filters)
                    cur_chain.append(t)
                    t = self.chains[t]
                cur_chain.append(t)
                visited = visited + cur_chain

                mask = np.zeros(chain_max_dim)
                for t in cur_chain:
                    bn = self.bn_for_conv[t]
                    cur_mask = (torch.abs(bn.weight) > 0).cpu().numpy()
                    mask = np.logical_or(mask, cur_mask)

                inactive_filter = np.where(mask == 0)[0]
                if len(inactive_filter) > 0:
                    for t in cur_chain:
                        filters_to_prune_per_layer[t] = list(inactive_filter.astype(int))
                        if len(inactive_filter) == bn.weight.size(0):
                            filters_to_prune_per_layer[t] = filters_to_prune_per_layer[t][:-2]
                
        return filters_to_prune_per_layer

    def get_valid_flops(self):
        in_channels = self.conv_in_channels.copy()
        out_channels = self.conv_out_channels.copy()
        visited = []
        for conv_idx in self.activation_to_conv:
            if not conv_idx in visited:
                cur_chain = []
                t = conv_idx
                chain_max_dim = self.activation_to_conv[t].weight.size(0)
                while t in self.chains:
                    num_filters = self.activation_to_conv[t].weight.size(0)
                    chain_max_dim = np.maximum(chain_max_dim, num_filters)
                    cur_chain.append(t)
                    t = self.chains[t]
                cur_chain.append(t)
                visited = visited + cur_chain

                mask = np.zeros(chain_max_dim)
                for t in cur_chain:
                    bn = self.bn_for_conv[t]
                    cur_mask = (torch.abs(bn.weight) > 0).cpu().numpy()
                    mask = np.logical_or(mask, cur_mask)

                inactive_filter = np.where(mask == 0)[0]
                if len(inactive_filter) > 0:
                    for t in cur_chain:
                        out_channels[t] -= len(inactive_filter)
                        if len(inactive_filter) == bn.weight.size(0):
                            out_channels[t] = 2

                        next_conv_idx = self.next_conv[t] if t in self.next_conv else None
                        if next_conv_idx:
                            for next_conv_i in next_conv_idx:
                                next_conv = self.activation_to_conv[next_conv_i]
                                if (next_conv.groups != next_conv.out_channels or next_conv.groups != next_conv.in_channels):
                                    in_channels[next_conv_i] = out_channels[t]

        flops = 0
        for k in self.activation_to_conv:
            flops += self.cost_map[k] * in_channels[k] * out_channels[k]
        flops += out_channels[k] * self.num_cls
                
        return flops

    def prune_conv_layer_segment(self, layer_index, filter_range):
        filters_begin = filter_range[0]
        filters_end = filter_range[1]
        pruned_filters = int(filters_end - filters_begin + 1)
        # Retrive conv based on layer_index
        conv = self.activation_to_conv[layer_index]
        next_bn = self.bn_for_conv[layer_index]
        next_conv_idx = self.next_conv[layer_index] if layer_index in self.next_conv else None

        # Surgery on the conv layer to be pruned
        # dw-conv, reduce groups as well
        if conv.groups == conv.out_channels:
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
        # If next is dw-conv, don't bother, since it is chained, so it will be pruned properly
        if next_conv_idx:
            for next_conv_i in next_conv_idx:
                next_conv = self.activation_to_conv[next_conv_i]
                if (next_conv.groups != next_conv.out_channels or next_conv.groups != next_conv.in_channels):
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
