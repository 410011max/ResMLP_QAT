import torch.nn as nn
import torch.nn.functional as F

class LRLinear(nn.Module):
    def __init__(self, ratio, in_channel, out_channel, bias=True):
        super().__init__()
        self.bias = bias
        self.lr = (ratio != None) and (ratio != 1.0)
        self.sample_ratio = ratio if self.lr else 1.0
        self.num_components = int(round(self.sample_ratio * min(in_channel, out_channel)))
        if self.lr:
            self.VT = nn.Linear(in_channel, self.num_components, bias=False)
            self.U = nn.Linear(self.num_components, out_channel, bias=bias)
        else:
            self.fc = nn.Linear(in_channel, out_channel, bias=bias)

    def forward(self,x,scaling_factor=None):
        if self.lr:
            x = self.VT(x)
            x = self.U(x)
        else:
            x = self.fc(x)

        return x

class LRLinearSuper(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super().__init__()
        self.bias = bias
        self.num_components = min(in_channel, out_channel)
        self.VT = nn.Linear(in_channel, self.num_components, bias=False)
        self.U = nn.Linear(self.num_components, out_channel, bias=bias)
        self.samples = {}
        self.set_sample_config(1.0)

    def set_sample_config(self, sample_ratio, adaptive_column=False):
        self.sample_ratio = sample_ratio
        self.adaptive_column = adaptive_column
        self._sample_parameters()
        
    def _sample_parameters(self):
        sample_dim = int(round(self.sample_ratio * self.num_components))
        if self.adaptive_column:
            sample_vector = (self.VT.weight.norm(dim=1)*self.U.weight.norm(dim=0)).argsort(descending=True)[:sample_dim]
            self.samples['VT_weight'] = self.VT.weight[sample_vector,:]
            self.samples['U_weight'] = self.U.weight[:,sample_vector]
        else:
            self.samples['VT_weight'] = self.VT.weight[:sample_dim,:]
            self.samples['U_weight'] = self.U.weight[:,:sample_dim]
        if self.bias:
            self.samples['bias'] = self.U.bias

    def forward(self,x,scaling_factor=None):
        x = F.linear(x, self.samples['VT_weight'])
        if self.bias:
            x = F.linear(x, self.samples['U_weight'], self.samples['bias'])
        else:
            x = F.linear(x, self.samples['U_weight'])

        return x
