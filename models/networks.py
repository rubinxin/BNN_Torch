import torch
import torch.nn as nn
import numpy as np

class AppendLayer(nn.Module):
    def __init__(self, noise=1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_var = nn.Parameter(torch.FloatTensor(1, 1))

        nn.init.constant_(self.log_var, val=np.log(noise))

    def forward(self, x):
        return torch.cat((x, self.log_var * torch.ones_like(x)), dim=1)

class Net(nn.Module):
    def __init__(self, n_inputs, dropout_p, n_units=[50, 50, 50], actv='tanh', hetero=False, noise=1e-3):
        super(Net, self).__init__()
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.fc1 = nn.Linear(n_inputs, n_units[0])
        self.fc2 = nn.Linear(n_units[0], n_units[1])
        self.fc3 = nn.Linear(n_units[1], n_units[2])
        self.out1 = nn.Linear(n_units[2], 1)
        self.out2 = nn.Linear(n_units[2], 1)
        self.out3 = AppendLayer(noise=noise)

        self.hetero = hetero

        if actv == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()


    def forward(self, x):

        x = self.activation(self.fc1(x))
        x = self.dropout(x)

        x = self.activation(self.fc2(x))
        x = self.dropout(x)

        x = self.activation(self.fc3(x))

        x = self.dropout(x)

        if self.hetero:
            mean = self.out1(x)
            log_var = self.out2(x)
            return torch.cat((mean, log_var), dim=1)
        else:
            x = self.out1(x)
            return self.out3(x)

def negative_log_likelihood(output, true):
    mean, log_var = output[:, 0].view((-1, 1)), output[:, 1].view((-1, 1))
    precision = torch.exp(-log_var)
    nll = - torch.mean(torch.sum(- 0.5*precision * (true - mean)**2 - 0.5 * log_var, 1), 0)
    return nll
