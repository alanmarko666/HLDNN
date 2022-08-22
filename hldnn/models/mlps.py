import numpy as np

import torch.nn as nn
from torch.nn import LeakyReLU

def ProgressiveSequence(start, end, layers):
    return np.linspace(
        start, end,
        layers + 1,
        dtype='int'
    )


def ProgressiveMLP(start, end, layers_n, homogeneous=False,last_activation=nn.LeakyReLU, activation=nn.LeakyReLU):
    if (layers_n == 0):
        return nn.Identity()

    size_list = ProgressiveSequence(start, end, layers_n)
    return CreateMLP(
        size_list=size_list,
        last_activation=last_activation, activation=activation, bias=not homogeneous
    )

def WideMLP(in_size,out_size,middle,n_mid_layers,homogeneous=False,last_activation=nn.LeakyReLU, activation=nn.LeakyReLU):
    size_list=[in_size]+[middle for i in range(n_mid_layers)]+[out_size]
    return CreateMLP(
        size_list=size_list,
        last_activation=last_activation, activation=activation, bias=not homogeneous
    )


def CreateMLP(size_list, bias=False,last_activation=nn.LeakyReLU, activation=nn.LeakyReLU, dropout_p=0):
    return nn.Sequential(
        *(
            nn.Sequential(
                nn.Linear(size_list[i_lay], size_list[i_lay + 1], bias=(bias if i_lay != len(size_list) - 2 else bias)),
                #nn.Dropout(dropout_p) if i_lay != len(size_list) - 2 else nn.Identity(),
                activation() if i_lay != len(size_list) - 2 else last_activation()
            )
            for i_lay in range(len(size_list) - 1)
        )
    )