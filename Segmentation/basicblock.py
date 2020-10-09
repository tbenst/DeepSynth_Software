#########################################################################################
# Copyright 2020 The Board of Trustees of Purdue University and the Purdue Research Foundation. All rights reserved.
# Script for demo the models. 
# Usage: python train.py 
# Author: purdue micro team
# Date: 1/17/2020
#########################################################################################

import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable


def conv_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv3d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model


def conv_trans_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.ConvTranspose3d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model

def build_conv_block(dim):
    conv_block = []
    conv_block += [nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1),
                   nn.BatchNorm3d(dim),
                   nn.LeakyReLU(0.2, inplace=True)]

    conv_block += [nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1),
                   nn.BatchNorm3d(dim)]

    return nn.Sequential(*conv_block)

    

def maxpool():
    pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_2(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block(in_dim,out_dim,act_fn),
        nn.Conv3d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
    )
    return model    


def conv_block_3(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block(in_dim,out_dim,act_fn),
        conv_block(out_dim,out_dim,act_fn),
        nn.Conv3d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
    )
    return model
