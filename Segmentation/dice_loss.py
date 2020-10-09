#########################################################################################
# Copyright 2020 The Board of Trustees of Purdue University and the Purdue Research Foundation. All rights reserved.
# Script for demo the models. 
# Usage: python train.py 
# Author: purdue micro team
# Date: 1/17/2020
#########################################################################################


import torch
from torch.autograd import Function
from itertools import repeat
import numpy as np
import torch.nn.functional as F
# Intersection = dot(A, B)
# Union = dot(A, A) + dot(B, B)
# The Dice loss function is defined as
# 1/2 * intersection / union
#
# The derivative is 2[(union * target - 2 * intersect * input) / union^2]


def dice_loss(input,target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    # 
    input = input.squeeze(1)
    target = target.squeeze(1)
    # print input.size()
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    uniques=np.unique(target.cpu().data[0].numpy())
    assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

    # probs=F.softmax(input)

    # print(input)
    # print(probs)

    probs = input
    num=probs*target#b,c,h,w--p*g
    num=torch.sum(num,dim=3)#b,c,h
    num=torch.sum(num,dim=2)
    num=torch.sum(num,dim=1)
    

    den1=probs*probs#--p^2
    den1=torch.sum(den1,dim=3)#b,c,h
    den1=torch.sum(den1,dim=2)
    den1=torch.sum(den1,dim=1)
    

    den2=target*target#--g^2
    den2=torch.sum(den2,dim=3)#b,c,h
    den2=torch.sum(den2,dim=2)#b,c
    den2=torch.sum(den2,dim=1)
    

    dice=2*(num/(den1+den2))
    # print dice.size()
    dice_eso=dice#we ignore bg dice val, and take the fg

    dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz

    return dice_total


# class DiceLoss(Function):
#     def __init__(self, *args, **kwargs):
#         pass

#     def forward(self, input, target, save=True):
#         if save:
#             self.save_for_backward(input, target)
#         eps = 0.000001
#         _, result_ = input.max(1)
#         result_ = torch.squeeze(result_)
#         if input.is_cuda:
#             result = torch.cuda.FloatTensor(result_.size())
#             self.target_ = torch.cuda.FloatTensor(target.size())
#         else:
#             result = torch.FloatTensor(result_.size())
#             self.target_ = torch.FloatTensor(target.size())
#         result.copy_(result_)
#         self.target_.copy_(target)
#         target = self.target_

#         intersect = torch.dot(result, target)
#         # binary values so sum the same as sum of squares
#         result_sum = torch.sum(result)
#         target_sum = torch.sum(target)
#         union = result_sum + target_sum + (2*eps)

#         # the target volume can be empty - so we still want to
#         # end up with a score of 1 if the result is 0/0
#         IoU = intersect / union
#         print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
#             union, intersect, target_sum, result_sum, 2*IoU))
#         out = torch.FloatTensor(1).fill_(2*IoU)
#         self.intersect, self.union = intersect, union
#         return out

#     def backward(self, grad_output):
#         input, _ = self.saved_tensors
#         intersect, union = self.intersect, self.union
#         target = self.target_
#         gt = torch.div(target, union)
#         IoU2 = intersect/(union*union)
#         pred = torch.mul(input[:, 1], IoU2)
#         dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
#         grad_input = torch.cat((torch.mul(dDice, -grad_output[0]),
#                                 torch.mul(dDice, grad_output[0])), 0)
#         return grad_input , None

# def dice_loss(input, target):
#     return DiceLoss()(input, target)

# def dice_error(input, target):
#     eps = 0.000001
#     _, result_ = input.max(1)
#     result_ = torch.squeeze(result_)
#     if input.is_cuda:
#         result = torch.cuda.FloatTensor(result_.size())
#         target_ = torch.cuda.FloatTensor(target.size())
#     else:
#         result = torch.FloatTensor(result_.size())
#         target_ = torch.FloatTensor(target.size())
#     result.copy_(result_.data)
#     target_.copy_(target.data)
#     target = target_
    
#     intersect = torch.dot(result, target)

#     result_sum = torch.sum(result)
#     target_sum = torch.sum(target)
#     union = result_sum + target_sum + 2*eps
#     intersect = np.max([eps, intersect])
#     # the target volume can be empty - so we still want to
#     # end up with a score of 1 if the result is 0/0
#     IoU = intersect / union
# #    print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
# #        union, intersect, target_sum, result_sum, 2*IoU))
#     return 2*IoU