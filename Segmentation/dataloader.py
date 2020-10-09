#########################################################################################
# Copyright 2020 The Board of Trustees of Purdue University and the Purdue Research Foundation. All rights reserved.
# Script for demo the models. 
# Usage: python train.py 
# Author: purdue micro team
# Date: 1/17/2020
#########################################################################################

import os.path
import torchvision.transforms as transforms
import random
from torch.utils.data.dataset import Dataset
import numpy as np
import torch
from skimage import io
import math

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

class PairDataset(Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.num_vol = opt.num_vol
        self.block = opt.block

        self.A_group = []
        self.B_group = []

        for i in range(self.num_vol):
            self.dir_A = os.path.join(opt.dataroot, opt.phase + '/syn' +str(i+1))
            self.dir_B = os.path.join(opt.dataroot, opt.phase + '/gt'  +str(i+1))

            self.A_paths = make_dataset(self.dir_A)
            self.B_paths = make_dataset(self.dir_B)

            self.A_paths = sorted(self.A_paths)
            self.B_paths = sorted(self.B_paths)

            self.A_group.append(self.A_paths)
            self.B_group.append(self.B_paths)

    def __getitem__(self, index):


        preprocess = transforms.Compose([transforms.ToTensor()])
        A_stack = torch.zeros(self.block,self.block,self.block)
        B_stack = torch.zeros(self.block,self.block,self.block)

        for i in range(self.block):
            A_path_group = self.A_group[index % self.num_vol]
            B_path_group = self.B_group[index % self.num_vol]
            A_path = A_path_group[i]
            B_path = B_path_group[i]
            A = io.imread(A_path)
            B = io.imread(B_path)#/255.0

            

            # A = preprocess(A)
            # B = preprocess(B)

            A_stack[i,:,:] = torch.from_numpy(A)
            B_stack[i,:,:] = torch.from_numpy(B)



        A_stack = A_stack.unsqueeze(0)
        B_stack = B_stack.unsqueeze(0)


        return A_stack,B_stack
            

    def __len__(self):
        return self.num_vol


class HeatDataset(Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.num_vol = opt.num_vol

        self.A_group = []
        self.B_group = []
        self.C_group = []

        for i in range(self.num_vol):
            self.dir_A = os.path.join(opt.dataroot, opt.phase + '/syn' +str(i+1))
            self.dir_B = os.path.join(opt.dataroot, opt.phase + '/gt'  +str(i+1))
            self.dir_C = os.path.join(opt.dataroot, opt.phase + '/bnd'  +str(i+1))

            self.A_paths = make_dataset(self.dir_A)
            self.B_paths = make_dataset(self.dir_B)
            self.C_paths = make_dataset(self.dir_C)

            self.A_paths = sorted(self.A_paths)
            self.B_paths = sorted(self.B_paths)
            self.C_paths = sorted(self.C_paths)

            self.A_group.append(self.A_paths)
            self.B_group.append(self.B_paths)
            self.C_group.append(self.C_paths)

    def __getitem__(self, index):


        preprocess = transforms.Compose([transforms.ToTensor()])
        A_stack = torch.zeros(64,64,64)
        B_stack = torch.zeros(64,64,64)
        C_stack = torch.zeros(64,64,64)

        for i in range(64):
            A_path_group = self.A_group[index % self.num_vol]
            B_path_group = self.B_group[index % self.num_vol]
            C_path_group = self.C_group[index % self.num_vol]

            A_path = A_path_group[i]
            B_path = B_path_group[i]
            C_path = C_path_group[i]

            A = io.imread(A_path)
            B = io.imread(B_path)
            C = io.imread(C_path)
            

            # A = preprocess(A)
            # B = preprocess(B)

            A_stack[i,:,:] = torch.from_numpy(A)
            B_stack[i,:,:] = torch.from_numpy(B)
            C_stack[i,:,:] = torch.from_numpy(C)


        A_stack = A_stack.unsqueeze(0)
        B_stack = B_stack.unsqueeze(0)
        C_stack = C_stack.unsqueeze(0)


        return A_stack,B_stack, C_stack
            

    def __len__(self):
        return self.num_vol


class testDataset(Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.dir_A = os.path.join(opt.dataroot)

        self.A_paths = make_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths)
        self.A_size = len(self.A_paths)

        h,w = torch.from_numpy(io.imread(self.A_paths[0])).shape

        print(str(self.A_size) + " h " + str(h)+ " w " + str(w))
        self.h = int(math.ceil(h / 32.0)) * 32 +32
        self.w = int(math.ceil(w / 32.0)) * 32 +32
        self.z = int(math.ceil(self.A_size / 32.0)) * 32 + 32

        print(str(self.z) + " " + str(self.h) + " " + str(self.w))

        h_vol = ((self.h-32)/32)
        w_vol = ((self.w-32)/32) 
        z_vol = ((self.z-32)/32)

        self.h_vol = h_vol
        self.w_vol = w_vol
        self.z_vol = z_vol

        self.big_stack = torch.zeros(self.z,self.h,self.w)
        self.total_vol = h_vol * w_vol * z_vol
        self.A_stack = []
        A_padded = np.zeros((self.h-32, self.w-32))
        for i in range(self.A_size):
            A = io.imread(self.A_paths[i])
            A_padded[0:h, 0:w] = A 
            self.big_stack[16+i,16:self.h-16,16:self.w-16] = torch.from_numpy(A_padded)

        for k in range(z_vol):
            for j in range(h_vol):
                for i in range(w_vol):
                    self.A_stack.append(self.big_stack[32*k : 32*(k+1) +32 , 32*j : 32*(j+1) +32, 32*i : 32*(i+1)+32])




        # h,w = torch.from_numpy(io.imread(self.A_paths[0])).shape

        # print(str(self.A_size) + " h " + str(h)+ " w " + str(w))
        # self.h = int(math.ceil(h / 32.0)) * 32 +32
        # self.w = int(math.ceil(w / 32.0)) * 32 +32
        # self.z = int(math.ceil(self.A_size / 32.0)) * 32 + 32

        # print(str(self.z) + " " + str(self.h) + " " + str(self.w))

        # h_vol = ((self.h-32)/32)
        # w_vol = ((self.w-32)/32) 
        # z_vol = ((self.z-32)/32)

        # self.h_vol = h_vol
        # self.w_vol = w_vol
        # self.z_vol = z_vol

        # self.big_stack = torch.zeros(self.w,self.h,self.z)
        # self.total_vol = h_vol * w_vol * z_vol
        # self.A_stack = []
        # A_padded = np.zeros((self.w-32,self.h-32))
        # for i in range(self.A_size):
        #     A = io.imread(self.A_paths[i])
        #     A_padded[0:w, 0:h] = A 
        #     self.big_stack[16:self.w-16,16:self.h-16,16+i] = torch.from_numpy(A_padded)

        # for k in range(w_vol):
        #     for j in range(h_vol):
        #         for i in range(z_vol):
        #             self.A_stack.append(self.big_stack[32*k : 32*(k+1) +32 , 32*j : 32*(j+1) +32, 32*i : 32*(i+1)+32])

    def getsize(self):
        return self.z-32, self.h -32, self.w -32, self.z_vol,self.h_vol,self.w_vol

    def __getitem__(self, index):

        A_stack = self.A_stack[index].unsqueeze(0)

        return A_stack
            

    def __len__(self):
        return self.total_vol



class testDataset_all(Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.dir_A = os.path.join(opt.dataroot)

        self.A_paths = make_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths)
        self.A_size = len(self.A_paths)

        self.h,self.w = torch.from_numpy(io.imread(self.A_paths[0])).shape

        print(str(self.A_size) + " h " + str(self.h)+ " w " + str(self.w))

        self.big_stack = torch.zeros(self.A_size,self.h,self.w)

        for i in range(self.A_size):
            A = io.imread(self.A_paths[i])
            self.big_stack[i,:,:] = torch.from_numpy(A)
            
    def getsize(self):
        return self.A_size, self.h, self.w

    def __getitem__(self, index):

        return self.big_stack
            

    def __len__(self):
        return 1