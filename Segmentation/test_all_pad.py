from model import *
import numpy as np
import argparse
from dataloader import *
import torch.utils.data
import os
import torchvision.utils as v_utils
import scipy.misc
import visdom
import torch.nn.functional as F
import math
import time
from collections import OrderedDict

from functools import partial
import pickle
pickle.load = partial(pickle.load, encoding="latin1")

pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm3d):
        module.track_running_stats= True
    else:
        for name, module1 in module._modules.items():
            module1 = recursion_change_bn(module1)

def recursion_change_bn2(module):
    if isinstance(module, torch.nn.BatchNorm3d):
        module.num_batches_tracked= True
    else:
        for name, module1 in module._modules.items():
            module1 = recursion_change_bn2(module1)

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot",required=True,help="specify data folder")
parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--dataname', required=True, help="specify data name")
parser.add_argument('--epoch', required=True, help="specify epoch")
parser.add_argument('--block', type=int, default=64, help="block size")

args = parser.parse_args()

batchsize = 1
img_dir = args.dataroot
save_dir = "./result/" + args.dataname + "/" + args.name + "/"
model_name = "./checkpoint/" + args.name + "/unet_" +args.epoch +".pkl"
block = args.block

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(save_dir + args.epoch+ '/img/'):
    os.makedirs(save_dir + args.epoch+ '/img/')

dataset = testDataset_all(args)
img_batch = torch.utils.data.DataLoader(dataset,batch_size=batchsize,shuffle=False, num_workers=2)

#unet = torch.load(model_name)
unet = torch.load(model_name, map_location=lambda storage, loc: storage, pickle_module=pickle)
unet.cuda()
model = unet
for name, module in model._modules.items():
    recursion_change_bn(model)
for name, module in model._modules.items():
    recursion_change_bn2(model)
#unet.cuda()

print(sum(p.numel() for p in unet.parameters() if p.requires_grad))
print(sum(p.numel() for p in unet.parameters()))

size_z, size_h, size_w = dataset.getsize()

w_p = int(math.ceil(size_w/float(block/2))*(block/2))
h_p = int(math.ceil(size_h/float(block/2))*(block/2))
z_p = int(math.ceil(size_z/float(block/2))*(block/2))

padz = z_p-size_z
padh = h_p-size_h
padw = w_p-size_w

print("x: " + str(h_p) + ", y: " + str(w_p) + ", z: "+ str(z_p))

tic = time.clock()

for i, data in enumerate(img_batch):

        inputs = Variable(data).cuda()


        # inputs_p = np.zeros([1,1,z_p,h_p,w_p])

        input_numpy = inputs.data[0].cpu().numpy()
        input_padded = np.pad(input_numpy, ((0, padz), (0, padh), (0, padw)), 'reflect')
        input_inference = np.pad(input_padded, ((int(block/4),int(block/4)), (int(block/4), int(block/4)), (int(block/4), int(block/4))), 'reflect')

        # inputs_p[0,0,0:size_z,0:size_h,0:size_w] = inputs.data[0].cpu().numpy()

        # inputs_pad = np.zeros([1,1,z_p+(block/2),h_p+(block/2),w_p+(block/2)])

        inputs_sub = np.zeros([1,1,block,block,block])

        output = np.zeros([z_p,h_p,w_p])



        # inputs_pad[0,0,(block/4):z_p+(block/4),(block/4):h_p+(block/4),(block/4):w_p+(block/4)] = inputs_p


        for kk in range(0,int(w_p/(block/2))):

            for jj in range(0,int(h_p/(block/2))):

                for ii in range(0,int(z_p/(block/2))):

                    inputs_sub[0,0,:,:,:] = input_inference[int(block/2)*ii:int(block/2)*(ii+2),int(block/2)*jj:int(block/2)*(jj+2),int(block/2)*kk:int(block/2)*(kk+2)]

                    inputs_sub = torch.from_numpy(inputs_sub).float()

                    inputs_sub = inputs_sub.cuda()

                    inputs_sub = Variable(inputs_sub)

                    output_sub = model(inputs_sub)

                    output_sub = torch.squeeze(output_sub)
                    
                    output_sub = output_sub.data.cpu().numpy()
                    output_sub[output_sub > 0.5] = 1

                    output_sub[output_sub <= 0.5] = 0

                    output[int(block/2)*ii:int(block/2)*(ii+1),int(block/2)*jj:int(block/2)*(jj+1),int(block/2)*kk:int(block/2)*(kk+1)] = output_sub[int(block/4):int(3*block/4),int(block/4):int(3*block/4),int(block/4):int(3*block/4)]

                    inputs_sub = inputs_sub.data.cpu().numpy()
                    # print("x: " + str(ii) + ", y: " + str(jj) + ", z: "+ str(kk))


toc = time.clock()
print(toc - tic)

for i in range(size_z):
    scipy.misc.imsave(save_dir+args.epoch+'/img/' + str(i+1) + '.png', output[i,:,:])




   
