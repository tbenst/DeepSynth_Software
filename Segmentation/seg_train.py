#########################################################################################
# Copyright 2020 The Board of Trustees of Purdue University and the Purdue Research Foundation. All rights reserved.
# Script for demo the models. 
# Usage: python train.py 
# Author: purdue micro team
# Date: 1/17/2020
#########################################################################################


from model import *
import numpy as np
import argparse
from dataloader import *
import torch.utils.data
import os
import torchvision.utils as v_utils
import scipy.misc
import visdom
from dice_loss import *
#from boundary_loss import HDDTBinaryLossContour, CannyLossContour


parser = argparse.ArgumentParser()
parser.add_argument("--batchsize",type=int, default=1, help="batch size")
parser.add_argument("--epoch",type=int, default=2000, help="training epoch")
parser.add_argument("--lr",type=float, default=0.0001, help="learning rate")
parser.add_argument("--dataroot",required=True,help="specify data folder")
parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
parser.add_argument("--num_gpu",type=int,default=1,help="number of gpus")
parser.add_argument("--phase",type=str,default="train",help="number of gpus")
parser.add_argument("--num_vol",type=int, default=64, help="num training volume")
parser.add_argument("--continue_train",action='store_true', help='continue training: load the latest model')
parser.add_argument('--start_epoch', type=int, default=1, help='the starting epoch count')
parser.add_argument('--block', type=int, default=64, help='block size')
args = parser.parse_args()

torch.cuda.manual_seed(1)


batchsize = args.batchsize
epoch = args.epoch
lr = args.lr
img_dir = args.dataroot
save_dir = "./checkpoint/" + args.name + "/"

block = args.block
# print(save_dir)

if not os.path.exists(save_dir):
	os.makedirs(save_dir)
if not os.path.exists(save_dir + 'img/'):
    os.makedirs(save_dir + 'img/')
	
#vis = visdom.Visdom(port=8888)

dataset = PairDataset(args)
#dataset = HeatDataset(args)
img_batch = torch.utils.data.DataLoader(dataset,batch_size=batchsize,shuffle=True, num_workers=2)


# unet = nn.DataParallel(encoder_decoder(1,1,32),device_ids=[i for i in range(args.num_gpu)]).cuda()
if args.continue_train:
	unet = torch.load(save_dir + 'unet_' + str(args.start_epoch) + '.pkl').cuda()
    # mapped_state_dict = OrderedDict()
    # for key, value in checkpoint['state_dict'].items():
    #     print(key)
    #     mapped_key = key
    #     mapped_state_dict[mapped_key] = value
    #     if 'running_var' in key:
    #         mapped_state_dict[key.replace('running_var', 'num_batches_tracked')] = torch.zeros(1).to(device)
    # model.load_state_dict(mapped_state_dict)
else:
	unet = nn.DataParallel(encoder_decoder(1,1,8),device_ids=[i for i in range(args.num_gpu)]).cuda()





gen_optimizer = torch.optim.Adam(unet.parameters(),lr=lr)

#loss_cri = nn.MSELoss().cuda()

loss_cri = nn.BCELoss().cuda()

#bound_loss = HDDTBinaryLossContour()
#bound_loss = CannyLossContour()

plot_data = {'X': [], 'Y': []}
plot_data['X'].append(0)
plot_data['Y'].append(0)

# vis.line(
#     X=np.array(plot_data['X']),
#     Y=np.array(plot_data['Y']),
#     opts={'title': 'loss over time','xlabel': 'epoch', 'ylabel': 'loss'},
#     win= 'lossline')


plot_accu = {'X': [], 'Y': []}
plot_accu['X'].append(0)
plot_accu['Y'].append(0)

# vis.line(
#     X=np.array(plot_accu['X']),
#     Y=np.array(plot_accu['Y']),
#     opts={'title': 'accuracy over time','xlabel': 'epoch', 'ylabel': 'accuracy'},
#     win= 'accyline')

for i in range(epoch):
    accu_avg = 0
    accu = 0
    for j, data in enumerate(img_batch):
        gen_optimizer.zero_grad()
        inputs, labels = data
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        y_pred= unet(inputs)
        loss_seg = dice_loss(y_pred, labels).cuda()
        loss_seg_mse = 10 * loss_cri(y_pred, labels).cuda()
        #loss_bound = 0.01 * bound_loss(y_pred, bound).cuda()
        #loss_seg = loss_seg_mse
        loss = loss_seg_mse + loss_seg
        #loss = loss_seg_mse


        loss.backward()

        gen_optimizer.step()
        print("epoch " + str(i + args.start_epoch) + " loss " + str(loss.data) + " loss_dice " + str(loss_seg.data)+ " loss_BCE " + str(loss_seg_mse.data))
        # print("epoch " + str(i) + " loss " + str(loss.data[0]))
        
        org_img = inputs.data.squeeze()
        pred_img = y_pred.data.squeeze()
        tmp_label = labels.data.squeeze()

        # lab_img =labels.cpu().data[0].numpy().squeeze()
        # input_img = inputs.cpu().data[0].numpy().squeeze()


        # new_pred = pred_img[32,:,:].squeeze()
        pred_img[pred_img>= 0.5] = 1
        pred_img[pred_img< 0.5] = 0


        indice_match = torch.sum(torch.eq(pred_img,tmp_label))

        # print (indice_match)
        accu += indice_match/(float(block*block*block))



    accu_avg = accu
    
    plot_data['X'].append(i)
    plot_data['Y'].append(loss.data.cpu())
    # vis.line(
    #     X=np.array(plot_data['X']),
    #     Y=np.array(plot_data['Y']),
    #     opts={'title': 'loss over time','xlabel': 'epoch', 'ylabel': 'loss'},
    #     update='replace',
    #     win= 'lossline')
    

    plot_accu['X'].append(i)
    plot_accu['Y'].append(accu_avg.cpu())
    #vis.image(pred_img[32,:,:].cpu(),opts=dict(title='seg', caption='seg'),win= 'seg')
    #vis.image(tmp_label[32,:,:].cpu(),opts=dict(title='label', caption='label'),win= 'label')
    #vis.image(org_img[32,:,:].cpu(),opts=dict(title='org', caption='org'),win= 'org')


    # vis.line(
    #     X=np.array(plot_accu['X']),
    #     Y=np.array(plot_accu['Y']),
    #     opts={'title': 'accuracy over time','xlabel': 'epoch', 'ylabel': 'accuracy'},
    #     update='replace',
    #     win= 'accyline')


    torch.save(unet, save_dir + 'unet_' + str(args.start_epoch+i+1) + '.pkl')
        



torch.save(unet, save_dir + 'unet.pkl')
