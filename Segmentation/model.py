#########################################################################################
# Copyright 2020 The Board of Trustees of Purdue University and the Purdue Research Foundation. All rights reserved.
# Script for demo the models. 
# Usage: python train.py 
# Author: purdue micro team
# Date: 1/17/2020
#########################################################################################


from basicblock import *
import torch.nn.functional as F

class encoder_decoder(nn.Module):
	def __init__(self, in_dim, out_dim, num_filter):
		super(encoder_decoder, self).__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.num_filter = num_filter
		act_fn = nn.LeakyReLU(0.2, inplace=True)

		print("\n--------Initialing encoder_decoder---------\n")

		self.down_1 = conv_block_2(self.in_dim,self.num_filter,act_fn)
		self.pool_1 = maxpool()
		self.down_2 = conv_block_2(self.num_filter*1,self.num_filter*2,act_fn)
		self.pool_2 = maxpool()
		self.down_3 = conv_block_2(self.num_filter*2,self.num_filter*4,act_fn)
		self.pool_3 = maxpool()
		# self.down_4 = conv_block_2(self.num_filter*4,self.num_filter*8,act_fn)
		# self.pool_4 = maxpool()

		self.bridge = conv_block_2(self.num_filter*4,self.num_filter*8,act_fn)

		# self.trans_1 = conv_trans_block(self.num_filter*16,self.num_filter*8,act_fn)
		# self.up_1 = conv_block_2(self.num_filter*16,self.num_filter*8,act_fn)
		self.trans_2 = conv_trans_block(self.num_filter*8,self.num_filter*4,act_fn)
		self.up_2 = conv_block_2(self.num_filter*8,self.num_filter*4,act_fn)
		self.trans_3 = conv_trans_block(self.num_filter*4,self.num_filter*2,act_fn)
		self.up_3 = conv_block_2(self.num_filter*4,self.num_filter*2,act_fn)
		self.trans_4 = conv_trans_block(self.num_filter*2,self.num_filter*1,act_fn)
		self.up_4 = conv_block_2(self.num_filter*2,self.num_filter*1,act_fn)

		self.out = nn.Sequential(
			nn.Conv3d(self.num_filter,self.out_dim,3,1,1),
			nn.Sigmoid(),
		)

	def forward(self, input):
		down_1 = self.down_1(input)
		pool_1 = self.pool_1(down_1)
		down_2 = self.down_2(pool_1)
		pool_2 = self.pool_2(down_2)
		down_3 = self.down_3(pool_2)
		pool_3 = self.pool_3(down_3)
		# down_4 = self.down_4(pool_3)
		# pool_4 = self.pool_4(down_4)

		bridge = self.bridge(pool_3)

		# trans_1 = self.trans_1(bridge)
		# concat_1 = torch.cat([trans_1,down_4],dim=1)
		# up_1 = self.up_1(concat_1)

		trans_2 = self.trans_2(bridge)
		concat_2 = torch.cat([trans_2,down_3],dim=1)
		up_2 = self.up_2(concat_2)
		trans_3 = self.trans_3(up_2)
		concat_3 = torch.cat([trans_3,down_2],dim=1)
		up_3 = self.up_3(concat_3)
		trans_4 = self.trans_4(up_3)
		concat_4 = torch.cat([trans_4,down_1],dim=1)
		up_4 = self.up_4(concat_4)

		out = self.out(up_4)



		return out




class unet_resnet(nn.Module):
	def __init__(self, in_dim, out_dim, num_filter):
		super(unet_resnet, self).__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.num_filter = num_filter
		act_fn = nn.LeakyReLU(0.2, inplace=True)

		print("\n--------Initialing encoder_decoder---------\n")

		self.down_1 = conv_block_2(self.in_dim,self.num_filter,act_fn)
		self.pool_1 = maxpool()
		self.down_2 = conv_block_2(self.num_filter*1,self.num_filter*2,act_fn)
		self.pool_2 = maxpool()
		self.down_3 = conv_block_2(self.num_filter*2,self.num_filter*4,act_fn)
		self.pool_3 = maxpool()
		self.down_4 = conv_block_2(self.num_filter*4,self.num_filter*8,act_fn)
		self.pool_4 = maxpool()

		self.resblock1 = build_conv_block(self.num_filter)
		self.resblock2 = build_conv_block(self.num_filter*2)
		self.resblock3 = build_conv_block(self.num_filter*4)
		self.resblock4 = build_conv_block(self.num_filter*8)

		self.bridge = conv_block_2(self.num_filter*8,self.num_filter*16,act_fn)

		self.trans_1 = conv_trans_block(self.num_filter*16,self.num_filter*8,act_fn)
		self.up_1 = conv_block_2(self.num_filter*16,self.num_filter*8,act_fn)
		self.trans_2 = conv_trans_block(self.num_filter*8,self.num_filter*4,act_fn)
		self.up_2 = conv_block_2(self.num_filter*8,self.num_filter*4,act_fn)
		self.trans_3 = conv_trans_block(self.num_filter*4,self.num_filter*2,act_fn)
		self.up_3 = conv_block_2(self.num_filter*4,self.num_filter*2,act_fn)
		self.trans_4 = conv_trans_block(self.num_filter*2,self.num_filter*1,act_fn)
		self.up_4 = conv_block_2(self.num_filter*2,self.num_filter*1,act_fn)

		self.act_res = nn.LeakyReLU(0.2, inplace=True)

		self.out = nn.Sequential(
			nn.Conv3d(self.num_filter,self.out_dim,3,1,1),
			nn.Sigmoid(),
		)

	def forward(self, input):
		down_1 = self.down_1(input)
		pool_1 = self.pool_1(down_1)
		down_2 = self.down_2(pool_1)
		pool_2 = self.pool_2(down_2)
		down_3 = self.down_3(pool_2)
		pool_3 = self.pool_3(down_3)
		down_4 = self.down_4(pool_3)
		pool_4 = self.pool_4(down_4)

		resblock1 = self.resblock1(down_1) + down_1
		resblock1 = self.act_res(resblock1)

		resblock2 = self.resblock2(down_2) + down_2
		resblock2 = self.act_res(resblock2)

		resblock3 = self.resblock3(down_3) + down_3
		resblock3 = self.act_res(resblock3)

		resblock4 = self.resblock4(down_4) + down_4
		resblock4 = self.act_res(resblock4)

		bridge = self.bridge(pool_4)

		trans_1 = self.trans_1(bridge)
		concat_1 = torch.cat([trans_1,resblock4],dim=1)
		up_1 = self.up_1(concat_1)
		trans_2 = self.trans_2(up_1)
		concat_2 = torch.cat([trans_2,resblock3],dim=1)
		up_2 = self.up_2(concat_2)
		trans_3 = self.trans_3(up_2)
		concat_3 = torch.cat([trans_3,resblock2],dim=1)
		up_3 = self.up_3(concat_3)
		trans_4 = self.trans_4(up_3)
		concat_4 = torch.cat([trans_4,resblock1],dim=1)
		up_4 = self.up_4(concat_4)

		out = self.out(up_4)



		return out




class encoder_decoder_heat(nn.Module):
	def __init__(self, in_dim, out_dim, num_filter):
		super(encoder_decoder_heat, self).__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.num_filter = num_filter
		act_fn = nn.LeakyReLU(0.2, inplace=True)

		print("\n--------Initialing encoder_decoder---------\n")

		self.down_1 = conv_block_2(self.in_dim,self.num_filter,act_fn)
		self.pool_1 = maxpool()
		self.down_2 = conv_block_2(self.num_filter*1,self.num_filter*2,act_fn)
		self.pool_2 = maxpool()
		self.down_3 = conv_block_2(self.num_filter*2,self.num_filter*4,act_fn)
		self.pool_3 = maxpool()
		self.down_4 = conv_block_2(self.num_filter*4,self.num_filter*8,act_fn)
		self.pool_4 = maxpool()

		self.bridge = conv_block_2(self.num_filter*8,self.num_filter*16,act_fn)

		self.trans_1 = conv_trans_block(self.num_filter*16,self.num_filter*8,act_fn)
		self.up_1 = conv_block_2(self.num_filter*16,self.num_filter*8,act_fn)
		self.trans_2 = conv_trans_block(self.num_filter*8,self.num_filter*4,act_fn)
		self.up_2 = conv_block_2(self.num_filter*8,self.num_filter*4,act_fn)
		self.trans_3 = conv_trans_block(self.num_filter*4,self.num_filter*2,act_fn)
		self.up_3 = conv_block_2(self.num_filter*4,self.num_filter*2,act_fn)
		self.trans_4 = conv_trans_block(self.num_filter*2,self.num_filter*1,act_fn)
		self.up_4 = conv_block_2(self.num_filter*2,self.num_filter,act_fn)


		self.up_5 = conv_block_2(self.num_filter,self.num_filter*2,act_fn)
		self.up_6 = conv_block_2(self.num_filter*2,self.num_filter*1,act_fn)


		self.segout = nn.Sequential(
			nn.Conv3d(self.num_filter,1,3,1,1),
			nn.Sigmoid(),
		)

		self.heatout = nn.Sequential(
			nn.Conv3d(self.num_filter,1,3,1,1),
			nn.Tanh(),
		)

	def forward(self, input):
		down_1 = self.down_1(input)
		pool_1 = self.pool_1(down_1)
		down_2 = self.down_2(pool_1)
		pool_2 = self.pool_2(down_2)
		down_3 = self.down_3(pool_2)
		pool_3 = self.pool_3(down_3)
		down_4 = self.down_4(pool_3)
		pool_4 = self.pool_4(down_4)

		bridge = self.bridge(pool_4)

		trans_1 = self.trans_1(bridge)
		concat_1 = torch.cat([trans_1,down_4],dim=1)
		up_1 = self.up_1(concat_1)
		trans_2 = self.trans_2(up_1)
		concat_2 = torch.cat([trans_2,down_3],dim=1)
		up_2 = self.up_2(concat_2)
		trans_3 = self.trans_3(up_2)
		concat_3 = torch.cat([trans_3,down_2],dim=1)
		up_3 = self.up_3(concat_3)
		trans_4 = self.trans_4(up_3)
		concat_4 = torch.cat([trans_4,down_1],dim=1)
		up_4 = self.up_4(concat_4)

		segout = self.segout(up_4)

		# up_5 = self.up_5(up_4)
		# up_6 = self.up_6(up_5)

		# heatout = self.heatout(up_6)



		return segout




class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.conv11 = nn.Conv3d(1, 8, 3, stride=1, padding=1)
		self.bn11 = nn.BatchNorm3d(8)
		self.conv12 = nn.Conv3d(8, 8, 3, stride=1, padding=1)
		self.bn12 = nn.BatchNorm3d(8)
#		self.pool1 = nn.MaxPool3d(2, stride=2, return_indices=True)

		self.conv21 = nn.Conv3d(8, 8, 3, stride=1, padding=1)
		self.bn21 = nn.BatchNorm3d(8)
		self.conv22 = nn.Conv3d(8, 8, 3, stride=1, padding=1)
		self.bn22 = nn.BatchNorm3d(8)
#		self.pool2 = nn.MaxPool3d(2, stride=2, return_indices=True)

		self.conv31 = nn.Conv3d(8, 8, 3, stride=1, padding=1)
		self.bn31 = nn.BatchNorm3d(8)
		self.conv32 = nn.Conv3d(8, 8, 3, stride=1, padding=1)
		self.bn32 = nn.BatchNorm3d(8)

#		self.unpool2 = nn.MaxUnpool3d(2, stride=2)
		self.conv41 = nn.Conv3d(8, 8, 3, stride=1, padding=1)
		self.bn41 = nn.BatchNorm3d(8)
		self.conv42 = nn.Conv3d(8, 8, 3, stride=1, padding=1)
		self.bn42 = nn.BatchNorm3d(8)

#		self.unpool1 = nn.MaxUnpool3d(2, stride=2)
		self.conv51 = nn.Conv3d(8, 8, 3, stride=1, padding=1)
		self.bn51 = nn.BatchNorm3d(8)
		self.conv52 = nn.Conv3d(8, 1, 3, stride=1, padding=1)
		self.bn52 = nn.BatchNorm3d(1)

	def forward(self, x):
		x = F.relu(self.bn11(self.conv11(x)))
		x = F.relu(self.bn12(self.conv12(x)))
		size1 = x.size()
		x,ind1 = F.max_pool3d(x, kernel_size=2, stride=2, return_indices=True)
#		x,ind1 = self.pool1(F.relu(self.bn12(self.conv12(x))))

		x = F.relu(self.bn21(self.conv21(x)))
		x = F.relu(self.bn22(self.conv22(x)))
		size2 = x.size()
		x,ind2 = F.max_pool3d(x, kernel_size=2, stride=2, return_indices=True)
#		x,ind2 = self.pool2(F.relu(self.bn22(self.conv22(x))))

		x = F.relu(self.bn31(self.conv31(x)))
		x = F.relu(self.bn32(self.conv32(x)))

		x = F.max_unpool3d(x, ind2, kernel_size=2, stride=2, output_size=size2)
#		x = F.relu(self.bn41(self.conv41(self.unpool2(x,ind2))))
		x = F.relu(self.bn41(self.conv41(x)))
		x = F.relu(self.bn42(self.conv42(x)))

		x = F.max_unpool3d(x, ind1, kernel_size=2, stride=2, output_size=size1)
#		x = F.relu(self.bn51(self.conv51(self.unpool1(x,ind1))))
		x = F.relu(self.bn51(self.conv51(x)))
		x = F.tanh(self.bn52(self.conv52(x)))

		return x