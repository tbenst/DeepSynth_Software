# DeepSynth segmentation
This repository contains training and testing code for the deepsynth u-net segmentation data part.

## Installation
This project is built using [PyTorch](https://pytorch.org), a high-level library for Deep Learning. We tested the code under the environment of Ubuntu 18.04 with CUDA-10.0. 

Note: This package has been developed and compiled to run on a Linux machine and will not work with on any other machine that is running a different OS such Windows OS.
 

### Software installation requirements:
Please run the following command to create an virtual environment with required packages using [Anaconda](https://www.anaconda.com):  
`conda env create -f spcyclegan.yml`

or 

We recommend using a virtual environment to install these packages and running the software in.
You can use `pip` to install these packages.
1. Pytorch:
  * torch==1.0.1.post2
  * torchfile==0.1.0
  * torchvision==0.2.2
2. visdom==0.1.8.8
3. numpy==1.16.3
4. scipy==1.2.1
5. scikit-image==0.15.0
5. dominate==2.3.1

## Repository Structure 
This repository is structured in the following way: 
* `checkpoints/`: pretrained model parameters.
* `result/`:example images generated when inferecing the provided pre-trained models.
* `test_data/`: example input images.
* `train_data/`: training data with binary data and synthetic data to train the segmentation models.
*  `spcyclegan.yml`: defines the necessary packages necessary used to generate the virtual environment.

## Dataset
Two data samples are provided as in ./train_data/train_wsm and ./train_data/train_immu
To use the training data, first unzip it from the ./train_data folder
Where gt and syn folders contain the synthetic binary data and synthetic data, respectively.
Testing data are located at ./test_data/WS200 and ./test_data/immu_32


## Running the code

###Training command example:

To use the training data, first unzip it from the ./train_data folder

CUDA_VISIBLE_DEVICES=1 python seg_train.py --dataroot ./train_data/train_wsm --name wsm_example  --phase train

###Testing command example:

python test_all_pad.py --dataroot ./test_data/immu_32 --name immu_BCE_DICE --dataname immu_test --epoch 1000

## Pre-trained models

There are pre-trained models and sample images in './checkpoints' provided in the folder

## References
The SpCycleGAN method in this paper is a modified version of the CycleGAN, for detail of the implementation please check out the original CycleGAN code if you would like to understand more and reproduce the results.
[[Paper]](https://arxiv.org/pdf/1703.10593.pdf)

The code generation also includes references and modification from the implementation from:
[CycleGAN Torch](https://github.com/junyanz/CycleGAN)

### Related Projects and Copyright Information

####[CycleGAN](https://github.com/junyanz/CycleGAN): Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks 
CycleGAN: [[Project]](https://junyanz.github.io/CycleGAN/) [[Paper]](https://arxiv.org/pdf/1703.10593.pdf) [[Torch]](https://github.com/junyanz/CycleGAN) 
--------------------------- LICENSE FOR CycleGAN ---------------------
Copyright (c) 2017, Jun-Yan Zhu and Taesung Park
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#### [pix2pix](https://github.com/phillipi/pix2pix): Image-to-image translation with conditional adversarial nets 
Pix2pix:  [[Project]](https://phillipi.github.io/pix2pix/) [[Paper]](https://arxiv.org/pdf/1611.07004v1.pdf) [[Torch]](https://github.com/phillipi/pix2pix)

####[pytorch-DCGAN](https://github.com/pytorch/examples/tree/master/dcgan).

## Contact
If you have any questions about our work or code, please send us an e-mail at [imart@ecn.purdue.edu](mailto:imart@ecn.purdue.edu).
