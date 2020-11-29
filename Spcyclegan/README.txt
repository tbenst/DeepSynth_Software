<!---
#Readme for DeepSynth Package 2/10/2020
#Copyright 2020 - The Board of Trustees of Purdue University - All rights reserved
#This is the training and testing code for the deepsynth synthetic data generation part.

#This software is covered by US patents and copyright.
#This source code is to be used for academic research purposes only, and no commercial use is allowed.
-->
# Terms of Use
*****
Term of Use and License
Version 1.1
February 10, 2020


This work was partially supported by a George M. O’Brien Award from the National Institutes of Health under 
grant NIH/NIDDK P30 DK079312 and the endowment of the Charles William Harrison 
Distinguished Professorship at Purdue University.

Copyright and Intellectual Property

The software/code is distributed under Creative Commons license
Attribution-NonCommercial-ShareAlike - CC BY-NC-SA

You are free to:
* Share — copy and redistribute the material in any medium or format
* Adapt — remix, transform, and build upon the material
* The licensor cannot revoke these freedoms as long as you follow the license terms.

Under the following terms:
* Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use. (See below for paper citation)
* NonCommercial — You may not use the material for commercial purposes.
* ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
* No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

For more information see:
https://creativecommons.org/licenses/by-nc-sa/4.0/


The data is distributed under Creative Commons license
Attribution-NonCommercial-NoDerivs - CC BY-NC-ND

You are free to:
* Share — copy and redistribute the material in any medium or format
* The licensor cannot revoke these freedoms as long as you follow the license terms.

Under the following terms:
* Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
* NonCommercial — You may not use the material for commercial purposes.
* NoDerivatives — If you remix, transform, or build upon the material, you may not distribute the modified material.
* No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

For More Information see:
https://creativecommons.org/licenses/by-nc-nd/4.0/



Attribution
In any publications you produce based on using our software/code or data we ask that you cite the following paper:

K. W. Dunn, C. Fu, D. J. Ho, S. Lee, S. Han, P. Salama, and E. J. Delp, "DeepSynth: Three-dimensional nuclear segmentation 
of biological images using neural networks trained with synthetic data," Scientific Reports, Volume 9, 
Article number: 18295, December 2019. DOI: 10.1038/s41598-019-54244-5


Privacy Statement
We are committed to the protection of personal privacy and have adopted a policy to  protect information about individuals. 
When using our software/source code we do not collect any information about you or your use of the code.

How to Contact Us

The source code/software and data is distributed "as is." 
We do not accept any responsibility for the operation of the software.
We can only provided limited help in running our code. 
Note: the software/code developed under Linux. 
We do not do any work with the Windows OS and cannot provide much help in running our software/code on the Windows OS

If you have any questions contact: imart@ecn.purdue.edu


# DeepSynth
This repository contains the code for training/testing the model for synthetic nuclei generation.


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
* `checkpoints/`: pretrained model parameters and example images generated in each epoch of training.
* `data/`: files reading different datasets.
* `datasets/`: example input images.
* `models/`: files defining model structure.
* `options/`: files defining the arguments for training/testing.
* `results/`: generated synthetic images as a result of testing.
* `util/`: helper files to generate html page to view example images generated in each epoch of training.
*  `spcyclegan.yml`: defines the necessary packages necessary used to generate the virtual environment.

### Dataset
Two examples are provided in `./datasets/wsm_128` and `./datasets/immu`, 
where the trainA and trainB subfolders contain the synthetic binary data and original data, respectively.


## Running the code
If you are using Anaconda, please activate the conda environment before running the code.  
`conda activate spcyclegan`

### Training command example:

Please change the commands according to your own testing options.

CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataroot ./datasets/immu --name immu --model cycle_lo_seg --no_dropout --which_model_netG resnet_9blocks --input_nc 1 --output_nc 1 --phase train --identity 0 > log_file/log_file_200114.out &



### Testing command example:
Example 1: to test the provided trained model,
Change the commands according to your own testing options

CUDA_VISIBLE_DEVICES=0 python test.py --dataroot ./datasets/immu --name immu_spcyclegan_package --model test_seg --phase test --which_model_netG resnet_9blocks --no_dropout --display_id 0 --dataset_mode single --which_direction AtoB --which_epoch 200 --loadSize 256 --fineSize 256 --input_nc 1 --output_nc 1 --how_many 512 > log_file/log_file_200114test.out

python train.py --dataroot ./datasets/zfish-b115-25x-0.9x --name zfish-b115-25x-0.9x --model cycle_lo_seg --no_d
ropout --which_model_netG resnet_9blocks --input_nc 1 --output_nc 
1 --phase train --identity 0

Example 2: to test the models generated with the training command example above

CUDA_VISIBLE_DEVICES=0 python test.py --dataroot ./datasets/immu --name immu --model test_seg --phase test --which_model_netG resnet_9blocks --no_dropout --display_id 0 --dataset_mode single --which_direction AtoB --which_epoch 200 --loadSize 256 --fineSize 256 --input_nc 1 --output_nc 1 --how_many 512 > log_file/log_file_200114test.out

python test.py --dataroot ./datasets/zfish-b115-25x-0.9x --name zfish-b115-25x-0.9x --model test_seg --phase test --which_model_netG resnet_9blocks --no_dropout --display_id 0 --dataset_mode single --which_direction AtoB --which_epoch 200 --loadSize 256 --fineSize 256 --input_nc 1 --output_nc 1 --how_many 512

### Pre-trained models

Pre-trained models can be found in `./checkpoints`.

Images generated from each epoch can be found in `./checkpoints/web/images`.

### Example Results from the training and testing command examples
The example results are provided in `./results/immu_spcyclegan/immu/images` and `./results/wsm_spcyclegan/wsm512/images`

## References
The SpCycleGAN method in this paper is a modified version of the CycleGAN method. For implementation details, please check out the original CycleGAN code if you would like to understand more and reproduce the results.
[[Paper]](https://arxiv.org/pdf/1703.10593.pdf)
The code generation also includes references and modification from the implementation from:
[CycleGAN Torch](https://github.com/junyanz/CycleGAN)

### Related Projects and Copyright Information

####[CycleGAN](https://github.com/junyanz/CycleGAN): Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks 
CycleGAN: [[Project]](https://junyanz.github.io/CycleGAN/) [[Paper]](https://arxiv.org/pdf/1703.10593.pdf) [[Torch]](https://github.com/junyanz/CycleGAN) 
--------------------------- LICENSE for CycleGAN ---------------------
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
