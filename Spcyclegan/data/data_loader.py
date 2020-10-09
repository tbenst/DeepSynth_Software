#########################################################################################
# Copyright 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation. All rights reserved.
# Script for demo the models. 
# Usage: python train.py 
# Author: purdue micro team
# Date: 12/20/2019
#########################################################################################
def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
