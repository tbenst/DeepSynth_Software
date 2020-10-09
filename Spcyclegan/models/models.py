#########################################################################################
# Copyright 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation. All rights reserved.
# Script for demo the models. 
# Usage: python train.py 
# Author: purdue micro team
# Date: 12/20/2019
#########################################################################################

def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'cycle_lo':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_lo_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'cycle_lo_seg':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_lo_seg_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'cycle_lo_seg_3D':
        assert(opt.dataset_mode == 'unaligned_3D')
        from .cycle_lo_seg_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'cycle_lo_seg_couple':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_lo_seg_couple_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'cycle_lo_heat':
        assert(opt.dataset_mode == 'unaligned_heat')
        from .cycle_lo_heat_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'cycle_lo_heat_separate':
        assert(opt.dataset_mode == 'unaligned_heat')
        from .cycle_lo_heat_separate_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    elif opt.model == 'test_seg':
        assert(opt.dataset_mode == 'single')
        from .test_seg_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
