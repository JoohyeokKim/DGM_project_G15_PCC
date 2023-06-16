#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:liruihui
@file: train.py 
@time: 2019/09/17
@contact: ruihuili.lee@gmail.com
@github: https://liruihui.github.io/
@description: 
"""
import os
import pprint
pp = pprint.PrettyPrinter()
from datetime import datetime

from Generation.config import opts

if __name__ == '__main__':

    if opts.phase == "train":
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        opts.log_dir = os.path.join(opts.log_dir,current_time)
        if not os.path.exists(opts.log_dir):
            os.makedirs(opts.log_dir)
    #opts.log_dir = "log/20210124-0059"
    print('checkpoints:', opts.log_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu
    print("GPU:",opts.gpu)
    from Generation.model import Model

    model = Model(opts)
    # model.train()
    print("training mode is:", opts.train_test_mode)
    
    model.train_lgan_only_latent()



