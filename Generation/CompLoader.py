import numpy as np
# import warnings
# import h5py
from torch.utils.data import Dataset
from glob import glob
# from Common import point_operation
import os
# from torchvision import transforms
# from Common import data_utils as d_utils
# from Common import point_operation
# import torch
import random


class CompLoader(Dataset):
    def __init__(self, root, category='all', mode='train'):
        self.data_root = root
        self.category = category
        self.categories = ['airplane', 'car', 'chair', 'guitar', 'table']
        self.datalist_class = {}
        self.datalist_class_val = {}
        self.phase=mode
        if self.category in self.categories:
            self.fixed_category = True
            self.datalist_class[self.category], self.datalist_class_val[self.category] = self.gather(self.category)
        else:
            self.fixed_category = False
            for c in self.categories:
                self.datalist_class[c], self.datalist_class_val[c] = self.gather(c)

    def __len__(self):
        if self.phase == 'train':
            target_set = self.datalist_class
        else:
            target_set = self.datalist_class_val
        if self.fixed_category == False:
            count = 0
            for c in self.categories:
                count += len(target_set[c])
            return count
        else:
            return len(target_set[self.category])
    def __getitem__(self, idx):
        if self.fixed_category == True:
            category = self.category
        else:
            # category = random.choice(self.categories)
            if self.phase=='train':
                category = self.categories[idx//180]
            else:
                category = self.categories[idx//20]
        if self.phase == 'train':
            chosen = (self.datalist_class[category])[idx%180]
        else:
            chosen = (self.datalist_class_val[category])[idx%20]
        gt = np.load(os.path.join(chosen, 'gt_pc.npy')).astype(np.float32).T
        partial_list = glob(os.path.join(chosen, 'part*'))
        partial = np.load(random.choice(partial_list)).astype(np.float32).T
        # shapegt = len(gt)
        # cat = np.concatenate([gt, partial], axis=0)
        # cat = point_operation.normalize_point_cloud(cat)
        # gt = cat[:shapegt,:].T
        # partial = cat[shapegt:,:].T

        return gt, partial, category
    def gather(self, c):
        folder = os.path.join(self.data_root, c)
        object_list = sorted(glob(os.path.join(folder, "*/models")))
        length = len(object_list)
        return object_list[:180], object_list[180:]

