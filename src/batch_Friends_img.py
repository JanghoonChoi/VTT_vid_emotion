import sys,os,time,cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from th_utils import *
from utils import imread_to_rgb, crop_img, emo2txt
import matplotlib.pyplot as plt


class Friends_dataset(Dataset):
    def __init__(self, ops):
        self.ops = ops
        self.fr_dict = ops.fr_dict
        self.len = self.ops.iter_step*self.ops.batch_size  #ops.valid_size #13369
        self.win_size = ops.win_size
    
    def __len__(self):
        return self.len
        
    def __getitem__(self, idx):
        ops = self.ops
        af_dict = self.fr_dict
        
        # randomly choose emotion - wrt neutral probability
        rndv = th_rand()
        if rndv < ops.neutral_ratio:
            # neutral
            emo = 6
        else:
            # non neutral, choose again from 0~5
            emo = th_randint(0, 6)
#        emo = th_randint(0, 7)
    
        # label vec
        batch_emo = np.zeros(7)
        batch_emo[emo] = 1
        
        # randomly choose image sample and import
        rndi = th_randint(len(af_dict[emo]))
        sel_dict = af_dict[emo][rndi]
        
        if self.win_size == 1:
            img = imread_to_rgb(ops.friends_path + sel_dict['img'][-1])
            batch_img = img
            batch_img = torch.Tensor( batch_img.transpose(2,0,1).copy() )
            batch_emo = torch.Tensor( batch_emo.copy() )
        else:
            batch_imgs = []
            batch_emos = []
            for i in range(self.win_size):
                img = imread_to_rgb(ops.friends_path + sel_dict['img'][i])
                batch_imgs.append(img)
                batch_emos.append(batch_emo)
                
            batch_img = torch.Tensor( np.array(batch_imgs).transpose(0,3,1,2).copy() )
            batch_emo = torch.Tensor( np.array(batch_emos).copy() )
            
        # returns
        return batch_img, batch_emo
    
    
