import sys,os,time,cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from th_utils import *
from utils import imread_to_rgb, crop_img, emo2txt
import matplotlib.pyplot as plt



class FER_Dataset(Dataset):
    def __init__(self, ops):
        self.ops = ops
        self.fer_dict = ops.fer_dict
        self.len = len(self.fer_dict.keys())
    
    def __len__(self):
        return self.len
        
    def __getitem__(self, idx):
        ops = self.ops
        fer_dict = self.fer_dict

        sel_dict = fer_dict[idx]
        batch_emo = np.zeros(7)
        batch_emo[sel_dict['em']] = 1.
        batch_img = np.expand_dims(sel_dict['img'], -1).repeat(3, -1)/255.
        
        # returns
        batch_img = torch.Tensor( batch_img.transpose(2,0,1).copy() )
        batch_emo = torch.Tensor( batch_emo.copy() )
        
        return batch_img, batch_emo
    
    

