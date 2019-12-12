import sys,os,time
import numpy as np

# config params
img_sz = 224               # batch image size

# train params
train_mode = True
from_chkpt = False         # init from checkpoint / init from scratch
make_valid = False          # generate new validation set
chkpt_path = '../../chkpts/'
weight_path = '../../weights/'

batch_size = 4
valid_size = 500*batch_size
neutral_ratio = 0.33        # proportion of neutral samples in batch
aug_marg = 0.50             # bounding box margin param
aug_prob = 0.10             # probability of data augmentation
aug_crop = 0.10             # shift/resize augmentation
win_size = 4                # 1 for static image, 4 for sequence

epoc_step = int(1e+1)
iter_step = int(1e+6)
save_step = int(1e+4)
errd_step = 50
vald_step = 10*errd_step
maxn_chkpt = 50

learning_rate = 1e-4
w_decay       = 5e-5
loss_lambda   = 0.5

# affectnet/friends path
if train_mode:
    affectnet_path = '/home/jhchoi/datasets3/affectnet/'
    af_dict = np.load('../../dicts/affectnet_parsed.npy', allow_pickle=True).item()
    raf_path = '/home/jhchoi/datasets4/RAF/'
    raf_dict = np.load('../../dicts/raf_parsed.npy', allow_pickle=True).item()
    friends_path = '/home/jhchoi/datasets2/friends/'
    fr_dict = np.load('../../dicts/friends_parsed_new.npy', allow_pickle=True).item()
    fer_path = '/home/jhchoi/datasets4/FER/'
    fer_dict = np.load('../../dicts/fer_parsed.npy', allow_pickle=True).item()

# test params
