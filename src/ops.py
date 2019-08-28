import sys,os,time
import numpy as np

# config params
img_sz = 224               # batch image size

# train params
do_train = True
from_chkpt = False         # init from checkpoint / init from scratch
make_valid = False         # generate new validation set
chkpt_path = '../../chkpts/'

batch_size = 16
valid_size = 500*batch_size
neutral_ratio = 0.5        # proportion of neutral samples in batch
aug_prob = 0.5             # probability of data augmentation
aug_crop = 0.1             # shift/resize augmentation

epoc_step = int(1e+1)
iter_step = int(1e+6)
save_step = int(1e+4)
errd_step = 50
vald_step = 10*errd_step
maxn_chkpt = 50

learning_rate = 1e-4
w_decay       = 1e-5
loss_lambda   = 0.5

# affectnet/friends path
if do_train:
    affectnet_path = '/home/jhchoi/datasets3/affectnet/'
    af_dict = np.load('../../dicts/affectnet_parsed.npy', allow_pickle=True).item()
    friends_path = '/home/jhchoi/datasets3/friends_p/'
    fr_dict = np.load('../../dicts/friends_parsed.npy', allow_pickle=True).item()

# test params
