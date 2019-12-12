import sys,os,time,cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#torch.multiprocessing.set_sharing_strategy('file_system')

import ops
from resnet_tsm import resnet18 as resnet
from loss_def import emo_loss
from batch_Friends_img import Friends_dataset
from utils import get_dtstr

if not ops.train_mode:
    print 'not in training mode, exit'
    sys.exit()
    


# In[ ]:


# define dataloader
emo_dataset = Friends_dataset(ops)
emo_loader = DataLoader(dataset=emo_dataset, batch_size=ops.batch_size, num_workers=8)

# define resent
net = resnet().cuda()
loss_eval = emo_loss(ops.loss_lambda).cuda()

# define optim
optim = torch.optim.Adam(net.parameters(), lr=ops.learning_rate, weight_decay=ops.w_decay, amsgrad=True)

# init
ckpt = torch.load(ops.weight_path+'/resnet18_init_weights.tar')
net.load_state_dict(ckpt['model_state_dict'])

# restore chkpt
if ops.from_chkpt:
    chkpt_file = sorted(os.listdir(ops.chkpt_path))[-1]
    
    if chkpt_file:
        ckpt = torch.load(ops.chkpt_path+chkpt_file)
        resume_step = ckpt['step']
        today_dt = ckpt['logdt']
        net.load_state_dict(ckpt['model_state_dict'])
        optim.load_state_dict(ckpt['optim_state_dict'])
        
    else:
        os.error('no checkpoint')

    print 'continue from checkpoint: ',
    print chkpt_file
else:
    print 'start training from scratch...'
    resume_step = 0
    # init text file
    today_dt = str(time.strftime("%m-%d-%H-%M", time.gmtime()))
    output_text = open("err_logs/log_"+ today_dt + ".txt","w")
    output_text.close()
    
    


# In[ ]:


# # gen valid set
# if ops.make_valid:
#     print 'generate new valid set...'
#     friends_dataset = Friends_dataset(ops.fr_dict, ops)
#     friends_loader = DataLoader(dataset=friends_dataset, batch_size=ops.batch_size, num_workers=4)
    
#     val_img = []
#     val_emo = []
    
#     for i, bat in enumerate(friends_loader):
#         sys.stdout.write("\r"+str(i+1)+'/'+str(ops.valid_size//ops.batch_size))
#         bat_img = bat[0].numpy()
#         bat_emo = bat[1].numpy()
#         val_img.append(bat_img)
#         val_emo.append(bat_emo)
        
#     val_img = np.array(val_img)
#     val_emo = np.array(val_emo)
    
#     print 'saving valid set...'
#     np.savez('../../dicts/friends_valid.npz', val_img=val_img, val_emo=val_emo)
#     print '\n saved valid set, exiting'
#     sys.exit()

# else:
#     print 'load valid set...'
#     valid_npz = np.load('../../dicts/friends_valid.npz', allow_pickle=True)
#     val_img = valid_npz['val_img']
#     val_emo = valid_npz['val_emo']

# print '> complete'
    


# In[ ]:


tic = time.time()
for epoch in range(ops.epoc_step):
    for i, bat in enumerate(emo_loader, 0):
        step = resume_step + epoch*ops.iter_step + i

        # bat = (img, emo)
        bat_img = bat[0].cuda()
        bat_emo = bat[1].cuda()
        toc0 = time.time() - tic

        # fit using batch data
        tic = time.time()
        # batch forward
        net.train()
        bat_prd = net.forward(bat_img)
        # get loss
        loss = loss_eval(bat_emo.flatten(0,1), bat_prd.flatten(0,1))
        
        # optim step
        learning_rate_use = ops.learning_rate
        optim.param_groups[0]['lr'] = learning_rate_use
        optim.zero_grad()
        loss.backward()
        optim.step()
        toc1 = time.time() - tic
        
        # validation step
#         if step % ops.vald_step == 0:
#             # eval mode
#             net.eval()
#             val_times = int(ops.valid_size/ops.batch_size)
#             valid_all = np.zeros(val_times)
#             v_acc_all = np.zeros(val_times)

#             tic = time.time()
#             for val_i in range(val_times):
#                 sys.stdout.write("\r"+str(val_i+1)+'/'+str(val_times))
#                 # get i-th batch from valid set
#                 val_i_img = torch.Tensor(val_img[val_i]).cuda()
#                 val_i_emo = torch.Tensor(val_emo[val_i]).cuda()
#                 # forward 
#                 with torch.no_grad():
#                     val_i_prd = net.forward(val_i_img)
#                 # total loss/err 
#                 val_i_err = loss_eval(val_i_emo, val_i_prd)
#                 # cumulate errors
#                 valid_all[val_i] = val_i_err.detach().cpu().numpy()
#                 # cumulate accuracy
#                 val_i_emo_am = val_i_emo.argmax(dim=1)
#                 val_i_prd_am = val_i_prd.argmax(dim=1)
#                 val_i_acc = torch.sum(val_i_emo_am == val_i_prd_am).type(torch.DoubleTensor) / float(ops.batch_size)
#                 v_acc_all[val_i] = val_i_acc.detach().cpu().numpy()

#             toc2 = time.time() - tic
#             print " elaptime: " + str('%.6f'%toc2)
#             print " valerror: " + str(valid_all.mean())
#             print " accurate: " + str(v_acc_all.mean())
        
    
        
        # error step
        if step % ops.errd_step == 0:
            net.eval()
            bat_prd = net.forward(bat_img)
            loss = loss_eval(bat_emo.flatten(0,1), bat_prd.flatten(0,1))
            error = loss.detach().cpu().numpy()
            accu = (torch.sum( bat_prd.flatten(0,1).argmax(dim=1) == bat_emo.flatten(0,1).argmax(dim=1) ).type(torch.DoubleTensor) / float(ops.batch_size*ops.win_size)).detach().cpu().numpy()

            print "step:" + str(step) + ', error: ' + str(error) + ', accuracy: ' + str(accu)
            print '  lr: ' + str(learning_rate_use) + ', elaptime: ' + str('%.6f'%toc0) + ', ' + str('%.6f'%toc1)

            # open - write - close
            output_text = open("err_logs/log_"+ today_dt + ".txt","a")
            output_text.write("%s, %s, %s, %s, %s\n" % (step, error, accu, 0., 0.))
            output_text.close()
            net.train()

        # save step
        if step % ops.save_step == 0:
            # remove oldest
            ckpt_nums = len(sorted(os.listdir(ops.chkpt_path)))
            if ckpt_nums == ops.maxn_chkpt:
                os.remove(ops.chkpt_path+'/'+sorted(os.listdir(ops.chkpt_path))[0])
            # save new
            torch.save({'step' : step, 'logdt' : today_dt,
                        'model_state_dict' : net.state_dict(), 'optim_state_dict' : optim.state_dict()
                        }, ops.chkpt_path+'chkpt_'+get_dtstr()+'_'+str(step)+'.tar')
            print "saved chkpt"
            
        tic = time.time()


# In[ ]:




