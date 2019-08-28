#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# preliminaries
import sys,os,time,cv2,json
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import ops

# for video file generation
from utils import imread_to_rgb, crop_img

DB_PATH = '/home/jhchoi/datasets2/friends/'
MD_PATH = '../../anno/'
window_size = 1
crop_size = ops.img_sz


# In[ ]:


def hmsf_to_fnum(hmsf, fps):
    hms = hmsf.split(';')[0].split(':')
    f   = hmsf.split(';')[1]
    
    return (int(hms[0])*60*60 + int(hms[1])*60 + int(hms[2]))*fps + int(f)

def emo_char_idx(emo):
    # 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
    if emo == 'angry' or emo == 'anger':
        return 0
    elif emo == 'disgust':
        return 1
    elif emo == 'fear':
        return 2
    elif emo == 'happy' or emo == 'happiness':
        return 3
    elif emo == 'sad' or emo == 'sadness':
        return 4
    elif emo == 'surprise':
        return 5
    elif emo == 'neutral':
        return 6
    else:
        'error'

def coor_change(coor):
    spl = coor.split('  ')
    if len(spl)>1:
        return int(spl[0])
    else:
        return int(coor)
        


# In[ ]:


# # remove all files
# epi_list = ['ep01', 'ep02', 'ep03', 'ep04', 'ep05', 'ep06', 'ep07', 'ep08', 'ep09', 'ep10']

# for epdir in epi_list:
#     flist = os.listdir(DB_PATH+'/'+epdir+'_p')
#     for fname in flist:
#         os.remove(DB_PATH+'/'+epdir+'_p/'+fname)
    
        


# In[ ]:


epi_list = ['ep01', 'ep02', 'ep03', 'ep04', 'ep05', 'ep06', 'ep07', 'ep08', 'ep09', 'ep10']
val_dict = dict()
emo_stat = list()
numl_all = np.zeros(7)

for i in range(7):
    val_dict[i] = dict()
#     for epi in epi_list:
#         val_dict[i][epi] = dict()

        
for epi_i in range(len(epi_list)):
    print '\n'+str(epi_i)
    
    # open 
    with open(MD_PATH+'s01_'+epi_list[epi_i]+'_tag2_visual_Final_180809.json') as md:
        epi_md = json.load(md)

    epi_md = epi_md['visual_results']

    # every period
    for i in range(len(epi_md)):
        sys.stdout.write("\r"+str(i+1)+'/'+str(len(epi_md)))
        # per num
        pnum = int(epi_md[i]['period_num'])
        
        # start-end
        stime = hmsf_to_fnum(epi_md[i]['start_time'], 24)
        etime = hmsf_to_fnum(epi_md[i]['end_time'], 24)
        # img
        imname = epi_md[i]['image']
        if imname[0:2] == 'Fr':
            pfnum = int(epi_md[i]['image'].split('.')[0].split('_')[1])
        else:
            pfnum = int(epi_md[i]['image'].split('.')[0][1:])
        # person
        pid_md = epi_md[i]['person'][0]

        # for every person
        for char in pid_md.keys():
            emo = pid_md[char][0]['emotion'].lower()
            face_bb = pid_md[char][0]['face_rect']

            if emo == 'none' or face_bb['max_x'] == 'none':
                continue

            # face xy
            face_bb = [coor_change(face_bb['min_x']), coor_change(face_bb['min_y']), coor_change(face_bb['max_x']), coor_change(face_bb['max_y'])]
            
            if face_bb[0] >= face_bb[2] or face_bb[1] >= face_bb[3]:
                continue
                
            # 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
            emo_idx = emo_char_idx(emo);    emo_stat.append(emo_idx)

            bat_img_temp = list() #np.zeros([crop_size, crop_size, window_size])
            
            for i in range(window_size):
                pfnum_batch = pfnum-window_size+1+i
                # fimg
                frame_img = cv2.imread(DB_PATH+'/'+epi_list[epi_i]+'/'+str('%05d' % pfnum_batch)+'.jpg')
                face_crop = frame_img[face_bb[1]:face_bb[3], face_bb[0]:face_bb[2]]
                #face_resz = (255*(face_crop)).astype(np.uint8)
                face_resz = cv2.resize(face_crop, (crop_size,crop_size))
                savename = epi_list[epi_i]+'_p/'+epi_list[epi_i]+'_'+str('%04d' % pnum)+'_'+char+'_'+str(emo_idx)+'_'+str('%05d' % pfnum_batch)+'.jpg'
                cv2.imwrite(DB_PATH+savename, face_resz)
                bat_img_temp.append(savename)

            # save to dict
            numl = numl_all[emo_idx]
            val_dict[emo_idx][numl] = dict()
            val_dict[emo_idx][numl]['img'] = savename
            val_dict[emo_idx][numl]['emo'] = emo_idx
            numl_all[emo_idx] += 1
            
#             val_dict[emo_idx][epi_list[epi_i]][str(pnum)+'_'+char] = dict()
#             val_dict[emo_idx][epi_list[epi_i]][str(pnum)+'_'+char]['crop'] = bat_img_temp
#             val_dict[emo_idx][epi_list[epi_i]][str(pnum)+'_'+char]['bb'] = face_bb
            
        

np.save('../../dicts/friends_valid.npy', val_dict)



# In[ ]:


# tt = 0
# for i in range(7):
#     for j in val_dict[i].keys():
#         tt += 1
# print tt


# In[ ]:




