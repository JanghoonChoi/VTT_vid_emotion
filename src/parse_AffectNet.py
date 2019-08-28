# preliminaries
import sys,os,time,cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import imread_to_rgb, img_rgb2bw

DB_PATH = '/home/jhchoi/datasets3/affectnet/'
af_dict = dict()
# CK: 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
#FER: 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
#AF : 0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt, 8: None, 9: Uncertain, 10: No-Face

def emo_af2fer(af):
    trm = [6, 3, 4, 5, 2, 1, 0, -1, -1, -1, -1]
    return trm[af]

for i in range(7):
    af_dict[i] = dict()
    

csv_train = np.genfromtxt(DB_PATH + '/training.csv', dtype=str)[1:]
csv_valid = np.genfromtxt(DB_PATH + '/validation.csv', dtype=str)[1:]
db_all = np.concatenate((csv_train, csv_valid))
numl_all = np.zeros(7)


# In[ ]:


for i in range(db_all.size):
    ll = db_all[i]
    ls = ll.split(',')
    
    if emo_af2fer(int(ls[6])) < 0: continue
    if ls[1] == 'NULL' : continue
    
    emo = emo_af2fer(int(ls[6]))
    numl = numl_all[emo]
    af_dict[emo][numl] = dict()
    af_dict[emo][numl]['img'] = ls[0] # 123/aaa.jpg
    af_dict[emo][numl]['gt']  = [int(ls[1]), int(ls[2]), int(ls[3]), int(ls[4])] # x,y,w,h
    af_dict[emo][numl]['emo'] = emo
    af_dict[emo][numl]['val'] = float(ls[-2])
    af_dict[emo][numl]['aro'] = float(ls[-1])
    numl_all[emo] += 1

np.save('../../dicts/affectnet_parsed.npy', af_dict)


