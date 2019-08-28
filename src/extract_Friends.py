#!/usr/bin/env python
# coding: utf-8

# In[1]:


# extract all frame images from Friends video file (*.mkv)
# preliminaries
import sys,os,time,cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io

DB_PATH = '/home/jhchoi/datasets2/friends/'


# In[ ]:


mkvlist = [f for f in os.listdir(DB_PATH) if f[-3:]=='mkv']
mkvlist = np.sort(mkvlist)

for epi_i, epi_f in enumerate(mkvlist):
    print epi_f
    
    epi_dir = 'ep' + str('%02d' % (epi_i+1))
    epi_vid = cv2.VideoCapture(DB_PATH + epi_f)
    fnum = 0;    r = True
    while r:
        sys.stdout.write("\r"+str(fnum))
        imfname = str('%05d' % fnum) + '.jpg'
        r, f = epi_vid.read()
        cv2.imwrite(DB_PATH+'/'+epi_dir + '/' + imfname, f)
        
        fnum += 1
    print ''


# In[ ]:




