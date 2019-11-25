import os, time
import numpy as np
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import pylab

def eval_errval():
    timeNow = time.localtime()
    timeStr = str('%02d'%timeNow.tm_mon)+'-'+str('%02d'%timeNow.tm_mday)+','+str('%02d'%timeNow.tm_hour)+':'+str('%02d'%timeNow.tm_min)+':'+str('%02d'%timeNow.tm_sec)
    print timeStr
    
    all_files = np.array(os.listdir("err_logs/"))
    all_files.sort()

    for selfile in all_files:
        if selfile[-3:] != 'txt':
            all_files = np.delete(all_files, np.where(all_files==selfile))
    
    sel_newest = pylab.genfromtxt("err_logs/"+all_files[-1],delimiter=',')
    if sel_newest.ndim < 2:
        print 'wait...'
        return
    
#     if sel_newest.shape[0] > 100:
#         sel_newest = sel_newest[100:,:]
    #(step, error, accu, valid_all.mean(), v_acc_all.mean())
    nums = sel_newest[:,0] 
    errs = sel_newest[:,1]
    acus = sel_newest[:,2]
    vals = sel_newest[:,3]
    accs = sel_newest[:,4]

    print int(nums[-1])
    fig, ax = plt.subplots(2,1)
    fig.set_figheight(8)
    fig.set_figwidth(7)
    ax[0].set_title('updated: ' + timeStr,fontsize=10)
    ax[0].plot(nums,errs, nums,vals)
    ax[1].plot(nums,acus, nums,accs)
    
    ax[0].grid(linestyle=':'); ax[1].grid(linestyle=':');
    plt.tight_layout()
    
    plt.savefig('plot.png')
    plt.close(fig)

while True:
    eval_errval()
    time.sleep(120)

