import torch
import numpy as np

# torch implementation of np.random.choice
def th_choice(a, p=None):
    """ torch implementation of np.random.choice(), x1.1~1.5 slower than original function """
    # preliminaries
    a_l = len(a)
    if p is None:
        idx = torch.randperm(a_l)
        return a[idx[0]]
        
    elif torch.sum(p) < 1.:
        print torch.sum(p),' p.sum() not 1'
    
    # accumulative prob
    pa = torch.cumsum(p,0)
    
    # random (0,1)
    trnd = torch.rand(1)[0]
    
    # find
    idx = (torch.argmax(pa < trnd)+1) % a_l
    return a[idx]


def th_rand(n=1):
    """ proxy to torch.rand(n)[0] """
    if n == 1:
        return float(torch.rand(n)[0])
    else:
        return torch.rand(n).numpy()

    
def th_rand_rng(low, high, n=1):
    """ pull uniform random sample(s) from [a,b) """
    if n == 1:
        return (high-low)*float(torch.rand(n)[0])+low
    else:
        return (high-low)*torch.rand(n).numpy()+low
    
    
def th_randint(low, high=None, size=1):
    """ proxy to torch.randint(low,high,(size,)) """
    if high is None:    ilow = 0;    ihigh = low
    else:    ilow = low;    ihigh = high
        
    if size == 1:
        return torch.randint(low=ilow, high=ihigh, size=(size,)).numpy()[0]
    else:
        return torch.randint(low=ilow, high=ihigh, size=(size,)).numpy()

    
# calculate receptive field
def calc_RF(net, cu=False, sz=511):
    # net eval, init
    if cu:   net = net.cpu()
    net.eval()
    net.zero_grad()
    # img -> out
    img = torch.autograd.Variable(torch.rand([1,3,sz,sz])*1e+30, requires_grad=True)
    out = net(img)
    # loss(out, out_g)
    oH = out.shape[2];    oW = out.shape[3];
    out_g = torch.Tensor(out.detach().numpy().copy())
    out_g[:,:,oH//2,oW//2].fill_(-1e+30)
    loss = torch.sum(torch.abs(torch.sub(out,out_g)))
    loss.backward()
    # find position with nonzero gradients
    xa = np.sum( img.grad.detach().numpy()[0].sum(axis=0),  axis=0)
    idxs = np.where(xa != 0)[0]
    rf_sz = idxs.max()-idxs.min()+1
    
    if rf_sz >= sz:
        print 'image_size == RF_size : RF_size may be bigger. Try again with bigger image size.'
    
    return rf_sz


    