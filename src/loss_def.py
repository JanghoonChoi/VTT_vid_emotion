import torch
import torch.nn as nn

def loss_neutral(e0, e1):
    # loss term only for neutral emotion
    # based on cross entropy
    # e0-gt, e1-estimate where shape=(batch_size, 7)
    
    # convert to neutral / non-neutral labels
    e0_n = torch.cat((e0[:,[-1]], e0[:,:-1].sum(dim=1,keepdim=True)),dim=1)
    e1_n = torch.cat((e1[:,[-1]], e1[:,:-1].sum(dim=1,keepdim=True)),dim=1)
    # cross-entropy loss
    cr_ent = nn.CrossEntropyLoss()
    loss = cr_ent(e1_n, e0_n.argmax(dim=1).long())
    
    return loss


def loss_emotion(e0, e1):
    # loss term for all emotions
    # based on cross entropy
    # cross-entropy loss
    cr_ent = nn.CrossEntropyLoss()
    loss = cr_ent(e1, e0.argmax(dim=1).long())
    
    return loss
    
    
def loss_total(e0, e1, lmb):
    # total combined loss with balancing term
    loss = lmb*loss_neutral(e0,e1) + (1.-lmb)*loss_emotion(e0,e1)
    
    return loss



    