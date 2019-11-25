import torch
import torch.nn as nn


class emo_loss(nn.Module):
    def __init__(self, loss_lambda):
        super(type(self), self).__init__()
        # balancing term
        self.loss_lambda = loss_lambda
        # bce loss obj
        self.bce_loss = nn.BCEWithLogitsLoss()
        

    def loss_neutral(self, e0, e1):
        # loss term only for neutral emotion
        # based on cross entropy
        # e0-gt, e1-estimate where shape=(batch_size, 7)

        # convert to neutral / non-neutral labels
        e0_n = torch.cat((e0[:,[-1]], e0[:,:-1].sum(dim=1,keepdim=True))   ,dim=1) # gt label
        e1_n = torch.cat((e1[:,[-1]], e1[:,:-1].max(dim=1,keepdim=True)[0]),dim=1) # predicted label
        # cross-entropy loss
        loss = self.bce_loss(e1_n, e0_n)

        return loss


    def loss_emotion(self, e0, e1):
        # loss term for all emotions
        # based on cross entropy

        # cross-entropy loss
        loss = self.bce_loss(e1, e0)

        return loss


    def forward(self, e0, e1):
        # total combined loss with balancing term
        loss = self.loss_lambda*self.loss_neutral(e0,e1) + (1.-self.loss_lambda)*self.loss_emotion(e0,e1)

        return loss



    