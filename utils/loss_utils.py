import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    """
    FocalLoss的引入主要是为了解决目标检测中正负样本数量极不平衡问题
    """
    def __init__(self, gamma=0, alpha=None, size_average=True):        
        super(FocalLoss, self).__init__()        
        self.gamma = gamma        
        self.alpha = alpha        
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])        
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)        
        self.size_average = size_average    
    
    def forward(self, input, target):        
        if input.dim()>2:            
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W            
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C            
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C        
        target = target.view(-1,1)        
        
        logpt = F.log_softmax(input, dim=-1)        
        logpt = logpt.gather(1,target)        
        logpt = logpt.view(-1)        
        pt = Variable(logpt.data.exp())        
        
        if self.alpha is not None:            
            if self.alpha.type()!=input.data.type():                
                self.alpha = self.alpha.type_as(input.data)            
            at = self.alpha.gather(0,target.data.view(-1))            
            logpt = logpt * Variable(at)        
        
        loss = -1 * (1-pt)**self.gamma * logpt        
        if self.size_average: return loss.mean()        
        else: return loss.sum()

class CostSensitiveCE(nn.Module):
    """
    CostSensitiveCE也是用于正负样本不均衡问题，CostSensitiveCE分类算法的主要目标是使得cost最小化
    """
    def __init__(self, gamma, num_class_list, device):
        super(CostSensitiveCE, self).__init__()
        self.num_class_list = num_class_list
        self.device = device
        self.csce_weight = torch.FloatTensor(np.array([(min(self.num_class_list) / N)**gamma for N in self.num_class_list])).to(self.device)

    def forward(self, input, target):
        csce_loss = F.cross_entropy(input=input, target=target, weight=self.csce_weight)
        #This criterion computes the cross entropy loss between input and target.
        return csce_loss

