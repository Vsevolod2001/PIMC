import torch 
from torch import nn
import oscillator
from oscillator import basic_oscillator
class KL_with_S(nn.Module):
    def __init__(self,S):
        super().__init__()
        self.S=S
    def forward(self,x,log_prob):
        loss=0
        M=len(x)
        for i in range(M):
            loss+=self.S(x[i])
        loss=loss-torch.sum(log_prob)
        loss=loss/M
        return loss

S_osc=basic_oscillator.get_S()
KL_osc=KL_with_S(S_osc)