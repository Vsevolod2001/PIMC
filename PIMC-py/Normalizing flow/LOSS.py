import torch 
from torch import nn
import NFoscillator
from NFoscillator import basic_oscillator
class KL_with_S(nn.Module):
    def __init__(self,S):
        super().__init__()
        self.S=S
    def forward(self,x,log_abs_det):
        S=self.S(x)
        loss=torch.mean(S-log_abs_det)
        return loss

S_osc=basic_oscillator.get_S()
KL_osc=KL_with_S(S_osc)