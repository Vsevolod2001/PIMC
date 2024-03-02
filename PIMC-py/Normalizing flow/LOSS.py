import torch 
from torch import nn
import numpy as np
pi=torch.tensor(np.pi)
from NFconstants import N_nod
import NFoscillator
from NFoscillator import basic_oscillator
from NFrel_oscillator import nr_rel_oscillator
from NFrel_oscillator import ur_rel_oscillator
from NFur_oscillator import basic_ur
from NFrel_oscillator import rel_oscillator_100
from NFrel_oscillator import unit_rel
from NFRelAbs import rel_abs


class KL_with_S(nn.Module):
    def __init__(self,S):
        super().__init__()
        self.S=S
    def forward(self,x,log_abs_det):
        S=self.S(x)
        loss=torch.mean(S-log_abs_det)
        return loss-(N_nod / 2) * (1+torch.log(2*pi))

S_osc=basic_oscillator.get_S()
KL_osc=KL_with_S(S_osc)

S_rel_nr=nr_rel_oscillator.get_S()
KL_rel_nr=KL_with_S(S_rel_nr)

S_ur=basic_ur.get_S()
KL_ur=KL_with_S(S_ur)

S_rel_ur=ur_rel_oscillator.get_S()
KL_rel_ur=KL_with_S(S_rel_ur)

S_100=rel_oscillator_100.get_S()
KL_rel_100=KL_with_S(S_100)

S_abs=rel_abs.get_S()
KL_abs=KL_with_S(S_abs)

S_1rel=unit_rel.get_S()
KL_1rel=KL_with_S(S_1rel)