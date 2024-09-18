import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN
import numpy as np

class Model(nn.Module):
    """
    RevIN-Linear
    """
    def __init__(self, args):
        super(Model, self).__init__()

        self.revin_layer = RevIN(args.enc_in, device=args.device)
        self.Linear = nn.Linear(args.seq_len, args.pred_len)
        

    def forward(self, x):

        # add the REVIN layer nomrmlisation at the input 
        x_in = self.revin_layer(x, 'norm')
        
        x = self.Linear(x_in.permute(0,2,1)).permute(0,2,1)

        # do the revesrve norm on the output
        x_out = self.revin_layer(x, 'denorm')

        return x_out # [Batch, Output length, Channel]

