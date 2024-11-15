import numpy as np
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

class colorNeighbour(nn.Module):
    def __init__(self,out_samples=20, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.out_samples = out_samples
        self.mul_parameters = torch.ones(out_samples,3,1,1)
        self.noise = torch.randn(out_samples,3,1,1)/8   # allow 0.75x-1.25x adjustment
        self.mul_parameters = torch.abs(self.mul_parameters+self.noise)
        self.mul_parameters=nn.Parameter(self.mul_parameters)
        # if cuda:
        #     self.mul_parameters=nn.Parameter(self.mul_parameters.cuda())
    
    def forward(self,x):
        assert x.shape[0]==1,"Batch size is not 1"
        x = x*self.mul_parameters
        # x = x.squeeze(0)
        # x_chunked = torch.tensor(x.chunk(self.out_samples,0))
        return x
    
if __name__ == '__main__':
    random_input = torch.randn((1,3,2,2))
    model = colorNeighbour(out_samples=2)
    for param in model.parameters():
        print(type(param), param.data)
    print(random_input)
    out = model(random_input)
    print(out)

