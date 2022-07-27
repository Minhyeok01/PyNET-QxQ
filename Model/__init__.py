import os
from importlib import import_module
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        print('Making model...')

        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        if args.n_GPUs==1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        module = import_module('Model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        self.model_name = args.model

    def forward(self, x):
            return self.model(x)
