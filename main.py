import torch

import DataLoaders
import Model
import Inferencer
from option import args


torch.manual_seed(args.seed)

loader = DataLoaders.Data(args)
model = Model.Model(args)
inferencer = Inferencer.get_Inferencer(args, model, loader)

inferencer.test()
