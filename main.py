import torch

import DataLoaders
import Models
import Trainers
from option import args


torch.manual_seed(args.seed)

loader = DataLoaders.Data(args)
model = Models.Model(args)
trainer = Trainers.get_trainer(args, model, loader)

trainer.test()
