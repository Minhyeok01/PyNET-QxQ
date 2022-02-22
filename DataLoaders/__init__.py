from importlib import import_module
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class Data:
    def __init__(self, args):
        
        module_test = import_module('DataLoaders.' +  args.data_test.lower())
        testset = getattr(module_test, args.data_test)(args, data_type="test")
        self.loader_test = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
        print("# of testset:", testset.__len__())
