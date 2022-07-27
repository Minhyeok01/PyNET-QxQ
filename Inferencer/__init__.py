from importlib import import_module


def get_Inferencer(args, model, loader):
    print("initailize inferencer")
    module = import_module('Inferencer.' +  args.inferencer.lower())
    return module.make_Inferencer(args, model, loader)
