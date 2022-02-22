from importlib import import_module


def get_trainer(args, model, loader):
    print("trainer init")
    module = import_module('Trainers.' +  args.trainer.lower())
    return module.make_trainer(args, model, loader)
