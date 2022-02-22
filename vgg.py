from torchvision import models
import torch.nn as nn
import torch


CONTENT_LAYER = 'relu_16'


def vgg_custom(vgg_name, device, vgg_path=None):

    pretrained = (vgg_path is None)

    if vgg_name == "vgg19":
        vgg = models.vgg19(pretrained=pretrained)
    elif vgg_name == "vgg16":
        vgg = models.vgg16(pretrained=pretrained)
    elif vgg_name == "vgg16_bn":
        vgg = models.vgg16_bn(pretrained=pretrained)
    elif vgg_name == "vgg13":
        vgg = models.vgg13(pretrained=pretrained)
    elif vgg_name == "vgg13_bn":
        vgg = models.vgg13_bn(pretrained=pretrained)
    elif vgg_name == "vgg11":
        vgg = models.vgg11(pretrained=pretrained)
    elif vgg_name == "vgg11_bn":
        vgg = models.vgg11_bn(pretrained=pretrained)
    else:
        raise ValueError("Unexpected vgg_name")

    if vgg_path is not None:
        vgg.load_state_dict(torch.load(vgg_path), strict=False)

    vgg = vgg.features
    model = nn.Sequential()

    i = 0
    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)
        if name == CONTENT_LAYER:
            break

    model = model.to(device)
    model = torch.nn.DataParallel(model)

    for param in model.parameters():
        param.requires_grad = False

    for param in vgg.parameters():
        param.requires_grad = False

    return model
