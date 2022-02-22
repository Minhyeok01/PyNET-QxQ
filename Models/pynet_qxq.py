import torch.nn as nn
import torch

from Utils.utils import get_gray_image


def make_model(args, parent=False):
    return PyNET_QxQ(args)


class PyNET_QxQ(nn.Module):
    def __init__(self, args, instance_norm_level_1=False):
        super(PyNET_QxQ, self).__init__()
        instance_norm_level_1 = args.instance_norm_level_1
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        
        self.conv_l1_d1 = ConvMultiBlock(4, 16, 3, instance_norm=False)

        # -------------------------------------
        self.conv_l1_d3 = ConvMultiBlock(16, 16, 5, instance_norm=False)
        self.conv_l1_d5 = ConvMultiBlock(48, 16, 7, instance_norm=instance_norm_level_1) #

        self.conv_l1_d6 = ConvMultiBlock(48, 16, 9, instance_norm=instance_norm_level_1) #
        self.conv_l1_d7 = ConvMultiBlock(64, 16, 9, instance_norm=instance_norm_level_1) #
        self.conv_l1_d8 = ConvMultiBlock(64, 16, 9, instance_norm=instance_norm_level_1) #
        self.conv_l1_d9 = ConvMultiBlock(64, 16, 9, instance_norm=instance_norm_level_1) #

        self.conv_l1_d10 = ConvMultiBlock(64, 16, 7, instance_norm=instance_norm_level_1) #
        self.conv_l1_d12 = ConvMultiBlock(64, 16, 5, instance_norm=instance_norm_level_1)#
        self.conv_l1_d14 = ConvMultiBlock(48, 16, 3, instance_norm=False)

        self.conv_t0 = Upsample_PS(16, 16, 3, 2)
        
        # -------------------------------------

        self.conv_l0_d1 = ConvLayer(17, 3, kernel_size=3, stride=1, relu=False)
        self.output_l0 = nn.Sigmoid()

    def level_1(self, conv_l1_d1):

        conv_l1_d3 = self.conv_l1_d3(conv_l1_d1)
        conv_l1_d4 = torch.cat([conv_l1_d3, conv_l1_d1], 1)

        conv_l1_d5 = self.conv_l1_d5(conv_l1_d4)

        conv_l1_d6 = self.conv_l1_d6(conv_l1_d5)
        conv_l1_d7 = self.conv_l1_d7(conv_l1_d6) + conv_l1_d6
        conv_l1_d8 = self.conv_l1_d8(conv_l1_d7) + conv_l1_d7
        conv_l1_d9 = self.conv_l1_d9(conv_l1_d8) + conv_l1_d8

        conv_l1_d10 = self.conv_l1_d10(conv_l1_d9)
        conv_l1_d11 = torch.cat([conv_l1_d10, conv_l1_d1], 1)

        conv_l1_d12 = self.conv_l1_d12(conv_l1_d11)
        conv_l1_d13 = torch.cat([conv_l1_d12, conv_l1_d1], 1)

        conv_l1_d14 = self.conv_l1_d14(conv_l1_d13)
        conv_t0 = self.conv_t0(conv_l1_d14)

        return conv_t0

    def level_0(self, conv_t0, gray):
        conc = torch.cat([conv_t0, gray], 1)
        conv_l0_d1 = self.conv_l0_d1(conc)
        output_l0 = self.output_l0(conv_l0_d1)

        return output_l0

    def forward(self, x):
        gray = get_gray_image(x, self.device)
        conv_l1_d1 = self.conv_l1_d1(x)
        
        conv_t0 = self.level_1(conv_l1_d1)
        output_l0 = self.level_0(conv_t0, gray)

        return output_l0


class ConvMultiBlock(nn.Module):

    def __init__(self, in_channels, out_channels, max_conv_size, instance_norm):

        super(ConvMultiBlock, self).__init__()
        self.max_conv_size = max_conv_size

        self.conv_3a = ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, instance_norm=instance_norm)
        self.conv_3b = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, instance_norm=instance_norm)

        if max_conv_size >= 5:
            self.conv_5a = ConvLayer(in_channels, out_channels, kernel_size=5, stride=1, instance_norm=instance_norm)
            self.conv_5b = ConvLayer(out_channels, out_channels, kernel_size=5, stride=1, instance_norm=instance_norm)

        if max_conv_size >= 7:
            self.conv_7a = ConvLayer(in_channels, out_channels, kernel_size=7, stride=1, instance_norm=instance_norm)
            self.conv_7b = ConvLayer(out_channels, out_channels, kernel_size=7, stride=1, instance_norm=instance_norm)

        if max_conv_size >= 9:
            self.conv_9a = ConvLayer(in_channels, out_channels, kernel_size=9, stride=1, instance_norm=instance_norm)
            self.conv_9b = ConvLayer(out_channels, out_channels, kernel_size=9, stride=1, instance_norm=instance_norm)

    def forward(self, x):

        out_3 = self.conv_3a(x)
        output_tensor = self.conv_3b(out_3)

        if self.max_conv_size >= 5:
            out_5 = self.conv_5a(x)
            out_5 = self.conv_5b(out_5)
            output_tensor = torch.cat([output_tensor, out_5], 1)

        if self.max_conv_size >= 7:
            out_7 = self.conv_7a(x)
            out_7 = self.conv_7b(out_7)
            output_tensor = torch.cat([output_tensor, out_7], 1)

        if self.max_conv_size >= 9:
            out_9 = self.conv_9a(x)
            out_9 = self.conv_9b(out_9)
            output_tensor = torch.cat([output_tensor, out_9], 1)

        return output_tensor


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, relu=True, instance_norm=False):

        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2

        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        self.instance_norm = instance_norm
        self.instance = None
        self.relu = None

        if instance_norm:
            self.instance = nn.InstanceNorm2d(out_channels, affine=True)

        if relu:
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        out = self.reflection_pad(x)
        out = self.conv2d(out)

        if self.instance_norm:
            out = self.instance(out)

        if self.relu:
            out = self.relu(out)

        return out


class Upsample_PS(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, upscaling_factor, stride=1):

        super(Upsample_PS, self).__init__()

        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)

        self.upchannel = torch.nn.Conv2d(in_channels, out_channels*(upscaling_factor**2), kernel_size, stride)
        self.pixel_shuffle = nn.PixelShuffle(upscaling_factor)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.upchannel(out)
        out = self.pixel_shuffle(out)

        return out
