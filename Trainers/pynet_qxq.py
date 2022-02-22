import torch
import math

from .base import AbstractTrainer
from Utils.msssim import MSSSIM
from Utils.vgg import vgg_custom
from Utils.utils import normalize_batch


def make_trainer(args, model, loader):
    return PyNetTrainer(args, model, loader)


class PyNetTrainer(AbstractTrainer):
    def __init__(self, args, model, loader):
        super().__init__(args, model, loader)

        self.MAE_loss = torch.nn.L1Loss()
        self.metric = args.metric
        self.VGG_loss = vgg_custom(vgg_name="vgg19", device=self.device, vgg_path=None)
        self.MSE_loss = torch.nn.MSELoss()
        self.MS_SSIM = MSSSIM()

    def calculate_metrics(self, batch):
        loss_mse_eval = 0
        loss_psnr_eval = 0
        loss_vgg_eval = 0
        loss_ssim_eval = 0

        x, y = batch
        x = x.to(self.device, non_blocking=True).float()
        y = y.to(self.device, non_blocking=True).float()
    
        # print("==================", x.shape, y.shape)
        enhanced = self.model(x)
        enhanced = enhanced.float()
        # print("==================", enhanced.shape)
        
        metrics = {}
        if self.metric:
            loss_mse_temp = self.MSE_loss(enhanced, y).item()

            loss_mse_eval += loss_mse_temp
            loss_psnr_eval += 20 * math.log10(1.0 / math.sqrt(loss_mse_temp))

            # MS-SSIM loss
            loss_ssim_eval += self.MS_SSIM(y, enhanced).item() # 변경

            # VGG loss
            enhanced_vgg_eval = self.VGG_loss(normalize_batch(enhanced)).detach()
            target_vgg_eval = self.VGG_loss(normalize_batch(y)).detach()
            loss_vgg_eval += self.MSE_loss(enhanced_vgg_eval, target_vgg_eval).item()
            
            metrics["psnr"] = loss_psnr_eval
            metrics["msssim"] = loss_ssim_eval
            metrics["mse"] = loss_mse_eval
            metrics["vgg"] = loss_vgg_eval
        else:   
            metrics["psnr"] = 0
            metrics["msssim"] = 0
            metrics["mse"] = 0
            metrics["vgg"] = 0
            
        return metrics, enhanced
