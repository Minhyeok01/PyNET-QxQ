from abc import *
import numpy as np
import os
from ptflops import get_model_complexity_info
import json
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from collections import OrderedDict
from Utils.loggers import  AverageMeterSet
from Utils.msssim import MSSSIM
from Utils.utils import sigmoid_flops_counter_hook


class AbstractTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, loader):
        self.args = args     
        self.model = model
        self.loader_test = loader.loader_test

        # commom args options
        self.export_root = args.export_root

        # cpu/gpu/multi-gpus options
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        if args.n_GPUs==1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # loss definitions
        self.MSE_loss = torch.nn.MSELoss()
        self.MS_SSIM = MSSSIM()

        # etc
        self.tensor_to_pil = transforms.ToPILImage()


    def test(self):
        print('Test best model with test set!')

        torch.backends.cudnn.deterministic = True
        torch.manual_seed(0)
        np.random.seed(0)
        
        # make result path
        self.result_path = self.export_root + "/test_images"
        self._create_folder(self.result_path)
        
        best_model = torch.load(os.path.join(self.export_root, 'models', 'best_acc_model.pth'), map_location=self.device)
        keys = best_model.keys()
        values = best_model.values()
        new_keys = []
        
        for key in keys:
            new_key = key[7:]
            new_keys.append(new_key)
        new_dict = OrderedDict(list(zip(new_keys, values)))
        
        self.model.load_state_dict(new_dict, strict=False)
        self.model.eval()
       
        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.loader_test)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                
                metrics, enhanced = self.calculate_metrics(batch)
                    
                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                if self.args.metric:
                    description = '[TEST]: ' + ', '.join(k + ' {:.3f}'.format(v) for k, v in average_meter_set.averages().items())
                else:
                    description = '[TEST] '
                tqdm_dataloader.set_description(description)

                self.save_img_file(self.result_path, enhanced, self.loader_test, batch_idx)
                _, _, w, h = batch[0].shape
            
            if self.args.metric:
                self.log_path = self.export_root + "/logs"
                self._create_folder(self.log_path)
                macs, params = get_model_complexity_info(self.model, (4, w, h), as_strings=True, print_per_layer_stat=False, verbose=False, custom_modules_hooks={nn.Sigmoid: sigmoid_flops_counter_hook})

                average_metrics = average_meter_set.averages()
                average_metrics["FLOPS"] = macs
                average_metrics["#parameters"] = params

                with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
                    json.dump(average_metrics, f, indent=4)

                print("Model Performance Metrics")
                print(json.dumps(average_metrics, indent=2))

    def _create_folder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)
    
    def save_img_file(self, path, enhanced, loader, idx):
        result_path = path
        visual_dataset = loader.dataset
        j = idx

        enhanced_norm= torch.squeeze(enhanced.float().detach().cpu())
        enhanced_img = self.tensor_to_pil(enhanced_norm)
        
        file_name_orig = '{0}_orig.png'.format(os.path.basename(visual_dataset.input_img_fnames[j].rstrip('.png')))
    
        enhanced_img.save(os.path.join(result_path, file_name_orig))  

