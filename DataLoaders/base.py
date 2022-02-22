import numpy as np
import os; os.getcwd()
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from glob import glob
import numpy as np
from PIL import Image, ImageStat
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

from Utils.utils import CFA_Filter


class AbstractDateset(Dataset):

    def __init__(self, data_type):

        data_path = self.args.dir_data
        input_folder = self.args.input_folder
    
        self.file_type = self.args.file_type
        self.output_file_type = self.args.output_file_type
    
        self.input_path = os.path.join(data_path, data_type, input_folder)
        self.input_img_fnames = sorted(glob(os.path.join(self.input_path, '*.png')))
        self.dataset_size = len(self.input_img_fnames)

        if self.args.metric:
            output_folder = self.args.output_folder
            self.output_path = os.path.join(data_path, data_type, output_folder)
            self.output_img_fnames = sorted(glob(os.path.join(self.output_path, '*.png')))

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        input_fname = self.input_img_fnames[idx]
        input_tmp = cv2.imread(input_fname, cv2.IMREAD_UNCHANGED)
        h, w = input_tmp.shape
        filters = CFA_Filter(w, h, 'QxQ', rgb=False)
        
        normfactor = 255.0  # 8 bit
        if 'uint16' in str(input_tmp.dtype):
            normfactor = 1023.0   # 10 bit
        input_arr = np.asarray(input_tmp).astype(np.float32) / normfactor
        input_tensor = self.to_tensor(filters.apply_4ch_filter(input_arr))

        if self.args.metric:
            output_fname = self.output_img_fnames[idx]

            # To PIL 8bit Image object
            output_tmp = cv2.imread(output_fname, cv2.IMREAD_UNCHANGED)
            normfactor = 1.0  # 8 bit
            if 'uint16' in str(output_tmp.dtype):
                normfactor = 4.0   # 10 bit
            
            output_img = Image.fromarray(np.round_(np.asarray(output_tmp)[:, :, ::-1].astype(np.float32) / normfactor, decimals=0).astype(np.uint8))
            output_tensor = self.to_tensor(output_img)

            return input_tensor, output_tensor
        else:
            return input_tensor, input_tensor
        