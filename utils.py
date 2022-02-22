import numpy as np
import torch
import os


mapper = {"Bayer": 2, "Quad": 4, "QxQ": 8}
NUM_DEFAULT_TRAIN_EPOCHS = [50, 25, 17, 17, 8, 8]


def get_last_iter(level):

    saved_models = [int((model_file.split("_")[-1]).split(".")[0])
                    for model_file in os.listdir("models/")
                    if model_file.startswith("pynet_level_" + str(level))]

    if len(saved_models) > 0:
        return np.max(saved_models)
    else:
        return -1


def normalize_batch(batch):
    # Normalize batch using ImageNet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std


def sigmoid_flops_counter_hook(module, input, output):
    active_elements_count = output.numel() * 4
    module.__flops__ += int(active_elements_count)


class CFA_Filter():
    def __init__(self, width, height, cfa_type, rgb):
        self.w= width
        self.h = height
        self.half_w = width // 2
        self.half_h = height // 2
        
        if not rgb:
            self.g2_idx = 3
            self.cfa_filters = np.zeros((height, width, 4), dtype=bool)
        else:
            self.g2_idx = 1
            self.cfa_filters = np.zeros((height, width, 3), dtype=bool)
        
        self.set_filter(cfa_type)

    def set_filter(self, cfa_type):
        mod = mapper[cfa_type]
        half_mod = mod // 2

        for j in range(half_mod):
            for i in range(half_mod):
                self.cfa_filters[i::mod, j::mod, 1] = True  # G1 filter
                self.cfa_filters[i::mod, j+half_mod::mod, 0] = True  # R filter
                self.cfa_filters[i+half_mod::mod, j::mod, 2] = True  # B filter
                self.cfa_filters[i+half_mod::mod, j+half_mod::mod, self.g2_idx] = True  # G2 filter

    def apply_4ch_filter(self, img):

        input_combined = np.zeros((self.half_h, self.half_w, 4))

        input_combined[:, :, 0] = (
                img[self.cfa_filters[:, :, 2]].reshape(self.half_h, self.half_w))  # B
        input_combined[:, :, 1] = (
                img[self.cfa_filters[:, :, 1]].reshape(self.half_h, self.half_w))  # G1
        input_combined[:, :, 2] = (
                img[self.cfa_filters[:, :, 0]].reshape(self.half_h, self.half_w))  # R
        input_combined[:, :, 3] = (
                img[self.cfa_filters[:, :, 3]].reshape(self.half_h, self.half_w))  # G2

        return input_combined.astype(np.float32)

    def apply_1ch_filter(self, img, grayscale=True):
        input_combined = np.zeros((self.h, self.w, 3))
        input_combined[self.cfa_filters] = img[self.cfa_filters]

        if grayscale:
            input_combined = np.sum(input_combined, axis=2)

        return input_combined

    def apply_1ch_filter_to_tensor(self, input_tensor, device):
        input_arr = input_tensor.cpu().detach().numpy().squeeze(0).transpose((1, 2, 0))
        applied_np = self.apply_1ch_filter(input_arr, True).reshape((1, 1, self.h, self.w))
        applied_tensor = torch.from_numpy(applied_np).float().to(device)

        return applied_tensor


def get_gray_image(input_tensor, device, cfa_type='QxQ'):
    input_arr = input_tensor.cpu().detach().numpy()
    batch_size , _, height, width = input_arr.shape
    mod = mapper[cfa_type]
    half_mod = mod // 2
    gray = np.zeros((batch_size, 1, height*2, width*2))

    for batch in range(batch_size):
        for x in range(half_mod):
            for y in range(half_mod):
                gray[batch, 0, y::mod, x::mod] = input_arr[batch, 1, y::half_mod, x::half_mod]
                gray[batch, 0, y+half_mod::mod, x::mod] = input_arr[batch, 0, y::half_mod, x::half_mod]
                gray[batch, 0, y::mod, x+half_mod::mod] = input_arr[batch, 2, y::half_mod, x::half_mod]
                gray[batch, 0, y+half_mod::mod, x+half_mod::mod] = input_arr[batch, 3, y::half_mod, x::half_mod]

    return torch.from_numpy(gray).float().to(device)
