import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import random
import numpy as np
from utils.image_utils import load_img


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class DataLowTrain(Dataset):
    def __init__(self, rgb_dir, resize=None, img_options=None):
        super(DataLowTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'low')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'high')))

        self.inp_filenames = [os.path.join(rgb_dir, 'low', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'high', x) for x in tar_files if is_image_file(x)]

        self.resize = resize

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        low_path = self.inp_filenames[index_]
        high_path = self.tar_filenames[index_]

        low_img = Image.open(low_path).convert('RGB')
        if self.resize is not None : 
            low_img = low_img.resize((self.resize, self.resize), resample=Image.BILINEAR) 
        high_img = Image.open(high_path).convert('RGB')
        if self.resize is not None : 
            high_img = high_img.resize((self.resize, self.resize), resample=Image.BILINEAR) 


        w, h = high_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            low_img = TF.pad(low_img, (0, 0, padw, padh), padding_mode='reflect')
            high_img = TF.pad(high_img, (0, 0, padw, padh), padding_mode='reflect')
        
        low_img = TF.to_tensor(low_img)
        high_img = TF.to_tensor(high_img)

        hh, ww = high_img.shape[1], high_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = low_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = high_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        filename = os.path.splitext(os.path.split(low_path)[-1])[0]

        return tar_img, inp_img, filename


class DataLowValid(Dataset):
    def __init__(self, rgb_dir, resize=None, img_options=None):
        super(DataLowValid, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'low')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'high')))

        self.inp_filenames = [os.path.join(rgb_dir, 'low', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'high', x) for x in tar_files if is_image_file(x)]

        self.resize = resize

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target


    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        # ps = self.ps

        low_path = self.inp_filenames[index_]
        high_path = self.tar_filenames[index_]

        low_img = Image.open(low_path).convert('RGB')
        if self.resize is not None : 
            low_img = low_img.resize((self.resize, self.resize), resample=Image.BILINEAR) 
        high_img = Image.open(high_path).convert('RGB')
        if self.resize is not None : 
            high_img = high_img.resize((self.resize, self.resize), resample=Image.BILINEAR) 
        
        low_img = TF.to_tensor(low_img)
        high_img = TF.to_tensor(high_img)


        filename = os.path.splitext(os.path.split(low_path)[-1])[0]

        return high_img, low_img, filename
