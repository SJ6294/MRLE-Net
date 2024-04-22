import torch
import numpy as np
import cv2
from pytorch_msssim import ssim

def load_img(filepath):
    image = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (256,256))
    return image 

def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def torchSSIM(tar_img, prd_img):
    return ssim(tar_img, prd_img, data_range=1.0, size_average=True)

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

