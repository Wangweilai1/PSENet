# dataloader add 3.0 scale
# dataloader add filer text
import numpy as np
import math
from PIL import Image
from torch.utils import data
import cv2
import torchvision.transforms as transforms


def pad_img_to_32Int(img, fill_values =255):
    h,w = img.shape[:2]
    if w < h :        
        len_shortage = int(math.ceil(w / 32.0) * 32) - w
        img = np.pad(img,pad_width =[[0,0],[0,len_shortage],[0,0]], mode='constant', constant_values=fill_values)        
    else:
        len_shortage = int(math.ceil(h / 32.0) * 32) - h
        img = np.pad(img,pad_width =[[0,len_shortage],[0,0],[0,0]], mode='constant', constant_values=fill_values)
    return img

def scale(img, long_size=2240):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    
    return pad_img_to_32Int(img), scale

class TestLoader(data.Dataset):
    def __init__(self, img, part_id=0, part_num=1, long_size=2240):
        self.imgSrc = img
        self.long_size = long_size
    def __len__(self):
        return 1
    def __getitem__(self, index):
        scaled_img, ratio = scale(self.imgSrc, self.long_size)
        scaled_img = Image.fromarray(scaled_img)
        scaled_img = scaled_img.convert('RGB')
        scaled_img = transforms.ToTensor()(scaled_img)
        scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)
        
        return ratio, scaled_img