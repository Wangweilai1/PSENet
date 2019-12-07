# dataloader add 3.0 scale
# dataloader add filer text
import numpy as np
import math
from PIL import Image
from torch.utils import data
import util
import cv2
import random
import torchvision.transforms as transforms
import torch

ic15_root_dir = './data/ICDAR2015/Challenge4/'
ic15_test_data_dir = ic15_root_dir + 'ch4_test_images/'
ic15_test_gt_dir = ic15_root_dir + 'ch4_test_localization_transcription_gt/'
test_data_dir = "./TestImgs/"

random.seed(123456)

def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print img_path
        raise
    return img

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

class IC15TestLoader(data.Dataset):
    def __init__(self, part_id=0, part_num=1, long_size=2240):
        data_dirs = [ic15_test_data_dir]
        
        self.img_paths = []
        
        for data_dir in data_dirs:
            img_names = util.io.ls(data_dir, '.jpg')
            img_names.extend(util.io.ls(data_dir, '.png'))

            img_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)
            
            self.img_paths.extend(img_paths)

        part_size = len(self.img_paths) / part_num
        l = part_id * part_size
        r = (part_id + 1) * part_size
        self.img_paths = self.img_paths[l:r]
        self.long_size = long_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        img = get_img(img_path)

        scaled_img, _ = scale(img, self.long_size)
        img = scaled_img.copy()
        scaled_img = Image.fromarray(scaled_img)
        scaled_img = scaled_img.convert('RGB')
        scaled_img = transforms.ToTensor()(scaled_img)
        scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)
        
        return img[:, :, [2, 1, 0]], scaled_img

class TestLoader(data.Dataset):
    def __init__(self, part_id=0, part_num=1, long_size=2240):
        data_dirs = [test_data_dir]
        
        self.img_paths = []
        
        for data_dir in data_dirs:
            img_names = util.io.ls(data_dir, '.jpg')
            img_names.extend(util.io.ls(data_dir, '.png'))

            img_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)
            
            self.img_paths.extend(img_paths)

        part_size = len(self.img_paths) / part_num
        l = part_id * part_size
        r = (part_id + 1) * part_size
        self.img_paths = self.img_paths[l:r]
        self.long_size = long_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        img = get_img(img_path)
        scaled_img, scale_fate = scale(img, self.long_size)
        #img = scaled_img.copy()
        #cv2.imwrite("./a.jpg", scaled_img)
        scaled_img = Image.fromarray(scaled_img)
        scaled_img = scaled_img.convert('RGB')
        scaled_img = transforms.ToTensor()(scaled_img)
        scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)
        
        return img[:, :, [2, 1, 0]], scaled_img, scale_fate