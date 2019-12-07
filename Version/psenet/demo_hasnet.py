import os
import cv2
import sys
import time
import collections
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from config import *
from torch.autograd import Variable
from torch.utils import data

from dataLoader import TestLoader
import fpn_resnet as models
#import fpn_resnet_dcn as models
# c++ version pse based on opencv 3+
from pse import pse
# python pse
# from pypse import pse as pypse
import glob
import shutil
#import onnx
USE_TF = False

class Detector():
    def __init__(self, model_path):
        # init Model
        #self._model = torch.jit.load(model_path)
        #self._model = onnx.load(model_path)

        self._model = models.resnet50(pretrained=True, num_classes=7, scale=1)
        print(len(list(self._model.parameters())))
        print(self._model.conv1.weight)
        return
        for param in self._model.parameters():
            param.requires_grad = False

        if torch.cuda.is_available() and GPU:
            self._model = self._model.cuda()
        else:
            self._model = self._model.cpu()

        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
            d = collections.OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            self._model.load_state_dict(d)
        else:
            print("No checkpoint found at '{}'".format(model_path))
        
        self._model.eval()
        example = torch.rand(1, 3, 800, 800)
        example = Variable(example.cuda())
        #torch.onnx.export(self._model, example, "model.proto", verbose=True)
        traced_script_module = torch.jit.trace(self._model, (example))
        traced_script_module.save("./model.pt")
        
    
    def detect(self, img):
        startTime0 = time.time()
        self.bboxes = []
        data_loader = TestLoader(img, long_size=DETE_IMG_SIZE)
        test_loader = torch.utils.data.DataLoader(
            data_loader,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            drop_last=True)
        for idx, (scale, img) in enumerate(test_loader):

            if torch.cuda.is_available() and GPU:
                #img = Variable(img.cuda(), volatile=True)
                img = Variable(img.cuda())
            else:
                #img = Variable(img.cpu(), volatile=True)
                img = Variable(img.cpu())
            #org_img = org_img.numpy().astype('uint8')[0]
            #text_box = org_img.copy()

            outputs = self._model(img)

            score = torch.sigmoid(outputs[:, 0, :, :])
            outputs = (torch.sign(outputs - DETE_BINARY_TH) + 1) / 2

            text = outputs[:, 0, :, :]
            kernels = outputs[:, 0:7, :, :] * text

            score = score.data.cpu().numpy()[0].astype(np.float32)
            text = text.data.cpu().numpy()[0].astype(np.uint8)
            kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
            cv2.imwrite("./7.jpg", kernels[0]*255)
            #cv2.imwrite("./6.jpg", kernels[1]*255)
            #cv2.imwrite("./5.jpg", kernels[2]*255)
            #cv2.imwrite("./4.jpg", kernels[3]*255)
            #cv2.imwrite("./3.jpg", kernels[4]*255)
            #cv2.imwrite("./2.jpg", kernels[5]*255)
            #cv2.imwrite("./1.jpg", kernels[6]*255)
            if USE_TF:
                mask_res, label_values = pse(kernels, 5.0)
                mask_res = np.array(mask_res)
                mask_res_resized = cv2.resize(mask_res, (mask_res.shape[1], mask_res.shape[0]), interpolation=cv2.INTER_NEAREST)
                boxes = []
                for label_value in label_values:
                    #(y,x)
                    points = np.argwhere(mask_res_resized==label_value)
                    points = points[:, (1,0)]
                    rect = cv2.minAreaRect(points)
                    box = cv2.boxPoints(rect) / (scale, scale)
                    box = box.astype('int32')
                    self.bboxes.append(box.reshape(-1))
                return
                    
            # c++ version pse
            pred = pse(kernels, 5.0)
            # python version pse
            # pred = pypse(kernels, 5.0)
            if(len(pred) == 0):
                continue
            self.bboxes = pred
            #print(self.bboxes, scale)
            #self.bboxes = self.bboxes / scale
            #self.bboxes = self.bboxes.astype('int32').tolist()
#             label = pred
#             label_num = np.max(label) + 1
#             whereup = 0
#             startTime = time.time()
#             for i in range(1, label_num):
#                 startTime1 = time.time()
#                 points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]
#                 whereup = whereup + time.time() - startTime1
                
#                 if points.shape[0] < DETE_MIN_AREA:
#                     continue

#                 score_i = np.mean(score[label == i])
#                 if score_i < DETE_MIN_SCORE:
#                     continue
#                 #if i == 2:
#                     #print(points)
#                 rect = cv2.minAreaRect(points)
#                 bbox = cv2.boxPoints(rect) / (scale/2, scale/2)
#                 bbox = bbox.astype('int32')
#                 self.bboxes.append(bbox.reshape(-1))
            print("Later:", time.time() - startTime)
            print("Total:", time.time() - startTime0)
            #print(bboxes)

def draw_bbox(img, bboxes, output_path):
    for bbox in bboxes:
            cv2.drawContours(img, [bbox.reshape(4, 2)], -1, (0, 255, 0), 2)      
    cv2.imwrite(output_path, img)
    
if __name__ == '__main__':
    output_Path = "./TestResult/"
    if os.path.exists(output_Path):
        shutil.rmtree(output_Path)
    os.makedirs(output_Path)
    
    model_path = "./checkpoints/checkpoint.pth.tar"
    #model_path = "./model.pt"
    dete_line = Detector(model_path)
    
    imgList = glob.glob(os.path.join("./TestImgs/", "*.jpg"))
    startTime = time.time()
    for img_path in imgList:
        print(img_path)
        img = cv2.imread(img_path)
        
        if torch.cuda.is_available() and GPU:
                torch.cuda.synchronize()
        start = time.time()

        dete_line.detect(img)
        fileSave = output_Path + os.path.split(img_path)[1]
        draw_bbox(img, dete_line.bboxes, fileSave)

        if torch.cuda.is_available() and GPU:
                torch.cuda.synchronize()
        end = time.time()
        print("Time is {0}s".format(end - start))
    print("Total is {0}s".format(time.time() - startTime))