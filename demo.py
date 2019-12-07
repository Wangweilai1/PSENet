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

from torch.autograd import Variable
from torch.utils import data

from dataset import TestLoader
import models
import util
# c++ version pse based on opencv 3+
from pse import pse
# python pse
#from pypse import pse as pypse
import glob
import shutil

def extend_3c(img):
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img = np.concatenate((img, img, img), axis=2)
    return img

def debug(idx, img_paths, imgs, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    col = []
    for i in range(len(imgs)):
        row = []
        for j in range(len(imgs[i])):
            # img = cv2.copyMakeBorder(imgs[i][j], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
            row.append(imgs[i][j])
        res = np.concatenate(row, axis=1)
        col.append(res)
    res = np.concatenate(col, axis=0)
    img_name = img_paths[idx].split('/')[-1]
    print (idx, '/', len(img_paths), img_name)
    cv2.imwrite(output_root + img_name, res)

def write_result_as_txt(image_name, bboxes, path):
    filename = util.io.join_path(path, 'res_%s.txt'%(image_name))
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        line = "%d, %d, %d, %d, %d, %d, %d, %d\n"%tuple(values)
        lines.append(line)
    util.io.write_lines(filename, lines)

def polygon_from_points(points):
    """
    Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """
    resBoxes=np.empty([1, 8],dtype='int32')
    resBoxes[0, 0] = int(points[0])
    resBoxes[0, 4] = int(points[1])
    resBoxes[0, 1] = int(points[2])
    resBoxes[0, 5] = int(points[3])
    resBoxes[0, 2] = int(points[4])
    resBoxes[0, 6] = int(points[5])
    resBoxes[0, 3] = int(points[6])
    resBoxes[0, 7] = int(points[7])
    pointMat = resBoxes[0].reshape([2, 4]).T
    return plg.Polygon(pointMat)

def test(args):
    if os.path.exists("./TestResult/"):
        shutil.rmtree("./TestResult/")
    os.makedirs("./TestResult/")
    data_loader = TestLoader(long_size=args.long_size)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=True)

    # Setup Model
    if args.arch == "resnet50":
        model = models.resnet50(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet101":
        model = models.resnet101(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet152":
        model = models.resnet152(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet34":
        model = models.resnet34(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet18":
        model = models.resnet18(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "se_resnet_50":
        model = models.se_resnet_50(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "se_resnext_50":
        model = models.se_resnext_50(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "dcn_resnet50":
        model = models.dcn_resnet50(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet50_lstm":
        model = models.resnet50_lstm(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet50_aspp":
        model = models.resnet50_aspp(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet50_psp":
        model = models.resnet50_psp(pretrained=True, num_classes=7, scale=args.scale)
    
    for param in model.parameters():
        param.requires_grad = False

    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()
    
    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
            
            # model.load_state_dict(checkpoint['state_dict'])
            d = collections.OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)

            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            sys.stdout.flush()
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            sys.stdout.flush()

    model.eval()
    
    #example = torch.rand(1, 3, 800, 800)
    #example = Variable(example.cuda())
    #traced_script_module = torch.jit.trace(model, (example))
    #traced_script_module.save("./model.pt")
    #torch.onnx.export(model, example, "model.onnx", verbose=True)
        
    total_frame = 0.0
    total_time = 0.0
    for idx, (org_img, img, scale) in enumerate(test_loader):
        print('progress: %d / %d'%(idx, len(test_loader)))
        sys.stdout.flush()

        if torch.cuda.is_available():
            img = Variable(img.cuda(), volatile=True)
        else:
            img = Variable(img.cpu(), volatile=True)
        org_img = org_img.numpy().astype('uint8')[0]
        text_box = org_img.copy()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()

        outputs = model(img)

        score = torch.sigmoid(outputs[:, 0, :, :])
        outputs = (torch.sign(outputs - args.binary_th) + 1) / 2

        text = outputs[:, 0, :, :]
        kernels = outputs[:, 0:args.kernel_num, :, :] * text

        score = score.data.cpu().numpy()[0].astype(np.float32)
        text = text.data.cpu().numpy()[0].astype(np.uint8)
        kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
        #cv2.imwrite("./6.jpg", kernels[0]*255)
        #cv2.imwrite("./5.jpg", kernels[1]*255)
        #cv2.imwrite("./4.jpg", kernels[2]*255)
        #cv2.imwrite("./3.jpg", kernels[3]*255)
        #cv2.imwrite("./2.jpg", kernels[4]*255)
        #cv2.imwrite("./1.jpg", kernels[5]*255)
        #cv2.imwrite("./0.jpg", kernels[6]*255)
        # c++ version pse
        #kernels = kernels[:-1]
        pred = pse(kernels, args.min_kernel_area / (args.scale * args.scale))
        # python version pse
        #pred = pypse(kernels, args.min_kernel_area / (args.scale * args.scale))
        if(len(pred) == 0):
            continue
        #scale = (org_img.shape[0] * 1.0 / pred.shape[0], org_img.shape[1] * 1.0 / pred.shape[1])
        #print(org_img.shape, pred.shape, scale)
        label = pred
        label_num = np.max(label) + 1
        bboxes = []
        for i in range(1, label_num):
            points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

            if points.shape[0] < args.min_area / (args.scale * args.scale):
                continue

            score_i = np.mean(score[label == i])
            if score_i < args.min_score:
                continue

            rect = cv2.minAreaRect(points)
            bbox = cv2.boxPoints(rect) / (scale, scale)
            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        total_frame += 1
        total_time += (end - start)
        print('fps: %.2f'%(total_frame / total_time))
        sys.stdout.flush()

        for bbox in bboxes:
            cv2.drawContours(text_box, [bbox.reshape(4, 2)], -1, (0, 0, 255), 4)

        image_name = data_loader.img_paths[idx].split('/')[-1].split('.')[0]
        #write_result_as_txt(image_name, bboxes, './TestResult/')
        
        #text_box = cv2.resize(text_box, (text.shape[1], text.shape[0]))
        debug(idx, data_loader.img_paths, [[text_box]], './TestResult/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet50')
    parser.add_argument('--resume', nargs='?', type=str, default='./checkpoints/Total_ic19_resnet50_bs_16_ep_600_pretrain_ic17/checkpoint.pth.tar',  
    #parser.add_argument('--resume', nargs='?', type=str, default='./checkpoints/ic19_mlt_resnet50_bs_16_ep_600_pretrain_ic17/checkpoint.pth.tar',  
                        help='Path to previous saved model to restart from')
    parser.add_argument('--binary_th', nargs='?', type=float, default=0.5,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--kernel_num', nargs='?', type=int, default=7,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--scale', nargs='?', type=int, default=1,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--long_size', nargs='?', type=int, default=800,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--min_kernel_area', nargs='?', type=float, default=5.0,
                        help='min kernel area')
    parser.add_argument('--min_area', nargs='?', type=float, default=50.0,
                        help='min area')
    parser.add_argument('--min_score', nargs='?', type=float, default=0.60,
                        help='min score')
    
    args = parser.parse_args()
    test(args)
