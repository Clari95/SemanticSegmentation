import os
import torch

import argparser as parser
import models
import models_best
import data_test as data
import mean_iou_evaluate

import cv2
import json

import glob

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

from sklearn.metrics import accuracy_score



def evaluate(model, data_loader):
    
    args = parser.arg_parse()
    ''' set model to evaluate mode '''
    model.eval()
    preds = []
    gts = [] #ground truth
    print('start evaluate')
    with torch.no_grad(): # do not need to caculate information for gradient during eval
        for idx, (imgs, gt) in enumerate(data_loader):
            imgs = imgs.cuda()
            pred = model(imgs)
           
            
            _, pred = torch.max(pred, dim = 1)

            pred = pred.cpu().numpy().squeeze()
            gt = gt.numpy().squeeze()
            
            preds.append(pred)
            gts.append(gt)
        
        
    gts = np.concatenate(gts)
    
    preds = np.concatenate(preds)

    np.save(args.save_dir + 'preds.npy', preds) 		    
    return mean_iou_evaluate.mean_iou_score(preds, gts)#maybe gts preds#, preds#accuracy_score(gts, preds)


if __name__ == '__main__':

    
    args = parser.arg_parse()

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    test_loader = torch.utils.data.DataLoader(data.DATA_TEST(args, mode='test'),
                                              batch_size=args.test_batch, 
                                              num_workers=args.workers,
                                              shuffle=False)
    ''' prepare mode '''
    #load best model
    if(args.resume =='model_best.pth.tar?dl=1'):
        model = models.Net(args).cuda()
    else:
        model = models_best.Net(args).cuda()
    #save predictions

    ''' resume save model '''
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)

    #acc = evaluate(model, test_loader)
    #print('Testing Accuracy: {}'.format(acc))
    preds = []
    for idx, imgs in enumerate(test_loader):
            imgs = imgs.cuda()
            pred = model(imgs)
            
            _, pred = torch.max(pred, dim = 1)

            pred = pred.cpu().numpy().squeeze()
            
            
            preds.append(pred)
      
    
    #prediction as list --> save in directory: predictions --save_dir
    preds = np.concatenate(preds)

    for idx, pred_img in enumerate(preds):
        
        if idx<10:
          name = '000' + str(idx)+ '.png'
        elif idx<100 :
          name = '00' + str(idx)+ '.png'
        elif idx<1000:
          name = '0' + str(idx)+ '.png'
        else:
          name = str(i)+ '.png'
        im = Image.fromarray(pred_img.astype('uint8'))
        save_pred = os.path.join(args.save_dir, name)

        im.save(save_pred)
