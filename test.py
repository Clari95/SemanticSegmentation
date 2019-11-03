import os
import torch

import argparser as parser
import models
import data
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
            #print('model predicted')
            
            _, pred = torch.max(pred, dim = 1)

            pred = pred.cpu().numpy().squeeze()
            gt = gt.numpy().squeeze()
            
            preds.append(pred)
            gts.append(gt)
        
        
    gts = np.concatenate(gts)
    
    #prediction as list --> save in directory: predictions --save_dir
    preds = np.concatenate(preds)

    np.save(args.save_dir + 'preds.npy', preds) 		    
    return mean_iou_evaluate.mean_iou_score(preds, gts)#maybe gts preds#, preds#accuracy_score(gts, preds)

def creat_json():

    img_dir = "hw2_data/val/img" # Enter Directory of all images 
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)
    files.sort() 
    data = []
    print('test')
    for f1 in files:
        img = cv2.imread(f1)
       # data.append(img)
        data.append([os.path.basename(f1),os.path.basename(f1)])
        #data.append('seg/'+ os.path.basename(f1))
        print(type(data))

    with open('test.json', 'w') as json_file:
        json.dump(data, json_file)

if __name__ == '__main__':

   # create_json()
    
    args = parser.arg_parse()

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    test_loader = torch.utils.data.DataLoader(data.DATA(args, mode='test'),
                                              batch_size=args.test_batch, 
                                              num_workers=args.workers,
                                              shuffle=True)
    ''' prepare mode '''
    #load best model
    model = models.Net(args).cuda()
    #save predictions

    ''' resume save model '''
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)

    acc = evaluate(model, test_loader)
    print('Testing Accuracy: {}'.format(acc))

    save_dir = args.save_dir
    save_pred = os.path.join(args.save_dir)
    preds_img = np.load(args.save_dir +'preds.npy')
    for idx, pred_img in enumerate(preds_img):
        
        if idx<10:
          name = '/000' + str(idx)+ '.png'
        elif idx<100 :
          name = '/00' + str(idx)+ '.png'
        elif idx<1000:
          name = '/0' + str(idx)+ '.png'
        else:
          name = str(i)+ '.png'
        im = Image.fromarray(pred_img.astype('uint8'))
        im.save(save_pred + name, 'JPEG')
