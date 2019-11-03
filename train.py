import os
import torch
from tensorboardX import SummaryWriter
from torch import nn

import argparser as parser
import models
import data as data_c
import test

import numpy as np

#from test import evaluate

def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)


if __name__=='__main__':

    args = parser.arg_parse()
    

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    #''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(data_c.DATA(args, mode='train'),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(data_c.DATA(args, mode='val'),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=False)

    print(args.lr)
    print(args.epoch)


    ''' load model '''
    print('===> prepare model ...')
   # print(args.size())
    model = models.Net(args)

    checkpoint = torch.load('./log/model_best.pth.tar')
    model.load_state_dict(checkpoint)

    ###Hier auskomentieren##
    model.cuda() # load model to gpu

    #print(model)

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()


    ''' setup optimizer '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))


    ''' train model '''
    print('===> start training ...')
    iters = 0
    best_acc = 0
    for epoch in range(1, args.epoch+1):

        model.train()

        for idx, (imgs, cls) in enumerate(train_loader):

            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(train_loader))
            iters += 1

            ''' move data to gpu '''
            ###hier auskommentieren
            imgs, cls = imgs.cuda(), cls.cuda()

            ''' forward path '''
            output = model(imgs)
             #_, output = torch.max(output, dim = 1)

            ''' compute loss, backpropagation, update parameters '''
           # cls = torch.squeeze(cls)
            loss = criterion(output, cls) # compute loss

            optimizer.zero_grad()         # set grad of all parameters to zero
            loss.backward()               # compute gradient for each parameters
            optimizer.step()              # update parameters

            ''' write out information to tensorboard '''
            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
            train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

            print(train_info)
           
        print(epoch%args.val_epoch)
        if epoch%args.val_epoch == 0:
            #print('evaluate')
            ''' evaluate the model '''
            acc = test.evaluate(model, val_loader)
            writer.add_scalar('val_acc', acc, iters)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))

            ''' save best model '''
            if acc > best_acc:
                save_model(model, os.path.join(args.save_dir, 'model_best.pth.tar'))
                best_acc = acc

        ''' save model '''
        save_model(model, os.path.join(args.save_dir, 'model_{}.pth.tar'.format(epoch)))
