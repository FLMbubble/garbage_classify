import math
import numpy as np
import os
import cv2
import random
import shutil
import time
import copy
import itertools
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
from easydict import EasyDict
from PIL import Image
from torch.utils.data import dataloader
from tqdm.auto import tqdm

from tensorboardX import SummaryWriter, writer
import torch
import torchvision
from Model.dataset import create_dataset
from Model.mobilenetv2 import ClassifyNet,MobileNetV2,SimpleHead,Combine
from datetime import datetime

device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
labels = {0: 'Plastic Bottle', 1: 'Hats', 2: 'Newspaper', 3: 'Cans', 4: 'Glassware', 5: 'Glass Bottle', 6: 'Cardboard', 7: 'Basketball',
            8: 'Paper', 9: 'Metalware', 10: 'Disposable Chopsticks', 11: 'Lighter', 12: 'Broom', 13: 'Old Mirror', 14: 'Toothbrush',
            15: 'Dirty Cloth', 16: 'Seashell', 17: 'Ceramic Bowl', 18: 'Paint bucket', 19: 'Battery', 20: 'Fluorescent lamp', 21: 'Tablet capsules',
            22: 'Orange Peel', 23: 'Vegetable Leaf', 24: 'Eggshell', 25: 'Banana Peel'}

config = EasyDict({
    "num_classes": 26, # 分类数，即输出层的维度
    "reduction": 'mean', # mean, max, Head部分池化采用的方式
    "HEIGHT": 224,
    "WIDTH": 224,
    "batch_size": 16, # 鉴于CPU容器性能，太大可能会导致训练卡住
    "eval_batch_size": 10,
    "epochs": 50, # 请尝试修改以提升精度
    "lr": 0.0001, # 请尝试修改以提升精度
    "decay_type": 'constant', # 请尝试修改以提升精度
    "momentum": 0.9, # 请尝试修改以提升精度
    "weight_decay": 2.0, # 请尝试修改以提升精度
    "device":device,
    "backbone":"./results/mobilenetv2/final/mobilenetv2.pth",
    "dataset_path": "./datasets/data/garbage_26x100",
    "features_path": "./results/mobilenetv2/garbage_26x100_features", # 临时目录，保存冻结层Feature Map，可随时删除
    "save_ckpt_epochs": 1,
    "save_ckpt_path": './results/mobilenetv2/ckpt',
    "pretrained_ckpt": './results/mobilenetv2/ckpt/transfer49.pth',
    "export_path": './results/mobilenetv2/final' ,
    "sum_path":'./results/mobilenetv2/runs'   
})
def train(train_data_loader,config):
    net=ClassifyNet().to(config.device)
    # backbone=MobileNetV2().to(config.device)
    # head=SimpleHead().to(config.device)
    # optimizer=optim.Adam(itertools.chain(backbone.parameters(),head.parameters()),config.lr)
    optimizer=optim.Adam(net.parameters(),config.lr)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    'max',
                                                    factor=0.5,
                                                    patience=3)
    criterion=nn.CrossEntropyLoss()
    best_loss=1e9
    best_model_weights=copy.deepcopy(net.state_dict())
    # best_model_weights=copy.deepcopy(dict(backbone.state_dict().items()+head.state_dict().items()))
    # backbone.load_state_dict(torch.load(config.backbone,map_location=config.device))
    
    # backbone.requires_grad_=False
    # backbone.train()
    # head.train()
    net.train()
    for epoch in range(config.epochs):
        # net.train()
        total=0
        correct=0
        for batch_idx,(x,y)in tqdm(enumerate(train_data_loader,1)):
            x=x.to(device)
            y=y.to(device)
            pred_y=net(x)
            # feature=backbone(x)
            # pred_y=head(feature)

            # backbone.zero_grad()
            # head.zero_grad()
            net.zero_grad()
            loss=criterion(pred_y,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss<best_loss:
                best_model_weights=copy.deepcopy(net.state_dict())
                # best_model_weights=copy.deepcopy(dict(backbone.state_dict().items()+head.state_dict().items()))
                best_loss=loss
            _,predicted=torch.max(pred_y,1)
            total+=y.shape[0]#计算所有标签数量
            correct+=(predicted==y).sum()#计算预测正确数量
               
        print('step:' + str(epoch + 1) + '/' + str(config.epochs) + ' || Total Loss: %.4f' % (loss)+'|| Accuracy Rate: %.4f' %(correct/total))
        if writer:
            writer.add_scalars('training_loss', {'train': loss}, epoch+1)
            writer.add_scalars('training_accuracy', {'train': correct/total}, epoch+1)
        if epoch%config.save_ckpt_epochs==0:
            torch.save(best_model_weights,os.path.join(config.save_ckpt_path,str(epoch)+'.pth'))
    print('Finish Training')

def pre_train(train_data_loader,config):
    # net=ClassifyNet().to(config.device)
    backbone=MobileNetV2().to(config.device)
    head=SimpleHead().to(config.device)
    backbone.load_state_dict(torch.load(config.backbone,map_location=config.device))
    net=Combine(backbone,head)
    # optimizer=optim.Adam(itertools.chain(backbone.parameters(),head.parameters()),config.lr)
    optimizer=optim.Adam(net.parameters(),config.lr)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    'max',
                                                    factor=0.5,
                                                    patience=3)
    criterion=nn.CrossEntropyLoss()
    # net.load_state_dict(torch.load(config.pretrained_ckpt,map_location=config.device))
    best_loss=1e9
    best_model_weights=copy.deepcopy(net.state_dict())
    # best_model_weights=copy.deepcopy(dict(backbone.state_dict().items()+head.state_dict().items()))
    # backbone.load_state_dict(torch.load(config.backbone,map_location=config.device))
    
    # backbone.requires_grad_=False
    # backbone.train()
    # head.train()
    net.train()
    for epoch in range(config.epochs):
        # net.train()
        total=0
        correct=0
        losses=[]
        for batch_idx,(x,y)in tqdm(enumerate(train_data_loader,1)):
            x=x.to(device)
            y=y.to(device)
            pred_y=net(x)
            # feature=backbone(x)
            # pred_y=head(feature)

            # backbone.zero_grad()
            # head.zero_grad()
            net.zero_grad()
            loss=criterion(pred_y,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            _,predicted=torch.max(pred_y,1)
            total+=y.shape[0]#计算所有标签数量
            correct+=(predicted==y).sum()#计算预测正确数量

        avg_loss=sum(losses)/len(losses) 
        if avg_loss<best_loss:
                best_model_weights=copy.deepcopy(net.state_dict())
                # best_model_weights=copy.deepcopy(dict(backbone.state_dict().items()+head.state_dict().items()))
                best_loss=avg_loss    
        print('step:' + str(epoch + 1) + '/' + str(config.epochs) + ' || Total Loss: %.4f' % (avg_loss)+'|| Accuracy Rate: %.4f' %(correct/total))
        if writer:
            writer.add_scalars('training_loss', {'train': avg_loss}, epoch+1)
            writer.add_scalars('training_accuracy', {'train': correct/total}, epoch+1)
        if epoch%config.save_ckpt_epochs==0:
            torch.save(best_model_weights,os.path.join(config.save_ckpt_path,'transfer'+str(epoch)+'.pth'))
    print('Finish Training')

def extract_features(net,config):
    if not os.path.exists(config.features_path):
        os.makedirs(config.features_path)
    data_loader = create_dataset(config=config)
    step_size=len(data_loader)
    if step_size == 0:
        raise ValueError("The step_size of dataset is zero. Check if the images count of train dataset is more \
            than batch_size in config.py")
    for i,(x,y)in tqdm(enumerate(data_loader,1)):
        features_path = os.path.join(config.features_path, f"feature_{i}.npy")
        label_path = os.path.join(config.features_path, f"label_{i}.npy")
        if not os.path.exists(features_path) or not os.path.exists(label_path):
            image = x
            label = y
            features = net(image)
            np.save(features_path, features.data.numpy())
            np.save(label_path, label.numpy())
        print(f"Complete the batch {i+1}/{step_size}")
    return

def fine_tune(train_data_loader,config):
    # net=ClassifyNet().to(config.device)
    backbone=MobileNetV2().to(config.device)
    head=SimpleHead().to(config.device)
    net=Combine(backbone,head)
    # optimizer=optim.Adam(itertools.chain(backbone.parameters(),head.parameters()),config.lr)
    optimizer=optim.Adam(net.parameters(),config.lr)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    'max',
                                                    factor=0.5,
                                                    patience=3)
    criterion=nn.CrossEntropyLoss()
    net.load_state_dict(torch.load(config.pretrained_ckpt,map_location=config.device))
    best_loss=1e9
    best_model_weights=copy.deepcopy(net.state_dict())
    # best_model_weights=copy.deepcopy(dict(backbone.state_dict().items()+head.state_dict().items()))
    # backbone.load_state_dict(torch.load(config.backbone,map_location=config.device))
    
    # backbone.requires_grad_=False
    # backbone.train()
    # head.train()
    net.train()
    for epoch in range(config.epochs):
        # net.train()
        total=0
        correct=0
        losses=[]
        for batch_idx,(x,y)in tqdm(enumerate(train_data_loader,1)):
            x=x.to(device)
            y=y.to(device)
            pred_y=net(x)
            # feature=backbone(x)
            # pred_y=head(feature)

            # backbone.zero_grad()
            # head.zero_grad()
            net.zero_grad()
            loss=criterion(pred_y,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            _,predicted=torch.max(pred_y,1)
            total+=y.shape[0]#计算所有标签数量
            correct+=(predicted==y).sum()#计算预测正确数量

        avg_loss=sum(losses)/len(losses) 
        if avg_loss<best_loss:
                best_model_weights=copy.deepcopy(net.state_dict())
                # best_model_weights=copy.deepcopy(dict(backbone.state_dict().items()+head.state_dict().items()))
                best_loss=avg_loss    
        print('step:' + str(epoch + 1) + '/' + str(config.epochs) + ' || Total Loss: %.4f' % (avg_loss)+'|| Accuracy Rate: %.4f' %(correct/total))
        if writer:
            writer.add_scalars('training_loss', {'train': avg_loss}, epoch+1)
            writer.add_scalars('training_accuracy', {'train': correct/total}, epoch+1)
        if epoch%config.save_ckpt_epochs==0:
            torch.save(best_model_weights,os.path.join(config.save_ckpt_path,'Finetune'+str(epoch)+'.pth'))
    print('Finish Training')

if __name__=='__main__':
    if not os.path.exists(config.features_path):
        os.makedirs(config.features_path)
    if not os.path.exists(config.save_ckpt_path):
        os.makedirs(config.save_ckpt_path)
    if not os.path.exists(config.pretrained_ckpt):
        os.makedirs(config.pretrained_ckpt)
    if not os.path.exists(config.export_path):
        os.makedirs(config.export_path)
    writer=SummaryWriter(os.path.join(config.sum_path, "{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())))
    data_loader=create_dataset(config)
    # train(data_loader,config)
    pre_train(data_loader,config)
    # fine_tune(data_loader,config)
    print("Finish all!!!")
    pass