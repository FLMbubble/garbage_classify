import math
import numpy as np
import os
import cv2
import random
import shutil
import time
import copy
import logging

import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
from easydict import EasyDict
from PIL import Image
from tqdm.auto import tqdm
from Configs.config import config,labels
import torchvision.transforms as T
from torch.utils.data import DataLoader, dataloader, dataset
from torchvision.datasets import ImageFolder
import torch
import torchvision
from Model.dataset import create_dataset
from Model.mobilenetv2 import ClassifyNet,Combine,MobileNetV2,SimpleHead
from datetime import datetime

def test(test_data_loader,config,logger):
    device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # net=ClassifyNet().to(device)
    backbone=MobileNetV2().to(device)
    head=SimpleHead().to(device)
    net=Combine(backbone,head)
    bestloss=1e9
    best_acc=0
    model_all_path=select_ckpt(config)
    criterion=nn.CrossEntropyLoss()
    best_model_weights=None
    print(model_all_path)
    for model in model_all_path:
        logger.info("Testing Model {}".format(model).center(60,'-'))
        start_time=time.time()
        net.load_state_dict(torch.load(os.path.join(config.save_ckpt_path,model),map_location=device))
        net.eval()
        correct=0
        total=0
        losses=[]
        with torch.no_grad():
            for batch_idx,(x,y) in tqdm(enumerate(test_data_loader,1)):
                x=x.to(device)
                y=y.to(device)
                pred_y=net(x)

                loss=criterion(pred_y,y)
                losses.append(loss.item())
                _,predicted=torch.max(pred_y,1)
                total+=y.shape[0]#计算所有标签数量
                correct+=(predicted==y).sum()#计算预测正确数量
        avg_loss=sum(losses)/len(losses) 
        # if avg_loss<bestloss:
        #     bestloss=avg_loss
        #     best_model_weights=copy.deepcopy(net.state_dict())
        acc=correct/total
        if best_acc<acc:
            best_acc=acc
            best_model_weights=copy.deepcopy(net.state_dict())
        print(model+ '|| Total Loss: %.4f' % (loss)+'|| Accuracy Rate: %.4f' %(correct/total))
        logger.info("Testing Time {:.6f} s | Avg Testing Loss {:.6f} | Avg Accuracy Rate {:.6f}".format(time.time()-start_time, loss,correct/total))
    torch.save(best_model_weights,os.path.join(config.export_path,'best.pth'))
    print('Finish Testing')

def to_mo(in_model_path,out_model_path):
    net=ClassifyNet()
    net.load_state_dict(torch.load(in_model_path))
    weights=copy.deepcopy(net.state_dict())
    torch.save(weights,out_model_path,_use_new_zipfile_serialization=False)

def select_ckpt(config):
    model_all_path=[]
    for files in os.listdir(config.save_ckpt_path):
        if 'transfer' in files.split('.')[0]:
            if files.split('.')[-1] == 'pth':
                model_all_path.append(files)
    return model_all_path

def check_best(test_data_loader,config):
    data_path = os.path.join(config.dataset_path,'val')
    transforms = T.Compose([
        T.ToTensor(),  # 转化为张量
        T.Normalize(mean=[0.485*255, 0.456*255, 0.406*255],std=[0.229*255, 0.224*255, 0.225*255])
    ])
    dataset=ImageFolder(data_path,transform=transforms)
    data_loader=DataLoader(dataset,config.batch_size)

    device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # net=ClassifyNet().to(device)
    backbone=MobileNetV2().to(device)
    head=SimpleHead().to(device)
    net=Combine(backbone,head)

    bestloss=1e9
    criterion=nn.CrossEntropyLoss()
    net.load_state_dict(torch.load(config.BEST,map_location=device))
    net.eval()
    correct=0
    total=0
    losses=[]
    with torch.no_grad():
        # for batch_idx,(x,y) in tqdm(enumerate(test_data_loader,1)):
        for batch_idx,(x,y) in tqdm(enumerate(data_loader,1)):
            x=x.to(device)
            y=y.to(device)
            pred_y=net(x)

            loss=criterion(pred_y,y)
            losses.append(loss.item())
            _,predicted=torch.max(pred_y,1)
            print('pred',predicted)
            print('real',y)
            total+=y.shape[0]#计算所有标签数量
            correct+=(predicted==y).sum()#计算预测正确数量
    avg_loss=sum(losses)/len(losses) 
    print('|| Total Loss: %.4f' % (loss)+'|| Accuracy Rate: %.4f' %(correct/total))
    print('Finish Checking')

def visualize(config):
    data_path = os.path.join(config.dataset_path,'val')
    transforms = T.Compose([
        T.ToTensor(),  # 转化为张量
        T.Normalize(mean=[0.485*255, 0.456*255, 0.406*255],std=[0.229*255, 0.224*255, 0.225*255])
    ])
    ori_transforms=T.Compose([
        T.ToTensor(),  # 转化为张量
    ])
    save_path="./results/mobilenetv2/val"
    dataset=ImageFolder(data_path,transform=transforms)
    ori_dataset=ImageFolder(data_path,transform=ori_transforms)
    data_loader=DataLoader(dataset,24)
    ori_loader=DataLoader(ori_dataset,24)
    # net=ClassifyNet()
    backbone=MobileNetV2()
    head=SimpleHead()
    net=Combine(backbone,head)
    net.load_state_dict(torch.load(config.BEST))
    net.eval()
    ori_loader=list(ori_loader)

    ori_imgs=[ori_loader[i][0].numpy() for i in range(len(ori_loader))]
    with torch.no_grad():
        for batch_idx,(x,y) in tqdm(enumerate(data_loader,1)):
            imgs=ori_imgs[batch_idx-1]
            pred_y=net(x)
            _,pred_y=torch.max(pred_y,1)
            print('pred',pred_y)
            print('real',y)
            fig=plt.figure(num='val:'+str(batch_idx))
            fig.subplots_adjust(bottom=.01, left=.01, right=.99, top=.99, wspace=.01,hspace=.30)
            
            for i in range(4 * 6):
                if i>=imgs.shape[0]:
                    break
                ax=fig.add_subplot(4, 6, i + 1)
                img=np.array((imgs[i]*255).tolist(),dtype=np.uint8).transpose((1,2,0))
                # img=cv2.resize(img)
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')
                ax.set_title("{}\n{}".format(labels[int(y[i])],labels[int(pred_y[i])]),color=("green"if int(y[i])==int(pred_y[i]) else "red"),fontsize=6)

            fig.savefig(os.path.join(save_path,'val'+str(batch_idx)+'.jpg'),bbox_inches='tight')
        print("Finish Visualize!!")

def test_one(img,net):
    transforms = T.Compose([
        T.ToTensor(),  # 转化为张量
        T.Normalize(mean=[0.485*255, 0.456*255, 0.406*255],std=[0.229*255, 0.224*255, 0.225*255])
    ])
    img=transforms.__call__(img)
    net.eval()
    with torch.no_grad():
        pred_y=net(img.unsqueeze(0))
    _,pred_y=torch.max(pred_y,1)
    return labels[int(pred_y[0])]
if __name__=='__main__':
    if not os.path.exists(config.export_path):
        os.makedirs(config.export_path)
    if not os.path.exists(config.TEST_LOG_PATH):
        os.makedirs(config.TEST_LOG_PATH)
    # logging.basicConfig(level=logging.INFO,format="%(message)s",handlers=[logging.FileHandler(os.path.join(config.TEST_LOG_PATH, "{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now()))), logging.StreamHandler()])
    # logger=logging.getLogger()#获取 logger对象，此处为root对象
    # test_data_loader=create_dataset(config,training=False)
    # test(test_data_loader,config,logger)
    # check_best(test_data_loader,config)
    # select_ckpt(config)
    # in_model_path='./results/mobilenetv2/final/best.pth'
    # out_model_path='./results/mobilenetv2/final/mo.pth'
    # to_mo(in_model_path,out_model_path)
    visualize(config)
    # backbone=MobileNetV2()
    # head=SimpleHead()
    # net=Combine(backbone,head)
    # net.load_state_dict(torch.load(config.BEST))
    # img=cv2.imread('./datasets/data/garbage_26x100/val/00_00/00037.jpg')
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # label=test_one(img,net)
    # print(label)
    print("Finish all!!!")
