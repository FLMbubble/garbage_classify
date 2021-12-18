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
from torch.utils.data import dataloader
from tqdm.auto import tqdm
from Configs.config import config

import torch
import torchvision
from Model.dataset import create_dataset
from Model.mobilenetv2 import ClassifyNet
from datetime import datetime

def test(test_data_loader,config,logger):
    device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    net=ClassifyNet().to(device)
    bestloss=1e9
    model_all_path=[]
    criterion=nn.CrossEntropyLoss()
    best_model_weights=None
    for files in os.listdir(config.save_ckpt_path):
        if files.split('.')[-1] == 'pth':
            model_all_path.append(files)
    print(model_all_path)
    for model in model_all_path:
        logger.info("Testing Model {}".format(model).center(60,'-'))
        start_time=time.time()
        net.load_state_dict(torch.load(os.path.join(config.save_ckpt_path,model),map_location=device))
        net.eval()
        correct=0
        total=0
        with torch.no_grad():
            for batch_idx,(x,y) in tqdm(enumerate(test_data_loader,1)):
                x=x.to(device)
                y=y.to(device)
                pred_y=net(x)

                loss=criterion(pred_y,y)

                if loss<bestloss:
                    bestloss=loss
                    best_model_weights=copy.deepcopy(net.state_dict())
                _,predicted=torch.max(pred_y,1)
                total+=y.shape[0]#计算所有标签数量
                correct+=(predicted==y).sum()#计算预测正确数量
        print(model+ '|| Total Loss: %.4f' % (loss)+'|| Accuracy Rate: %.4f' %(correct/total))
        logger.info("Testing Time {:.6f} s | Avg Testing Loss {:.6f} | Avg Accuracy Rate {:.6f}".format(time.time()-start_time, loss,correct/total))
    torch.save(best_model_weights,os.path.join(config.export_path,'best.pth'))
    print('Finish Testing')

if __name__=='__main__':
    if not os.path.exists(config.export_path):
        os.makedirs(config.export_path)
    if not os.path.exists(config.TEST_LOG_PATH):
        os.makedirs(config.TEST_LOG_PATH)
    logging.basicConfig(level=logging.INFO,format="%(message)s",handlers=[logging.FileHandler(os.path.join(config.TEST_LOG_PATH, "{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now()))), logging.StreamHandler()])
    logger=logging.getLogger()#获取 logger对象，此处为root对象
    test_data_loader=create_dataset(config,training=False)
    test(test_data_loader,config,logger)
    print("Finish all!!!")
