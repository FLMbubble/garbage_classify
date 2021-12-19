import os
import torch
import torchvision

import torchvision.transforms as T
from torch.utils.data import DataLoader, dataloader, dataset
from torchvision.datasets import ImageFolder

def create_dataset(config,training=True):
    data_path = os.path.join(config.dataset_path, 'train' if training else 'val')

    if training:
        transforms = T.Compose([
        T.RandomResizedCrop(size=config.HEIGHT),
        T.RandomHorizontalFlip(0.5),  # 进行随机水平翻转
        T.RandomVerticalFlip(0.5),  # 进行随机竖直翻转
        T.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
        T.ToTensor(),  # 转化为张量
        T.Normalize(mean=[0.485*255, 0.456*255, 0.406*255],std=[0.229*255, 0.224*255, 0.225*255]),  # 归一化
    ])
    else:
        transforms = T.Compose([
        T.Resize((int(config.HEIGHT/0.875),int(config.WIDTH/0.875))),
        T.CenterCrop(config.WIDTH),
        T.ToTensor(),  # 转化为张量
        T.Normalize(mean=[0.485*255, 0.456*255, 0.406*255],std=[0.229*255, 0.224*255, 0.225*255]),  # 归一化
    ])
    dataset = ImageFolder(data_path, transform=transforms)
    dataloader=DataLoader(dataset,config.batch_size,True,num_workers=0)
    return dataloader

def PairLabel(config):
    data_path = config.dataset_path
    transforms = T.Compose([
        T.RandomResizedCrop(size=config.HEIGHT),
        T.ToTensor(),  # 转化为张量
    ])
    dataset=ImageFolder(data_path,transform=transforms)
    data_loader=DataLoader(dataset,config.batch_size)
    batch_idx,(x,y)=next(enumerate(data_loader))
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    fig=plt.figure(num='label')
    fig.subplots_adjust(bottom=.01, left=.01, right=.99, top=.99, wspace=.01,hspace=.15)
    for i in range(2 * 5):
        ax=fig.add_subplot(2, 5, i + 1)
        img=np.array((x[i].numpy()*255).tolist(),dtype=np.uint8).transpose((1,2,0))
        # img=cv2.resize(img)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        ax.set_title("{}".format(int(y[i]))),
    fig.savefig(os.path.join(config.dataset_path,'label.jpg'),bbox_inches='tight')

def Raw_dataset(config):
    data_path = config.dataset_path
    transforms = T.Compose([
        T.ToTensor(),  # 转化为张量
    ])
    dataset=ImageFolder(data_path,transform=transforms)
    data_loader=DataLoader(dataset,config.batch_size)
    return data_loader
if __name__=='__main__':
    from easydict import EasyDict
    config=EasyDict({
        "dataset_path":"./datasets/data/label",
        "HEIGHT":224,
        "WIDTH":224,
        "batch_size":32
    })
    # data_loader=create_dataset(config)
    # batch_idx,(x,y)=next(enumerate(data_loader))
    # print(x.shape,y.shape)
    # print(y)
    PairLabel(config)