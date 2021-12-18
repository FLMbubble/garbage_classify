import os
import torch
import torchvision

import torchvision.transforms as T
from torch.utils.data import DataLoader, dataloader
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
        T.Resize(int(config.HEIGHT/0.875),int(config.WIDTH/0.875)),
        T.CenterCrop(config.WIDTH),
        T.ToTensor(),  # 转化为张量
        T.Normalize(mean=[0.485*255, 0.456*255, 0.406*255],std=[0.229*255, 0.224*255, 0.225*255]),  # 归一化
    ])
    dataset = ImageFolder(data_path, transform=transforms)
    dataloader=DataLoader(dataset,config.batch_size,True,num_workers=0)
    return dataloader

if __name__=='__main__':
    from easydict import EasyDict
    config=EasyDict({
        "dataset_path":"./datasets/data/garbage_26x100",
        "HEIGHT":224,
        "WIDTH":224,
        "batch_size":32
    })
    data_loader=create_dataset(config)
    batch_idx,(x,y)=next(enumerate(data_loader))
    print(x.shape,y.shape)
    print(y)