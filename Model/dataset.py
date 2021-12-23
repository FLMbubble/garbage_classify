import os
import torch
import torchvision

import torchvision.transforms as T
from torch.utils.data import DataLoader, dataloader, dataset
from torchvision.datasets import ImageFolder

def create_dataset(config,training=True):
    """
    创建数据集迭代器
    args:
    config:配置

    returns:
    data_loader:数据集迭代器
    """
    data_path = os.path.join(config.dataset_path, 'train' if training else 'val')

    if training:#训练数据增强
        transforms = T.Compose([
        T.RandomResizedCrop(size=config.HEIGHT),
        T.RandomHorizontalFlip(0.5),  # 进行随机水平翻转
        T.RandomVerticalFlip(0.5),  # 进行随机竖直翻转
        T.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
        T.ToTensor(),  # 转化为张量
        T.Normalize(mean=[0.485*255, 0.456*255, 0.406*255],std=[0.229*255, 0.224*255, 0.225*255]),  # 归一化
    ])
    else:#验证数据归一化
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
    """
    可视化数字标签与各类图片对应关系
    args:
    config:配置
    """
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
    for i in range(3 * 9):
        if i>=26:
            break
        ax=fig.add_subplot(3, 9, i + 1)
        img=np.array((x[i].numpy()*255).tolist(),dtype=np.uint8).transpose((1,2,0))
        # img=cv2.resize(img)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        ax.set_title("{}".format(int(y[i]))),
    fig.savefig(os.path.join(config.dataset_path,'label.jpg'),bbox_inches='tight')

def Raw_dataset(config):
    """
    不经任何处理，返回图像数据迭代器
    args:
    config:配置

    returns:
    data_loader:包含原始图像信息的数据迭代器
    """
    data_path = config.dataset_path
    transforms = T.Compose([
        T.ToTensor(),  # 转化为张量
    ])
    dataset=ImageFolder(data_path,transform=transforms)
    data_loader=DataLoader(dataset,config.batch_size)
    return data_loader

def show_enhance(config):
    """
    展示数据增强结果
    args:
    config:配置
    """
    data_path = os.path.join(config.train_path, 'train')
    transforms = T.Compose([
    T.RandomResizedCrop(size=config.HEIGHT),
    T.RandomHorizontalFlip(0.5),  # 进行随机水平翻转
    T.RandomVerticalFlip(0.5),  # 进行随机竖直翻转
    T.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
    T.ToTensor() # 转化为张量
    ])
    raws = T.Compose([
    T.ToTensor() # 转化为张量
    ])
    torch.manual_seed(0)
    # dataset = ImageFolder(data_path, transform=transforms)
    old_set=ImageFolder(data_path,transform=raws)
    # data_loader=DataLoader(dataset,config.batch_size,True,num_workers=0)
    old_loader=DataLoader(old_set,config.batch_size,True,num_workers=0)
    # batch_idx,(x,y)=next(enumerate(data_loader))
    _,(ox,oy)=next(enumerate(old_loader))
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    fig=plt.figure(num='enhance')
    fig.subplots_adjust(bottom=.01, left=.01, right=.99, top=.99, wspace=.01,hspace=.15)
    for i in range(4*5):
        if i>=20:
            break
        ax=fig.add_subplot(4, 5, i + 1)
        
        if (i//5)//2<1:
            img=np.array((ox[i-(i//5)*5].numpy()*255).tolist(),dtype=np.uint8).transpose((1,2,0))
            label=oy[i-(i//5)*5]
        else:
            img=np.array((ox[i-(i//5-1)*5].numpy()*255).tolist(),dtype=np.uint8).transpose((1,2,0))
            label=oy[i-(i//5-1)*5]
        # elif (i//5)<2:
        #     img=np.array((ox[i-(i//5)*5].numpy()*255).tolist(),dtype=np.uint8).transpose((1,2,0))
        #     img=transforms.__call__(img)
        #     img=img.numpy()
        # elif (i//5)<3:
        #     img=np.array((ox[i-(i//5-1)*5].numpy()*255).tolist(),dtype=np.uint8).transpose((1,2,0))
        # else:
        #     img=np.array((ox[i-(i//5-1)*5].numpy()*255).tolist(),dtype=np.uint8).transpose((1,2,0))
        if (i//5)%2==1:
            img=Image.fromarray(img.astype('uint8')).convert('RGB')
            img=transforms.__call__(img)
            img=img.numpy().transpose((1,2,0))
        
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        ax.set_title("{}".format(int(label))),
    fig.savefig(os.path.join(config.dataset_path,'enhance.jpg'),bbox_inches='tight')

def select_ckpt(config):
    """
    选择迁移学习训练后模型的路径
    args:
    config:配置

    returns:
    model_all_path:迁移学习训练后模型的路径
    """
    model_all_path=[]
    for files in os.listdir(config.save_ckpt_path):
        if 'transfer' in files.split('.')[0]:
            if files.split('.')[-1] == 'pth':
                model_all_path.append(files)
    return model_all_path

def select_raw_ckpt(config):
    """
    选择整网从头训练后模型的路径
    args:
    config:配置

    returns:
    model_all_path:整网从头训练后模型的路径
    """
    model_all_path=[]
    for files in os.listdir(config.save_ckpt_path):
        if 'pre' in files.split('.')[0]:
            if files.split('.')[-1] == 'pth':
                model_all_path.append(files)
    return model_all_path

if __name__=='__main__':
    from easydict import EasyDict
    config=EasyDict({
        "train_path":"./datasets/data/garbage_26x100",
        "dataset_path":"./datasets/data/label",
        "HEIGHT":224,
        "WIDTH":224,
        "batch_size":32
    })
    # data_loader=create_dataset(config)
    # batch_idx,(x,y)=next(enumerate(data_loader))
    # print(x.shape,y.shape)
    # print(y)
    # PairLabel(config)
    show_enhance(config)