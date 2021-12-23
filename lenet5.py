import os
import matplotlib.pyplot as plt
import numpy as np
import copy

import torch
import torch.nn as nn
from torch.nn.modules import loss
import torch.optim as optim 
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import ToTensor
from torchvision import datasets
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter
from datetime import datetime

device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
DATA_PATH = 'datasets/data/MNIST/'
CKPT_DIR = 'results/lenet/ckpt'
DEMO_PATH='results/lenet/demo'
SUM_PATH='results/lenet/runs'
writer = SummaryWriter(os.path.join(SUM_PATH, "{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())))
def create_dataset(data_dir, training=True, batch_size=32, resize=(32, 32),
                   rescale=1/(255*0.3081), shift=-0.1307/0.3081, buffer_size=64):
    """
    获取MNIST数据集，并返回数据迭代器
    args:
    data_dir:数据集存放路径
    training:训练/验证集选择标志
    batch_size:单次训练batch大小
    resize:调整后图片大小
    rescale:像素归一化因子
    shift:偏移(未使用)
    buffer_size:(未使用)

    returns:
    train_data_loader:训练数据迭代器
    test_data_loader:验证数据迭代器
    """

    transforms=T.Compose([
        T.Resize(resize),
        T.ToTensor(),
        T.Normalize([0],[1]),
    ])

    train_dataset=datasets.MNIST(root=data_dir,train=True,transform=transforms,download=True)
    test_dataset=datasets.MNIST(root=data_dir,train=False,transform=transforms,download=True)

    train_data_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
    test_data_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=0)

    return train_data_loader,test_data_loader

class LeNet5(nn.Module):
    """
    Lenet5网络定义
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        batch_size=x.size()[0]
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x).view(batch_size,-1)# flatten

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


def train(data_dir, ckpt_dir,epochs=10,lr=0.01, num_epochs=3,writer=None):
    """
    训练lenet5
    args:
    data_dir:数据集路径
    ckpt_dir:check point文件路径
    epochs:训练轮数
    lr:初始学习率
    num_epochs:check point文件保存间隔轮数(未使用)
    writer:tensorboard日志生成器
    """
    train_data_loader,_=create_dataset(data_dir)
    net = LeNet5().to(device)
    optimizer=optim.Adam(net.parameters(),lr)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    'max',
                                                    factor=0.5,
                                                    patience=3)
    criterion=nn.CrossEntropyLoss()
    best_loss=1e9
    best_model_weights=copy.deepcopy(net.state_dict())

    for epoch in range(epochs):
        net.train()
        losses=[]
        for batch_idx,(x,y)in tqdm(enumerate(train_data_loader,1)):
            x=x.to(device)
            y=y.to(device)
            pred_y=net(x)

            loss=criterion(pred_y,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        

        avg_loss=sum(losses)/len(losses)
        if avg_loss<best_loss:
            best_model_weights=copy.deepcopy(net.state_dict())
            best_loss=avg_loss
        print('step:' + str(epoch + 1) + '/' + str(epochs) + ' || Total Loss: %.4f' % (loss))
        if writer:
            writer.add_scalars('training_loss', {'train': avg_loss}, epoch+1)
        torch.save(best_model_weights,os.path.join(ckpt_dir,str(epoch)+'.pth'))
    print('Finish Training')

def test(data_dir,model_dir):
    """
    测试lenet5,保存验证集上表现最好的模型
    args:
    model_dir:模型存放路径
    """
    _,test_data_loader=create_dataset(data_dir)
    net=LeNet5().to(device)
    bestloss=1e9
    model_all_path=[]
    criterion=nn.CrossEntropyLoss()
    best_model_weights=None
    for files in os.listdir(model_dir):
        if files.split('.')[-1] == 'pth':
            model_all_path.append(files)
    for model in model_all_path:
        net.load_state_dict(torch.load(os.path.join(model_dir,model),map_location=device))
        net.eval()
        correct=0
        total=0
        with torch.no_grad():
            losses=[]
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
            if avg_loss<bestloss:
                bestloss=avg_loss
                best_model_weights=copy.deepcopy(net.state_dict())
                
        print(model+ '|| Total Loss: %.4f' % (avg_loss)+'|| Accuracy Rate: %.4f' %(correct/total))
    torch.save(best_model_weights,os.path.join(model_dir,'best.pth'))
    print('Finish Testing')

def show(data_dir,model_path,demo_path,show_num=20,show_rows=2,show_cols=10):
    """
    随机抽取训练集和测试集中数据可视化lenet5识别结果
    args:
    data_dir:数据集路径
    model_path:测试模型路径
    demo_path:可视化结果存放路径
    show_num:可视化识别结果数量
    show_rows:展示行数
    show_cols:展示列数
    """
    train_data_loader,test_data_loader=create_dataset(data_dir)
    _,(train_img_datas,train_img_idxes)=next(enumerate(train_data_loader))
    _,(test_img_datas,test_img_idxes)=next(enumerate(test_data_loader))
    print("check trainset:image shape|{}".format(train_img_datas.shape))
    print("check testset:image shape|{}".format(test_img_datas.shape))
    model=LeNet5().to(device)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.eval()
    train_real_idx=train_img_idxes.numpy()[:show_num]
    test_real_idx=test_img_idxes.numpy()[:show_num]
    train_show_imgs=train_img_datas.numpy()[:show_num]
    test_show_imgs=test_img_datas.numpy()[:show_num]
    train_eva_idx=[]
    test_eva_idx=[]

    with torch.no_grad():
        for i in range(show_num):
            x1=train_img_datas[i].to(device)
            x2=test_img_datas[i].to(device)
            y1=model(x1.unsqueeze(0))
            y2=model(x2.unsqueeze(0))
            if torch.cuda.is_available():
                y1=y1.cpu()
                y2=y2.cpu()
            _,y1=torch.max(y1,1)
            y1=int(y1[0])
            _,y2=torch.max(y2,1)
            y2=int(y2[0])

            train_eva_idx.append(y1)
            test_eva_idx.append(y2)
    plot_gallery(train_show_imgs,train_real_idx,train_eva_idx,demo_path,'train results',show_rows,show_cols)
    plot_gallery(test_show_imgs,test_real_idx,test_eva_idx,demo_path,'test results',show_rows,show_cols)

    pass


def plot_gallery(images,real_idxes,eva_idxes,path,name,n_row=0, n_col=0, h=32, w=32,show=True):  # 3行4列
    """
    展示多张图片
    :param images: numpy array 格式的图片
    :param titles: 图片标题
    :param h: 图像reshape的高
    :param w: 图像reshape的宽
    :param n_row: 展示行数
    :param n_col: 展示列数
    :return: 
    """
    # show pictures
    fig=plt.figure(num=name,figsize=(1.8 * n_col, 1.8 * n_row))
    fig.subplots_adjust(bottom=.01, left=.01, right=.99, top=.99, wspace=.01,hspace=.15)
    for i in range(n_row * n_col):
        ax=fig.add_subplot(n_row, n_col, i + 1)
        ax.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        ax.set_title("real:{} eval:{}".format(str(real_idxes[i]),str(eva_idxes[i])),
        color=("green"if real_idxes[i]==eva_idxes[i] else "red"))
    fig.savefig(os.path.join(path,name+'.jpg'),bbox_inches='tight')
    if show:
        fig.show()
if __name__=='__main__':
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    if not os.path.exists(DEMO_PATH):
        os.makedirs(DEMO_PATH)
    if not os.path.exists(SUM_PATH):
        os.makedirs(SUM_PATH)
    train_loader,_=create_dataset(DATA_PATH)
    # bidx,(x,y)=next(enumerate(train_loader))
    # print(x.shape)
    # train(DATA_PATH,CKPT_DIR,writer=writer)
    test(DATA_PATH,CKPT_DIR)
    show(DATA_PATH,os.path.join(CKPT_DIR,'best.pth'),DEMO_PATH)
    print('Finish all !!!')
