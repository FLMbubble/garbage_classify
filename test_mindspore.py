# 1. 导入相关包
import os
import cv2
import numpy as np
import mindspore as ms
import torch
import glob
from mindspore import nn
from mindspore import Tensor
from easydict import EasyDict
from mindspore import context
from mindspore.train.serialization import load_checkpoint
from src_mindspore.mobilenetv2 import MobileNetV2Backbone, mobilenet_v2  # 模型定义脚本
from src_mindspore.dataset import create_dataset
from tqdm.auto import tqdm
from mindspore.train.model import Model
import matplotlib.pyplot as plt


# 2.系统测试部分标签与该处一致，请不要改动
# 垃圾分类数据集标签，以及用于标签映射的字典。
index = {'00_00': 0, '00_01': 1, '00_02': 2, '00_03': 3, '00_04': 4, '00_05': 5, '00_06': 6, '00_07': 7,
         '00_08': 8, '00_09': 9, '01_00': 10, '01_01': 11, '01_02': 12, '01_03': 13, '01_04': 14,
         '01_05': 15, '01_06': 16, '01_07': 17, '02_00': 18, '02_01': 19, '02_02': 20, '02_03': 21,
         '03_00': 22, '03_01': 23, '03_02': 24, '03_03': 25}
inverted = {0: 'Plastic Bottle', 1: 'Hats', 2: 'Newspaper', 3: 'Cans', 4: 'Glassware', 5: 'Glass Bottle', 6: 'Cardboard', 7: 'Basketball',
            8: 'Paper', 9: 'Metalware', 10: 'Disposable Chopsticks', 11: 'Lighter', 12: 'Broom', 13: 'Old Mirror', 14: 'Toothbrush',
            15: 'Dirty Cloth', 16: 'Seashell', 17: 'Ceramic Bowl', 18: 'Paint bucket', 19: 'Battery', 20: 'Fluorescent lamp', 21: 'Tablet capsules',
            22: 'Orange Peel', 23: 'Vegetable Leaf', 24: 'Eggshell', 25: 'Banana Peel'}

## 生成 main.py 时请勾选此 cell

# 3. NoteBook 模型调整参数部分，你可以根据自己模型需求修改、增加、删除、完善部分超参数
# 训练超参
config = EasyDict({
    "num_classes": 26,
    "reduction": 'mean',
    "image_height": 224,
    "image_width": 224,
    "eval_batch_size": 10,
    "epochs":25,
    "dataset_path":'./datasets/data/garbage_26x100',
    "batch_size": 24, # 鉴于CPU容器性能，太大可能会导致训练卡住
    "export_path": './results/mobilenetv2.mindir',
    "save_ckpt_path": './results/ckpt_mobilenetv2',
    "class_index": index,
})

# 4. 自定义模型Head部分
class GlobalPooling(nn.Cell):
    def __init__(self, reduction='mean'):
        super(GlobalPooling, self).__init__()
        if reduction == 'max':
            self.mean = ms.ops.ReduceMax(keep_dims=False)
        else:
            self.mean = ms.ops.ReduceMean(keep_dims=False)

    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x


class MobileNetV2Head(nn.Cell):
    def __init__(self, input_channel=1280, hw=7, num_classes=1000, reduction='mean', activation="None"):
        super(MobileNetV2Head, self).__init__()
        if reduction:
            self.flatten = GlobalPooling(reduction)
        else:
            self.flatten = nn.Flatten()
            input_channel = input_channel * hw * hw
        self.dense = nn.Dense(input_channel, num_classes, weight_init='ones', has_bias=False)
        if activation == "Sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "Softmax":
            self.activation = nn.Softmax()
        else:
            self.need_activation = False

    def construct(self, x):
        x = self.flatten(x)
        x = self.dense(x)
        if self.need_activation:
            x = self.activation(x)
        return x

def image_process(image):
    """Precess one image per time.
    
    Args:
        image: shape (H, W, C)
    """
    mean=[0.485*255, 0.456*255, 0.406*255]
    std=[0.229*255, 0.224*255, 0.225*255]
    image = (np.array(image) - mean) / std

#     image = image.transpose((2,0,1))
    image = image.transpose((2,0,1))
    # print(image.shape)

    img_tensor = Tensor(np.array([image], np.float32))
    # img_tensor = torch.Tensor(np.array([image]))
    
    
    return img_tensor

def predict(image):
    """
    加载模型和模型预测
    主要步骤:
        1.图片处理,此处尽量与训练模型数据处理一致
        2.用加载的模型预测图片的类别
    :param image: OpenCV 读取的图片对象，数据类型是 np.array，shape (H, W, C)
    :return: string, 模型识别图片的类别, 
            包含 'Plastic Bottle','Hats','Newspaper','Cans'等共 26 个类别
    """
    # -------------------------- 实现图像处理部分的代码 ---------------------------
    # 该处是与 NoteBook 训练数据预处理一致；
    # 如使用其它方式进行数据处理，请修改完善该处，否则影响成绩
    image = cv2.resize(image,(config.image_height, config.image_width))
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = image_process(image)
    
    # -------------------------- 实现模型预测部分的代码 ---------------------------
    logits=network(image)
    pred = np.argmax(logits.asnumpy(), axis=1)[0]

    
    return inverted[int(pred)]

def visualize(config):
    
    """
    以最优模型在验证集上所有数据做推理，并可视化推理结果
    args:
    config:推理配置
    """
    data_path = config.dataset_path
    save_path="./results/mobilenetv2/val"
    data_all=[]
    labels_all=[]
    files_all=glob.glob(pathname=os.path.join(data_path,'*/*.jpg'))
    for files in files_all:
        # print(files)
        if 'jpg' in files.split('.')[-1]:
            print(files.split('.')[-1],files.split('\\')[-2])
            data_all.append(files)
            labels_all.append(files.split('\\')[-2])

    # net=ClassifyNet()
    total_num=len(data_all)
    idx=0
    pic=1
    for num in range(len(data_all)):
        if idx>=total_num:
            break

        fig=plt.figure(num='val:'+str(pic))
        fig.subplots_adjust(bottom=.01, left=.01, right=.99, top=.99, wspace=.01,hspace=.30)
        
        for i in range(4 * 6):
            if idx>=total_num:
                break
            image = cv2.imread(data_all[idx])
            pred=predict(image)
            real=inverted[index[labels_all[idx]]]
            ax=fig.add_subplot(4, 6, i + 1)
            img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            # img=cv2.resize(img)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
            ax.set_title("{}\n{}".format(real,pred),color=("green"if real==pred else "red"),fontsize=6)
            idx+=1

        fig.savefig(os.path.join(save_path,'val'+str(pic)+'.jpg'),bbox_inches='tight')
        pic+=1
        print("Finish :{}".format(pic))
    print("Finish Visualize!!")

def test_model(config):
    eval_dataset = create_dataset(config=config,training=False)
    step_size = eval_dataset.get_dataset_size()
    
    backbone = MobileNetV2Backbone()
    # Freeze parameters of backbone.

    head = MobileNetV2Head(input_channel=backbone.out_channels, num_classes=config.num_classes, reduction=config.reduction)
    network = mobilenet_v2(backbone, head)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')   
    print('validating the model...')
    accs=[]
    for epoch in range(config.epochs):   
        load_checkpoint(os.path.join(config.save_ckpt_path,'mobilenetv2-'+str(epoch+1)+'.ckpt'), net=network)
        eval_model = Model(network, loss, metrics={'acc', 'loss'})
        acc = eval_model.eval(eval_dataset, dataset_sink_mode=False)
        accs.append(acc)
    print(acc)
if __name__=='__main__':
    os.environ['GLOG_v'] = '2'  # Log Level = Error
    has_gpu = (os.system('command -v nvidia-smi') == 0)
    print('Excuting with', 'GPU' if has_gpu else 'CPU', '.')
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU' if has_gpu else 'CPU')
    # backbone = MobileNetV2Backbone()
    # head = MobileNetV2Head(input_channel=backbone.out_channels, num_classes=config.num_classes, reduction=config.reduction)
    # network = mobilenet_v2(backbone, head)

    # 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
    # 如果你的模型是在 results 文件夹下的模型，则 model_path = './results/ckpt_mobilenetv2/mobilenetv2-4.ckpt'

    # model_path = './results/ckpt_mobilenetv2/mobilenetv2-25.ckpt'
    # load_checkpoint(model_path, net=network)
    # image_path = './datasets/data/garbage_26x100/val/00_00/00037.jpg'

    # # 使用 opencv 读取图片
    # image = cv2.imread(image_path)


    # 打印返回结果
    # print(predict(image))

    # visualize(config)
    test_model(config)