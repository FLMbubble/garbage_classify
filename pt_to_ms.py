from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor
import torch
import os

from mindspore.train.serialization import export, load_checkpoint, load_param_into_net
from mindspore import Tensor
import numpy as np
from src_mindspore.mobilenetv2 import MobileNetV2Backbone, mobilenet_v2
from pre_train import MobileNetV2Head
from easydict import EasyDict

index = {'00_00': 0, '00_01': 1, '00_02': 2, '00_03': 3, '00_04': 4, '00_05': 5, '00_06': 6, '00_07': 7,
         '00_08': 8, '00_09': 9, '01_00': 10, '01_01': 11, '01_02': 12, '01_03': 13, '01_04': 14, 
         '01_05': 15, '01_06': 16, '01_07': 17, '02_00': 18, '02_01': 19, '02_02': 20, '02_03': 21,
         '03_00': 22, '03_01': 23, '03_02': 24, '03_03': 25}
inverted = {0: 'Plastic Bottle', 1: 'Hats', 2: 'Newspaper', 3: 'Cans', 4: 'Glassware', 5: 'Glass Bottle', 6: 'Cardboard', 7: 'Basketball',
            8: 'Paper', 9: 'Metalware', 10: 'Disposable Chopsticks', 11: 'Lighter', 12: 'Broom', 13: 'Old Mirror', 14: 'Toothbrush',
            15: 'Dirty Cloth', 16: 'Seashell', 17: 'Ceramic Bowl', 18: 'Paint bucket', 19: 'Battery', 20: 'Fluorescent lamp', 21: 'Tablet capsules',
            22: 'Orange Peel', 23: 'Vegetable Leaf', 24: 'Eggshell', 25: 'Banana Peel'}

config = EasyDict({
    "num_classes": 26, # 分类数，即输出层的维度
    "reduction": 'mean', # mean, max, Head部分池化采用的方式
    "image_height": 224,
    "image_width": 224,
    "batch_size": 24, # 鉴于CPU容器性能，太大可能会导致训练卡住
    "eval_batch_size": 10,
    "epochs": 25, # 请尝试修改以提升精度
    "lr_max": 0.001, # 请尝试修改以提升精度
    "decay_type": 'consine', # 请尝试修改以提升精度
    "momentum": 0.95, # 请尝试修改以提升精度
    "weight_decay": 1.05, # 请尝试修改以提升精度
    "dataset_path": "./datasets/data/garbage_26x100",
    "features_path": "./results/garbage_26x100_features", # 临时目录，保存冻结层Feature Map，可随时删除
    "class_index": index,
    "save_ckpt_epochs": 1,
    "save_ckpt_path": './results/ckpt_mobilenetv2',
    "pretrained_ckpt": './src_mindspore/mobilenetv2-200_1067_cpu_gpu.ckpt',
    "export_path": './results/mobilenetv2.mindir'
    
})


def ckpt2ms(in_path,out_path):
    backbone=MobileNetV2Backbone()
    head=MobileNetV2Head()
    net=mobilenet_v2(backbone,head)
    par_dict=net.trainable_params()
    for name in par_dict:
        print('========================ckpt_name',name)
    param_dict = load_checkpoint(in_path)
    print('*'*30)
    for name in param_dict:
        print('========================ckpt_name',name)
    # load the parameter into net
    load_param_into_net(net, param_dict)
    input = np.random.uniform(0.0, 1.0, size=[32, 3, 224, 224]).astype(np.float32)
    export(net, Tensor(input), file_name=out_path, file_format='MINDIR')
    pass

def pt2ckpt(in_path,out_path):


    print("hello")
    par_dict = torch.load(in_path,map_location='cpu')
    # print(par_dict)
    new_params_list = []

    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]

        print('========================py_name',name)
        if name.endswith('normalize.bias'):
            name = name[:name.rfind('normalize.bias')]
            name = name + 'normalize.beta'
        elif name.endswith('normalize.weight'):
            name = name[:name.rfind('normalize.weight')]
            name = name + 'normalize.gamma'
        elif name.endswith('.running_mean'):
            name = name[:name.rfind('.running_mean')]
            name = name + '.moving_mean'
        elif name.endswith('.running_var'):
            name = name[:name.rfind('.running_var')]
            name = name + '.moving_variance'
        print('========================ms_name',name)

        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        new_params_list.append(param_dict)

    save_checkpoint(new_params_list,  out_path)

if __name__=='__main__':
    # in_path='./results/mobilenetv2/final/best.pth'
    # out_path='./results/mobilenetv2/final/best.ckpt'
    # mindir_path='./results/mobilenetv2/final/best.mindir'
    # # pt2ckpt(in_path,out_path)
    # ckpt2ms(out_path,mindir_path)
    

    backbone = MobileNetV2Backbone()
    # 导出带有Softmax层的模型
    head = MobileNetV2Head(input_channel=backbone.out_channels, num_classes=config.num_classes,
                        reduction=config.reduction, activation='Softmax')
    network = mobilenet_v2(backbone, head)
    CKPT='mobilenetv2-25.ckpt'
    load_checkpoint(os.path.join(config.save_ckpt_path, CKPT), net=network)

    input = np.random.uniform(0.0, 1.0, size=[1, 3, 224, 224]).astype(np.float32)
    export(network, Tensor(input), file_name=config.export_path, file_format='MINDIR')
    print("Finish convert mindir!!!")