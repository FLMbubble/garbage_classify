# garbage_classify
本仓库实现了报告中所涉及的所有功能，在此对各文件做简要叙述。

#### Configs

###### config.py

存放pytorch框架代码训练、测试所使用的配置。以及标签对应关系。

###### datasets

###### garbage_26x100

26类图片数据集(包括训练集与测试集)

###### label

garbage_26x100抽取的部分图片，用于标签匹配

###### MNIST

MNIST数据集

#### Model

###### dataset.py

训练数据迭代器的创建、标签可视化匹配、数据增强效果可视化、模型路径读取

###### mobilenetv2

Mobilenetv2的pytorch实现，FC层构建的线性分类器的实现，图像分类网络的实现

#### image_classification

mindspore框架模型在android上部署项目文件

#### PytorchAndroid

pytorch框架模型在android上部署项目文件

#### results

###### ckpt_mobilenetv2

mindspore框架模型的check point文件

###### garbage_26x100_features

mindspore框架下mobilenetv2构成的backbone所提取的图像特征文件

###### lenet

pytorch框架下lenet5训练模型，运行时数据(损失函数值，识别准确率)，识别结果可视化图像

###### mobilenetv2

pytorch框架下图像分类网络训练、测试运行时数据，识别结果可视化

#### release

pytorch模型、mindspore模型部署app的apk文件

#### src_mindspore

###### dataset.py

训练数据迭代器的创建、标签可视化匹配、数据增强效果可视化、模型路径读取

###### mobilenetv2

Mobilenetv2的mindspore实现，FC层构建的线性分类器的实现，图像分类网络的实现

##### lenet5.py

pytorch框架MNIST数据集构建，lenet5搭建、训练、验证实现文件

##### pre_train.py

mindspore框架整网微调实现文件

##### pt_to_ms.py

pytorch框架模型转化为mindspore框架check point文件或ms模型脚本

##### pth2pt.py

pth文件转化为适用于端侧推理部署的pt文件脚本

##### test_mindspore.py

测试mindspore框架模型识别准确率与损失函数值脚本

##### test_torch.py

测试、选择pytorch框架下模型，模型识别结果可视化

##### train_main.py

mindspore框架迁移学习实现脚本(冻结backbone，微调FC)

##### train_torch.py

pytorch框架训练脚本(整网从头训练，加载预训练模型微调)

