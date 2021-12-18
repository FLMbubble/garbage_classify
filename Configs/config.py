from easydict import EasyDict
labels = {0: 'Plastic Bottle', 1: 'Hats', 2: 'Newspaper', 3: 'Cans', 4: 'Glassware', 5: 'Glass Bottle', 6: 'Cardboard', 7: 'Basketball',
            8: 'Paper', 9: 'Metalware', 10: 'Disposable Chopsticks', 11: 'Lighter', 12: 'Broom', 13: 'Old Mirror', 14: 'Toothbrush',
            15: 'Dirty Cloth', 16: 'Seashell', 17: 'Ceramic Bowl', 18: 'Paint bucket', 19: 'Battery', 20: 'Fluorescent lamp', 21: 'Tablet capsules',
            22: 'Orange Peel', 23: 'Vegetable Leaf', 24: 'Eggshell', 25: 'Banana Peel'}

config = EasyDict({
    "num_classes": 26, # 分类数，即输出层的维度
    "reduction": 'mean', # mean, max, Head部分池化采用的方式
    "HEIGHT": 224,
    "WIDTH": 224,
    "batch_size": 32, # 鉴于CPU容器性能，太大可能会导致训练卡住
    "eval_batch_size": 10,
    "epochs": 50, # 请尝试修改以提升精度
    "lr": 0.01, # 请尝试修改以提升精度
    "decay_type": 'constant', # 请尝试修改以提升精度
    "momentum": 0.9, # 请尝试修改以提升精度
    "weight_decay": 2.0, # 请尝试修改以提升精度
    "dataset_path": "./datasets/data/garbage_26x100",
    "features_path": "./results/mobilenetv2/garbage_26x100_features", # 临时目录，保存冻结层Feature Map，可随时删除
    "save_ckpt_epochs": 1,
    "save_ckpt_path": './results/mobilenetv2/ckpt',
    "pretrained_ckpt": './results/mobilenetv2/pretrained',
    "export_path": './results/mobilenetv2/final' ,
    "sum_path":'./results/mobilenetv2/runs',
    "TRAIN_LOG_PATH":'./results/mobilenetv2/runs/train/log',
    "TEST_LOG_PATH":'./results/mobilenetv2/runs/test/log'   
})