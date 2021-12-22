import torch
import torchvision
import copy
from Model.mobilenetv2 import MobileNetV2,SimpleHead,Combine

def pth2pt(in_path,out_path):
    backbone=MobileNetV2()
    head=SimpleHead()
    net=Combine(backbone,head)
    net.load_state_dict(torch.load(in_path,map_location='cpu'))
    net.eval()
    example = torch.rand(1, 3, 224, 224)
    mobile_module = torch.jit.trace(net, example)
    mobile_module.save(out_path)
    pass

def to_lowpth(in_path):
    backbone=MobileNetV2()
    head=SimpleHead()
    net=Combine(backbone,head)
    net.load_state_dict(torch.load(in_path,map_location='cpu'))
    weights=copy.deepcopy(net.state_dict())
    torch.save(weights,in_path,_use_new_zipfile_serialization=False)
    print("Finish!")
if __name__=='__main__':
    in_path='./results/mobilenetv2/final/best.pth'
    out_path='./results/mobilenetv2/final/best.pt'
    to_lowpth(in_path)
    pth2pt(in_path,out_path)
    print("Finish Converted!!!")