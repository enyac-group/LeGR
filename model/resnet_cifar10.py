'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class DownsampleA(nn.Module):  
  def __init__(self, nIn, nOut, stride):
    super(DownsampleA, self).__init__() 
    assert stride == 2    
    self.out_channels = nOut
    self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)   

  def forward(self, x):   
    x = self.avg(x)  
    if self.out_channels-x.size(1) > 0:
        return torch.cat((x, torch.zeros(x.size(0), self.out_channels-x.size(1), x.size(2), x.size(3), device='cuda')), 1) 
    else:
        return x

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = DownsampleA(in_planes, planes, stride)
            #self.shortcut = nn.Sequential(
            #    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            #    nn.BatchNorm2d(planes)
            #)

    def forward(self, x):
        x = F.relu(self.shortcut(x) + self.conv(x))
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.features = [conv_bn(3, 16, 1)]
        self.features.append(self._make_layer(block, 16, num_blocks[0], stride=1))
        self.features.append(self._make_layer(block, 32, num_blocks[1], stride=2))
        self.features.append(self._make_layer(block, 64, num_blocks[2], stride=2))
        self.features.append(nn.AvgPool2d(8))
        self.features = nn.Sequential(*self.features)
        
        self.classifier = nn.Sequential(nn.Linear(64, num_classes))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if type(m) in [nn.Conv2d, nn.Linear, nn.BatchNorm2d]:
                m.reset_parameters()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.classifier[0].in_features)
        x = self.classifier(x)
        return x


def ResNet20(num_classes=10):
    return ResNet(BasicBlock, [3,3,3], num_classes=num_classes)

def ResNet32(num_classes=10):
    return ResNet(BasicBlock, [5,5,5], num_classes=num_classes)

def ResNet44(num_classes=10):
    return ResNet(BasicBlock, [7,7,7], num_classes=num_classes)

def ResNet56(num_classes=10):
    return ResNet(BasicBlock, [9,9,9], num_classes=num_classes)


def test():
    net = ResNet20()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
