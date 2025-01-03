import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
import pdb
from collections import OrderedDict

# 定义带两个卷积路径和一条捷径的残差基本块类
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):  # 初始化函数，in_planes为输入通道数，planes为输出通道数，步长默认为1
        super(BasicBlock, self).__init__()
        # 定义第一个卷积，默认卷积前后图像大小不变但可修改stride使其变化，通道可能改变
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # 定义第一个批归一化
        self.bn1 = nn.BatchNorm2d(planes)
        # 定义第二个卷积，卷积前后图像大小不变，通道数不变
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # 定义第二个批归一化
        self.bn2 = nn.BatchNorm2d(planes)

        # 定义一条捷径，若两个卷积前后的图像尺寸有变化(stride不为1导致图像大小变化或通道数改变)，捷径通过1×1卷积用stride修改大小
        # 以及用expansion修改通道数，以便于捷径输出和两个卷积的输出尺寸匹配相加
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    # 定义前向传播函数，输入图像为x，输出图像为out
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 第一个卷积和第一个批归一化后用ReLU函数激活
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 第二个卷积和第二个批归一化后与捷径相加
        out = F.relu(out)  # 两个卷积路径输出与捷径输出相加后用ReLU激活
        return out


# 定义残差网络ResNet18
class ResNet(nn.Module):
    # 定义初始函数，输入参数为残差块，残差块数量，默认参数为分类数10
    def __init__(self, block, num_blocks, class_num=10):
        super(ResNet, self).__init__()
        # 设置第一层的输入通道数
        self.in_planes = 64

        # 定义输入图片先进行一次卷积与批归一化，使图像大小不变，通道数由3变为64得两个操作
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 定义第一层，输入通道数64，有num_blocks[0]个残差块，残差块中第一个卷积步长自定义为1
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # 定义第二层，输入通道数128，有num_blocks[1]个残差块，残差块中第一个卷积步长自定义为2
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # 定义第三层，输入通道数256，有num_blocks[2]个残差块，残差块中第一个卷积步长自定义为2
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # 定义第四层，输入通道数512，有num_blocks[3]个残差块，残差块中第一个卷积步长自定义为2
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 定义全连接层，输入512*block.expansion个神经元，输出10个分类神经元
        self.linear = nn.Linear(512 * block.expansion, class_num)
        self.is_on_client = None
        self.is_on_server = None
        
        
    # 定义创造层的函数，在同一层中通道数相同，输入参数为残差块，通道数，残差块数量，步长
    def _make_layer(self, block, planes, num_blocks, stride):
        # strides列表第一个元素stride表示第一个残差块第一个卷积步长，其余元素表示其他残差块第一个卷积步长为1
        strides = [stride] + [1] * (num_blocks - 1)
        # 创建一个空列表用于放置层
        layers = []
        # 遍历strides列表，对本层不同的残差块设置不同的stride
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))  # 创建残差块添加进本层
            self.in_planes = planes * block.expansion  # 更新本层下一个残差块的输入通道数或本层遍历结束后作为下一层的输入通道数
        return nn.Sequential(*layers)  # 返回层列表

    # 定义前向传播函数，输入图像为x，输出预测数据
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 第一个卷积和第一个批归一化后用ReLU函数激活
        out = self.layer1(out)  # 第一层传播
        out = self.layer2(out)  # 第二层传播
        out = self.layer3(out)  # 第三层传播
        out = self.layer4(out)  # 第四层传播
        out = F.avg_pool2d(out, 4)  # 经过一次4×4的平均池化
        out = out.view(out.size(0), -1)  # 将数据flatten平坦化
        out = self.linear(out)  # 全连接传播
        return out

    def client_params_requires_grad_(self, requires_grad):
        for p in self.client_parameters():
            p.requires_grad_(requires_grad)

    def server_params_requires_grad_(self, requires_grad):
        for p in self.server_parameters():
            p.requires_grad_(requires_grad)
            
    def client_parameters(self):
        return [p for (n, p) in self.named_parameters() if self.is_on_client(n)]
    
    def server_parameters(self):
        return [p for (n, p) in self.named_parameters() if self.is_on_server(n)]
    
    #参数
    def split_server_and_client_params(self, client_mode, layers_to_client, adapter_hidden_dim=-1, dropout=0.):
        device = next(self.parameters()).device
        if self.is_on_client is not None:
            raise ValueError('This model has already been split across clients and server.')
        assert client_mode in ['none', 'res_layer', 'inp_layer', 'out_layer', 'adapter', 'interpolate', 'finetune'] 
        # Prepare
        if layers_to_client is None:  # no layers to client
            layers_to_client = []
        if client_mode == 'res_layer' and len(layers_to_client) is None:
            raise ValueError(f'No residual blocks to finetune. Nothing to do')
        is_on_server = None
        
        # Set requires_grad based on `train_mode`
        if client_mode in ['none', None]:
            # no parameters on the client
            def is_on_client(name):
                return False
        elif 'res_layer' in client_mode:
            # Specific residual blocks are sent to client (available layers are [1, 2, 3, 4])
            def is_on_client(name):
                return any([f'layer{i}' in name for i in layers_to_client])
        elif client_mode in ['inp_layer']:
            # First convolutional layer is sent to client
            def is_on_client(name):
                return (name in ['conv1.weight', 'bn1.weight', 'bn1.bias'])  # first conv + bn
            self.drop_i = nn.Dropout(dropout)
        elif client_mode in ['out_layer']:
            # Final linear layer is sent to client
            def is_on_client(name):
                return (name in ['linear.weight', 'linear.bias'])  # final fc
            self.drop_o = nn.Dropout(dropout)
        elif client_mode in ['adapter']:
            # Train adapter modules (+ batch norm)
            def is_on_client(name):
                return ('adapter' in name) or ('bn1' in name) or ('bn2' in name)
            # Add adapter modules
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for block in layer.children():
                    # each block is of type `ResidualBlock`
                    block.add_adapters(dropout)
        elif client_mode == 'interpolate':  # both on client and server
            is_on_client = lambda _: True
            is_on_server = lambda _: True
        elif client_mode == 'finetune':  # all on client
            is_on_client = lambda _: True
            is_on_server = lambda _: False
        else:
            raise ValueError(f'Unknown client_mode: {client_mode}')
        if is_on_server is None:
            def is_on_server(name): 
                return not is_on_client(name)
        
        self.is_on_client = is_on_client
        self.is_on_server = is_on_server
        self.to(device)
    
    def client_state_dict(self):
        return OrderedDict((n, p) for (n, p) in self.state_dict().items() if self.is_on_client(n))
    
    def server_state_dict(self):
        return OrderedDict((n, p) for (n, p) in self.state_dict().items() if self.is_on_server(n))

    
def customized_resnet18(pretrained: bool = False, class_num=10,progress: bool = True) -> ResNet:
    res18 = ResNet(BasicBlock, [2, 2, 2, 2],class_num=class_num)


    # Change BN to GN
    res18.bn1 = nn.GroupNorm(num_groups=32, num_channels=64)

    res18.layer1[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)

    res18.layer2[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)

    res18.layer3[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)

    res18.layer4[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)

    assert len(dict(res18.named_parameters()).keys()) == len(
        res18.state_dict().keys()), 'More BN layers are there...'

    return res18

class tiny_ResNet(nn.Module):
    # 定义初始函数，输入参数为残差块，残差块数量，默认参数为分类数10
    def __init__(self, block, num_blocks, class_num=10):
        super(tiny_ResNet, self).__init__()
        # 设置第一层的输入通道数
        self.in_planes = 64

        # 定义输入图片先进行一次卷积与批归一化，使图像大小不变，通道数由3变为64得两个操作
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 定义第一层，输入通道数64，有num_blocks[0]个残差块，残差块中第一个卷积步长自定义为1
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # 定义第二层，输入通道数128，有num_blocks[1]个残差块，残差块中第一个卷积步长自定义为2
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # 定义第三层，输入通道数256，有num_blocks[2]个残差块，残差块中第一个卷积步长自定义为2
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # 定义第四层，输入通道数512，有num_blocks[3]个残差块，残差块中第一个卷积步长自定义为2
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 定义全连接层，输入512*block.expansion个神经元，输出10个分类神经元
        self.linear = nn.Linear(512 * block.expansion, class_num)

    # 定义创造层的函数，在同一层中通道数相同，输入参数为残差块，通道数，残差块数量，步长
    def _make_layer(self, block, planes, num_blocks, stride):
        # strides列表第一个元素stride表示第一个残差块第一个卷积步长，其余元素表示其他残差块第一个卷积步长为1
        strides = [stride] + [1] * (num_blocks - 1)
        # 创建一个空列表用于放置层
        layers = []
        # 遍历strides列表，对本层不同的残差块设置不同的stride
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))  # 创建残差块添加进本层
            self.in_planes = planes * block.expansion  # 更新本层下一个残差块的输入通道数或本层遍历结束后作为下一层的输入通道数
        return nn.Sequential(*layers)  # 返回层列表

    # 定义前向传播函数，输入图像为x，输出预测数据
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 第一个卷积和第一个批归一化后用ReLU函数激活
        out = self.layer1(out)  # 第一层传播
        out = self.layer2(out)  # 第二层传播
        out = self.layer3(out)  # 第三层传播
        out = self.layer4(out)  # 第四层传播
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)  # 全连接传播
        return out

def tiny_resnet18(pretrained: bool = False, class_num=200,progress: bool = True) -> ResNet:
    res18 = tiny_ResNet(BasicBlock, [2, 2, 2, 2],class_num=class_num)


    # Change BN to GN
    res18.bn1 = nn.GroupNorm(num_groups=32, num_channels=64)

    res18.layer1[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)

    res18.layer2[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)

    res18.layer3[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)

    res18.layer4[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)

    assert len(dict(res18.named_parameters()).keys()) == len(
        res18.state_dict().keys()), 'More BN layers are there...'

    return res18
# def resnet18(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0, num_classes=10,**kwargs: Any):
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=32, num_channels=group_channels)
#
#     model = ResNet(BasicBlock, [2, 2, 2, 2],num_classes=num_classes)
#     return model


# def resnet34(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0, **kwargs: Any):
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#
#     model = ResNet(BasicBlock, [3, 4, 6, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['resnet34'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def resnet50(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0, **kwargs: Any):
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#
#     model = ResNet(Bottleneck, [3, 4, 6, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['resnet50'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def resnet101(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0, **kwargs: Any):
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#
#     model = ResNet(Bottleneck, [3, 4, 23, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['resnet101'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def resnet152(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0, **kwargs: Any):
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#
#     model = ResNet(Bottleneck, [3, 8, 36, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['resnet152'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def resnext50_32x4d(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0,
#                     **kwargs: Any) -> ResNet:
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 4
#     model = ResNet(Bottleneck, [3, 4, 6, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['resnext50_32x4d'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def resnext101_32x8d(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0,
#                      **kwargs: Any) -> ResNet:
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 8
#     model = ResNet(Bottleneck, [3, 4, 23, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['resnext101_32x8d'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def wide_resnet50_2(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0,
#                     **kwargs: Any) -> ResNet:
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#     kwargs['width_per_group'] = 64 * 2
#     model = ResNet(Bottleneck, [3, 4, 6, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['wide_resnet50_2'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def wide_resnet101_2(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0,
#                      **kwargs: Any) -> ResNet:
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#     kwargs['width_per_group'] = 64 * 2
#     model = ResNet(Bottleneck, [3, 4, 23, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['wide_resnet101_2'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)