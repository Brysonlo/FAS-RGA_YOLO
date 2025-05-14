# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import time
from torch.optim import AdamW

__all__ = ('Conv', 'Conv2', 'LightConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus', 'GhostConv',
           'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'RepConv')


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
class FalConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.conv = GEPdecompose(self.conv,bn=False, relu=False, groups=1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]:i[0] + 1, i[1]:i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__('cv2')
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)
class GEPdecompose(nn.Module):
    """
    GEP decomposition class
    """

    def __init__(self, conv_layer, rank=1, init=True, alpha=1.0, bn=False, relu=False, groups=1):
        """
        Initialize FALCON layer.

        :param conv_layer: standard convolution layer
        :param rank: rank of GEP
        :param init: whether initialize FALCON with decomposed tensors
        :param relu: whether use relu function
        :param groups: number of groups in 1*1 conv
        """

        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.bn_ = bn
        self.relu_ = relu
        self.device = conv_layer.weight.device

        # get weight and bias
        weight = conv_layer.weight.data
        bias = conv_layer.bias
        if bias is not None:
            bias = bias.data

        out_channels, in_channels, _, _ = weight.shape
        self.out_channels = int(out_channels * self.alpha)
        self.in_channels = int(in_channels * self.alpha)

        if self.rank == 1:
            self.point_wise = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False).to(self.device)
            self.depth_wise = nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.kernel_size[0] // 2,
                bias=False,
                groups=self.out_channels).to(self.device)
            self.batch_norm = nn.BatchNorm2d(self.out_channels)
            if init:
                self.decompose(conv_layer.weight, self.point_wise.weight, self.depth_wise.weight)

        else:
            for i in range(self.rank):
                setattr(self, 'point_wise' + str(i),
                        nn.Conv2d(in_channels=self.in_channels,
                                  out_channels=self.out_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  bias=False).to(self.device))
                setattr(self, 'depth_wise' + str(i),
                        nn.Conv2d(in_channels=self.out_channels,
                                  out_channels=self.out_channels,
                                  kernel_size=conv_layer.kernel_size,
                                  stride=conv_layer.stride,
                                  padding=conv_layer.kernel_size[0] // 2,
                                  bias=False,
                                  groups=self.out_channels).to(self.device))
            self.batch_norm = nn.BatchNorm2d(self.out_channels)
            if init:
                if alpha == 1.0:
                    self.decompose_rank(conv_layer.weight)
                else:
                    self.width_mul(conv_layer.weight)
                    self.decompose_rank(conv_layer.weight)

        self.stride = conv_layer.stride

        if groups != 1:
            self.group1x1(groups)

    def forward(self, input_):
        """
        Run forward propagation

        :param input_: input feature maps
        :return: out: output tensor of forward propagation
        """
        if self.rank == 1:
            out = self.depth_wise(self.point_wise(input_))
        else:
            for i in range(self.rank):
                if i == 0:
                    out = getattr(self, 'point_wise' + str(i))(input_)
                    out = getattr(self, 'depth_wise' + str(i))(out)
                else:
                    out += getattr(self, 'depth_wise' + str(i)) \
                        (getattr(self, 'point_wise' + str(i))(input_))
        if self.bn_:
            out = self.batch_norm(out)
        if self.relu_:
            out = nn.SiLU(out, True)
        return out

    def decompose(self, conv, point_wise, depth_wise, learning_rate=0.001, steps=600):
        """
        GEP decomposes standard convolution kernel

        :param conv: standard convolution kernel
        :param point_wise: decomposed pointwise convolution kernel
        :param depth_wise: decomposed depthwise convolution kernel
        :param learning_rate: learning rate
        :param steps: training steps for decomposing
        """

        conv.requires_grad = False
        point_wise.requires_grad = True
        depth_wise.requires_grad = True

        criterion = nn.MSELoss()
        optimizer = AdamW({point_wise, depth_wise}, lr=learning_rate)
        start_time = time.time()
        for step in range(steps):
            if steps in (400, 700):
                learning_rate = learning_rate / 10
                optimizer = AdamW({point_wise, depth_wise}, lr=learning_rate)
            optimizer.zero_grad()
            kernel_pred = point_wise.to('cpu')  * depth_wise.to('cpu')
            loss = criterion(kernel_pred, conv.to('cpu'))
            loss.backward()
            optimizer.step()
            if step % 100 == 99:
                print(f'loss = {loss}, time = {time.time() - start_time}%d')
                start_time = time.time()

    def decompose_rank(self, kernel, learning_rate=5e-3, steps=600):
        """
        GEP decomposes standard convolution kernel with different rank

        :param conv: standard convolution kernel
        :param learning_rate: learning rate
        :param steps: training steps for decomposing
        """

        kernel.requires_grad = False
        param = {self.depth_wise0.weight, self.point_wise0.weight}
        for i in range(self.rank):
            getattr(self, 'point_wise' + str(i)).weight.requires_grad = True
            getattr(self, 'depth_wise' + str(i)).weight.requires_grad = True
            if i != 0:
                param.add(getattr(self, 'point_wise' + str(i)).weight)
                param.add(getattr(self, 'point_wise' + str(i)).weight)

        criterion = nn.MSELoss()
        optimizer = AdamW(param, lr=learning_rate)
        start_time = time.time()
        for step in range(steps):
            if steps in (400, 700):
                learning_rate = learning_rate / 10
                optimizer = AdamW(param, lr=learning_rate)
            optimizer.zero_grad()
            for i in range(self.rank):
                if i == 0:
                    kernel_pred = getattr(self, \
                                          'point_wise' + str(i)).weight.cuda() * \
                                  getattr(self, 'depth_wise' + str(i)).weight.cuda()
                else:
                    kernel_pred += getattr(self, \
                                           'point_wise' + str(i)).weight.cuda() * getattr(self, \
                                                                                          'depth_wise' + str(
                                                                                              i)).weight.cuda()
            loss = criterion(kernel_pred, kernel.cuda())
            loss.backward()
            optimizer.step()
            if step % 100 == 99:
                print(f'step {step + 1}: loss = {loss}, time = {time.time() - start_time}')
                start_time = time.time()

    def group1x1(self, group_num):
        """
        Replace 1x1 pointwise convolution in FALCON with 1x1 group convolution

        :param group_num: number of groups for 1x1 group
        """
        if self.rank == 1:
            point_wise = self.point_wise.weight.data
            if point_wise.shape[1] != 3:
                self.point_wise = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                    groups=group_num).to(self.device)
                in_length = int(self.in_channels / group_num)
                out_length = int(self.out_channels / group_num)
                for i in range(group_num):
                    self.point_wise.weight.data[
                    (i * out_length):((i + 1) * out_length), 0:(in_length), :, :] = \
                        point_wise[
                        (i * out_length):((i + 1) * out_length), (i * in_length):((i + 1) * in_length), :, :]
        else:
            for i in range(self.rank):
                point_wise = getattr(self, 'point_wise' + str(i)).weight.data
                if point_wise.shape[1] != 3:
                    setattr(self, 'point_wise' + str(i),
                            nn.Conv2d(in_channels=self.in_channels,
                                      out_channels=self.out_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=False,
                                      groups=group_num).to(self.device))
                    in_length = int(self.in_channels / group_num)
                    out_length = int(self.out_channels / group_num)
                    for j in range(group_num):
                        getattr(self, \
                                'point_wise' + str(i)).weight.data[
                        (j * out_length):((j + 1) * out_length), 0:(in_length), :, :] = \
                            point_wise[
                            (j * out_length):((j + 1) * out_length), (j * in_length):((j + 1) * in_length), :, :]
