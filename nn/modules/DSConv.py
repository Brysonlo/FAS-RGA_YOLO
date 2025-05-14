
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv,FalConv
from .transformer import TransformerBlock
import time
from torch.optim import AdamW

def autopad(k, p=None):
    if p is None:
        p = k//2 if isinstance(k,int) else [x//2 for x in k] #auto-pad
    return p

class Conv(nn.Module):
    #Standard convolution
    def __init__(self,c1,c2,k=1,s=1,p=None,g=1,act=True): #ch_in,ch_out,kernel,stride,padding,groups
        super(Conv,self).__init__()
        self.conv = nn.Conv2d(c1,c2,k,s,autopad(k,p),groups=g,bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act,nn.Module) else nn.Identity())

    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self,x):
        return self.act(self.conv(x))

class DySnakeConv(nn.Module):
    def __init__(self, inc, ouc, k=3, act=True) -> None:
        super().__init__()

        self.conv_0 = FalConv(inc, ouc, k, act=act)
        self.conv_x = DSConv(inc, ouc, 0, k)
        self.conv_y = DSConv(inc, ouc, 1, k)
        self.conv_ds = Conv(ouc * 3, ouc, k=1, act=act)

    def forward(self, x):
        return self.conv_ds(torch.cat([self.conv_0(x), self.conv_x(x), self.conv_y(x)], dim=1))

class DSConv(nn.Module):
    def __init__(self,in_ch,out_ch,morph,kernel_size=3,if_offset=True,extend_scope=1,act=True):
        """
                The Dynamic Snake Convolution
                :param in_ch: input channel
                :param out_ch: output channel
                :param kernel_size: the size of kernel
                :param extend_scope: the range to expand (default 1 for this method)
                :param morph: the morphology of the convolution kernel is mainly divided into two types
                                along the x-axis (0) and the y-axis (1) (see the paper for details)
                :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
                :param device: set on gpu
        """

        super(DSConv,self).__init__()
        #use the <offset_conv>to learn the deformable offset
        self.offset_conv = nn.Conv2d(in_ch,2 * kernel_size,3,padding=1)
        self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.kernel_size = kernel_size
        #two types of the DSConv (along.x-axis and y-axis)
        self.dsc_conv_x = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(kernel_size,1),
            stride=(kernel_size,1),
            padding=0,
        )

        self.dsc_conv_y = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(1,kernel_size),
            stride=(1,kernel_size),
            padding=0,
        )

        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.act = nn.SiLU() if act is True else (act if isinstance(act,nn.Module)else nn.Identity())
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset

    def forward(self, f):
        offset = self.offset_conv(f)
        offset = self.bn(offset)
        offset = torch.tanh(offset)
        input_shape = f.shape
        dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph)
        deformed_feature = dsc.deform_conv(f, offset, self.if_offset)

        if self.morph == 0:
            x = self.dsc_conv_x(deformed_feature.type(f.dtype))
            x = self.gn(x)
            x = self.act(x)
            return x
        else:
            x = self.dsc_conv_y(deformed_feature.type(f.dtype))
            x = self.gn(x)
            x = self.act(x)
            return x


class DSC(object):
    def __init__(self,input_shape,kernel_size,extend_scope,morph):
        self.num_points = kernel_size
        self.width = input_shape[2]
        self.height = input_shape[3]
        self.morph = morph
        self.extend_scope = extend_scope
        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]

    def _coordinate_map_3D(self, offset, if_offset):
        device = offset.device
        # offset
        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)
        y_center = torch.arange(0, self.width).repeat([self.height])
        y_center = y_center.reshape(self.height, self.width)
        y_center = y_center.permute(1, 0)
        y_center = y_center.reshape([-1, self.width, self.height])
        y_center = y_center.repeat([self.num_points, 1,1]).float()
        y_center = y_center.unsqueeze(0)

        x_center = torch.arange(0, self.height).repeat([self.width])
        x_center = x_center.reshape(self.width, self.height)
        x_center = x_center.permute(0, 1)
        x_center = x_center.reshape([-1, self.width, self.height])
        x_center = x_center.repeat([self.num_points, 1, 1]).float()
        x_center = x_center.unsqueeze(0)

        if self.morph == 0:
            """
            Initialize the kernel and flatten the kernel
            y:only need 0
            x:-num_points//2 num_points//2 (Determined by the kernel size)
            !!The related PPT will be submitted later,and the PPT will contain the whole changes of each step
            """
            y = torch.linspace(0, 0, 1)
            x = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )
            y,x = torch.meshgrid(y,x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)
            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0) #[B * K * K, W, H]
            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)  # [B*K*K,W,H]

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(device)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(device)
            y_offset_new = y_offset.detach().clone()

            if if_offset:
                y_offset = y_offset.permute(1,0, 2, 3)
                y_offset_new = y_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)
                y_offset_new[center] = 0
                for index in range(1, center):
                    y_offset_new[center+index] = (y_offset_new[center + index - 1] + y_offset[center + index])
                    y_offset_new[center-index] = (y_offset_new[center - index + 1] + y_offset[center - index])
                y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(device)
                y_new = y_new.add(y_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                    self.num_batch, self.num_points * self.width, 1 * self.height
                    ])
            x_new = x_new.reshape(
                [self.num_batch, self.num_points,1,self.width,self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            return y_new,x_new
        else:
            y = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )
            x = torch.linspace(0, 0, 1)
            y,x = torch.meshgrid(y,x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)
            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1)

            y_new = y_new.to(device)
            x_new = x_new.to(device)
            x_offset_new = x_offset.detach().clone()

            if if_offset:
                x_offset = x_offset.permute(1, 0, 2, 3)
                x_offset_new = x_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)
                x_offset_new[center] = 0
                for index in range(1, center):
                    x_offset_new[center + index] = (x_offset_new[center + index - 1] + y_offset[center + index])
                    x_offset_new[center - index] = (x_offset_new[center - index + 1] + y_offset[center - index])
                x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(device)
                x_new = y_new.add(x_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            return y_new, x_new

    """
            input: input feature map [N,C,D,W,H]；coordinate map [N,K*D,K*W,K*H] 
            output: [N,1,K*D,K*W,K*H]  deformed feature map
    """

    def _bilinear_interpolate_3D(self, input_feature, y, x):
        device = input_feature.device
        y = y.reshape([-1]).float()
        x = x.reshape([-1]).float()

        zero = torch.zeros([]).int()
        max_y = self.width - 1
        max_x = self.height - 1

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)

        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(
            self.num_batch, self.num_channels, self.width, self.height)
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
        dimension = self.height * self.width

        base = torch.arange(self.num_batch) * dimension
        base = base.reshape([-1, 1]).float()

        repeat = torch.ones([self.num_points * self.width * self.height
                             ]).unsqueeze(0)
        repeat = repeat.float()

        base = torch.matmul(base, repeat)
        base = base.reshape([-1])

        base = base.to(device)

        base_y0 = base + y0 * self.height
        base_y1 = base + y1 * self.height

        # top rectangle of the neighbourhood volume
        index_a0 = base_y0 - base + x0
        index_c0 = base_y0 - base + x1

        # bottom rectangle of the neighbourhood volume
        index_a1 = base_y1 - base + x0
        index_c1 = base_y1 - base + x1

        # get 8 grid values
        value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(device)
        value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(device)
        value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(device)
        value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(device)

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)

        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(
            self.num_batch, self.num_channels, self.width, self.height)
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
        dimension = self.height * self.width
        base = torch.arange(self.num_batch) * dimension
        base = base.reshape([-1, 1]).float()
        repeat = torch.ones([self.num_points * self.width * self.height
                             ]).unsqueeze(0)
        repeat = repeat.float()

        base = torch.matmul(base, repeat)
        base = base.reshape([-1])

        base = base.to(device)

        base_y0 = base + y0 * self.height
        base_y1 = base + y1 * self.height

        # top rectangle of the neighbourhood volume
        index_a0 = base_y0 - base + x0
        index_c0 = base_y0 - base + x1

        # bottom rectangle of the neighbourhood volume
        index_a1 = base_y1 - base + x0
        index_c1 = base_y1 - base + x1

        # get 8 grid values
        value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(device)
        value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(device)
        value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(device)
        value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(device)

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y + 1)
        y1 = torch.clamp(y1, zero, max_y + 1)
        x0 = torch.clamp(x0, zero, max_x + 1)
        x1 = torch.clamp(x1, zero, max_x + 1)

        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()

        vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(device)
        vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(device)
        vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(device)
        vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(device)

        outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 +
                   value_c1 * vol_c1)

        if self.morph == 0:
            outputs = outputs.reshape([
                self.num_batch,
                self.num_points * self.width,
                1 * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        else:
            outputs = outputs.reshape([
                self.num_batch,
                1 * self.width,
                self.num_points * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        return outputs

    def deform_conv(self, input, offset, if_offset):
        y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input, y, x)
        return deformed_feature

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
            out = F.relu(out, True)
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
            kernel_pred = point_wise.cuda() * depth_wise.cuda()
            loss = criterion(kernel_pred, conv.cuda())
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



if __name__ == '__main__':
    input = torch.randn(1,128,8,8)
    dsconv = DySnakeConv(128,256)
    output = dsconv(input)
    print(output.shape)

