from collections import OrderedDict



from nets.CSPdarknet53_tiny import darknet53_tiny

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from utils.cfg import USE_ATTEN

# from mmcv.cnn import constant_init, kaiming_init
from nets.CSPdarknet53_tiny import Resblock_body
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

#-------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   Conv2d + BatchNormalization + LeakyReLU
#-------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        # nn.GroupNorm
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        BasicConv(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, return_mask=0,reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.pool_c = nn.AdaptiveAvgPool2d((1, 1))
        self.return_mask=return_mask
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        # t=a_w * a_h
        # t1=self.pool_c(t).squeeze(0).squeeze(1).squeeze(1)

        # weight=torch.softmax(t1,dim=0)
        # ####tm=torch.mean(t,axis=1,keepdim=True)
        out = identity * a_w * a_h
        # batch_cams = (weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(0) * t).sum(dim=1)
        if self.return_mask:
            return out,a_w * a_h
        return out

class ContextBlock2d(nn.Module):
    def __init__(self, inplanes, planes):
        super(ContextBlock2d, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1),
        )
        self.reset_parameters()

    def reset_parameters(self):
    #     kaiming_init(self.conv_mask, mode="fan_in")
        self.conv_mask.inited = True
    #     last_zero_init(self.channel_add_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()

        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        beta1 = context_mask
        beta2 = torch.transpose(beta1, 1, 2)
        atten = torch.matmul(beta2, beta1)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context, atten

    def forward(self, x):
        # [N, C, 1, 1]
        context, atten = self.spatial_pool(x)
        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        out = x + channel_add_term

        return out, atten
class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes,show_atten=USE_ATTEN):
        super(YoloBody, self).__init__()
        #  backbone
        self.show_atten=show_atten
        if self.show_atten:
            print('注意力模块加载')
            self.attention0=CoordAtt(512,512,return_mask=1)
            self.attention1 = CoordAtt(256, 256)
            # self.attentionP5 = CoordAtt(256, 256)
            # self.attentionP4 = CoordAtt(384, 384)
            # self.attention100  = ContextBlock2d(512, 512)
            # self.attention101 = ContextBlock2d(256,256)
        self.backbone = darknet53_tiny(None)

        self.conv_for_P5 = BasicConv(512,256,1)
        self.yolo_headP5 = yolo_head([512, num_anchors * (5 + num_classes)],256)

        self.upsample = Upsample(256,128)
        self.yolo_headP4 = yolo_head([256, num_anchors * (5 + num_classes)],384)
        self._initialize_weights()


    def _initialize_weights(self, pretrain=True):
        print('init weights...')
        for name, m in self.named_modules():
            print(name,'初始化')
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #---------------------------------------------------#
        #   生成CSPdarknet53_tiny的主干模型
        #   feat1的shape为26,26,256
        #   feat2的shape为13,13,512
        #---------------------------------------------------#
        feat1, feat2 = self.backbone(x)
        if self.show_atten:
            feat2 ,atten=self.attention0(feat2)
            feat1 = self.attention1(feat1)
            # feat2 ,atten2 = self.attention100(feat2)
            # feat1,atten1 = self.attention101(feat1)
        # 13,13,512 -> 13,13,256
        P5 = self.conv_for_P5(feat2)
        # if self.show_atten:
        #     P5=self.attentionP5(P5)
        # 13,13,256 -> 13,13,512 -> 13,13,75(voc)
        out0 = self.yolo_headP5(P5) 

        # 13,13,256 -> 13,13,128 -> 26,26,128
        P5_Upsample = self.upsample(P5)
        # 26,26,256 + 26,26,128 -> 26,26,384
        P4 = torch.cat([P5_Upsample,feat1],axis=1)
        # if self.show_atten:
        #     P4=self.attentionP4(P4)
        # 26,26,384 -> 26,26,256 -> 26,26,75
        out1 = self.yolo_headP4(P4)
        if USE_ATTEN:
            return out0, out1,atten
        return out0,out1

class YoloBody4(nn.Module):
    '''注意力可视化'''
    def __init__(self, num_anchors, num_classes, show_atten=USE_ATTEN):
        super(YoloBody, self).__init__()
        #  backbone
        self.show_atten = show_atten
        if self.show_atten:
            print('注意力模块加载')
            # self.attention0 = CoordAtt(512, 512)
            # self.attention1 = CoordAtt(256, 256)
            # self.attentionP5 = CoordAtt(256, 256)
            # self.attentionP4 = CoordAtt(384, 384)
            self.attention0,atten2 = ContextBlock2d(512, 512)
            self.attention1,atten1 = ContextBlock2d(256,256)
        self.backbone = darknet53_tiny(None)

        self.conv_for_P5 = BasicConv(512, 256, 1)
        self.yolo_headP5 = yolo_head([512, num_anchors * (5 + num_classes)], 256)

        self.upsample = Upsample(256, 128)
        self.yolo_headP4 = yolo_head([256, num_anchors * (5 + num_classes)], 384)
        self._initialize_weights()

    def _initialize_weights(self, pretrain=True):
        print('init weights...')
        for name, m in self.named_modules():
            print(name, '初始化')
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # ---------------------------------------------------#
        #   生成CSPdarknet53_tiny的主干模型
        #   feat1的shape为26,26,256
        #   feat2的shape为13,13,512
        # ---------------------------------------------------#
        feat1, feat2 = self.backbone(x)
        if self.show_atten:
            feat2,atten2 = self.attention0(feat2)#(512, 512)
            feat1,atten1 = self.attention1(feat1)
        # 13,13,512 -> 13,13,256
        P5 = self.conv_for_P5(feat2)
        # if self.show_atten:
        #     P5=self.attentionP5(P5)
        # 13,13,256 -> 13,13,512 -> 13,13,75(voc)
        out0 = self.yolo_headP5(P5)

        # 13,13,256 -> 13,13,128 -> 26,26,128
        P5_Upsample = self.upsample(P5)
        # 26,26,256 + 26,26,128 -> 26,26,384
        P4 = torch.cat([P5_Upsample, feat1], axis=1)
        # if self.show_atten:
        #     P4=self.attentionP4(P4)
        # 26,26,384 -> 26,26,256 -> 26,26,75
        out1 = self.yolo_headP4(P4)

        return out0, out1,atten1,atten2

def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))
class YoloBody1(nn.Module):
    '''改为PANet结构'''
    def __init__(self, num_anchors, num_classes, show_atten=1):
        super(YoloBody, self).__init__()
        #  backbone
        self.show_atten = show_atten
        if self.show_atten:
            print('注意力模块加载')
            self.attention0 = CoordAtt(512, 512)
            self.attention1 = CoordAtt(256, 256)
            self.attention384=CoordAtt(384,384)
            # self.attention = ContextBlock2d(512, 512)
        self.backbone = darknet53_tiny(None)

        self.conv_for_P5 = BasicConv(512, 256, 1)
        self.yolo_headP5 = yolo_head([512, num_anchors * (5 + num_classes)], 256)

        self.down_sample = conv2d(128, 256, 3, stride=2)

        self.upsample = Upsample(256, 128)
        self.yolo_headP4 = yolo_head([256, num_anchors * (5 + num_classes)], 384)

        self.P4toHead=conv2d(384,384,3)
        self.P5toHead = conv2d(256,256, 3)
        self._initialize_weights()

    def _initialize_weights(self, pretrain=True):
        print('init weights...')
        for name, m in self.named_modules():
            print(name, '初始化')
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # ---------------------------------------------------#
        #   生成CSPdarknet53_tiny的主干模型
        #   feat1的shape为26,26,256
        #   feat2的shape为13,13,512
        # ---------------------------------------------------#
        feat1, feat2 = self.backbone(x)
        if self.show_atten:
            feat2 = self.attention0(feat2)
            feat1 = self.attention1(feat1)
        # 13,13,512 -> 13,13,256
        P5 = self.conv_for_P5(feat2)
        # 13,13,256 -> 13,13,512 -> 13,13,75(voc)
        P5=self.P5toHead(P5)####################***************

        out0 = self.yolo_headP5(P5)

        # 13,13,256 -> 13,13,128 -> 26,26,128
        P5_Upsample = self.upsample(P5)
        # 26,26,256 + 26,26,128 -> 26,26,384
        P4 = torch.cat([P5_Upsample, feat1], axis=1)
        P4=self.P4toHead(P4)################***************

        # 26,26,384 -> 26,26,256 -> 26,26,255
        out1 = self.yolo_headP4(P4)

        return out0, out1


class Resblock_body2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock_body2, self).__init__()
        self.out_channels = out_channels

        self.conv1 = BasicConv(in_channels, out_channels, 3)

        self.conv2 = BasicConv(out_channels // 2, out_channels // 2, 3)
        self.conv3 = BasicConv(out_channels // 2, out_channels // 2, 3)

        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

        self.conv5 = BasicConv(1024, out_channels, 1)
    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1(x)
        # 引出一个大的残差边route
        route = x

        c = self.out_channels
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, c // 2, dim=1)[1]
        # 对主干部分进行3x3卷积
        x = self.conv2(x)
        # 引出一个小的残差边route_1
        route1 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = torch.cat([x, route1], dim=1)

        # 对相接后的结果进行1x1卷积
        x = self.conv4(x)
        feat = x
        x = torch.cat([route, x], dim=1)

        # 利用最大池化进行高和宽的压缩
        # x = self.maxpool(x)
        x=self.conv5(x)
        return x, feat


class YoloBody2(nn.Module):
    '''增加PAN'''
    def __init__(self, num_anchors, num_classes, show_atten=USE_ATTEN):
        super(YoloBody2, self).__init__()
        #  backbone
        self.show_atten = show_atten
        if self.show_atten:
            print('注意力模块加载')
            self.attention0 = CoordAtt(512, 512)
            self.attention1 = CoordAtt(256, 256)
            # self.attentionP5 = CoordAtt(256, 256)
            # self.attentionP4 = CoordAtt(384, 384)
            # self.attention = ContextBlock2d(512, 512)
            ''' P5 = self.conv_for_P5(feat2)
        P5_Upsample = self.upsample(P5)
        P4 = torch.cat([P5_Upsample, feat1], axis=1)
        P41=self.conv_for_P41(P4)
        P41_Downsample=self.down_sample_P41(P41)
        P51=torch.cat([P5,P41_Downsample],axis=1)
        P51=self.conv_for_P51(P51)
        out0 = self.yolo_headP5(P51)

        out1 = self.yolo_headP4(P41)'''
        self.backbone = darknet53_tiny(None)

        self.conv_for_P5 = BasicConv(512, 256,1)
        self.yolo_headP5 = yolo_head([512, num_anchors * (5 + num_classes)], 256)

        self.upsample = Upsample(256, 128)
        self.yolo_headP4 = yolo_head([256, num_anchors * (5 + num_classes)], 256)

        self.conv_for_P51=BasicConv(512, 256, 1)
        self.conv_for_P41=BasicConv(384, 256, 1)
        self.down_sample_P41= conv2d(256,256,1,stride=2)


        self._initialize_weights()
    def _initialize_weights(self, pretrain=True):
        print('init weights...')
        for name, m in self.named_modules():
            print(name, '初始化')
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # ---------------------------------------------------#
        #   生成CSPdarknet53_tiny的主干模型
        #   feat1的shape为26,26,256
        #   feat2的shape为13,13,512
        # ---------------------------------------------------#
        feat1, feat2 = self.backbone(x)
        if self.show_atten:

            feat2 = self.attention0(feat2)
            feat1 = self.attention1(feat1)
        # 13,13,512 -> 13,13,256
        P5 = self.conv_for_P5(feat2)
        P5_Upsample = self.upsample(P5)
        P4 = torch.cat([P5_Upsample, feat1], axis=1)
        P41=self.conv_for_P41(P4)
        P41_Downsample=self.down_sample_P41(P41)
        P51=torch.cat([P5,P41_Downsample],axis=1)
        P51=self.conv_for_P51(P51)
        out0 = self.yolo_headP5(P51)

        out1 = self.yolo_headP4(P41)
        if self.show_atten:
            return out0,out1

        return out0, out1


class YoloBody3(nn.Module):
    '''密集连接'''
    def __init__(self, num_anchors, num_classes, show_atten=USE_ATTEN):
        super(YoloBody3, self).__init__()
        #  backbone
        self.show_atten = show_atten
        if self.show_atten:
            print('注意力模块加载')
            self.attention0 = CoordAtt(512, 512)
            self.attention1 = CoordAtt(256, 256)
            self.attentionP5 = CoordAtt(256, 256)
            self.attentionP4 = CoordAtt(384, 384)
            # self.attention = ContextBlock2d(512, 512)
        self.backbone = darknet53_tiny(None)

        self.conv_for_P5 = BasicConv(512, 256, 1)
        self.yolo_headP5 = yolo_head([512, num_anchors * (5 + num_classes)], 256)

        self.upsample = Upsample(256, 128)

        # self.yolo_headP4 = yolo_head([256, num_anchors * (5 + num_classes)], 384)
        self.yolo_headP4 = yolo_head([256, num_anchors * (5 + num_classes)], 512)

        self.upsample_for_feat3_1 = Upsample(512, 64)
        self.upsample_for_feat2_1 = Upsample(512, 64)
        self.conv_for_feat3_2 = BasicConv(512, 128, 1)
        self.conv_for_feat1_2 = BasicConv(256, 128, 1,stride=2)
        # self.conv_for_feat1_1 = BasicConv(256, 128, 1 )
        self.conv_for_feat2_2 = BasicConv(512, 256, 1 )
        self.conv_1 = BasicConv(256, 256, 1 )
        self.conv_2 = BasicConv(384, 384, 1 )

        self.resblock_body4 = Resblock_body2(512, 512)
        self.conv4 = BasicConv(1024, 32, 1)

        self._initialize_weights()

    def _initialize_weights(self, pretrain=True):
        print('init weights...')
        for name, m in self.named_modules():
            print(name, '初始化')
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # ---------------------------------------------------#
        #   生成CSPdarknet53_tiny的主干模型
        #   feat1的shape为26,26,256
        #   feat2的shape为13,13,512
        # ---------------------------------------------------#
        feat1, feat2 = self.backbone(x)
        feat3,_=self.resblock_body4(feat2)

        if self.show_atten:
            feat2 = self.attention0(feat2)
            feat1 = self.attention1(feat1)
        # 13,13,512 -> 13,13,256

        f3_1=self.upsample_for_feat3_1(feat3)
        f2_1=self.upsample_for_feat2_1(feat2)
        f1_2=self.conv_for_feat1_2(feat1)
        f2_2=self.conv_for_feat2_2(feat2)
        f3_2=self.conv_for_feat3_2(feat3)
        P5=torch.cat([f1_2,f2_2,f3_2],axis=1)
        P5 = self.conv_for_P5(P5)
        out0 = self.yolo_headP5(P5)
        P5_Upsample = self.upsample(P5)
        P4=torch.cat([feat1,f2_1,f3_1,P5_Upsample],axis=1)
        out1 = self.yolo_headP4(P4)

        # P5 = self.conv_for_P5(feat2)
        # if self.show_atten:
        #     P5=self.attentionP5(P5)
        # # 13,13,256 -> 13,13,512 -> 13,13,75(voc)
        # out0 = self.yolo_headP5(P5)
        #
        # # 13,13,256 -> 13,13,128 -> 26,26,128
        # P5_Upsample = self.upsample(P5)
        # # 26,26,256 + 26,26,128 -> 26,26,384
        # P4 = torch.cat([P5_Upsample, feat1], axis=1)
        # if self.show_atten:
        #     P4=self.attentionP4(P4)
        # # 26,26,384 -> 26,26,256 -> 26,26,255
        # out1 = self.yolo_headP4(P4)

        return out0, out1
class YoloBody5(nn.Module):
    '''密集连接 轻量型'''
    def __init__(self, num_anchors, num_classes, show_atten=USE_ATTEN):
        super(YoloBody5, self).__init__()
        #  backbone
        self.show_atten = show_atten
        if self.show_atten:
            print('注意力模块加载')
            self.attention0 = CoordAtt(512, 512)
            self.attention1 = CoordAtt(256, 256)
            # self.attentionP5 = CoordAtt(256, 256)
            # self.attentionP4 = CoordAtt(384, 384)
            # self.attention = ContextBlock2d(512, 512)
        self.backbone = darknet53_tiny(None)

        self.conv_for_P5 = BasicConv(384, 256, 1)
        self.yolo_headP5 = yolo_head([256, num_anchors * (5 + num_classes)], 256)

        self.upsample = Upsample(256, 128)

        # self.yolo_headP4 = yolo_head([256, num_anchors * (5 + num_classes)], 384)
        self.yolo_headP4 = yolo_head([256, num_anchors * (5 + num_classes)], 256)


        self.upsample_for_feat2_1 = Upsample(512, 128)

        self.conv_for_feat1_2 = BasicConv(256, 128, 1,stride=2)
        # self.conv_for_feat1_1 = BasicConv(256, 128, 1 )
        self.conv_for_feat2_2 = BasicConv(512, 256, 1 )

        self.conv_forP4_to_HEAD=BasicConv(512,256,1)

        # self.conv_1 = BasicConv(256, 256, 1 )
        # self.conv_2 = BasicConv(384, 384, 1 )

        # self.resblock_body4 = Resblock_body2(512, 512)
        # self.conv4 = BasicConv(1024, 32, 1)

        self._initialize_weights()

    def _initialize_weights(self, pretrain=True):
        print('init weights...')
        for name, m in self.named_modules():
            print(name, '初始化')
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # ---------------------------------------------------#
        #   生成CSPdarknet53_tiny的主干模型
        #   feat1的shape为26,26,256
        #   feat2的shape为13,13,512
        # ---------------------------------------------------#
        feat1, feat2 = self.backbone(x)

        if self.show_atten:
            feat2 = self.attention0(feat2)
            feat1 = self.attention1(feat1)
        # 13,13,512 -> 13,13,256


        f2_1=self.upsample_for_feat2_1(feat2)#128
        f1_2=self.conv_for_feat1_2(feat1)#128
        f2_2=self.conv_for_feat2_2(feat2)#256

        P5=torch.cat([f1_2,f2_2],axis=1)
        P5 = self.conv_for_P5(P5)#256
        out0 = self.yolo_headP5(P5)
        P5_Upsample = self.upsample(P5)
        P4=torch.cat([feat1,f2_1,P5_Upsample],axis=1)
        # P4=torch.cat([P4,f2_1,feat1],axis=1)
        P4=self.conv_forP4_to_HEAD(P4)
        out1 = self.yolo_headP4(P4)

        # P5 = self.conv_for_P5(feat2)
        # if self.show_atten:
        #     P5=self.attentionP5(P5)
        # # 13,13,256 -> 13,13,512 -> 13,13,75(voc)
        # out0 = self.yolo_headP5(P5)
        #
        # # 13,13,256 -> 13,13,128 -> 26,26,128
        # P5_Upsample = self.upsample(P5)
        # # 26,26,256 + 26,26,128 -> 26,26,384
        # P4 = torch.cat([P5_Upsample, feat1], axis=1)
        # if self.show_atten:
        #     P4=self.attentionP4(P4)
        # # 26,26,384 -> 26,26,256 -> 26,26,255
        # out1 = self.yolo_headP4(P4)

        return out0, out1