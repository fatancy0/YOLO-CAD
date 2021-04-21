import math
from random import shuffle

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
from utils.util import bbox_iou, merge_bboxes
from utils.cfg import USE_EIOU,USE_CIOU,LOSS_UNITE,Focal_LOSS

def Lf(x,alpha=1,beta=0.8,threshold=1):
    alpha=math.e*beta
    t=threshold
    beta=torch.tensor(beta,requires_grad=False).cuda()
    # print(beta.requires_grad,'beta')
    alpha=torch.tensor(alpha,requires_grad=False).cuda()
    xx=x.clone()
    for i in range(len(x)):
        if x[i] > t:
            # aa= -alpha * torch.log(beta) * x[i] + (2 * alpha * torch.log(beta) + alpha) / 4
            # print(aa.shape)
            # print(type(aa))
            # print(xx.shape)   xx[i]= -1*alpha * torch.log(beta) * x[i]  + ( alpha *(3 * torch.log(beta) +2)/9 )
            #             # print(aa) / 4
            #         if x[i]<=t and x[i]>0:
            #             xx[i]= -1*(alpha *2*torch.pow(x[i],3/2)  * (3 * torch.log(beta * x[i]) - 1)) / 9
            # print(type(xx))
            xx[i]= -1*alpha * torch.log(beta) * x[i]/t + (2 * alpha * torch.log(beta) + alpha)
            # print(aa) / 4
        if x[i]<=t and x[i]>0:
            xx[i]= -1*(alpha * x[i] * x[i] * (2 * torch.log(beta * x[i]) - 1)) / (4*t*t)
            # print(xx[i])
    # print('===========')
    # print(xx)
    return xx
def LfM(x,alpha=1,beta=0.7,threshold=1):
    alpha=math.e*beta
    t=threshold
    beta=torch.tensor(beta,requires_grad=False).cuda()
    # print(beta.requires_grad,'beta')
    alpha=torch.tensor(alpha,requires_grad=False).cuda()
    xx=x.clone()
    for i in range(len(x)):
        if x[i] > t:
            xx[i]=  xx[i]= -1*alpha * torch.log(beta) * x[i]  + ( alpha *(3 * torch.log(beta) +2)/9 )
            # print(aa) / 4
        if x[i]<=t and x[i]>0:
            xx[i]= xx[i]= -1*(alpha *2*torch.pow(x[i],3/2)  * (3 * torch.log(beta * x[i]) - 1)) / 9
            # print(xx[i])
    # print('===========')
    # print(xx)
    return xx
def Lf1(x,alpha=1,beta=0.8,threshold=1, k=0.02):
    '''大于threshold部分，其导数恒定为k'''
    alpha=math.e*beta
    t=threshold
    beta=torch.tensor(beta,requires_grad=False).cuda()
    # print(beta.requires_grad,'beta')
    alpha=torch.tensor(alpha,requires_grad=False).cuda()
    xx=x.clone()
    ''' lambda x: -1*alpha * np.log(beta)*x+alpha * np.log(beta)*t-(alpha * t  * (2 * np.log(beta ) - 1)/ 4),
                         lambda x: -1*(alpha * x * x * (2 * np.log(beta * x/t) - 1)) / (4*t)])'''
    if x> t:
        # xx=-1 * alpha * torch.log(beta) * x + alpha * torch.log(beta) * t - (alpha * t * (2 * torch.log(beta) - 1) / 4)
        ####改进版

        xx=(k/2)*torch.pow(x-t,2)-alpha* torch.log(beta) * x+ alpha * torch.log(beta) * t - (alpha * t * (2 * torch.log(beta) - 1) / 4)
        #######################################
    if x<=t and x>0:
        xx= -1*(alpha * x * x * (2 * torch.log(beta * x/t) - 1)) / (4*t)
        # print(xx[i])
    # print('===========')
    # print(xx)
    return xx
def Lf2(x,alpha=1,beta=0.7,threshold=40, Ht=1,  Hb=0.01,w=2):
    '''大于threshold部分，其导数为余弦值'''

    t=threshold
    Hb=torch.tensor(Hb,requires_grad=False).cuda()
    Ht = torch.tensor(Ht, requires_grad=False).cuda()
    w = torch.tensor(w, requires_grad=False).cuda()
    beta=torch.tensor(beta,requires_grad=False).cuda()
    # print(beta.requires_grad,'beta')
    # alpha=torch.tensor(alpha,requires_grad=False).cuda()
    alpha = -0.5 * (Ht - Hb) / (torch.log(beta))
    xx=x.clone()

    ''' lambda x: -1*alpha * np.log(beta)*x+alpha * np.log(beta)*t-(alpha * t  * (2 * np.log(beta ) - 1)/ 4),
                         lambda x: -1*(alpha * x * x * (2 * np.log(beta * x/t) - 1)) / (4*t)])'''
    if x> t:
        xx = (Ht - Hb) / w* torch.cos(w * (x - t)) + x * (Ht - Hb) / 2  - t * (Ht - Hb) / 2 - (Ht - Hb) / w- (alpha * t * (2 * torch.log(beta) - 1)/4)

        # xx=-1 * alpha * torch.log(beta) * x + alpha * torch.log(beta) * t - (alpha * t * (2 * torch.log(beta) - 1) / 4)
        ####改进版

        # xx=(k/2)*torch.pow(x-t,2)-alpha* torch.log(beta) * x+ alpha * torch.log(beta) * t - (alpha * t * (2 * torch.log(beta) - 1) / 4)
        #######################################
    if x<=t and x>0:
        xx= -1*(alpha * x * x * (2 * torch.log(beta * x/t) - 1)) / (4*t)
        # print(xx[i])
    # print('===========')
    # print(xx)
    return xx
# def Lf(x,alpha=1,beta=0.5):
#     x.detach().map_(x,lf )
#
#     return x
    #
    # lx=np.piecewise(x, [x > 1, x <= 1], [lambda x: -alpha * np.log(beta) * x + (2 * alpha * np.log(beta) + alpha) / 4,
    #                                   lambda x: -(alpha * x * x * (2 * np.log(beta * x) - 1)) / 4])
    #
    # return lx

def box_eiou(b1, b2):

    """
    输入为：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    返回为：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # 求出预测框左上角右下角
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # 求出真实框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 求真实框和预测框所有的iou
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / torch.clamp(union_area, min=1e-6)

    # 计算中心的差距
    center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)
    # print('center d',center_distance)
    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxes = torch.max(b1_maxes, b2_maxes)
    enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
    # 计算对角线距离
    enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
    eiou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min=1e-6)
    # print(eiou.shape)
    #     # print(b1_xy.shape)
    #     # print(b2_xy.shape)
    #     # print(MSELoss(b1_xy,b2_xy))
    # print('eiou0',eiou)
    # 找到包裹两个框的最大框的左上角和右下角
    union_mins = torch.min(b1_mins, b2_mins)
    union_maxes = torch.max(b1_maxes, b2_maxes)
    union_wh = torch.max(union_maxes - union_mins, torch.zeros_like(union_maxes))
    union_area = union_wh[..., 0] * union_wh[..., 1]
    w_distance=torch.pow((b1_wh[...,0] - b2_wh[...,0]),2)
    h_distance=torch.pow((b1_wh[...,1] - b2_wh[...,1]), 2)
    # print('eiou zhiqian',eiou)
    # eiou=eiou-w_distance/(union_wh[...,0]*union_wh[...,0])-h_distance/(union_wh[...,1]*union_wh[...,1])#version1
    '''version2'''
    # eiou = eiou - w_distance / (torch.pow(torch.max(b1_wh[...,0],b2_wh[...,0]),2)) -h_distance / (torch.pow(torch.max(b1_wh[...,1],b2_wh[...,1]),2))  # version2
    '''version2'''
    eiou = eiou -  w_distance / (torch.pow(torch.max(b1_wh[..., 0], b2_wh[..., 0]), 2)) -  h_distance / (
        torch.pow(torch.max(b1_wh[..., 1], b2_wh[..., 1]), 2))  # version2
    # print('w_distance / (torch.pow(torch.max(b1_wh[..., 0], b2_wh[..., 0]), 2))',w_distance / (torch.pow(torch.max(b1_wh[..., 0], b2_wh[..., 0]), 2)))
    '''VERSION3'''
    # print(b1_wh[..., 0])
    # print(torch.pow(b1_wh[..., 0],2))
    # print(w_distance)
    # print(w_distance / (torch.pow(b1_wh[..., 0],2)+torch.pow( b2_wh[..., 0]), 2))
    # eiou = eiou -  w_distance / (torch.pow(b1_wh[..., 0],2)+torch.pow( b2_wh[..., 0]), 2)-  h_distance / (
    #     torch.pow(b1_wh[..., 1],2)+torch.pow(b2_wh[..., 1], 2))
    # print('wd', w_distance / (torch.pow(torch.max(b1_wh[..., 0], b2_wh[..., 0]), 2)))
    # print('h distance',h_distance / (torch.pow(torch.max(b1_wh[...,1],b2_wh[...,1]),2)))

    # print('w_distance/(union_wh[...,0]*union_wh[...,0])',w_distance/(union_wh[...,0]*union_wh[...,0]))
    # print('h_distance/(union_wh[...,1]*union_wh[...,1]',h_distance/(union_wh[...,1]*union_wh[...,1]))
    # eiou=eiou-torch.sum( MSELoss(b1_xy,b2_xy))-torch.sum( MSELoss(b1_wh,b2_wh))
    # print('eiou1=',eiou)
    return eiou
def clip_by_tensor(t,t_min,t_max):
    t=t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result
def iou(b1,b2):#xywh
    # 求出预测框左上角右下角
    b1=b1.float()
    b2=b2.float()
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # 求出真实框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 求真实框和预测框所有的iou
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou_value = (intersect_area +0.001)/ (torch.clamp(union_area, min=1e-6)+0.001)
    return iou_value

def focal_eiou_loss(a,b  ):
    # return torch.pow(  1-bbox_iou(a,b) ,0.5) * torch.log(bbox_iou(a,b))*(1-box_eiou(a,b))
    # print(1-box_eiou(a,b),'1-box_eiou(a,b)')
    # print('torch.pow(   bbox_iou(a,b) ,0.5) * (1-box_eiou(a,b))',torch.pow(  1- bbox_iou(a,b) ,0.5) * (1-box_eiou(a,b)))
    # return -torch.log(iou(a,b))*torch.pow(  1- iou(a,b) ,0.5) * (1-box_eiou(a,b))
    # return  torch.pow( iou(a,b),0.4) * (1-box_eiou(a,b))
    return  (1+iou(a,b)) * (1-box_eiou(a,b))
def focal_eiou_lossv2(a,b  ):
    # print(bbox_iou(a, b, x1y1x2y2=False))
    # print(bbox_iou(a, b), 'iou')
    # return -torch.pow(  1-bbox_iou(a,b,x1y1x2y2=False) ,0.5) * torch.log(bbox_iou(a,b,x1y1x2y2=False))*(1-box_eiou(a,b))

    # print('torch.pow(   bbox_iou(a,b) ,0.5) * (1-box_eiou(a,b))',torch.pow(  1- bbox_iou(a,b) ,0.5) * (1-box_eiou(a,b)))
    return  (0.1+bbox_iou(a,b,x1y1x2y2=False))  * (1-box_eiou(a,b))#此方法有争议，好像效果不好是它造成的
def jaccard(_box_a, _box_b):
    b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
    b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
    b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
    b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2
    box_a = torch.zeros_like(_box_a)
    box_b = torch.zeros_like(_box_b)
    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
    box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)

    inter = inter[:, :, 0] * inter[:, :, 1]
    # 计算先验框和真实框各自的面积
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    # 求IOU
    union = area_a + area_b - inter
    return inter / union  # [A,B]
    
#---------------------------------------------------#
#   平滑标签
#---------------------------------------------------#
def smooth_labels(y_true, label_smoothing,num_classes):
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

def box_ciou(b1, b2):
    """
    输入为：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    返回为：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # 求出预测框左上角右下角
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # 求出真实框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 求真实框和预测框所有的iou
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / torch.clamp(union_area,min = 1e-6)

    # 计算中心的差距
    center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)
    
    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxes = torch.max(b1_maxes, b2_maxes)
    enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
    # 计算对角线距离
    enclose_diagonal = torch.sum(torch.pow(enclose_wh,2), axis=-1)
    ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal,min = 1e-6)
    
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[..., 0]/torch.clamp(b1_wh[..., 1],min = 1e-6)) - torch.atan(b2_wh[..., 0]/torch.clamp(b2_wh[..., 1],min = 1e-6))), 2)
    alpha = v / torch.clamp((1.0 - iou + v),min=1e-6)
    ciou = ciou - alpha * v
    return ciou
  
def clip_by_tensor(t,t_min,t_max):
    t=t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def MSELoss(pred,target):
    return (pred-target)**2
def BCELoss(pred,target,Focal_Loss=False):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    if Focal_Loss:
        output = -target * torch.log(pred)*(1-pred)*(1-pred)*0.9 - (1.0 - target) * torch.log(1.0 - pred)*pred*pred*0.1
    return output

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size, label_smooth=0, cuda=True, normalize=True):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size
        self.feature_length = [img_size[0]//32,img_size[0]//16]
        self.label_smooth = label_smooth

        self.ignore_threshold = 0.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        self.lambda_loc = 1.0
        self.cuda = cuda
        self.normalize = normalize
        self.count=0

    def forward(self, input, targets=None):
        #----------------------------------------------------#
        #   input的shape为  bs, 3*(5+num_classes), 13, 13
        #                   bs, 3*(5+num_classes), 26, 26
        #----------------------------------------------------#
        
        #-----------------------#
        #   一共多少张图片
        #-----------------------#
        bs = input.size(0)
        #-----------------------#
        #   特征层的高
        #-----------------------#
        in_h = input.size(2)
        #-----------------------#
        #   特征层的宽
        #-----------------------#
        in_w = input.size(3)

        #-----------------------------------------------------------------------#
        #   计算步长
        #   每一个特征点对应原来的图片上多少个像素点
        #   如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        #   如果特征层为26x26的话，一个特征点就对应原来的图片上的16个像素点
        #   stride_h = stride_w = 32、16、8
        #-----------------------------------------------------------------------#
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w


        #-------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        #-------------------------------------------------#
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        
        #-----------------------------------------------#
        #   输入的input一共有两个，他们的shape分别是
        #   batch_size, 3, 13, 13, 5 + num_classes
        #   batch_size, 3, 26, 26, 5 + num_classes
        #-----------------------------------------------#
        prediction = input.view(bs, int(self.num_anchors/2),
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        
        # 获得置信度，是否有物体
        conf = torch.sigmoid(prediction[..., 4])
        # 种类置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])

        #---------------------------------------------------------------#
        #   找到哪些先验框内部包含物体
        #   利用真实框和先验框计算交并比
        #   mask        batch_size, 3, in_h, in_w   有目标的特征点
        #   noobj_mask  batch_size, 3, in_h, in_w   无目标的特征点
        #   t_box       batch_size, 3, in_h, in_w, 4   中心宽高的真实值
        #   tconf       batch_size, 3, in_h, in_w   置信度真实值
        #   tcls        batch_size, 3, in_h, in_w, num_classes  种类真实值
        #----------------------------------------------------------------#
        mask, noobj_mask, t_box, tconf, tcls, box_loss_scale_x, box_loss_scale_y = self.get_target(targets, scaled_anchors,in_w, in_h,self.ignore_threshold)

        #---------------------------------------------------------------#
        #   将预测结果进行解码，判断预测结果和真实值的重合程度
        #   如果重合程度过大则忽略，因为这些特征点属于预测比较准确的特征点
        #   作为负样本不合适
        #----------------------------------------------------------------#
        noobj_mask, pred_boxes_for_ciou = self.get_ignore(prediction, targets, scaled_anchors, in_w, in_h, noobj_mask)

        if self.cuda:
            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            box_loss_scale_x, box_loss_scale_y= box_loss_scale_x.cuda(), box_loss_scale_y.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()
            pred_boxes_for_ciou = pred_boxes_for_ciou.cuda()
            t_box = t_box.cuda()

        box_loss_scale = 2 - box_loss_scale_x * box_loss_scale_y
        #---------------------------------------------------------------#
        #   计算预测结果和真实结果的CIOU

        #----------------------------------------------------------------#
        if USE_CIOU:

            if self.count==0:
                print('ciou')
            ciou = (1 - box_ciou(pred_boxes_for_ciou[mask.bool()], t_box[mask.bool()])) * box_loss_scale[mask.bool()]
            # ciou=focal_ciou_loss(pred_boxes_for_ciou[mask.bool()], t_box[mask.bool()],box_loss_scale[mask.bool()])

            loss_loc = torch.sum(ciou)
        ##############pred_boxes_for_ciou是调整后的预测框中心与宽高
        if USE_EIOU:
            if self.count == 0:
                print('使用eiou')
            eiou = focal_eiou_loss(pred_boxes_for_ciou[mask.bool()], t_box[mask.bool()])

            eiou = LfM(eiou) * box_loss_scale[mask.bool()]
            loss_loc=torch.sum(eiou)

        self.count+=1
        # 计算置信度的loss
        # loss_conf = torch.sum(BCELoss(conf, mask,Focal_Loss=True) * mask) + \
        #             torch.sum(BCELoss(conf, mask,Focal_Loss=True) * noobj_mask)
        loss_conf = torch.sum(BCELoss(conf, mask, Focal_Loss=Focal_LOSS) * mask) + \
                    torch.sum(BCELoss(conf, mask, Focal_Loss=Focal_LOSS) * noobj_mask)
        loss_cls = torch.sum(
            BCELoss(pred_cls[mask == 1], smooth_labels(tcls[mask == 1], self.label_smooth, self.num_classes)))
        loss = loss_conf * self.lambda_conf + loss_cls * self.lambda_cls + loss_loc * self.lambda_loc
        if LOSS_UNITE:
            if self.count == 1:
                print('use united loss')
            self.count+=1
            class_conf, class_pred = torch.max(pred_cls, -1, keepdim=False)
            conf_s=conf.clone()
            # conf_s.require_grad=False
            # loss_unite = torch.sum(MSELoss(class_conf[noobj_mask==1], conf_s[noobj_mask==1]))/100
            loss_unite = torch.sum(MSELoss(class_conf[mask == 0], conf[mask == 0])) /100#被忽略部分也考虑
            if self.count%100==0:
                print(loss_unite)
                print(loss_conf * self.lambda_conf + loss_cls * self.lambda_cls + loss_loc * self.lambda_loc)
            loss = loss_conf * self.lambda_conf + loss_cls * self.lambda_cls + loss_loc * self.lambda_loc + loss_unite

        


        if self.normalize:
            num_pos = torch.sum(mask)
            num_pos = torch.max(num_pos, torch.ones_like(num_pos))
        else:
            num_pos = bs/2

        return loss, num_pos

    def get_target(self, target, anchors, in_w, in_h, ignore_threshold):
        #-----------------------------------------------------#
        #   计算一共有多少张图片
        #-----------------------------------------------------#
        bs = len(target)
        #-------------------------------------------------------#
        #   获得当前特征层先验框所属的编号，方便后面对先验框筛选
        #-------------------------------------------------------#
        anchor_index = [[3,4,5],[1,2,3]][self.feature_length.index(in_w)]
        
        #-------------------------------------------------------#
        #   创建全是0或者全是1的阵列
        #-------------------------------------------------------#
        mask = torch.zeros(bs, int(self.num_anchors/2), in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, int(self.num_anchors/2), in_h, in_w, requires_grad=False)

        tx = torch.zeros(bs, int(self.num_anchors/2), in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, int(self.num_anchors/2), in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, int(self.num_anchors/2), in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, int(self.num_anchors/2), in_h, in_w, requires_grad=False)
        t_box = torch.zeros(bs, int(self.num_anchors/2), in_h, in_w, 4, requires_grad=False)
        tconf = torch.zeros(bs, int(self.num_anchors/2), in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, int(self.num_anchors/2), in_h, in_w, self.num_classes, requires_grad=False)

        box_loss_scale_x = torch.zeros(bs, int(self.num_anchors/2), in_h, in_w, requires_grad=False)
        box_loss_scale_y = torch.zeros(bs, int(self.num_anchors/2), in_h, in_w, requires_grad=False)
        for b in range(bs):
            if len(target[b])==0:
                continue
            #-------------------------------------------------------#
            #   计算出正样本在特征层上的中心点
            #-------------------------------------------------------#
            gxs = target[b][:, 0:1] * in_w
            gys = target[b][:, 1:2] * in_h
            
            #-------------------------------------------------------#
            #   计算出正样本相对于特征层的宽高
            #-------------------------------------------------------#
            gws = target[b][:, 2:3] * in_w
            ghs = target[b][:, 3:4] * in_h

            #-------------------------------------------------------#
            #   计算出正样本属于特征层的哪个特征点
            #-------------------------------------------------------#
            gis = torch.floor(gxs)
            gjs = torch.floor(gys)
            
            #-------------------------------------------------------#
            #   将真实框转换一个形式
            #   num_true_box, 4
            #-------------------------------------------------------#
            gt_box = torch.FloatTensor(torch.cat([torch.zeros_like(gws), torch.zeros_like(ghs), gws, ghs], 1))
            
            #-------------------------------------------------------#
            #   将先验框转换一个形式
            #   6, 4
            #-------------------------------------------------------#
            anchor_shapes = torch.FloatTensor(torch.cat((torch.zeros((self.num_anchors, 2)), torch.FloatTensor(anchors)), 1))
            #-------------------------------------------------------#
            #   计算交并比
            #   num_true_box, 6
            #-------------------------------------------------------#
            anch_ious = jaccard(gt_box, anchor_shapes)

            #-------------------------------------------------------#
            #   计算重合度最大的先验框是哪个
            #   num_true_box, 
            #-------------------------------------------------------#
            best_ns = torch.argmax(anch_ious,dim=-1)
            for i, best_n in enumerate(best_ns):
                if best_n not in anchor_index:
                    continue
                #-------------------------------------------------------------#
                #   取出各类坐标：
                #   gi和gj代表的是真实框对应的特征点的x轴y轴坐标
                #   gx和gy代表真实框的x轴和y轴坐标
                #   gw和gh代表真实框的宽和高
                #-------------------------------------------------------------#
                gi = gis[i].long()
                gj = gjs[i].long()
                gx = gxs[i]
                gy = gys[i]
                gw = gws[i]
                gh = ghs[i]
                if (gj < in_h) and (gi < in_w):
                    best_n = anchor_index.index(best_n)
                    #----------------------------------------#
                    #   noobj_mask代表无目标的特征点
                    #----------------------------------------#
                    noobj_mask[b, best_n, gj, gi] = 0
                    #----------------------------------------#
                    #   mask代表有目标的特征点
                    #----------------------------------------#
                    mask[b, best_n, gj, gi] = 1
                    #----------------------------------------#
                    #   tx、ty代表中心的真实值
                    #----------------------------------------#
                    tx[b, best_n, gj, gi] = gx
                    ty[b, best_n, gj, gi] = gy
                    #----------------------------------------#
                    #   tw、th代表宽高的真实值
                    #----------------------------------------#
                    tw[b, best_n, gj, gi] = gw
                    th[b, best_n, gj, gi] = gh
                    #----------------------------------------#
                    #   用于获得xywh的比例
                    #   大目标loss权重小，小目标loss权重大
                    #----------------------------------------#
                    box_loss_scale_x[b, best_n, gj, gi] = target[b][i, 2]
                    box_loss_scale_y[b, best_n, gj, gi] = target[b][i, 3]
                    #----------------------------------------#
                    #   tconf代表物体置信度
                    #----------------------------------------#
                    tconf[b, best_n, gj, gi] = 1
                    #----------------------------------------#
                    #   tcls代表种类置信度
                    #----------------------------------------#
                    tcls[b, best_n, gj, gi, target[b][i, 4].long()] = 1
                else:
                    print('Step {0} out of bound'.format(b))
                    print('gj: {0}, height: {1} | gi: {2}, width: {3}'.format(gj, in_h, gi, in_w))
                    continue
        t_box[...,0] = tx
        t_box[...,1] = ty
        t_box[...,2] = tw
        t_box[...,3] = th
        return mask, noobj_mask, t_box, tconf, tcls, box_loss_scale_x, box_loss_scale_y

    def get_ignore(self,prediction,target,scaled_anchors,in_w, in_h,noobj_mask):
        #-----------------------------------------------------#
        #   计算一共有多少张图片
        #-----------------------------------------------------#
        bs = len(target)
        #-------------------------------------------------------#
        #   获得当前特征层先验框所属的编号，方便后面对先验框筛选
        #-------------------------------------------------------#
        anchor_index = [[3,4,5],[1,2,3]][self.feature_length.index(in_w)]
        scaled_anchors = np.array(scaled_anchors)[anchor_index]

        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])  
        y = torch.sigmoid(prediction[..., 1])
        # 先验框的宽高调整参数
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # 生成网格，先验框中心，网格左上角
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs*self.num_anchors/2), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs*self.num_anchors/2), 1, 1).view(y.shape).type(FloatTensor)

        # 生成先验框的宽高
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        
        #-------------------------------------------------------#
        #   计算调整后的先验框中心与宽高
        #-------------------------------------------------------#
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x + grid_x
        pred_boxes[..., 1] = y + grid_y
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h
        for i in range(bs):
            pred_boxes_for_ignore = pred_boxes[i]
            #-------------------------------------------------------#
            #   将预测结果转换一个形式
            #   pred_boxes_for_ignore      num_anchors, 4
            #-------------------------------------------------------#
            pred_boxes_for_ignore = pred_boxes_for_ignore.view(-1, 4)
            #-------------------------------------------------------#
            #   计算真实框，并把真实框转换成相对于特征层的大小
            #   gt_box      num_true_box, 4
            #-------------------------------------------------------#
            if len(target[i]) > 0:
                gx = target[i][:, 0:1] * in_w
                gy = target[i][:, 1:2] * in_h
                gw = target[i][:, 2:3] * in_w
                gh = target[i][:, 3:4] * in_h
                gt_box = torch.FloatTensor(torch.cat([gx, gy, gw, gh],-1)).type(FloatTensor)

                #-------------------------------------------------------#
                #   计算交并比
                #   anch_ious       num_true_box, num_anchors
                #-------------------------------------------------------#
                anch_ious = jaccard(gt_box, pred_boxes_for_ignore)
                #-------------------------------------------------------#
                #   每个先验框对应真实框的最大重合度
                #   anch_ious_max   num_anchors
                #-------------------------------------------------------#
                anch_ious_max, _ = torch.max(anch_ious,dim=0)
                anch_ious_max = anch_ious_max.view(pred_boxes[i].size()[:3])
                noobj_mask[i][anch_ious_max>self.ignore_threshold] = 0
        return noobj_mask, pred_boxes

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


class Generator(object):
    def __init__(self,batch_size,
                 train_lines, image_size,
                 ):
        
        self.batch_size = batch_size
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        
    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        '''r实时数据增强的随机预处理'''
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # 调整目标框坐标
            box_data = np.zeros((len(box), 5))
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # 保留有效框
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            return image_data, box_data

        # resize image
        new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
        scale = rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        # correct boxes
        box_data = np.zeros((len(box),5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            box_data = np.zeros((len(box),5))
            box_data[:len(box)] = box

        return image_data, box_data

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, hue=.1, sat=1.5, val=1.5):
        '''random preprocessing for real-time data augmentation'''
        h, w = input_shape
        min_offset_x = 0.3
        min_offset_y = 0.3
        scale_low = 1-min(min_offset_x,min_offset_y)
        scale_high = scale_low+0.2

        image_datas = [] 
        box_datas = []
        index = 0

        place_x = [0,0,int(w*min_offset_x),int(w*min_offset_x)]
        place_y = [0,int(h*min_offset_y),int(h*min_offset_y),0]
        for line in annotation_line:
            # 每一行进行分割
            line_content = line.split()
            # 打开图片
            image = Image.open(line_content[0])
            image = image.convert("RGB") 
            # 图片的大小
            iw, ih = image.size
            # 保存框的位置
            box = np.array([np.array(list(map(int,box.split(',')))) for box in line_content[1:]])
            
            # 是否翻转图片
            flip = rand()<.5
            if flip and len(box)>0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0,2]] = iw - box[:, [2,0]]

            # 对输入进来的图片进行缩放
            new_ar = w/h
            scale = rand(scale_low, scale_high)
            if new_ar < 1:
                nh = int(scale*h)
                nw = int(nh*new_ar)
            else:
                nw = int(scale*w)
                nh = int(nw/new_ar)
            image = image.resize((nw,nh), Image.BICUBIC)

            # 进行色域变换
            hue = rand(-hue, hue)
            sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
            val = rand(1, val) if rand()<.5 else 1/rand(1, val)
            x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
            x[..., 0] += hue*360
            x[..., 0][x[..., 0]>1] -= 1
            x[..., 0][x[..., 0]<0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x[:,:, 0]>360, 0] = 360
            x[:, :, 1:][x[:, :, 1:]>1] = 1
            x[x<0] = 0
            image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) # numpy array, 0 to 1
            
            image = Image.fromarray((image*255).astype(np.uint8))
            # 将图片进行放置，分别对应四张分割图片的位置
            dx = place_x[index]
            dy = place_y[index]
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            
            index = index + 1
            box_data = []
            # 对box进行重新处理
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box
            
            image_datas.append(image_data)
            box_datas.append(box_data)

        # 将图片分割，放在一起
        cutx = np.random.randint(int(w*min_offset_x), int(w*(1 - min_offset_x)))
        cuty = np.random.randint(int(h*min_offset_y), int(h*(1 - min_offset_y)))

        new_image = np.zeros([h,w,3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        # 对框进行进一步的处理
        new_boxes = np.array(merge_bboxes(box_datas, cutx, cuty))

        if len(new_boxes) == 0:
            return new_image, []
        if (new_boxes[:,:4]>0).any():
            return new_image, new_boxes
        else:
            return new_image, []

    def generate(self, train = True, mosaic = True):
        while True:
            shuffle(self.train_lines)
            lines = self.train_lines
            inputs = []
            targets = []
            flag = True
            n = len(lines)
            for i in range(len(lines)):
                if mosaic == True:
                    if flag and (i+4) < n:
                        img,y = self.get_random_data_with_Mosaic(lines[i:i+4], self.image_size[0:2])
                        i = (i+4) % n
                    else:
                        img,y = self.get_random_data(lines[i], self.image_size[0:2], random=train)
                        i = (i+1) % n
                    flag = bool(1-flag)
                else:
                    img,y = self.get_random_data(lines[i], self.image_size[0:2], random=train)
                    i = (i+1) % n
                    
                if len(y)!=0:
                    boxes = np.array(y[:,:4],dtype=np.float32)
                    boxes[:,0] = boxes[:,0]/self.image_size[1]
                    boxes[:,1] = boxes[:,1]/self.image_size[0]
                    boxes[:,2] = boxes[:,2]/self.image_size[1]
                    boxes[:,3] = boxes[:,3]/self.image_size[0]

                    boxes = np.maximum(np.minimum(boxes,1),0)
                    boxes[:,2] = boxes[:,2] - boxes[:,0]
                    boxes[:,3] = boxes[:,3] - boxes[:,1]
    
                    boxes[:,0] = boxes[:,0] + boxes[:,2]/2
                    boxes[:,1] = boxes[:,1] + boxes[:,3]/2
                    y = np.concatenate([boxes,y[:,-1:]],axis=-1)
                    
                img = np.array(img,dtype = np.float32)

                inputs.append(np.transpose(img/255.0,(2,0,1)))              
                targets.append(np.array(y,dtype = np.float32))
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = targets
                    inputs = []
                    targets = []
                    yield tmp_inp, tmp_targets
