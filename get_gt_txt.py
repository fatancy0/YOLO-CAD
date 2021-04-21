#----------------------------------------------------#
#   获取测试集的ground-truth
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
import sys
import os
import glob
import xml.etree.ElementTree as ET
from utils.cfg import DATASET
'''
！！！！！！！！！！！！！注意事项！！！！！！！！！！！！！
# 这一部分是当xml有无关的类的时候，下方有代码可以进行筛选！
'''
#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

image_ids = open('F:\毕业\数据集\VOC2007/ImageSets/Main/val.txt').read().strip().split()
print(image_ids)
if DATASET==0:
    image_ids=open(r"F:\毕业\数据集\MIXuav\yolo_uav\val.txt").read().strip().split()
    if not os.path.exists("./input/uav"):
        os.makedirs("./input/uav")
    if not os.path.exists("./input/uav/ground-truth"):
        os.makedirs("./input/uav/ground-truth")
if DATASET==4:
    image_ids=open(r"F:\毕业\数据集\VOC2007test\ImageSets\Main\val.txt").read().strip().split()
    if not os.path.exists("./input/voc12"):
        os.makedirs("./input/voc12")
    if not os.path.exists("./input/voc12/ground-truth"):
        os.makedirs("./input/voc12/ground-truth")#意思是对voc12的测试集，为voc2007的test部分
if DATASET==6:
    image_ids=open(r"D:\dataset\yolo_uav\val.txt").read().strip().split()
    if not os.path.exists("./input/yolo_uav"):
        os.makedirs("./input/yolo_uav")
    if not os.path.exists("./input/yolo_uav/ground-truth"):
        os.makedirs("./input/yolo_uav/ground-truth")#意思是对voc12的测试集，为voc2007的test部分
if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/ground-truth"):
    os.makedirs("./input/ground-truth")

for image_id in image_ids:
    path="./input/ground-truth/"
    if DATASET==0:
        path="./input/uav/ground-truth/"
    if DATASET==4:
        path="./input/voc12/ground-truth/"
    if DATASET == 6:
        path = "./input/yolo_uav/ground-truth/"
    with open(path+image_id+".txt", "w") as new_f:

        if DATASET==0:
            root = ET.parse("F:\毕业\数据集\MIXuav\yolo_uav/Annotations/" + image_id + ".xml").getroot()
        if DATASET==1:
            root = ET.parse("F:\毕业\数据集/VOC2007/Annotations/" + image_id + ".xml").getroot()
        if DATASET == 4:
            root = ET.parse("F:\毕业\数据集\VOC2007test\Annotations/" + image_id + ".xml").getroot()
        if DATASET == 6:
            root = ET.parse("D:\dataset\yolo_uav\Annotations/" + image_id + ".xml").getroot()
        for obj in root.findall('object'):
            difficult_flag = False
            if obj.find('difficult')!=None:
                difficult = obj.find('difficult').text
                if int(difficult)==1:
                    difficult_flag = True
            obj_name = obj.find('name').text
            if DATASET==0:
                obj_name='uav'
            '''
            ！！！！！！！！！！！！注意事项！！！！！！！！！！！！
            # 这一部分是当xml有无关的类的时候，可以取消下面代码的注释
            # 利用对应的classes.txt来进行筛选！！！！！！！！！！！！
            '''
            # classes_path = 'model_data/voc_classes.txt'
            # class_names = get_classes(classes_path)
            # if obj_name not in class_names:
            #     continue

            bndbox = obj.find('bndbox')
            left = bndbox.find('xmin').text
            top = bndbox.find('ymin').text
            right = bndbox.find('xmax').text
            bottom = bndbox.find('ymax').text

            if difficult_flag:
                new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
            else:
                new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

print("Conversion completed!")
