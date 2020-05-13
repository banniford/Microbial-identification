#从VOC数据集的xml文件中提取位置信息并生成训练索引
import sys
import xml.etree.ElementTree as ET
import os
# print(sys.path)

classes=[]

def convert_annotation(year, image_id, list_file):
    in_file = open('neural_network/VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text



        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


def voc():
    indx = ('python neural_network/VOCdevkit/VOC2007/voc2ssd.py')
    p = os.system(indx)
    sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
    # print(p) 运行成功会返回0

    with open('neural_network/model_data/voc_classes.txt', 'r') as f:
        for line in f.readlines():  # 依次读取每行
            line = line.strip()  # 去掉每行头尾空白
            if line != '' and line not in classes:
                classes.append(line)

    wd = os.getcwd()[:]
    for year, image_set in sets:
        image_ids = open('neural_network/VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (year, image_set)).read().strip().split()
        list_file = open('neural_network/%s_%s.txt' % (year, image_set), 'w')
        for image_id in image_ids:
            list_file.write('%s/neural_network/VOCdevkit/VOC%s/JPEGImages/%s.jpg' % (wd, year, image_id))
            convert_annotation(year, image_id, list_file)
            list_file.write('\n')
        list_file.close()