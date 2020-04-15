import os
import random 
from utils.config import Config

xmlfilepath=r'neural_network/VOCdevkit/VOC2007/Annotations' #os相对于上层调用时得os路径
saveBasePath=r"neural_network//VOCdevkit/VOC2007/ImageSets/Main/"
 
trainval_percent=Config["trainval_percent"]  # 自己设定（训练集+验证集）所占（训练集+验证集+测试集）的比重
train_percent=Config["train_percent"] # 自己设定（训练集）所占（训练集+验证集）的比重

temp_xml = os.listdir(xmlfilepath)
total_xml = []
for xml in temp_xml:
    if xml.endswith(".xml"):
        total_xml.append(xml)

num=len(total_xml)  
list=range(num)  
tv=int(num*trainval_percent)  
tr=int(tv*train_percent)  
trainval= random.sample(list,tv)  
train=random.sample(trainval,tr)  
 
print("train and val size",tv)
print("traub suze",tr)
ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
fval = open(os.path.join(saveBasePath,'val.txt'), 'w')  
 
for i  in list:  
    name=total_xml[i][:-4]+'\n'  
    if i in trainval:  
        ftrainval.write(name)  
        if i in train:  
            ftrain.write(name)  
        else:  
            fval.write(name)  
    else:  
        ftest.write(name)  
  
ftrainval.close()  
ftrain.close()  
fval.close()  
ftest .close()
