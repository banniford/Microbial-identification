# Microbial-identification
 ## 前言

 Graduation project

本人博客：www.paniford.com

本项目主要使用arduino+Raspberry Pi 3B+deep learning（SSD网络）+python3 实现微生物图像识别。

项目结构

├─src
│  │  arduino_control.py
│  │  Microbial.py
│  │  mid_module.py
│  │  parameter.py
│  │  run.py （主程序）
│  │ 
│  └─neural_network
│      │  2007_test.txt
│      │  2007_train.txt
│      │  2007_val.txt
│      │  predict.py
│      │  train.py
│      │ 
│      ├─img（预测图片存放）
│      │      save_predict_img.md
│      │ 
│      ├─model_data （字体/迁移模型/类别）
│      │      README.md
│      │      simhei.ttf
│      │      ssd_weights.pth（迁移模型）
│      │      voc_classes.txt
│      │ 
│      ├─nets（ssd网络）
│      │      README.md
│      │      ssd.py
│      │      ssd_layers.py
│      │      ssd_training.py
│      │      vgg.py
│      │      原理图.png
│      │ 
│      ├─outputs
│      │      Epoch10-Loc0.7121-Conf2.2285.pth（训练后模型）
│      │      README.md
│      │ 
│      ├─utils （预测框生成/配置文件/数据集生成索引）
│      │      box_utils.py
│      │      config.py
│      │      voc_annotation.py
│      │ 
│      └─VOCdevkit（训练集）
│          └─VOC2007
│              │  voc2ssd.py
│              │ 
│              ├─Annotations（标签样本）
│              ├─ImageSets（索引）
│              │  └─Main
│              │          test.txt
│              │          train.txt
│              │          trainval.txt
│              │          val.txt
│              │ 
│              └─JPEGImages（图片样本）
└─ui（ui文件）
        Microbial.ui
        parameter.ui
        README.md

## 使用框架及工具包：

- python 3.7.3
- pytorch 1.4.0
- PyQt5
- numpy
- PIL
- opencv 3
- serial (arduino通信)
- labalimg

 

## 使用说明

### 准备条件

1）补充 `model_data` 目录下的迁移模型，具体看文件夹内 Readme

2）准备数据集并放置与 `VOCdevkit/VOC2007/JPEGImages` 文件夹下

3)   使用labimg生成与`VOCdevkit/VOC2007/JPEGImages`内图片对应的标签文件并存放于`VOCdevkit/VOC2007/Annotations`内

4） 启动`run.py` 在`模型训练`页面设置参数以及标签名称，一定要与标签文件内标注名称对应，多个标签空格隔开

5）训练完成后在`output`中产生的模型可以用于预测



### 注意事项

- 网络层报错检查类别文件，输入参数一定要和类别文件`neural_network/model_data/voc_classes.txt`中类别一一对应



## 目前进展

- 使用PyQt5进行了外观初步构建,由arduino控制界面，模型训练界面，图片预测界面构成

- 目前已实现功能主要分为三大部分

  - 摄像头控制部分

    - 检测并连接摄像头
  - arduino通讯检测及步进电机控制
    - opencv拉普拉斯算子计算清晰度

  ![摄像头控制](doc/1.png)

  - 模型训练部分
    - SSD目标检测网络
      - 参数更改
      - 网络中间层参数数量信息打印
    - 数据集校验，生成索引文件
    - 多线程开始/中止训练

  ![模型训练](doc/2.png)

  - 模型预测部分
    - 选择训练完成模型进行预测标注
    - 结果打印

  ![模型预测](doc/3.png)

  ![模型预测](doc/4.png)

- 目前可以进行模型的参数设定，训练以及识别，权重模型和数据集需要下载，详情看文件内Readme



## 致谢

SSD目标检测网络详解b站up主 Bubbliiiing 视频链接：https://www.bilibili.com/video/BV1A7411976Z

up主Blog：https://blog.csdn.net/weixin_44791964/article/list/1?t=1&orderby=UpdateTime