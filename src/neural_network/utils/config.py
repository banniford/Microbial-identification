# 配置文件
Config = {
    'num_classes': 2,#训练时种类类个数+1// 填写种类时获取
    'feature_maps': [38, 19, 10, 5, 3, 1], #中间特征层预测时卷积层尺寸 //不变
    'min_dim': 300, # 图片预处理转换后大小
    'steps': [8, 16, 32, 64, 100, 300], # 处理步长 //不变
    'min_sizes': [30, 60, 111, 162, 213, 264], # 最小框尺寸 //不变
    'max_sizes': [60, 111, 162, 213, 264, 315], # 最大框尺寸 //不变
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]], # 生成长方形框时比例 //不变
    'variance': [0.1, 0.2], #不变
    'clip': True, # np.clip clip函数裁剪最小值和最大值，防止低于0，超出1
    'name': 'VOC', # VOC数据集格式前缀
    "model_path":"neural_network/outputs/Epoch10-Loc0.7121-Conf2.2285.pth", #预测模型路径
    "classes_path":'neural_network/model_data/voc_classes.txt',# 类别
    "migrate_path":'neural_network/model_data/ssd_weights.pth',#训练迁移模型路径
    "model_image_size" : (300, 300, 3), # 输入参数
    "confidence": 0.5, # 置信度
    "Cuda":False,#“Cuda”:True # Gpu加速
    "bkg_label":0,# 背景标签 // 不变
    "top_k":200,# 置信度前200的框数 // 不变
    "conf_thresh":0.01, #分类预测阈值
    "nms_thresh":0.45,# 阈值 必须大于零！！
    "trainval_percent": 1,  # 自己设定（训练集+验证集）所占（训练集+验证集+测试集）的比重
    "train_percent": 1,  # 自己设定（训练集）所占（训练集+验证集）的比重
    "Batch_size":4,#每批次输出图片数量
    "lr":1e-5,#学习率
    "Epoch":10,#循环轮次
    "Start_iter":0,#从哪一组开始
    "loc_loss":1,
    "conf_loss":2.5
}
