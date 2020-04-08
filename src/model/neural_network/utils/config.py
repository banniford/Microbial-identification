# 配置文件
Config = {
    'num_classes': 2,#训练时种类类个数+1
    'feature_maps': [38, 19, 10, 5, 3, 1], #中间特征层预测时卷积层尺寸
    'min_dim': 300, # 图片预处理转换后大小
    'steps': [8, 16, 32, 64, 100, 300], # 处理步长
    'min_sizes': [30, 60, 111, 162, 213, 264], # 最小框尺寸
    'max_sizes': [60, 111, 162, 213, 264, 315], # 最大框尺寸
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]], # 生成长方形框时比例
    'variance': [0.1, 0.2],
    'clip': True, # np.clip clip函数裁剪最小值和最大值，防止低于0，超出1
    'name': 'VOC', # VOC数据集格式前缀
    "model_path": 'model_data/ssd_weights.pth',# 模型路径
    # "model_path":'outputs/1.pth',# 模型路径
    "classes_path": 'model_data/voc_classes.txt',# 类别
    "model_image_size" : (300, 300, 3), # 输入参数
    "confidence": 0.5, # 置信度
    "Cuda":False, # Gpu加速
    "bkg_label":0,# 背景标签
    "top_k":200,# 置信度前200的框数
    "conf_thresh":0.01,# 分类
    "nms_thresh":0.45,# 阈值 必须大于零！！
}
