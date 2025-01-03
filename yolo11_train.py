from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    model='yolo11n.pt',  # 预训练模型路径
    data=r'C:\Users\13617\Desktop\mycode\eda_match\datasets\eda\eda.yaml',  # 数据集配置文件路径
    epochs=100,  # 训练100个epoch
    batch=16,  # 每批次16张图片
    imgsz=640,  # 图像大小
    save=True,  # 保存训练过程中模型检查点
    save_period=10,  # 每10个epoch保存一次模型
    cache=False,  # 不使用缓存
    device='xpu',  # 
    workers=8,  # 数据加载使用8个线程
    project='datasets/eda',  # 项目目录
    name='yolo_train_run',  # 训练子目录名称
    exist_ok=False,  # 不覆盖现有的目录
    # pretrained=True,  # 使用预训练模型
    # optimizer='auto',  # 自动选择优化器
    # seed=42,  # 随机种子
    # deterministic=True,  # 使用确定性算法
    # rect=False,  # 不启用矩形训练
    # cos_lr=True,  # 启用余弦学习率调整
    # close_mosaic=10,  # 在最后10个epoch禁用马赛克数据增强
    # resume=False,  # 不从断点恢复训练
    # amp=True,  # 启用自动混合精度训练
    # fraction=1.0,  # 使用全部数据集进行训练
    # profile=False,  # 不启用ONNX/TensorRT速度分析
    # freeze=None,  # 不冻结层
    # lr0=0.01,  # 初始学习率
    # lrf=0.01,  # 最终学习率与初始学习率的比例
    # momentum=0.937,  # 动量（SGD）
    # weight_decay=0.0005,  # 权重衰减
    # warmup_epochs=3,  # 学习率预热的epoch数
    # warmup_momentum=0.8,  # 预热阶段的动量
    # warmup_bias_lr=0.1,  # 预热阶段的偏置学习率
    # box=7.5,  # 边框损失的权重
    # cls=0.5,  # 分类损失的权重
    # dfl=1.5,  # 分布焦点损失的权重
    # pose=12.0,  # 姿势估计损失的权重
    # kobj=2.0,  # 关键点目标性损失的权重
    # nbs=64,  # 标准化损失的批次大小
    # overlap_mask=True,  # 启用重叠目标掩膜合并
    # mask_ratio=4,  # 分割掩膜的下采样比例
    # dropout=0.0,  # Dropout率
    # val=True,  # 启用验证
    # plots=True  # 启用训练过程的可视化图表
)
