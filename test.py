from ultralytics import YOLO

model = YOLO("runs/detect/train_m/weights/best.pt") # 加载最佳权重
metrics = model.val(
    data='/home/share/clr/share/data/stenosisv2/data.yaml', 
    imgsz=512,
    # batch=64,
    split='test',
    # save_txt=True,
    conf=0.001,         # 设置检测的最小置信度阈值。较低的值会提高召回率，但也可能引入更多的假阳性。
    iou=0.7,            # 设置交并比（Intersection Over Union）阈值，用于非极大值抑制（Non-Maximum Suppression）。控制重复检测的消除。
    plots=True,
    save_conf=True,
    save_json=True,
    project='runs/detect',
    name='test_m',
    # verbose=True,
) # 验证数据集
print(metrics.box.map50, metrics.box.map) # 输出 mAP@50 值