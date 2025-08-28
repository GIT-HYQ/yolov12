from ultralytics import YOLO

model = YOLO("runs/detect/train_n/weights/best.pt") # 加载最佳权重
metrics = model.predict(
    source='/home/share/clr/share/data/stenosisv2/images/test',
    save=True,
    save_txt=True,
    conf=0.25,         # 设置检测的最小置信度阈值。较低的值会提高召回率，但也可能引入更多的假阳性。
    iou=0.7,           # 用于非极大值抑制 (NMS) 的 Intersection Over Union (IoU) 阈值。较低的值会通过消除重叠的框来减少检测结果，这对于减少重复项很有用。
    save_conf=True,
    project='runs/detect',
    name='predict_n',
)
# print(metrics) 