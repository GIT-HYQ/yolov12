from ultralytics import YOLO

model = YOLO('yolov12x.yaml')

# Train the model
results = model.train(
  data='/home/share/clr/share/data/stenosisv2/data.yaml',
  epochs=600, 
  batch=128, 
  imgsz=512,
  scale=0.5,  # N:0.5 S:0.9; M:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.0,  # N:0.0 S:0.05; M:0.15; L:0.15; X:0.2
  copy_paste=0.1,  # N:0.1 S:0.15; M:0.4; L:0.5; X:0.6
  device="0,1,2,3",
  project='runs/detect',
  name='train_x',
  patience=0,
)