from ultralytics import YOLO

# Load a model
# yaml会自动下载
model = YOLO("D:/LUOYUZHE/ultralytics-main/ultralytics/runs/detect/GC10 RFB SCCONV ADOWN/weights/best.pt")  # build a new model from scratch
# model = YOLO("d:/Data/yolov8s.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="D:/LUOYUZHE/ultralytics-main/ultralytics/assets/gc10.yaml",batch=8,workers=0,epochs=300,patience=200)

