from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO(r'D:/LUOYUZHE/ultralytics-main/ultralytics/runs/detect/NEU RFB SCCONV ADOWN/weights/best.pt')
    # 验证模型
    metrics=model.predict(source='D:/LUOYUZHE/NEU-DET/NEU-DET/val/images',save=True,save_conf=True,save_txt=True,name='output',device=0 )

