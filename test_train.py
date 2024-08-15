from ultralytics import YOLO

yolo = YOLO("yolov8n.pt")

yolo.train(data=r"D:\program\object detection\test_scratch\data.yaml",workers=0,epochs=100,batch=16)