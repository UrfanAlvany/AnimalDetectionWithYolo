from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')

model.train(data='C:/Users/user/Desktop/detectanimal/ultralytics/archive',
            epochs=20, imgsz=512, save_period=5, device="cpu", name="animal_classification")
