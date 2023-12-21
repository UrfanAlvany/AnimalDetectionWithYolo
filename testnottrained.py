from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO('yolov8n-cls.pt')
results = model('0df78ee76bafd3a9.jpg')
print(results.pandas().xyxy[0])  # For a single image

# To visualize the predictions on the images
results.show()


#image 1/1 C:\Users\user\Desktop\allv2\detectanimal\ultralytics\0df78ee76bafd3a9.jpg: 224x224 sloth_bear 0.93, American_black_bear 0.06, howler_monkey 0.00, brown_bear 0.00, lesser_panda 0.00, 24.4ms
#Speed: 0.0ms preprocess, 24.4ms inference, 0.0ms postprocess per image at shape (1, 3, 224, 224)