from ultralytics import YOLO

# Load your trained model
model = YOLO('best.pt')

# Load your test dataset
test_data = model('0df78ee76bafd3a9.jpg')

# Perform inference
results = model(test_data)

# Evaluate results
metrics = results.metrics()  # Get precision, recall, mAP, etc.

# Optionally, visualize some predictions
results.show()

#image 1/1 C:\Users\user\Desktop\allv2\detectanimal\ultralytics\0df78ee76bafd3a9.jpg: 512x512 Bear 0.96, Brown bear 0.04, Camel 0.00, Butterfly 0.00, Bull 0.00, 57.2ms
#Speed: 0.0ms preprocess, 57.2ms inference, 0.0ms postprocess per image at shape (1, 3, 512, 512)
