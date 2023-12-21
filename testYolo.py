import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def test_yolov8_model(model, test_loader):
    # Ensure the model is in evaluation mode
    model.eval()

    # Lists to store targets and predictions
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:  # Replace with your data loading method
            # Forward pass
            outputs = model(images)

            # Extract predictions (Modify this part based on your model's output format)
            preds = outputs.argmax(dim=1).cpu().numpy()

            # Collect all predictions and targets
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    # Compute metrics
    print("Classification Report:")
    print(classification_report(all_targets, all_preds))

    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))

