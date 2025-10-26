import os
from sklearn.metrics import confusion_matrix, classification_report
from ultralytics import YOLO
from PIL import Image

# Load model once
MODEL_PATH = "/Users/prateekgaurav/Desktop/PotholeDetectionProject/2ndReview/CodesToTrain/runs/detect/train22/weights/best.pt"
model = YOLO(MODEL_PATH)

# Test dataset folder structure: test/pothole/*.jpg, test/normal/*.jpg
TEST_DIR = "/Users/prateekgaurav/Desktop/PotholeDetectionProject/2ndReview/dataset/images/test"

y_true, y_pred = [], []

for label_dir in ["pothole", "normal"]:
    dir_path = os.path.join(TEST_DIR, label_dir)
    for f in os.listdir(dir_path):
        if not f.lower().endswith((".jpg", ".png")):
            continue
        img_path = os.path.join(dir_path, f)
        results = model(img_path)
        detections = results[0].boxes

        # Infer binary prediction
        pred_label = "normal"
        if len(detections) > 0:
            labels = [model.names[int(cls)] for cls in detections.cls]
            if "pothole" in labels:
                pred_label = "pothole"

        y_true.append(label_dir)
        y_pred.append(pred_label)

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=["pothole", "normal"])
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_true, y_pred, digits=4))