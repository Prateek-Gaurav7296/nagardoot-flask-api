from flask import Flask, request, jsonify
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import requests
import traceback

app = Flask(__name__)

# === CONFIG ===
MODEL_PATH = "https://huggingface.co/PrateekGaurav7296/pothole-detection/resolve/main/best.pt"
model = YOLO(MODEL_PATH)  # Load model once at startup

# === PREDICT ENDPOINT ===
@app.route("/predict", methods=["POST"])
def predict():
    """
    Production-ready pothole detection endpoint.
    
    Expects JSON payload:
    {
        "downloadUrl": "https://s3-bucket-url/image.jpg",
        "fileName": "image.jpg",
        "userId": "42"  (optional)
    }
    
    Returns JSON response:
    {
        "status": "success",
        "fileName": "image.jpg",
        "prediction": "pothole" | "normal",
        "confidence": 0.95,
        "userId": "42"
    }
    """
    try:
        # Parse request
        data = request.get_json(force=True)
        download_url = data.get("downloadUrl")
        file_name = data.get("fileName", "unknown.jpg")
        user_id = data.get("userId", None)

        # Validate required fields
        if not download_url:
            return jsonify({"error": "Missing required field: downloadUrl"}), 400

        print(f"⬇️ Downloading: {file_name}")

        # Download image from S3
        resp = requests.get(download_url, timeout=15)
        if resp.status_code != 200:
            return jsonify({"error": f"Failed to download image (HTTP {resp.status_code})"}), 400

        # Load image directly into memory (no temp file needed)
        image = Image.open(BytesIO(resp.content)).convert("RGB")

        # Run YOLO inference (YOLO accepts PIL Images directly)
        print(f"🔍 Running inference: {file_name}")
        results = model(image)
        detections = results[0].boxes

        # Parse detection results
        label = "normal"
        confidence = 0.0
        
        if len(detections) > 0:
            labels = [model.names[int(cls)] for cls in detections.cls]
            confs = detections.conf.tolist()

            if "pothole" in labels:
                pothole_idx = labels.index("pothole")
                label = "pothole"
                confidence = round(confs[pothole_idx], 3)

        # Prepare response (format required by Spring Boot backend)
        response_data = {
            "status": "success",
            "fileName": file_name,
            "prediction": label,
            "confidence": confidence,
            "userId": user_id
        }

        print(f"✅ [{file_name}] => {label.upper()} (confidence: {confidence})")
        return jsonify(response_data), 200

    except Exception as e:
        print("❌ Error:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "YOLOv10 Pothole Detection API is running"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)