from flask import Flask, request, jsonify
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import requests
import os
import tempfile
import traceback

app = Flask(__name__)

# === CONFIG ===
MODEL_PATH = "https://huggingface.co/PrateekGaurav7296/pothole-detection/resolve/main/best.pt"
model = YOLO(MODEL_PATH)  # ✅ Load model once

# === PREDICT ENDPOINT ===
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON payload:
    {
        "downloadUrl": "https://s3-bucket-name.s3.ap-south-1.amazonaws.com/image123.jpg",
        "fileName": "image123.jpg",
        "userId": "42"   # optional, for backend tracking
    }
    """
    try:
        data = request.get_json(force=True)
        download_url = data.get("downloadUrl")
        file_name = data.get("fileName", "unknown.jpg")
        user_id = data.get("userId", None)

        if not download_url:
            return jsonify({"error": "Missing required field: downloadUrl"}), 400

        print(f"⬇️ Downloading image: {download_url}")

        # === 1️⃣ Download image from S3 presigned URL ===
        resp = requests.get(download_url, timeout=15)
        if resp.status_code != 200:
            return jsonify({"error": f"Failed to download image (HTTP {resp.status_code})"}), 400

        # Save image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            temp_path = tmp.name
            image = Image.open(BytesIO(resp.content)).convert("RGB")
            image.save(temp_path)

        # === 2️⃣ Run YOLO inference ===
        print(f"🔍 Running inference on {file_name} ...")
        results = model(temp_path)
        detections = results[0].boxes

        # === 3️⃣ Interpret results ===
        label, confidence = "normal", 0.0
        if len(detections) > 0:
            labels = [model.names[int(cls)] for cls in detections.cls]
            confs = detections.conf.tolist()

            if "pothole" in labels:
                pothole_idx = labels.index("pothole")
                label = "pothole"
                confidence = round(confs[pothole_idx], 3)

        # === 4️⃣ Save annotated image (optional local) ===
        output_dir = "predictions"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, file_name.replace(".jpg", "_pred.jpg"))
        results[0].save(filename=output_file)

        # === 5️⃣ Clean up temp file ===
        os.remove(temp_path)

        # === 6️⃣ Prepare final response ===
        response_data = {
            "status": "success",
            "fileName": file_name,
            "prediction": label,
            "confidence": confidence,
            "userId": user_id,
            "annotated_image_path": f"/{output_file}",  # local path for now
        }

        print(f"✅ [{file_name}] => {label.upper()} ({confidence})")
        return jsonify(response_data), 200

    except Exception as e:
        print("❌ Exception:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "YOLOv10 Pothole Detection API is running"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)