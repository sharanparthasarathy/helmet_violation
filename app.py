import os
import sys
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "static", "uploads")
app.config["PREDICTED_FOLDER"] = os.path.join(BASE_DIR, "static", "predicted")

# Ensure folders
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["PREDICTED_FOLDER"], exist_ok=True)

# ---- YOLOv5 LOCAL IMPORT ----
YOLO_DIR = os.path.join(BASE_DIR, "yolov5")
sys.path.append(YOLO_DIR)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# Load model
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
model = DetectMultiBackend(MODEL_PATH, device="cpu")
stride = model.stride
names = model.names

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)

    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    output_path = os.path.join(app.config["PREDICTED_FOLDER"], "pred_" + filename)

    file.save(upload_path)

    # --- Preprocess image for YOLOv5 ---
    img0 = Image.open(upload_path).convert("RGB")
    img0 = np.array(img0)

    img = letterbox(img0, 640, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()
    img /= 255.0
    img = img.unsqueeze(0)

    # Inference
    pred = model(img)

    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45)[0]

    if pred is not None and len(pred):
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], img0.shape).round()

        # Draw results manually
        import cv2
        for *xyxy, conf, cls in pred:
            label = f"{names[int(cls)]} {conf:.2f}"
            cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])),
                          (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save output
    Image.fromarray(img0).save(output_path)

    return render_template(
        "result.html",
        input_image="uploads/" + filename,
        output_image="predicted/" + "pred_" + filename,
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
