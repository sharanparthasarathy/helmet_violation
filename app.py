from flask import Flask, render_template, request, redirect
import os
import sys
from werkzeug.utils import secure_filename
from PIL import Image
import torch

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['PREDICTED_FOLDER'] = os.path.join(BASE_DIR, 'static', 'predicted')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PREDICTED_FOLDER'], exist_ok=True)

# --- IMPORT YOLOv5 FROM LOCAL FOLDER ---
sys.path.append(os.path.join(BASE_DIR, "yolov5"))
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.augmentations import letterbox

# Load model
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
model = DetectMultiBackend(MODEL_PATH, device="cpu")
stride = model.stride
names = model.names

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file', None)
    if not file or file.filename == "":
        return redirect("/")

    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pred_path = os.path.join(app.config['PREDICTED_FOLDER'], "pred_" + filename)

    file.save(upload_path)

    # --- Inference ---
    img = Image.open(upload_path).convert('RGB')
    img = letterbox(img, 640, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float().unsqueeze(0) / 255.0

    pred = model(img)
    pred = non_max_suppression(pred, 0.25, 0.45)[0]

    # Draw boxes
    rendered = model.model.draw(img[0], pred, names)
    Image.fromarray(rendered).save(pred_path)

    return render_template(
        "result.html",
        input_image="uploads/" + filename,
        output_image="predicted/" + "pred_" + filename
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
