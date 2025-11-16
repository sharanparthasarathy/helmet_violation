from flask import Flask, render_template, request, redirect, url_for
import torch
import os
from werkzeug.utils import secure_filename
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['PREDICTED_FOLDER'] = os.path.join(BASE_DIR, 'static', 'predicted')

# Make sure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PREDICTED_FOLDER'], exist_ok=True)

# Load YOLOv5 model
model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path=os.path.join(BASE_DIR, 'best.pt'),
    source='github'
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    filename = secure_filename(file.filename)

    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    predicted_path = os.path.join(app.config['PREDICTED_FOLDER'],
                                  'pred_' + filename)

    # Save uploaded image
    file.save(upload_path)

    # YOLOv5 inference
    results = model(upload_path)
    results.render()

    # Save rendered result
    rendered_img = results.imgs[0]
    Image.fromarray(rendered_img).save(predicted_path)

    return render_template(
        'result.html',
        input_image='uploads/' + filename,
        output_image='predicted/' + 'pred_' + filename
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
