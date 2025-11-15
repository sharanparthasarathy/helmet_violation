from flask import Flask, render_template, request, redirect, url_for
import torch
import os
from werkzeug.utils import secure_filename
import shutil

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PREDICTED_FOLDER'] = 'static/predicted'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PREDICTED_FOLDER'], exist_ok=True)

# Load YOLOv5 model (use your trained best.pt)
#model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
model = torch.hub.load('./yolov5', 'custom', path='best.pt', source='local', force_reload=True)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check file input
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        predicted_path = os.path.join(app.config['PREDICTED_FOLDER'], 'pred_' + filename)

        # Save uploaded file
        file.save(upload_path)

        # Run YOLOv5 inference
        results = model(upload_path)
        results.render()  # modifies results.imgs with boxes/labels

        # Save rendered image manually (avoid YOLO auto-folder creation)
        rendered_img = results.ims[0]  # numpy array (image)
        from PIL import Image
        Image.fromarray(rendered_img).save(predicted_path)

        return render_template(
            'result.html',
            input_image='uploads/' + filename,
            output_image='predicted/' + 'pred_' + filename
        )

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
