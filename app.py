import os
import numpy as np
from flask import Flask, request, render_template_string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import io
import base64

app = Flask(__name__)

# ── Load model ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.h5")
IMG_SIZE   = 224

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# ── HTML Template ─────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Skin Cancer Detection</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: Arial, sans-serif; background: #f0f4f8; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
    .container { background: white; border-radius: 16px; padding: 40px; max-width: 520px; width: 90%; box-shadow: 0 4px 24px rgba(0,0,0,0.08); }
    h1 { font-size: 22px; color: #1a1a2e; margin-bottom: 6px; }
    .subtitle { font-size: 14px; color: #666; margin-bottom: 28px; }
    .upload-area { border: 2px dashed #c0c8d8; border-radius: 12px; padding: 32px; text-align: center; cursor: pointer; margin-bottom: 20px; transition: border-color 0.2s; }
    .upload-area:hover { border-color: #4a90d9; }
    .upload-area input { display: none; }
    .upload-label { font-size: 14px; color: #555; cursor: pointer; }
    .upload-label span { color: #4a90d9; font-weight: bold; }
    .preview { max-width: 100%; max-height: 260px; border-radius: 10px; margin: 16px auto; display: block; }
    .btn { width: 100%; padding: 14px; background: #4a90d9; color: white; border: none; border-radius: 10px; font-size: 16px; font-weight: bold; cursor: pointer; transition: background 0.2s; }
    .btn:hover { background: #357abd; }
    .result { margin-top: 24px; padding: 20px; border-radius: 12px; text-align: center; }
    .result.benign { background: #e8f5e9; border: 1px solid #81c784; }
    .result.malignant { background: #fce4ec; border: 1px solid #e57373; }
    .result-title { font-size: 22px; font-weight: bold; margin-bottom: 6px; }
    .result.benign .result-title { color: #2e7d32; }
    .result.malignant .result-title { color: #c62828; }
    .result-conf { font-size: 14px; color: #555; margin-bottom: 12px; }
    .result-note { font-size: 12px; color: #888; }
    .conf-bar-wrap { background: #e0e0e0; border-radius: 6px; height: 10px; margin: 10px 0; overflow: hidden; }
    .conf-bar { height: 100%; border-radius: 6px; transition: width 0.5s; }
    .benign .conf-bar { background: #4caf50; }
    .malignant .conf-bar { background: #e53935; }
    .warning { background: #fff3e0; border: 1px solid #ffb74d; border-radius: 10px; padding: 12px 16px; font-size: 12px; color: #e65100; margin-top: 16px; }
  </style>
</head>
<body>
<div class="container">
  <h1>Skin Cancer Detection</h1>
  <p class="subtitle">Upload a skin lesion image to check if it is benign or malignant.</p>

  <form method="POST" enctype="multipart/form-data">
    <div class="upload-area" onclick="document.getElementById('file').click()">
      <input type="file" id="file" name="file" accept="image/*" onchange="previewImage(event)">
      {% if preview %}
        <img src="{{ preview }}" class="preview" id="preview-img">
      {% else %}
        <div class="upload-label" id="upload-text">Click to <span>choose an image</span><br><br>Supports JPG, PNG</div>
        <img src="" class="preview" id="preview-img" style="display:none">
      {% endif %}
    </div>
    <button type="submit" class="btn">Analyse Image</button>
  </form>

  {% if result %}
  <div class="result {{ result.css }}">
    <div class="result-title">{{ result.label }}</div>
    <div class="result-conf">Confidence: {{ result.confidence }}%</div>
    <div class="conf-bar-wrap">
      <div class="conf-bar" style="width: {{ result.confidence }}%"></div>
    </div>
    <div class="result-note">{{ result.note }}</div>
  </div>
  <div class="warning">
    This tool is for educational purposes only. Always consult a qualified dermatologist for medical diagnosis.
  </div>
  {% endif %}
</div>

<script>
function previewImage(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(ev) {
    const img = document.getElementById('preview-img');
    const txt = document.getElementById('upload-text');
    img.src = ev.target.result;
    img.style.display = 'block';
    if (txt) txt.style.display = 'none';
  };
  reader.readAsDataURL(file);
}
</script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    result  = None
    preview = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            # Read and preprocess image
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Preview as base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            preview = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()

            # Predict
            img_resized = img.resize((IMG_SIZE, IMG_SIZE))
            arr = img_to_array(img_resized) / 255.0
            arr = np.expand_dims(arr, axis=0)
            pred = float(model.predict(arr, verbose=0)[0][0])

            if pred >= 0.5:
                label      = "MALIGNANT"
                css        = "malignant"
                confidence = round(pred * 100, 1)
                note       = "High risk detected. Please consult a dermatologist immediately."
            else:
                label      = "BENIGN"
                css        = "benign"
                confidence = round((1 - pred) * 100, 1)
                note       = "Low risk detected. Continue regular skin check-ups."

            result = {"label": label, "css": css, "confidence": confidence, "note": note}

    return render_template_string(HTML, result=result, preview=preview)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Skin Cancer Detection Web App")
    print("=" * 50)
    print("  Open your browser and go to:")
    print("  http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(debug=False, port=5000)