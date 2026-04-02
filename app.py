import os
import io
import base64
import numpy as np
from datetime import datetime
from flask import Flask, request, render_template_string, session, redirect, url_for, make_response
from PIL import Image
import onnxruntime as ort

app = Flask(__name__)
app.secret_key = "skincancer2024secretkey"

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "Model2_EffB4_No_meta.onnx")
IMG_SIZE    = 380   # EfficientNetB4 input size

print("Loading ONNX model (EfficientNet B4)...")
sess        = ort.InferenceSession(MODEL_PATH)
INPUT_NAME  = sess.get_inputs()[0].name
OUTPUT_NAME = sess.get_outputs()[0].name
print(f"Model loaded! Input: {INPUT_NAME} | Output: {OUTPUT_NAME}")

# 9 class labels — same order as model output
LABELS = [
    "Melanoma",
    "Melanocytic Nevus",
    "Basal Cell Carcinoma",
    "Actinic Keratosis",
    "Benign Keratosis",
    "Dermatofibroma",
    "Vascular Lesion",
    "Squamous Cell Carcinoma",
    "Unknown"
]

COLORS = [
    "#c0392b",   # Melanoma — red
    "#27ae60",   # Melanocytic Nevus — green
    "#8e44ad",   # Basal Cell Carcinoma — purple
    "#e67e22",   # Actinic Keratosis — orange
    "#2980b9",   # Benign Keratosis — blue
    "#16a085",   # Dermatofibroma — teal
    "#1abc9c",   # Vascular Lesion — cyan
    "#d35400",   # Squamous Cell Carcinoma — dark orange
    "#95a5a6",   # Unknown — gray
]

HIGH_RISK = {"Melanoma", "Basal Cell Carcinoma",
             "Squamous Cell Carcinoma", "Actinic Keratosis"}

RESULT_STORE = {}

# ── Preprocess image for EfficientNetB4 ──────────────────────────────────────
def preprocess(img):
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    # Normalize with ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std
    arr  = np.expand_dims(arr, axis=0)   # [1, 380, 380, 3]
    return arr.astype(np.float32)

# ── Softmax (model output may be logits) ─────────────────────────────────────
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Raleway:wght@300;400;500;600;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Raleway','Segoe UI',sans-serif;background:#f0f0ee;min-height:100vh}
nav{background:white;padding:18px 48px;display:flex;align-items:center;justify-content:space-between;box-shadow:0 1px 4px rgba(0,0,0,0.08)}
nav .brand{font-size:15px;font-weight:700;color:#e07b54;letter-spacing:1px}
nav ul{list-style:none;display:flex;gap:32px}
nav ul li a{text-decoration:none;font-size:12px;font-weight:600;color:#666;letter-spacing:1px;transition:color 0.2s}
nav ul li a:hover,nav ul li a.active{color:#e07b54}
.hero{background:linear-gradient(135deg,#2c2c2c 0%,#3d3d3d 100%);color:white;padding:80px 48px;text-align:center}
.hero h1{font-size:34px;font-weight:300;letter-spacing:4px;margin-bottom:14px}
.hero p{font-size:14px;color:#bbb;max-width:520px;margin:0 auto 32px;line-height:1.9}
.hero-btn{display:inline-block;padding:14px 40px;background:#e07b54;color:white;border-radius:4px;text-decoration:none;font-size:12px;font-weight:700;letter-spacing:2px}
.container{max-width:920px;margin:40px auto;padding:0 24px}
.card{background:white;border-radius:4px;box-shadow:0 2px 16px rgba(0,0,0,0.07);margin-bottom:28px;overflow:hidden}
.card-header{background:#f7f7f7;padding:18px 32px;border-bottom:1px solid #eee}
.card-header h2{font-size:13px;font-weight:700;color:#555;letter-spacing:1.5px;text-transform:uppercase}
.card-body{padding:32px}
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.fg{display:flex;flex-direction:column;gap:7px}
.fg.full{grid-column:1/-1}
.fg label{font-size:11px;font-weight:700;color:#999;letter-spacing:1px;text-transform:uppercase}
.fg input,.fg select{padding:11px 14px;border:1px solid #ddd;border-radius:4px;font-size:14px;font-family:inherit;color:#333;outline:none;transition:border-color 0.2s;background:#fafafa}
.fg input:focus,.fg select:focus{border-color:#e07b54;background:white}
.btn-primary{padding:13px 36px;background:#e07b54;color:white;border:none;border-radius:4px;font-size:12px;font-weight:700;letter-spacing:2px;cursor:pointer;margin-top:10px}
.btn-primary:hover{background:#c9663e}
.steps{display:flex;align-items:center;justify-content:center;gap:0;padding:28px 0 4px}
.si{display:flex;align-items:center;gap:8px;font-size:13px}
.sn{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px}
.si.active .sn{background:#e07b54;color:white}
.si.done .sn{background:#27ae60;color:white}
.si.inactive .sn{background:#ddd;color:#aaa}
.si.active span{color:#e07b54;font-weight:600}
.si.done span{color:#27ae60}
.si.inactive span{color:#bbb}
.sl{width:56px;height:2px;background:#ddd;margin:0 4px}
.sl.done{background:#27ae60}
.info-panel{background:#e8e8e6;display:grid;grid-template-columns:1fr 1fr;gap:0}
.panel-sec{padding:28px 32px}
.panel-sec:first-child{border-right:1px solid #ccc}
.panel-sec h3{font-size:12px;font-weight:700;color:#555;letter-spacing:1px;text-transform:uppercase;margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid #ccc}
.ir{display:flex;gap:8px;margin-bottom:10px;font-size:13px}
.ir .lb{color:#888;min-width:130px}
.ir .vl{color:#333;font-weight:500}
.upload-zone{border:2px dashed #ddd;border-radius:4px;padding:36px;text-align:center;cursor:pointer;transition:border-color 0.2s;background:#fafafa}
.upload-zone:hover{border-color:#e07b54}
.upload-zone input{display:none}
.upload-zone p{font-size:13px;color:#aaa;margin-top:8px}
.upload-zone p span{color:#e07b54;font-weight:600}
.pdf-btn{display:inline-block;padding:11px 28px;background:transparent;border:2px solid #e07b54;color:#e07b54;border-radius:4px;font-size:11px;font-weight:700;letter-spacing:2px;cursor:pointer;text-decoration:none;transition:all 0.2s;margin-top:14px}
.pdf-btn:hover{background:#e07b54;color:white}
.diag-box{background:#f9f9f9;border-left:4px solid #e07b54;padding:18px 24px;border-radius:0 4px 4px 0;margin:0 32px 24px}
.diag-box .dl{font-size:11px;color:#aaa;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px}
.diag-box .dv{font-size:22px;font-weight:700;color:#e07b54}
.diag-box .dc{font-size:12px;color:#888;margin-top:4px}
.disclaimer{background:#fff8f0;border:1px solid #f0d0b0;border-radius:4px;padding:14px 22px;margin:0 32px 28px;font-size:12px;color:#a0522d;line-height:1.7}
.nav-btns{display:flex;gap:14px;margin:0 32px 32px}
.nb{padding:11px 24px;border-radius:4px;font-size:11px;font-weight:700;letter-spacing:1px;cursor:pointer;text-decoration:none;display:inline-block;text-align:center}
.nb-out{border:2px solid #ccc;color:#666;background:white}
.nb-out:hover{border-color:#e07b54;color:#e07b54}
.nb-fill{background:#e07b54;color:white;border:2px solid #e07b54}
.nb-fill:hover{background:#c9663e}
.err{background:#fce4ec;border:1px solid #f48fb1;border-radius:4px;padding:11px 18px;color:#c62828;font-size:13px;margin-bottom:18px}
.chart-area{padding:28px 32px}
canvas{max-height:320px}
"""

NAVBAR = """<nav>
  <div class="brand">SKIN CANCER CLASSIFICATION</div>
  <ul>
    <li><a href="/">HOME</a></li>
    <li><a href="#">INFO</a></li>
    <li><a href="#">TOOLS</a></li>
    <li><a href="/upload" class="active">OUR SOLUTION</a></li>
    <li><a href="#">ABOUT US</a></li>
  </ul>
</nav>"""

HOME_HTML = """<!DOCTYPE html><html><head><title>Skin Cancer Classification</title>
<style>"""+CSS+"""</style></head><body>"""+NAVBAR+"""
<div class="hero">
  <h1>SKIN CANCER DETECTION</h1>
  <p>AI-powered skin lesion classification using EfficientNet B4 Deep Learning model.<br>
     Upload your dermoscopy image for instant 9-class analysis with 91% accuracy.</p>
  <a href="#form" class="hero-btn">GET STARTED</a>
</div>
<div class="container" id="form">
  <div class="steps">
    <div class="si active"><div class="sn">1</div><span>Patient Details</span></div>
    <div class="sl"></div>
    <div class="si inactive"><div class="sn">2</div><span>Upload Image</span></div>
    <div class="sl"></div>
    <div class="si inactive"><div class="sn">3</div><span>Results</span></div>
  </div>
  <div class="card">
    <div class="card-header"><h2>Patient Information</h2></div>
    <div class="card-body">
      {% if error %}<div class="err">{{ error }}</div>{% endif %}
      <form method="POST" action="/details">
        <div class="form-grid">
          <div class="fg"><label>Patient ID</label>
            <input type="text" name="patient_id" placeholder="e.g. ISIC 2029" value="{{ f.patient_id }}"></div>
          <div class="fg"><label>Full Name *</label>
            <input type="text" name="name" placeholder="e.g. John Doe" value="{{ f.name }}" required></div>
          <div class="fg"><label>Age *</label>
            <input type="number" name="age" placeholder="e.g. 34" min="1" max="120" value="{{ f.age }}" required></div>
          <div class="fg"><label>Gender *</label>
            <select name="gender" required>
              <option value="">Select</option>
              <option value="Male" {% if f.gender=='Male' %}selected{% endif %}>Male</option>
              <option value="Female" {% if f.gender=='Female' %}selected{% endif %}>Female</option>
              <option value="Other" {% if f.gender=='Other' %}selected{% endif %}>Other</option>
            </select></div>
          <div class="fg"><label>Anatomical Site *</label>
            <select name="site" required>
              <option value="">Select body area</option>
              <option value="Head/Neck">Head / Neck</option>
              <option value="Upper Extremity">Upper Extremity</option>
              <option value="Lower Extremity">Lower Extremity</option>
              <option value="Torso">Torso</option>
              <option value="Back">Back</option>
              <option value="Palms/Soles">Palms / Soles</option>
              <option value="Other">Other</option>
            </select></div>
          <div class="fg"><label>Contact Number</label>
            <input type="text" name="phone" placeholder="e.g. 9876543210" value="{{ f.phone }}"></div>
          <div class="fg full"><label>Medical History</label>
            <input type="text" name="history" placeholder="e.g. Eczema, None" value="{{ f.history }}"></div>
        </div>
        <button type="submit" class="btn-primary">CONTINUE TO IMAGE UPLOAD →</button>
      </form>
    </div>
  </div>
</div></body></html>"""

UPLOAD_HTML = """<!DOCTYPE html><html><head><title>Upload — Skin Cancer Classification</title>
<style>"""+CSS+"""</style></head><body>"""+NAVBAR+"""
<div class="container">
  <div class="steps">
    <div class="si done"><div class="sn">✓</div><span>Patient Details</span></div>
    <div class="sl done"></div>
    <div class="si active"><div class="sn">2</div><span>Upload Image</span></div>
    <div class="sl"></div>
    <div class="si inactive"><div class="sn">3</div><span>Results</span></div>
  </div>
  <div class="card">
    <div class="card-header"><h2>Patient Detail</h2></div>
    <div class="info-panel">
      <div class="panel-sec">
        <h3>Personal Information</h3>
        <div class="ir"><span class="lb">Patient ID:</span><span class="vl">{{ p.patient_id or 'N/A' }}</span></div>
        <div class="ir"><span class="lb">Name:</span><span class="vl">{{ p.name }}</span></div>
        <div class="ir"><span class="lb">Age:</span><span class="vl">{{ p.age }}</span></div>
        <div class="ir"><span class="lb">Gender:</span><span class="vl">{{ p.gender }}</span></div>
        <div class="ir"><span class="lb">Anatomical Site:</span><span class="vl">{{ p.site }}</span></div>
        <div class="ir"><span class="lb">Medical History:</span><span class="vl">{{ p.history or 'None' }}</span></div>
      </div>
      <div class="panel-sec">
        <h3>Upload Skin Lesion Image</h3>
        <form method="POST" action="/predict" enctype="multipart/form-data">
          <div class="upload-zone" onclick="document.getElementById('imgf').click()">
            <input type="file" id="imgf" name="file" accept="image/*" onchange="prev(event)" required>
            <div id="uicon" style="font-size:40px;color:#ddd">📷</div>
            <p id="utxt">Click to <span>choose image</span> &nbsp;·&nbsp; JPG, PNG</p>
            <img id="pimg" style="display:none;max-height:140px;border-radius:4px;margin-top:10px">
          </div>
          <div style="display:flex;gap:12px;margin-top:16px">
            <a href="/" style="flex:1;padding:12px;text-align:center;border:2px solid #ccc;color:#666;border-radius:4px;font-size:11px;font-weight:700;letter-spacing:1px;text-decoration:none">← BACK</a>
            <button type="submit" class="btn-primary" style="flex:2;margin:0">ANALYSE IMAGE →</button>
          </div>
        </form>
      </div>
    </div>
  </div>
</div>
<script>
function prev(e){const f=e.target.files[0];if(!f)return;const r=new FileReader();
r.onload=ev=>{const i=document.getElementById('pimg');i.src=ev.target.result;i.style.display='block';
document.getElementById('uicon').style.display='none';
document.getElementById('utxt').style.display='none';};r.readAsDataURL(f);}
</script></body></html>"""

RESULT_HTML = """<!DOCTYPE html><html><head><title>Results — Skin Cancer Classification</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>"""+CSS+"""</style></head><body>"""+NAVBAR+"""
<div class="container">
  <div class="steps">
    <div class="si done"><div class="sn">✓</div><span>Patient Details</span></div>
    <div class="sl done"></div>
    <div class="si done"><div class="sn">✓</div><span>Upload Image</span></div>
    <div class="sl done"></div>
    <div class="si active"><div class="sn">3</div><span>Results</span></div>
  </div>
  <div class="card">
    <div class="card-header"><h2>Patient Detail</h2></div>
    <div class="info-panel">
      <div class="panel-sec">
        <h3>Personal Information</h3>
        <div class="ir"><span class="lb">Patient ID:</span><span class="vl">{{ p.patient_id or 'N/A' }}</span></div>
        <div class="ir"><span class="lb">Name:</span><span class="vl">{{ p.name }}</span></div>
        <div class="ir"><span class="lb">Age:</span><span class="vl">{{ p.age }}</span></div>
        <div class="ir"><span class="lb">Gender:</span><span class="vl">{{ p.gender }}</span></div>
        <div class="ir"><span class="lb">Anatomical Site:</span><span class="vl">{{ p.site }}</span></div>
        <div class="ir"><span class="lb">Medical History:</span><span class="vl">{{ p.history or 'None' }}</span></div>
        <div class="ir"><span class="lb">Report Date:</span><span class="vl">{{ date }}</span></div>
      </div>
      <div class="panel-sec" style="display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center">
        <h3 style="align-self:stretch">Download Result</h3>
        <img src="{{ preview }}" style="max-height:130px;max-width:100%;margin:12px 0;border-radius:4px;border:1px solid #ddd">
        <p style="font-size:11px;color:#aaa;margin-bottom:12px">Click to download your full analysis report</p>
        <a href="/download_pdf" class="pdf-btn">⬇ GENERATE PDF</a>
      </div>
    </div>
  </div>
  <div class="card">
    <div class="card-header"><h2>Model Prediction — EfficientNet B4 (91% Accuracy)</h2></div>
    <div class="chart-area">
      <canvas id="chart"></canvas>
    </div>
    <div class="diag-box">
      <div class="dl">Primary Diagnosis</div>
      <div class="dv">{{ top_label }}</div>
      <div class="dc">
        Confidence: <b>{{ top_conf }}%</b> &nbsp;·&nbsp;
        Risk Level: <b style="color:{{ '#c0392b' if risk=='HIGH' else '#27ae60' }}">{{ risk }}</b>
        &nbsp;·&nbsp; {{ date }}
      </div>
    </div>
    <div class="disclaimer">
      ⚠ This AI tool is for educational and screening purposes only. It is NOT a substitute
      for professional medical diagnosis. Always consult a qualified dermatologist for proper diagnosis and treatment.
    </div>
    <div class="nav-btns">
      <a href="/upload" class="nb nb-out">← CHECK ANOTHER IMAGE</a>
      <a href="/" class="nb nb-fill">NEW PATIENT →</a>
    </div>
  </div>
</div>
<script>
new Chart(document.getElementById('chart').getContext('2d'),{
  type:'bar',
  data:{
    labels:{{ labels|tojson }},
    datasets:[{
      data:{{ values|tojson }},
      backgroundColor:{{ colors|tojson }},
      borderRadius:4,
      borderSkipped:false
    }]
  },
  options:{
    responsive:true,
    plugins:{
      legend:{display:false},
      tooltip:{callbacks:{label:c=>` Probability: ${(c.raw*100).toFixed(1)}%`}}
    },
    scales:{
      y:{beginAtZero:true,max:1.0,
         ticks:{callback:v=>v.toFixed(1)},
         grid:{color:'#f5f5f5'},
         title:{display:true,text:'Probability',font:{size:12}}},
      x:{grid:{display:false},ticks:{font:{size:11},maxRotation:15}}
    }
  }
});
</script></body></html>"""


@app.route("/", methods=["GET"])
def index():
    e = dict(patient_id="",name="",age="",gender="",site="",phone="",history="")
    return render_template_string(HOME_HTML, error=None, f=e)


@app.route("/details", methods=["POST"])
def details():
    f = request.form
    if not all([f.get("name"), f.get("age"), f.get("gender"), f.get("site")]):
        return render_template_string(HOME_HTML,
            error="Please fill all required fields marked with *", f=f)
    session["patient"] = dict(
        patient_id=f.get("patient_id",""),
        name=f["name"].strip(),
        age=f["age"],
        gender=f["gender"],
        site=f["site"],
        phone=f.get("phone",""),
        history=f.get("history","None") or "None"
    )
    return redirect(url_for("upload"))


@app.route("/upload", methods=["GET"])
def upload():
    p = session.get("patient")
    if not p: return redirect(url_for("index"))
    return render_template_string(UPLOAD_HTML, p=p)


@app.route("/predict", methods=["POST"])
def predict():
    p = session.get("patient")
    if not p: return redirect(url_for("index"))

    file = request.files.get("file")
    if not file or not file.filename:
        return redirect(url_for("upload"))

    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Thumbnail preview
    thumb = img.copy()
    thumb.thumbnail((300, 300))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=60)
    preview = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    # Run ONNX inference
    arr    = preprocess(img)
    output = sess.run([OUTPUT_NAME], {INPUT_NAME: arr})[0][0]  # shape: (9,)
    probs  = softmax(output)                                    # convert to probabilities
    values = [round(float(p), 4) for p in probs]

    top_idx   = int(np.argmax(probs))
    top_label = LABELS[top_idx]
    top_conf  = round(float(probs[top_idx]) * 100, 1)
    risk      = "HIGH" if top_label in HIGH_RISK else "LOW"
    date_str  = datetime.now().strftime("%d %b %Y, %I:%M %p")

    # Store for PDF
    pid = p.get("patient_id") or p.get("name", "patient")
    RESULT_STORE[pid] = dict(
        top_label=top_label, top_conf=top_conf,
        risk=risk, probs=values, date=date_str
    )
    session["last_pid"] = pid

    return render_template_string(RESULT_HTML,
        p=p, preview=preview,
        top_label=top_label, top_conf=top_conf,
        risk=risk, date=date_str,
        labels=LABELS, values=values, colors=COLORS
    )


@app.route("/download_pdf")
def download_pdf():
    p     = session.get("patient", {})
    pid   = session.get("last_pid", "")
    r     = RESULT_STORE.get(pid, {})
    date  = datetime.now().strftime("%d %B %Y %I:%M %p")
    probs = r.get("probs", [0]*9)

    rows = "".join(
        f"<tr>"
        f"<td style='padding:8px 14px;border-bottom:1px solid #eee;color:#555;font-size:13px'>{LABELS[i]}</td>"
        f"<td style='padding:8px 14px;border-bottom:1px solid #eee;font-size:13px;font-weight:600;color:#333'>{round(probs[i]*100,1)}%</td>"
        f"<td style='padding:8px 14px;border-bottom:1px solid #eee;width:200px'>"
        f"<div style='background:{COLORS[i]};height:10px;border-radius:3px;width:{min(round(probs[i]*100,1),100)}%'></div></td>"
        f"</tr>"
        for i in range(9)
    )

    risk_color = "#c0392b" if r.get("risk") == "HIGH" else "#27ae60"

    html = f"""<!DOCTYPE html><html><head>
    <style>
    body{{font-family:Arial,sans-serif;padding:48px;color:#333;max-width:740px;margin:0 auto}}
    h1{{color:#e07b54;font-size:22px;margin-bottom:4px;letter-spacing:1px}}
    .sub{{font-size:12px;color:#aaa;margin-bottom:28px;border-bottom:1px solid #eee;padding-bottom:16px}}
    h2{{font-size:11px;font-weight:700;color:#aaa;letter-spacing:2px;text-transform:uppercase;margin:24px 0 12px;padding-bottom:8px;border-bottom:1px solid #f0f0f0}}
    .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:8px}}
    .fi .lb{{font-size:11px;color:#aaa;margin-bottom:2px}}
    .fi .vl{{font-size:13px;font-weight:600;color:#333}}
    .result-box{{background:#fff3ee;border-left:4px solid #e07b54;padding:18px 24px;border-radius:0 4px 4px 0;margin:16px 0 24px}}
    .result-box .rv{{font-size:26px;font-weight:700;color:#e07b54;margin-bottom:6px}}
    .result-box .rc{{font-size:13px;color:#888}}
    table{{width:100%;border-collapse:collapse;margin-bottom:20px}}
    th{{padding:10px 14px;font-size:11px;color:#aaa;font-weight:700;text-transform:uppercase;text-align:left;background:#f9f9f9}}
    .disc{{background:#f9f9f9;border:1px solid #eee;padding:14px 18px;font-size:11px;color:#999;line-height:1.8;border-radius:4px}}
    </style></head><body>
    <h1>SKIN CANCER DETECTION — ANALYSIS REPORT</h1>
    <div class="sub">Generated: {date} &nbsp;·&nbsp; Model: EfficientNet B4 &nbsp;·&nbsp; Accuracy: 91%</div>

    <h2>Patient Information</h2>
    <div class="grid">
      <div class="fi"><div class="lb">Patient ID</div><div class="vl">{p.get('patient_id','N/A') or 'N/A'}</div></div>
      <div class="fi"><div class="lb">Full Name</div><div class="vl">{p.get('name','N/A')}</div></div>
      <div class="fi"><div class="lb">Age</div><div class="vl">{p.get('age','N/A')}</div></div>
      <div class="fi"><div class="lb">Gender</div><div class="vl">{p.get('gender','N/A')}</div></div>
      <div class="fi"><div class="lb">Anatomical Site</div><div class="vl">{p.get('site','N/A')}</div></div>
      <div class="fi"><div class="lb">Medical History</div><div class="vl">{p.get('history','None')}</div></div>
    </div>

    <h2>Diagnosis Result</h2>
    <div class="result-box">
      <div class="rv">{r.get('top_label','N/A')}</div>
      <div class="rc">
        Confidence: <b>{r.get('top_conf','N/A')}%</b> &nbsp;·&nbsp;
        Risk Level: <b style="color:{risk_color}">{r.get('risk','N/A')}</b> &nbsp;·&nbsp;
        {r.get('date', date)}
      </div>
    </div>

    <h2>Probability Distribution — All 9 Cancer Types</h2>
    <table>
      <tr><th>Diagnosis Type</th><th>Probability</th><th>Distribution</th></tr>
      {rows}
    </table>

    <div class="disc">
      ⚠ Disclaimer: This report is generated by an AI-powered tool (EfficientNet B4, 91% accuracy) for
      educational and screening purposes only. It is NOT a substitute for professional medical diagnosis
      or treatment. Please consult a certified dermatologist for accurate diagnosis and appropriate care.
    </div>
    </body></html>"""

    name_safe = p.get("name", "patient").replace(" ", "_")
    resp = make_response(html)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    resp.headers["Content-Disposition"] = f'attachment; filename=SkinCancer_Report_{name_safe}.html'
    return resp


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Skin Cancer Classification — EfficientNet B4")
    print("  Model Accuracy: 91% | Classes: 9")
    print("  Open browser: http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=False, port=5000)