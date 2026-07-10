from flask import Flask, request, jsonify, render_template_string
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import base64
import io
app = Flask(__name__)
CLASS_NAMES = [
    "cane", "cavallo", "elefante", "farfalla",
    "gallina", "gatto", "mucca", "pecora",
    "ragno", "scoiattolo"]
CLASS_NAMES_VI = {
    "cane": "Chó",
    "cavallo": "Ngựa",
    "elefante": "Voi",
    "farfalla": "Bướm",
    "gallina": "Gà",
    "gatto": "Mèo",
    "mucca": "Bò",
    "pecora": "Cừu",
    "ragno": "Nhện",
    "scoiattolo": "Sóc"}
class MobileNetV3SmallClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights=None)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes))
    def forward(self, x):
        return self.backbone(x)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileNetV3SmallClassifier(num_classes=10).to(device)
import os
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "mobilenet_v3_small", "best_acc_model.pth")
if os.path.exists(MODEL_PATH):
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")
else:
    print(f"WARNING: Model not found at {MODEL_PATH}")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),])
HTML = """
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nhận diện thú cưng</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d3a;
    --border-hover: #4f46e5;
    --accent: #4f46e5;
    --accent-light: #6366f1;
    --text: #e8e9f0;
    --text-muted: #6b7280;
    --success: #10b981;
    --radius: 16px;}
  body {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 48px 16px;}
  header {
    text-align: center;
    margin-bottom: 40px;}
  header h1 {
    font-size: 28px;
    font-weight: 600;
    letter-spacing: -0.5px;}
  header p {
    color: var(--text-muted);
    font-size: 14px;
    margin-top: 8px;}
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 32px;
    width: 100%;
    max-width: 480px;
    display: flex;
    flex-direction: column;
    gap: 24px;}
  /* Drop zone */
  #drop-zone {
    border: 2px dashed var(--border);
    border-radius: 12px;
    height: 260px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    position: relative;
    overflow: hidden;}
  #drop-zone:hover,
  #drop-zone.drag-over {
    border-color: var(--border-hover);
    background: rgba(79, 70, 229, 0.05);}
  #drop-zone.has-image {
    border-style: solid;
    border-color: var(--border);}
  #preview {
    display: none;
    width: 100%;
    height: 100%;
    object-fit: contain;
    border-radius: 10px;}
  #preview.visible { display: block; }
  .drop-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    pointer-events: none;}
  .plus-icon {
    width: 48px;
    height: 48px;
    border: 2px solid var(--border);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    color: var(--text-muted);}
  .drop-placeholder p {
    font-size: 14px;
    color: var(--text-muted);
    text-align: center;
    line-height: 1.5;}
  .drop-placeholder span {
    font-size: 12px;
    color: var(--text-muted);
    opacity: 0.6;}
  #file-input { display: none; }
  /* Button */
  #predict-btn {
    width: 100%;
    padding: 14px;
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 10px;
    font-size: 15px;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
    cursor: pointer;
    transition: background 0.2s, transform 0.1s;}
  #predict-btn:hover { background: var(--accent-light); }
  #predict-btn:active { transform: scale(0.98); }
  #predict-btn:disabled {
    background: var(--border);
    color: var(--text-muted);
    cursor: not-allowed;
    transform: none;}
  /* Result */
  #result {
    display: none;
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 12px;
    padding: 20px;}
  #result.visible { display: block; }
  #result .label {
    font-size: 12px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;}
  #result .animal-name {
    font-size: 26px;
    font-weight: 600;
    color: var(--success);}
  #result .confidence-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 12px;}
  #result .conf-bar-bg {
    flex: 1;
    height: 6px;
    background: var(--border);
    border-radius: 999px;
    overflow: hidden;}
  #result .conf-bar-fill {
    height: 100%;
    background: var(--success);
    border-radius: 999px;
    transition: width 0.5s ease;}
  #result .conf-text {
    font-size: 13px;
    color: var(--text-muted);
    min-width: 44px;
    text-align: right;}
  .top3 {
    margin-top: 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;}
  .top3-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 13px;
    color: var(--text-muted);}
  .top3-item .bar-wrap {
    flex: 1;
    margin: 0 10px;
    height: 4px;
    background: var(--border);
    border-radius: 999px;
    overflow: hidden;}
  .top3-item .bar {
    height: 100%;
    background: var(--border-hover);
    border-radius: 999px;
    opacity: 0.5;}
  .spinner {
    display: inline-block;
    width: 16px;
    height: 16px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    vertical-align: middle;
    margin-right: 8px;}
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<header>
  <h1>Nhận diện thú cưng</h1>
  <p>MobileNetV3 Small — Animals-10 Dataset</p>
</header>
<div class="card">
  <div id="drop-zone" onclick="document.getElementById('file-input').click()">
    <img id="preview" alt="preview">
    <div class="drop-placeholder" id="placeholder">
      <div class="plus-icon">+</div>
      <p>Nhấn để chọn ảnh<br>hoặc kéo thả vào đây</p>
      <span>Hỗ trợ paste ảnh từ clipboard (Ctrl+V)</span>
    </div>
  </div>
  <input type="file" id="file-input" accept="image/*">
  <button id="predict-btn" disabled onclick="predict()">Dự đoán</button>
  <div id="result">
    <div class="label">Kết quả dự đoán</div>
    <div class="animal-name" id="animal-name"></div>
    <div class="confidence-row">
      <div class="conf-bar-bg">
        <div class="conf-bar-fill" id="conf-bar" style="width:0%"></div>
      </div>
      <div class="conf-text" id="conf-text"></div>
    </div>
    <div class="top3" id="top3"></div>
  </div>
</div>
<script>
  let imageBase64 = null;
  const dropZone = document.getElementById('drop-zone');
  const preview = document.getElementById('preview');
  const placeholder = document.getElementById('placeholder');
  const predictBtn = document.getElementById('predict-btn');
  function loadImage(file) {
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = e => {
      imageBase64 = e.target.result;
      preview.src = imageBase64;
      preview.classList.add('visible');
      placeholder.style.display = 'none';
      dropZone.classList.add('has-image');
      predictBtn.disabled = false;
      document.getElementById('result').classList.remove('visible');
    };
    reader.readAsDataURL(file);}
  document.getElementById('file-input').addEventListener('change', e => {
    loadImage(e.target.files[0]);});
  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');});
  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');});
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    loadImage(e.dataTransfer.files[0]);
  });  
  document.addEventListener('paste', e => {
    const items = e.clipboardData.items;
    for (let item of items) {
      if (item.type.startsWith('image/')) {
        loadImage(item.getAsFile());
        break;}}});
  async function predict() {
    if (!imageBase64) return;
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<span class="spinner"></span>Đang dự đoán...';
    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageBase64 })
      });
      const data = await res.json();
      if (data.error) {
        alert('Lỗi: ' + data.error);
        return;
      }
      document.getElementById('animal-name').textContent = data.label_vi;
      document.getElementById('conf-bar').style.width = (data.confidence * 100).toFixed(1) + '%';
      document.getElementById('conf-text').textContent = (data.confidence * 100).toFixed(1) + '%';
      const top3El = document.getElementById('top3');
      top3El.innerHTML = data.top3.map(item => `
        <div class="top3-item">
          <span>${item.label_vi}</span>
          <div class="bar-wrap">
            <div class="bar" style="width:${(item.confidence*100).toFixed(1)}%"></div>
          </div>
          <span>${(item.confidence*100).toFixed(1)}%</span>
        </div>
      `).join('');
      document.getElementById('result').classList.add('visible');
    } catch(e) {
      alert('Không thể kết nối server.');
    } finally {
      predictBtn.disabled = false;
      predictBtn.innerHTML = 'Dự đoán';
    }
  }
</script>
</body>
</html>
"""
@app.route("/")
def index():
    return render_template_string(HTML)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_data = data["image"]
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
        top1_idx = probs.argmax().item()
        top1_conf = probs[top1_idx].item()
        top1_label = CLASS_NAMES[top1_idx]
        top3_indices = probs.topk(3).indices.tolist()
        top3 = [
            {
                "label": CLASS_NAMES[i],
                "label_vi": CLASS_NAMES_VI[CLASS_NAMES[i]],
                "confidence": probs[i].item()
            }
            for i in top3_indices]
        return jsonify({
            "label": top1_label,
            "label_vi": CLASS_NAMES_VI[top1_label],
            "confidence": top1_conf,
            "top3": top3
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(debug=True, port=5000)