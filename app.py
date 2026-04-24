import os
import io
import pickle
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request, render_template, jsonify
import timm
import base64

app = Flask(__name__)

# ─────────────────────────────────────────
# Model Definition (must match training)
# ─────────────────────────────────────────
class CattleBreedClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s.in21k_ft_in1k",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.5),
            nn.Linear(in_features, 1024),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# ─────────────────────────────────────────
# Load PKL
# ─────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class _CPUUnpickler(pickle.Unpickler):
    """Remaps any CUDA storage to CPU transparently."""
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(
                io.BytesIO(b),
                map_location="cpu",
                weights_only=False
            )
        return super().find_class(module, name)

with open("cattle_classifier.pkl", "rb") as f:
    checkpoint = _CPUUnpickler(f).load()

NUM_CLASSES  = checkpoint["num_classes"]
IDX_TO_CLASS = checkpoint["idx_to_class"]

model = CattleBreedClassifier(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ─────────────────────────────────────────
# Preprocessing (320px — matches training)
# ─────────────────────────────────────────
from torchvision import transforms

TRANSFORM = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# If top prediction below this %, show "Unpredictable"
CONFIDENCE_THRESHOLD = 20.0

def predict(img: Image.Image, topk: int = 5):
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    top_probs, top_idx = torch.topk(probs, topk)
    results = [
        {"breed": IDX_TO_CLASS[i.item()], "prob": round(p.item() * 100, 2)}
        for p, i in zip(top_probs, top_idx)
    ]
    uncertain = results[0]["prob"] < CONFIDENCE_THRESHOLD
    return results, uncertain


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Cannot read image: {e}"}), 400

    b64 = base64.b64encode(img_bytes).decode("utf-8")
    mime = file.content_type or "image/jpeg"
    img_data_url = f"data:{mime};base64,{b64}"

    predictions, uncertain = predict(img, topk=5)

    top_breed = "Unpredictable" if uncertain else predictions[0]["breed"]
    top_prob  = predictions[0]["prob"]

    return render_template(
        "result.html",
        predictions=predictions,
        image_data=img_data_url,
        top_breed=top_breed,
        top_prob=top_prob,
        uncertain=uncertain,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for programmatic use."""
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400
    file  = request.files["image"]
    img   = Image.open(io.BytesIO(file.read())).convert("RGB")
    preds, uncertain = predict(img, topk=5)
    return jsonify({
        "predictions": preds,
        "uncertain": uncertain,
        "result": "Unpredictable" if uncertain else preds[0]["breed"]
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)