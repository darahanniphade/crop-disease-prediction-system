from flask import Flask, render_template, request
import torch
from torchvision import transforms, models
from PIL import Image
import os

# Flask app
app = Flask(__name__)

# Define model
model = models.mobilenet_v2(weights=None)  # No warning
model.classifier[1] = torch.nn.Linear(1280, 23)  # ✅ Match trained model's output

# Load weights
model.load_state_dict(torch.load("crop_disease_model.pth", map_location=torch.device('cpu')))
model.eval()

# Class labels — 🔁 update with your actual 23 class names
classes = [
    'GUAVA_anthracnose', 'GUAVA_healthy', 'Healthy corn', 'INFECTED_CORN',
    'INFECTED_Pomo', 'INFECTED_STRAWBERRY', 'MANGO_Anthracnose', 'MANGO_Bacterial Canker',
    'MANGO_Cutting Weevil', 'MANGO_Die Back', 'MANGO_Gall Midge', 'MANGO_Healthy',
    'MANGO_Powdery Mildew', 'MANGO_Sooty Mould', 'RICE_blast', 'RICE_blight',
    'RICE_tungro', 'rottenpomo', 'SUGARCANE_Healthy', 'SUGARCANE_Mosaic',
    'SUGARCANE_RedRot', 'SUGARCANE_Rust', 'SUGARCANE_Yellow'
]


# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image file", 400

    file = request.files['image']
    if file.filename == '':
        return "Empty filename", 400

    image = Image.open(file.stream).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        label = classes[predicted.item()]

    return render_template('index.html', prediction=label)

if __name__ == '__main__':
    app.run(debug=True)
