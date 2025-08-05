import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# Paths
MODEL_PATH = "crop_disease_model.pth"
TEST_DIR = "test"  # directory containing test images

# Class labels (23 classes)
classes = [
    'GUAVA_anthracnose', 'GUAVA_healthy', 'Healthy corn', 'INFECTED_CORN',
    'INFECTED_Pomo', 'INFECTED_STRAWBERRY', 'MANGO_Anthracnose',
    'MANGO_Bacterial Canker', 'MANGO_Cutting Weevil', 'MANGO_Die Back',
    'MANGO_Gall Midge', 'MANGO_Healthy', 'MANGO_Powdery Mildew',
    'MANGO_Sooty Mould', 'RICE_blast', 'RICE_blight', 'RICE_tungro',
    'SUGARCANE_Healthy', 'SUGARCANE_Mosaic', 'SUGARCANE_RedRot',
    'SUGARCANE_Rust', 'SUGARCANE_Yellow', 'rottenpomo'
]

# Preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Build model architecture
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, len(classes))

# Load trained weights
torch.manual_seed(0)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# Run inference on all images in TEST_DIR
print(f"[INFO] Predicting on images in '{TEST_DIR}' directory...")
for img_name in os.listdir(TEST_DIR):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    img_path = os.path.join(TEST_DIR, img_name)
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error loading {img_name}: {e}")
        continue

    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        label = classes[pred.item()]
    print(f"📸 {img_name} → Predicted: {label}")
