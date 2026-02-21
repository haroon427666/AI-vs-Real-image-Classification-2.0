import torch
import torch.nn as nn
from torchvision.models import convnext_tiny
from torchvision import transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

class AIDetectorConvNeXt(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()

        self.backbone = convnext_tiny(pretrained=False)
        in_features = self.backbone.classifier[2].in_features

        self.backbone.classifier = nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.avgpool(features)
        return self.classifier(features)


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model(model_path):
    model = AIDetectorConvNeXt()
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model

def predict_image(model, image):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return predicted.item(), confidence.item(), probs.cpu().numpy()[0]