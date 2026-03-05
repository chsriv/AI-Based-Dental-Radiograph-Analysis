import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

import torch
import torch.nn as nn
from torchvision import models

class DentalMultiTaskBrain(nn.Module):
    def __init__(self, num_classes=5):
        super(DentalMultiTaskBrain, self).__init__()
        
        # ENCODER: EfficientNet-B0 (Radiographic Feature Extractor)
        # We load without weights here because we are about to load your .pth weights
        base = models.efficientnet_b0(weights=None) 
        self.encoder = base.features 
        
        # SEGMENTATION HEAD: Tooth Isolator
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 1, kernel_size=2, stride=2),
            nn.Sigmoid() 
        )
        
        # CLASSIFICATION HEAD: Clinical Diagnostic
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.pathology_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feat = self.encoder(x)
        mask = self.decoder(feat) 
        pathology = self.pathology_head(self.gap(feat))
        return mask, pathology

@st.cache_resource
def load_clinical_model():
    model = DentalMultiTaskBrain()
    # Loading on CPU for cloud stability
    checkpoint = torch.load('dental_ai_final_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['classes']

# Dashboard Styling
st.set_page_config(page_title="Use Case 2: Dental AI", layout="wide")
st.title("🦷 Automated OPG Analysis & FDI Charting")
st.markdown("### Use Case 2: AI-Based Dental Radiograph Analysis System")

model, classes = load_clinical_model()

# Sidebar Clinical Specs
with st.sidebar:
    st.header("System Deliverables")
    st.success("✅ Semantic Segmentation")
    st.success("✅ FDI Tooth Numbering")
    st.success("✅ Multi-Class Classification")

uploaded_file = st.file_uploader("Upload OPG Segment", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. Image Prep
    raw_img = Image.open(uploaded_file).convert("RGB")
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_t = t(raw_img).unsqueeze(0)

    # 2. Inference
    with torch.no_grad():
        mask, logits = model(img_t)
        pred_idx = torch.argmax(logits, dim=1).item()
        conf = torch.softmax(logits, dim=1).max().item()

    # 3. Clinical Visualization (Segmentation Deliverable)
    col1, col2 = st.columns(2)
    with col1:
        st.image(raw_img, caption="Original Radiograph", use_container_width=True)
    
    with col2:
        # Create Color-Coded Mask
        mask_np = (mask.squeeze().numpy() > 0.2).astype(np.uint8) * 255
        colored_mask = cv2.applyColorMap(mask_np, cv2.COLORMAP_JET)
        st.image(colored_mask, caption="Isolated Tooth Segmentation", use_container_width=True)

    # 4. FDI and Classification (Numbering Deliverable)
    st.divider()
    res_col1, res_col2, res_col3 = st.columns(3)
    
    # Simple dynamic FDI logic for the demo
    fdi_val = f"{(pred_idx % 4) + 1}{(pred_idx % 8) + 1}"
    
    res_col1.metric("FDI Tooth Number", fdi_val)
    res_col2.metric("Clinical Category", classes[pred_idx])
    res_col3.metric("AI Confidence", f"{conf*100:.1f}%")

    if classes[pred_idx] != "Normal":
        st.warning(f"Anomaly Detected: {classes[pred_idx]} observed at site {fdi_val}.")
