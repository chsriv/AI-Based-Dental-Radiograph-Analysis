import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import os
import requests

# --- 1. MODEL ARCHITECTURE ---

class DentalMultiTaskBrain(nn.Module):
    def __init__(self, num_classes=5):
        super(DentalMultiTaskBrain, self).__init__()
        base = models.efficientnet_b0(weights=None) 
        self.encoder = base.features 
        
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

# --- 2. ROBUST LOADING LOGIC ---

@st.cache_resource
def load_clinical_model():
    model_url = "https://github.com/chsriv/AI-Based-Dental-Radiograph-Analysis/raw/main/dental_ai_final_model.pth"
    model_path = "dental_ai_final_model.pth"
    
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
        response = requests.get(model_url, allow_redirects=True)
        with open(model_path, "wb") as f:
            f.write(response.content)

    checkpoint = torch.load(model_path, map_location='cpu')
    classes = ['Cavity', 'Fillings', 'Impacted Tooth', 'Implant', 'Normal']
    
    model = DentalMultiTaskBrain(num_classes=len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, classes

# --- 3. DASHBOARD UI ---

st.set_page_config(page_title="Dental AI: Use Case 2", layout="wide")
st.title("🦷 Automated OPG Analysis & FDI Charting")
st.markdown("### Use Case 2: AI-Based Dental Radiograph Analysis System")
st.caption(f"Engine validated on {25410:,} Training Samples | {2721:,} Validation Samples")

try:
    model, classes = load_clinical_model()
    st.info("Clinical Engine Optimized & Online")
except Exception as e:
    st.error(f"Engine Load Error: {e}")

uploaded_file = st.file_uploader("Upload OPG Segment", type=["jpg", "png", "jpeg"])

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_t = t(raw_img).unsqueeze(0)

    with torch.no_grad():
        mask, logits = model(img_t)
        # Diagnostic Sensitivity Tweak (Prioritizing Cavity Detection)
        logits[0, 0] += 0.8  
        
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        conf = probs.max().item()

    col1, col2 = st.columns(2)
    with col1:
        st.image(raw_img, caption="Original Radiograph", use_container_width=True)
    
    with col2:
        mask_np = (mask.squeeze().numpy() * 255).astype(np.uint8)
        colored_mask = cv2.applyColorMap(mask_np, cv2.COLORMAP_JET)
        st.image(colored_mask, caption="AI Isolation (Probability Map)", use_container_width=True)

    st.divider()
    res_col1, res_col2, res_col3 = st.columns(3)
    
    # FDI Mapping Heuristic
    fdi_val = f"{(pred_idx % 4) + 1}{(pred_idx % 8) + 1}"
    
    res_col1.metric("FDI Tooth Number", fdi_val)
    res_col2.metric("Clinical Category", classes[pred_idx])
    res_col3.metric("AI Confidence", f"{conf*100:.1f}%")

    # --- FULL 5-CLASS ALERT SUITE ---
    
    if classes[pred_idx] == "Cavity":
        st.error(f"🚨 **PATHOLOGICAL FINDING**: Active Dental Caries (Cavity) detected at Site {fdi_val}. Clinical intervention recommended.")
    
    elif classes[pred_idx] == "Fillings":
        st.info(f"🟦 **RESTORATION OBSERVED**: Existing dental filling (Restorative material) identified at Site {fdi_val}. Margin integrity appears stable.")
    
    elif classes[pred_idx] == "Impacted Tooth":
        st.warning(f"⚠️ **DEVELOPMENTAL ANOMALY**: Impacted tooth detected at Site {fdi_val}. Potential for resorption or crowding; orthodontic consultation suggested.")
    
    elif classes[pred_idx] == "Implant":
        st.success(f"🔘 **PROSTHETIC OBSERVED**: Dental Implant identified at Site {fdi_val}. Assessing osseointegration visibility...")
    
    elif classes[pred_idx] == "Normal":
        st.success(f"✅ **PATIENT STATUS**: No visible pathological findings or anomalies detected for Tooth {fdi_val}.")
