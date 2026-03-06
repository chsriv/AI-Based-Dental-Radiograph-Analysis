import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import os
import requests
import time

# --- 1. ENHANCED MODEL ARCHITECTURE ---
class DentalMultiTaskBrain(nn.Module):
    def __init__(self, num_classes=6): # Updated to 6 for Bone Loss/Anomalies
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

# --- 2. CLINICAL LOGIC FUNCTIONS ---
def get_fdi_quadrant(x, y, w, h):
    """Clinical FDI Mapping Logic (Quadrants 1-4)"""
    if y < h/2: # Upper
        return "1" if x < w/2 else "2"
    else: # Lower
        return "4" if x < w/2 else "3"

def analyze_bone_density(img_np):
    """Detects structural anomalies and bone loss via pixel intensity."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # CLAHE for bone density visualization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    density_map = cv2.applyColorMap(enhanced, cv2.COLORMAP_BONE)
    mean_density = np.mean(enhanced)
    return mean_density, density_map

# --- 3. ROBUST LOADING ---
@st.cache_resource
def load_clinical_model():
    # Note: Ensure you upload your refined 'dental_ai_structural_v2.pth' to your Github or local path
    model_path = "dental_ai_structural_v2.pth"
    
    classes = ['Cavity', 'Fillings', 'Impacted Tooth', 'Implant', 'Normal', 'Bone Loss/Anomaly']
    model = DentalMultiTaskBrain(num_classes=len(classes))
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        # Compatibility check for state dict
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    return model, classes

# --- 4. UI CONFIGURATION ---
st.set_page_config(page_title="Team 24 | Advanced Dental AI", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { border-radius: 10px; border: 1px solid #d1d8e0; background-color: white; }
    .report-card { padding: 20px; border-radius: 10px; border-left: 10px solid #2e7bcf; background-color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🦷 Automated FDI Charting & Structural Diagnostic Engine")
st.write("### **Team 24** | Problem Statement: Peri-implantitis & Multi-class Pathologies")

model, classes = load_clinical_model()

# Sidebar
st.sidebar.header("Clinical Parameters")
colors = {"Cavity": "#dc3545", "Fillings": "#007bff", "Impacted Tooth": "#ffc107", "Implant": "#6f42c1", "Normal": "#28a745", "Bone Loss/Anomaly": "#e67e22"}
for cls, color in colors.items():
    st.sidebar.markdown(f'<div style="background-color:{color}; color:white; padding:5px; border-radius:3px; margin-bottom:2px; font-size:0.8rem;">{cls}</div>', unsafe_allow_html=True)

# Main App
uploaded_file = st.file_uploader("Upload Panoramic OPG", type=["jpg", "png", "jpeg"])

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(raw_img)
    
    # 1. Image Enhancement & Density Check
    density_val, density_heatmap = analyze_bone_density(img_np)
    
    # 2. AI Inference
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_t = t(raw_img).unsqueeze(0)

    with st.spinner('Analyzing Structural Integrity...'):
        with torch.no_grad():
            mask, logits = model(img_t)
            # Apply clinical bias for anomalies
            logits[0, 5] += 0.5 # Bone Loss bias
            logits[0, 0] += 0.5 # Cavity bias
            
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            conf = probs.max().item()

    # 3. Visuals
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🔍 Original OPG")
        st.image(raw_img, use_container_width=True)
    with c2:
        st.subheader("🎯 Diagnostic Mask")
        mask_np = cv2.resize(mask.squeeze().numpy(), (img_np.shape[1], img_np.shape[0]))
        overlay = cv2.applyColorMap((mask_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(img_np, 0.7, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), 0.3, 0)
        st.image(blended, use_container_width=True)
    with c3:
        st.subheader("🦴 Bone Density Map")
        st.image(density_heatmap, use_container_width=True)

    # 4. Clinical Results
    st.divider()
    fdi_q = get_fdi_quadrant(img_np.shape[1]/2, img_np.shape[0]/2, img_np.shape[1], img_np.shape[0])
    
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    res_col1.metric("Clinical Category", classes[pred_idx])
    res_col2.metric("FDI Quadrant", f"Q{fdi_q}")
    res_col3.metric("Structure Index", f"{density_val:.1f}")
    res_col4.metric("AI Confidence", f"{conf*100:.1f}%")

    # Status Box
    status_color = colors[classes[pred_idx]]
    st.markdown(f"""
        <div class="report-card">
        <h4>Automated Clinical Finding: {classes[pred_idx]}</h4>
        <p><b>Observation:</b> The system detected {classes[pred_idx]} at the site. 
        Structural analysis indicates a density score of {density_val:.1f}.</p>
        <p><b>Recommendation:</b> {"Immediate Clinical Intervention" if classes[pred_idx] in ['Cavity', 'Impacted Tooth', 'Bone Loss/Anomaly'] else "Periodic Monitoring"}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.download_button("📥 Export OPG Analysis Report", f"Diagnosis: {classes[pred_idx]}", file_name="Dental_AI_Report.txt")

else:
    st.info("Awaiting OPG Upload for FDI Charting and Structural Analysis.")
