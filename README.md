# AI-Based Dental Radiograph Analysis System
### Use Case 2: Automated OPG Analysis and FDI Charting

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/framework-PyTorch-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Executive Summary
This repository houses a high-throughput clinical diagnostic engine designed to automate the interpretation of Dental Orthopantomograms (OPG). By integrating a multi-task convolutional neural network, the system executes simultaneous tooth segmentation, pathological classification, and ISO/FDI notation mapping. Engineered for reliability, the model was trained on a robust dataset of 25,410 clinical samples to minimize diagnostic oversight in high-volume dental environments.

---

## Tech Stack
[![My Skills](https://skillicons.dev/icons?i=py,pytorch,opencv,github)](https://skillicons.dev)

---

## System Architecture
The engine utilizes an EfficientNet-B0 backbone for feature extraction, bifurcating into a segmentation decoder and a classification head.

## 🎨 Classification & Visualization Legend
The system employs a standardized color-coding scheme to provide immediate diagnostic triage. Overlays are generated using a custom JET colormap where color intensity correlates with model confidence.

| Color | Clinical Classification | Description |
| :--- | :--- | :--- |
| 🔴 **Red** | **Cavity** | Active radiolucent lesions requiring immediate intervention. |
| 🔵 **Blue** | **Filling** | Existing restorative materials and prior dental history. |
| 🟣 **Purple** | **Implant** | Integrated prosthetic hardware and osseointegration status. |
| 🟡 **Yellow** | **Impacted Tooth** | Unerupted or obstructed dental structures. |
| 🟢 **Green** | **Normal** | Healthy dental anatomy with no detectable pathology. |

---

## 🦷 Automated FDI Charting Logic
Each detected segment is processed through a geometric centroid algorithm to assign the correct **ISO/FDI 2-Digit Notation**. 



1. **Centroid Calculation**: Identifying the center of the segmentation mask.
2. **Quadrant Detection**: Determining the dental arch position (1-4).
3. **FDI Assignment**: Mapping the tooth to its specific number (e.g., Tooth 21) for automated electronic health record (EHR) entry.



```mermaid
graph TD
    A[Input Radiograph] --> B[EfficientNet-B0 Encoder]
    B --> C[Feature Map Extraction]
    C --> D[Transposed Conv Decoder]
    C --> E[Global Average Pooling]
    D --> F[Semantic Mask: Segmentation]
    E --> G[Fully Connected Heads]
    G --> H[Class Prediction: Cavity/Normal/etc.]
    F & H --> I[ISO/FDI Logic Mapper]
    I --> J[Final Clinical Report]
