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
[![My Skills](https://skillicons.dev/icons?i=py,pytorch,opencv,svg,github,md)](https://skillicons.dev)

---

## System Architecture
The engine utilizes an EfficientNet-B0 backbone for feature extraction, bifurcating into a segmentation decoder and a classification head.



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
