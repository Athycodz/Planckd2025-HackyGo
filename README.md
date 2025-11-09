# Planckd2025-HackyGo

Planckd 2025 Hackathon — **Team HackyGo**  
Exploring Quantum Machine Learning with PennyLane 🧠⚡  

---

##  Track
**Quantum Machine Learning (QML)**

---

##  Problem Statement
> Handwritten digit classification (MNIST) using classical, quantum, and hybrid learning models.

---

## 👥 Team
- **Team Name:** HackyGo  
- **Member:** Atharva Sanjay Sakhare ([@Athycodz](https://github.com/Athycodz))  
- **Institute:** IIIT Sri City  

---

## 🚀 Project Overview

This project demonstrates a comparative study between a classical Support Vector Machine (SVM) and a hybrid quantum–classical neural model using **PennyLane**.  
The goal was to evaluate how quantum circuits enhance learning in low-dimensional feature spaces.

### Implemented Models:
1. **Classical SVM** — baseline accuracy: **0.87**  
2. **Hybrid Quantum-Classical Classifier** — final accuracy: **0.99**

### Model Highlights:
- PCA feature compression → 6 principal components  
- Quantum circuit: `AngleEmbedding` + `StronglyEntanglingLayers` (6 qubits)  
- Optimizer: Adam (25 epochs, lr=0.1)  
- Backend: `default.qubit` simulator  

---

## Folder Structure

Planckd2025-HackyGo/
├── code/
│ ├── quick_check.py # MNIST sanity check
│ ├── svm_baseline.py # Classical SVM model
│ └── hybrid_quantum_model.py # Quantum-classical hybrid model
│
├── results/
│ ├── svm_accuracy.txt
│ ├── quantum_accuracy.txt
│ └── loss_plot.png
│
├── report.pdf # Final project report
├── requirements.txt # Dependency list
├── README.md # Summary + team info
└── .gitignore

##  Results Summary

| Model Type        | Accuracy | Notes |
|--------------------|----------|-------|
| Classical SVM      | 0.8725   | Baseline using scikit-learn |
| Quantum Hybrid QNN | 0.9900   | 6 qubits, PennyLane backend |

---
