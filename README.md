# 🏥 AI-Augmented Neonatal Care System

## 🚀 Project Overview
This project is a **Dual-Engine AI System** designed to assist medical staff in the early identification of Low Birth Weight (LBW) and Preterm neonates. It combines **Machine Learning (Predictive Analytics)** and **Computer Vision (Image Analysis)** to provide a comprehensive risk assessment tool for neonatal care.

## 🛠️ Key Features (My Code)
This repository contains the complete source code for the application, featuring two core custom-built engines:

### **Engine A: Predictive Risk Stratification**
- **Purpose**: Predicts the risk of preterm birth based on maternal health vitals.
- **My Implementation**:
    - **Ensemble Learning Architecture**: I implemented a `VotingClassifier` dealing with complex health data, combining **XGBoost** (Gradient Boosting) and **Random Forest** models to achieve high recall and precision.
    - **Synthetic Data Generation**: Created a robust data simulation module to train and validate the model in the absence of private medical records.
    - **Explainable AI (XAI)**: Integrated **SHAP (SHapley Additive exPlanations)** to provide real-time visual explanations (Waterfall charts) for *why* a specific risk score was predicted, ensuring clinical trust.

### **Engine B: Visual Weight Estimator**
- **Purpose**: Estimates a newborn's weight using just a smartphone camera, without needing a weighing scale.
- **My Implementation**:
    - **Computer Vision Pipeline**: Built a custom image processing pipeline using **OpenCV** to perform:
        - Gray-scaling and Gaussian Blurring for noise reduction.
        - Canny Edge Detection and Morphological operations (Dilation/Erosion) to isolate the baby and reference object.
    - **Reference Object Calibration**: Implemented logic to auto-detect a standard reference object (like a Credit/ID Card) to calculate a dynamic `pixels-per-metric` ratio.
    - **Mathematical Modeling**: Derived a specific regression formula effectively mapping 2D contour area and perimeter to an estimated 3D weight in grams.

## ⚙️ How It Works

### 1. The Interface
The application is built with **Streamlit**, offering a clean, responsive web interface that works on both desktop and mobile devices.

### 2. Risk Engine Workflow
1.  **Input**: Doctor enters maternal age, hemoglobin levels, blood pressure, and birth history.
2.  **Processing**: The input is fed into the pre-trained Ensemble Model.
3.  **Output**:
    - **Risk Score**: A probability percentage of preterm birth.
    - **Clinical Alert**: "High Risk" (Red) or "Normal Risk" (Green) indicators.
    - **Report**: auto-generates a downloadable text report for valid application.

### 3. Vision Engine Workflow
1.  **Capture**: User captures a photo of the infant next to a reference ID card.
2.  **Image Processing**: The system scans the image to identify contours.
3.  **Analysis**:
    - Checks for "Calibration Quality" to warn about bad camera angles.
    - Calculates the baby's dimensions relative to the known ID card size.
4.  **Result**: Displays the baby's estimated weight and immediately flags if it falls below the **2.5kg (LBW)** threshold, suggesting "Kangaroo Mother Care" if needed.

## 📦 Tech Stack
- **Languages**: Python
- **Frontend**: Streamlit
- **ML/AI**: XGBoost, Scikit-Learn, SHAP, NumPy, Pandas
- **Computer Vision**: OpenCV (cv2), Imutils
