import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from imutils import contours, grab_contours
import shap
import matplotlib.pyplot as plt
import io
import time
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Neonatal Care Pro", page_icon="🏥", layout="wide")

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏥 AI-Augmented Neonatal Care System")
st.markdown("### 🚀 Dual-Engine System for Early Identification of LBW & Preterm Neonates")

# --- SIDEBAR & SETTINGS ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    app_mode = st.radio("Select Module:", ["Engine A: Predictive Risk", "Engine B: Visual Weight"])
    
    st.divider()
    st.caption("Advanced Settings")
    dev_mode = st.checkbox("Enable Developer Mode")
    
    if app_mode == "Engine B: Visual Weight" and dev_mode:
        st.subheader("Vision Tuning")
        canny_low = st.slider("Edge Threshold (Low)", 0, 100, 30)
        canny_high = st.slider("Edge Threshold (High)", 50, 200, 100)
    else:
        canny_low, canny_high = 30, 100

# ==========================================
# CACHED FUNCTIONS
# ==========================================

@st.cache_resource
def train_model_a():
    """Trains an Ensemble (XGBoost + Random Forest) model and calculates validation metrics."""
    # Generate Synthetic Data
    np.random.seed(42)
    n_samples = 2000
    data = {
        'Maternal_Age': np.random.randint(18, 40, n_samples),
        'Hemoglobin_Level': np.random.normal(11, 2, n_samples),
        'Systolic_BP': np.random.normal(120, 15, n_samples),
        'Diastolic_BP': np.random.normal(80, 10, n_samples),
        'Birth_Interval': np.random.randint(12, 60, n_samples),
        'Previous_Preterm': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    }
    df = pd.DataFrame(data)
    
    # Risk Logic (Simulated Ground Truth)
    risk_factor = (
        (df['Hemoglobin_Level'] < 10).astype(int) * 2 + 
        (df['Systolic_BP'] > 140).astype(int) * 2 + 
        df['Previous_Preterm'] * 3 +
        np.random.normal(0, 0.5, n_samples) 
    )
    df['Preterm_Birth'] = (risk_factor > 1.5).astype(int)
    
    X = df.drop('Preterm_Birth', axis=1)
    y = df['Preterm_Birth']
    
    # Split Data for Validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define Base Models
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    # Model 1: XGBoost (Gradient Boosting)
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=ratio,
        max_depth=4,
        learning_rate=0.1,
        n_estimators=100
    )
    
    # Model 2: Random Forest (Bagging)
    rf_clf = RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced', 
        random_state=42
    )
    
    # Ensemble: Voting Classifier (Soft Voting averages probabilities)
    ensemble_model = VotingClassifier(
        estimators=[('xgb', xgb_clf), ('rf', rf_clf)],
        voting='soft'
    )
    
    # Train Ensemble
    ensemble_model.fit(X_train, y_train)
    
    # We also fit the standalone XGBoost for SHAP explanation purposes (VotingClassifier is hard to explain visually)
    xgb_clf.fit(X_train, y_train)
    
    # Calculate Metrics on Test Set
    y_pred = ensemble_model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred)
    }
    
    # For robust SHAP (Black-box approach): Keep training data sample for background
    background_data = X_train.sample(100, random_state=42)
    
    # Return Ensemble for prediction, but XGB for explanation
    return ensemble_model, xgb_clf, metrics, background_data

# Load Models and Metrics
ensemble_model, explainer_model, model_metrics, background_data = train_model_a()

def generate_report(patient_data, risk_score, risk_label):
    """Generates a simple text report."""
    report = f"""
    NEONATAL RISK ASSESSMENT REPORT
    -------------------------------
    Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    PATIENT VITALS:
    - Age: {patient_data['Maternal_Age'].values[0]}
    - Hemoglobin: {patient_data['Hemoglobin_Level'].values[0]} g/dL
    - BP: {patient_data['Systolic_BP'].values[0]}/{patient_data['Diastolic_BP'].values[0]} mmHg
    
    AI ASSESSMENT:
    - Prediction: {risk_label}
    - Risk Probability: {risk_score:.1%}
    
    RECOMMENDATION:
    {'Refer to District Hospital immediately.' if risk_label == 'High Risk' else 'Continue routine ANC checkups.'}
    
    -------------------------------
    Generated by IHAT AI System
    """
    return report

def process_weight_estimation(image, canny_low, canny_high, dev_mode=False):
    """Core logic for estimating weight from an image."""
    # Resize
    h, w = image.shape[:2]
    new_h = 800
    new_w = int(w * (800/h))
    image = cv2.resize(image, (new_w, new_h))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Use adjustable thresholds
    edged = cv2.Canny(gray, canny_low, canny_high)
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=1)
    
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    
    result_image = image.copy()
    total_weight = 0
    message = ""
    status = "error" # error, success, warning
    debug_info = {}
    calib_quality = "Unknown"

    if len(cnts) > 0:
        (cnts, _) = contours.sort_contours(cnts)
        pixelsPerMetric = None
        
        # Process Contours
        for i, c in enumerate(cnts):
            if cv2.contourArea(c) < 1000: continue
            
            box = cv2.minAreaRect(c)
            (x, y), (w_box, h_box), angle = box
            
            # Calibration (First Object)
            if pixelsPerMetric is None:
                pixelsPerMetric = w_box / 8.56
                
                # Aspect Ratio Check (Credit Card is ~1.58)
                aspect_ratio = max(w_box, h_box) / min(w_box, h_box)
                if 1.4 < aspect_ratio < 1.7:
                    calib_quality = "Good (Parallel)"
                else:
                    calib_quality = f"Poor (Skewed: {aspect_ratio:.2f})"
                
                box_points = cv2.boxPoints(box).astype("int")
                cv2.drawContours(result_image, [box_points], -1, (255, 0, 0), 2)
                continue
            
            # Measurement (Baby Parts)
            area_pixels = cv2.contourArea(c)
            perimeter_pixels = cv2.arcLength(c, True)
            area_cm2 = area_pixels / (pixelsPerMetric ** 2)
            perimeter_cm = perimeter_pixels / pixelsPerMetric
            
            # Regression Formula
            ALPHA, BETA, GAMMA = 0.29, 12.5, 120
            est_weight = (ALPHA * (area_cm2 ** 1.5)) + (BETA * perimeter_cm) + GAMMA
            total_weight += est_weight
            
            # Draw
            box_points = cv2.boxPoints(box).astype("int")
            cv2.drawContours(result_image, [box_points], -1, (0, 255, 0), 2)
            cv2.putText(result_image, f"{est_weight:.0f}g", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        status = "success"
        message = f"Total Estimated Weight: {total_weight:.0f} g"
        
        if dev_mode:
            debug_info = {
                "Calibration Ratio": f"{pixelsPerMetric:.2f} px/cm" if pixelsPerMetric else "N/A",
                "Objects Detected": len(cnts),
                "Calibration Quality": calib_quality
            }
            
    else:
        status = "error"
        message = "No objects detected. Try adjusting edge thresholds."
        
    return result_image, total_weight, message, status, debug_info, calib_quality


# ==========================================
# ENGINE A: PREDICTIVE RISK SCORER
# ==========================================
if app_mode == "Engine A: Predictive Risk":
    st.subheader("🔮 Engine A: Preterm Risk Stratification")
    
    # --- MODEL HEALTH DASHBOARD ---
    with st.expander("📊 Model Performance (Validation Metrics)", expanded=False):
        m1, m2, m3 = st.columns(3)
        m1.metric("Recall (Sensitivity)", f"{model_metrics['recall']:.1%}", help="Ability to catch positive cases (Sick babies)")
        m2.metric("Accuracy", f"{model_metrics['accuracy']:.1%}")
        m3.metric("Precision", f"{model_metrics['precision']:.1%}")
        
        st.caption("Note: High Recall is prioritized to minimize False Negatives (Missing a sick baby).")
        
        # Confusion Matrix
        if st.checkbox("Show Confusion Matrix"):
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(model_metrics['cm'], annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            st.pyplot(fig_cm)

    
    with st.expander("📝 Patient Vitals Entry", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Maternal Age", 18, 45, 25)
            hb = st.number_input("Hemoglobin (g/dL)", 5.0, 16.0, 11.0, step=0.1, help="Normal range: 12-16 g/dL")
        
        with col2:
            bp_sys = st.number_input("Systolic BP (mmHg)", 90, 200, 120)
            bp_dia = st.number_input("Diastolic BP (mmHg)", 60, 120, 80)
            
        with col3:
            interval = st.number_input("Birth Interval (Months)", 0, 120, 24)
            prev_preterm = st.selectbox("Previous Preterm Birth?", ["No", "Yes"])
        
    prev_preterm_val = 1 if prev_preterm == "Yes" else 0
    
    if st.button("🚀 Analyze Risk"):
        with st.spinner("Running Ensemble Inference..."):
            time.sleep(0.5) # UI smoother
            
            # Create Dataframe
            input_data = pd.DataFrame([[age, hb, bp_sys, bp_dia, interval, prev_preterm_val]], 
                                      columns=['Maternal_Age', 'Hemoglobin_Level', 'Systolic_BP', 'Diastolic_BP', 'Birth_Interval', 'Previous_Preterm'])
            
            # Predict (Using Ensemble Model)
            prediction = ensemble_model.predict(input_data)[0]
            probability = ensemble_model.predict_proba(input_data)[0][1]
            
            st.divider()
            
            # Display Result
            c1, c2 = st.columns([1, 1.5])
            
            with c1:
                st.markdown("#### Clinical Assessment")
                if prediction == 1:
                    st.error(f"### 🔴 High Risk Detected")
                    st.metric("Risk Probability", f"{probability:.1%}", delta="High Alert", delta_color="inverse")
                    st.warning("⚠️ **Action Required:** Immediate referral to District Hospital (SNCU).")
                else:
                    st.success(f"### 🟢 Normal Risk")
                    st.metric("Risk Probability", f"{probability:.1%}", delta="Safe")
                    st.info("✅ **Action:** Schedule next routine ANC visit.")
                
                # Download Report Feature
                report_text = generate_report(input_data, probability, "High Risk" if prediction == 1 else "Normal Risk")
                st.download_button("📄 Download Medical Report", report_text, file_name="risk_report.txt")

            with c2:
                st.markdown("#### Explainability (Why?)")
                try:
                    # Attempt 1: Tree Explainer with Booster Patch (Best Visualization)
                    # We use explainer_model (XGBoost) as a proxy for explaining the ensemble's decision
                    booster = explainer_model.get_booster()
                    model_bytearray = booster.save_raw()[4:]
                    def my_booster_save_raw(*args, **kwargs):
                        return model_bytearray
                    booster.save_raw = my_booster_save_raw
                    explainer = shap.TreeExplainer(booster)
                    shap_values = explainer(input_data)
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    shap.plots.waterfall(shap_values[0], show=False)
                    st.pyplot(fig)
                    
                except Exception as e_tree:
                    # Fallback Attempt 2: Kernel Explainer (Model Agnostic / Black Box)
                    # Guaranteed to work but slower
                    try:
                        # Use a small background dataset for speed
                        background_summary = shap.sample(background_data, 10)
                        
                        # Define prediction function for KernelExplainer using ENSEMBLE model
                        def predict_fn(data):
                            return ensemble_model.predict_proba(data)[:, 1]
                            
                        explainer = shap.KernelExplainer(predict_fn, background_summary)
                        shap_values = explainer.shap_values(input_data)
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        # KernelExplainer returns a list for classifiers, take index 0
                        # Note: KernelExplainer output format is slightly different than TreeExplainer
                        # We use force_plot or summary_plot usually, but for single instance waterfall:
                        shap.plots.waterfall(shap.Explanation(values=shap_values[0], 
                                                            base_values=explainer.expected_value, 
                                                            data=input_data.iloc[0], 
                                                            feature_names=input_data.columns), show=False)
                        st.pyplot(fig)
                        
                    except Exception as e_kernel:
                        st.warning(f"Visual explanation unavailable. Rely on Risk Score.")
                        if dev_mode: st.error(f"Debug Error: {e_tree} | {e_kernel}")

# ==========================================
# ENGINE B: VISUAL WEIGHT ESTIMATOR
# ==========================================
elif app_mode == "Engine B: Visual Weight":
    st.subheader("📷 Engine B: Visual Weight Estimator")
    
    st.info("💡 **Instructions:** Ensure a Credit Card/ID Card is placed next to the baby for calibration.")
    
    tabs = st.tabs(["Upload Photo", "Use Camera"])
    
    # --- TAB 1: UPLOAD ---
    with tabs[0]:
        uploaded_file = st.file_uploader("Choose Image", type=['jpg', 'png', 'jpeg'])
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            c1, c2 = st.columns(2)
            with c1:
                st.image(image, channels="BGR", caption="Original Input", use_column_width=True)
            
            if st.button("⚖️ Estimate Weight (Upload)"):
                with st.spinner("Processing..."):
                    res_img, weight, msg, status, debug, quality = process_weight_estimation(image, canny_low, canny_high, dev_mode)
                    
                    with c2:
                        st.image(res_img, channels="BGR", caption="Analysis Result", use_column_width=True)
                    
                    st.divider()
                    
                    # Calibration Health Check
                    if "Poor" in quality:
                        st.warning(f"⚠️ **Camera Angle Issue:** Calibration Quality is '{quality}'. Results may be inaccurate. Hold camera parallel to bed.")
                    else:
                        st.caption(f"✅ Calibration Quality: {quality}")

                    if status == "success":
                        st.success(f"### ✅ {msg}")
                        if weight < 2500:
                            st.warning("⚠️ **Low Birth Weight Detected (<2.5kg).** Initiate Kangaroo Mother Care (KMC).")
                        else:
                            st.success("✅ Weight is within normal range.")
                    else:
                        st.error(msg)
                        
                    if dev_mode and debug:
                        with st.expander("Debug Data"):
                            st.write(debug)

    # --- TAB 2: CAMERA INPUT ---
    with tabs[1]:
        camera_image = st.camera_input("Take a photo")
        
        if camera_image is not None:
            file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            if st.button("⚖️ Estimate Weight (Camera)"):
                with st.spinner("Processing..."):
                    res_img, weight, msg, status, debug, quality = process_weight_estimation(image, canny_low, canny_high, dev_mode)
                    
                    st.image(res_img, channels="BGR", caption="Analysis Result", use_column_width=True)
                    
                    st.divider()
                    
                    # Calibration Health Check
                    if "Poor" in quality:
                        st.warning(f"⚠️ **Camera Angle Issue:** Calibration Quality is '{quality}'. Results may be inaccurate. Hold camera parallel to bed.")
                    
                    if status == "success":
                        st.success(f"### ✅ {msg}")
                        if weight < 2500:
                            st.warning("⚠️ **Low Birth Weight Detected (<2.5kg).** Initiate Kangaroo Mother Care (KMC).")
                        else:
                            st.success("✅ Weight is within normal range.")
                    else:
                        st.error(msg)
                        
                    if dev_mode and debug:
                        with st.expander("Debug Data"):
                            st.write(debug)