import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
import shap
import matplotlib.pyplot as plt
import optuna

# --- 1. Generate Synthetic Data ---
print("Generating data...")
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

# Risk Logic
risk_factor = (
    (df['Hemoglobin_Level'] < 10).astype(int) * 2 + 
    (df['Systolic_BP'] > 140).astype(int) * 2 + 
    df['Previous_Preterm'] * 3 +
    np.random.normal(0, 0.5, n_samples) 
)
df['Preterm_Birth'] = (risk_factor > 1.5).astype(int)

X = df.drop('Preterm_Birth', axis=1)
y = df['Preterm_Birth']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Advanced: Optuna Objective Function ---
def objective(trial):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': 'gbtree',
        'scale_pos_weight': float(np.sum(y == 0)) / np.sum(y == 1),
        'verbosity': 0,
        
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200)
    }
    
    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    recall = recall_score(y_test, preds)
    return recall

# --- 3. Run Optimization ---
print("🚀 Starting AI Optimization (Optuna)...")
optuna.logging.set_verbosity(optuna.logging.WARNING) # Reduce noise
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10) # Reduced to 10 for quicker test

print(f"✅ Best Recall Found: {study.best_value:.4f}")
print("   Best Params:", study.best_params)

# --- 4. Train Final Model with Best Params ---
print("Training Final Model with Best Parameters...")
best_params = study.best_params
best_params['objective'] = 'binary:logistic'
best_params['scale_pos_weight'] = float(np.sum(y == 0)) / np.sum(y == 1)

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_train, y_train)

# --- 5. Explainability (Fixed) ---
print("Generating SHAP Plot...")

# FIX: We create a generic Explainer and pass the PREDICT method, not the model object.
# This forces SHAP to treat it as a black-box function if tree-explainer fails, 
# which is much safer and avoids version conflicts.
# However, for XGBoost, TreeExplainer is best. Let's try passing the model directly 
# but ensuring data is passed correctly.

try:
    # Attempt 1: Standard TreeExplainer on the sklearn wrapper
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer(X_test)
except Exception as e:
    print(f"⚠️ TreeExplainer failed ({e}), switching to generic Explainer...")
    # Attempt 2: Generic Explainer (slower but safer)
    # We use a subset of data for background distribution to speed it up
    background = shap.maskers.Independent(X_train, max_samples=100)
    explainer = shap.Explainer(final_model.predict, background)
    shap_values = explainer(X_test[:10]) # Explain first 10 for speed

# Plot
print("Displaying plot... (Close popup to finish)")
# Check if shap_values is a list (generic explainer) or object (tree explainer)
if isinstance(shap_values, list):
    # For generic explainer on classification, it returns list of classes. 
    # We take index 1 (Positive class / Preterm)
    shap.plots.waterfall(shap_values[0]) 
else:
    # Tree explainer returns a single object
    shap.plots.waterfall(shap_values[0])

plt.show()