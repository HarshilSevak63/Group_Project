import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

# --- 1. Diabetes Dataset Processing ---

print("\n--- Diabetes Prediction ---")

# Load diabetes data
diabetes_data = pd.read_csv('diabetes.csv')  # <-- Replace filename

# Diabetes features and target
diabetes_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
diabetes_target = 'Outcome'

X_diabetes = diabetes_data[diabetes_features]
y_diabetes = diabetes_data[diabetes_target]

# Split
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)

# Scale
scaler_diabetes = StandardScaler()
X_train_d_scaled = scaler_diabetes.fit_transform(X_train_d)
X_test_d_scaled = scaler_diabetes.transform(X_test_d)

# Train RandomForest with all features
rf_model_diabetes = RandomForestClassifier(random_state=42)
rf_model_diabetes.fit(X_train_d_scaled, y_train_d)

# Evaluate full model
y_pred_d_full = rf_model_diabetes.predict(X_test_d_scaled)
full_accuracy_d = accuracy_score(y_test_d, y_pred_d_full)
print(f"Before Feature Selection (Diabetes) Accuracy: {full_accuracy_d:.4f}")

# --- Feature Selection ---
selector_d = SelectFromModel(rf_model_diabetes, prefit=True, threshold='median')
X_train_d_selected = selector_d.transform(X_train_d_scaled)
X_test_d_selected = selector_d.transform(X_test_d_scaled)

# Print important features
selected_features_d = [feature for feature, selected in zip(diabetes_features, selector_d.get_support()) if selected]
print(f"Selected Diabetes Features: {selected_features_d}")

# Train RandomForest on selected features
rf_model_diabetes_selected = RandomForestClassifier(random_state=42)
rf_model_diabetes_selected.fit(X_train_d_selected, y_train_d)

# Evaluate reduced model
y_pred_d_selected = rf_model_diabetes_selected.predict(X_test_d_selected)
selected_accuracy_d = accuracy_score(y_test_d, y_pred_d_selected)
print(f"After Feature Selection (Diabetes) Accuracy: {selected_accuracy_d:.4f}")

# Save models and scaler
joblib.dump(rf_model_diabetes_selected, 'diabetes_rf_model.pkl')
joblib.dump(scaler_diabetes, 'diabetes_scaler.pkl')
joblib.dump(selector_d, 'diabetes_feature_selector.pkl')

# --- 2. Heart Disease Dataset Processing ---

print("\n--- Heart Disease Prediction ---")

# Load heart disease data
heart_data = pd.read_csv('heart.csv')  # <-- Replace filename

# Heart features and target
heart_features = ['age', 'sex', 'cp', 'trestbps', 'chol',
                  'fbs', 'restecg', 'thalach', 'exang',
                  'oldpeak', 'slope', 'ca', 'thal']
heart_target = 'target'

X_heart = heart_data[heart_features]
y_heart = heart_data[heart_target]

# Split
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)

# Scale
scaler_heart = StandardScaler()
X_train_h_scaled = scaler_heart.fit_transform(X_train_h)
X_test_h_scaled = scaler_heart.transform(X_test_h)

# Train RandomForest with all features (for feature selection)
rf_model_heart_full = RandomForestClassifier(random_state=42)
rf_model_heart_full.fit(X_train_h_scaled, y_train_h)

# Evaluate full model
y_pred_h_full = rf_model_heart_full.predict(X_test_h_scaled)
full_accuracy_h = accuracy_score(y_test_h, y_pred_h_full)
print(f"Before Feature Selection (Heart) Accuracy: {full_accuracy_h:.4f}")

# --- Feature Selection ---
selector_h = SelectFromModel(rf_model_heart_full, prefit=True, threshold='mean')  # use mean importance
X_train_h_selected = selector_h.transform(X_train_h_scaled)
X_test_h_selected = selector_h.transform(X_test_h_scaled)

# Print important features
selected_features_h = [feature for feature, selected in zip(heart_features, selector_h.get_support()) if selected]
print(f"Selected Heart Disease Features: {selected_features_h}")


# Train RandomForest again on selected features
rf_model_heart_selected = RandomForestClassifier(random_state=42)
rf_model_heart_selected.fit(X_train_h_selected, y_train_h)

# Evaluate reduced model
y_pred_h_selected = rf_model_heart_selected.predict(X_test_h_selected)
selected_accuracy_h = accuracy_score(y_test_h, y_pred_h_selected)
print(f"After Feature Selection (Heart) Accuracy: {selected_accuracy_h:.4f}")

# Save models and scaler
joblib.dump(rf_model_heart_selected, 'heart_rf_model_selected.pkl')
joblib.dump(scaler_heart, 'heart_scaler.pkl')
joblib.dump(selector_h, 'heart_feature_selector.pkl')

print("\nBoth models trained, optimized, and saved successfully!")