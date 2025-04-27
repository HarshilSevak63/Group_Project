import joblib
import numpy as np

# Load the models and scalers
model_diabetes = joblib.load('diabetes_rf_model.pkl')
scaler_diabetes = joblib.load('diabetes_scaler.pkl')
selector_diabetes = joblib.load('diabetes_feature_selector.pkl')

model_heart = joblib.load('heart_rf_model_selected.pkl')
scaler_heart = joblib.load('heart_scaler.pkl')
selector_heart = joblib.load('heart_feature_selector.pkl')

# # Take inputs from the user for diabetes prediction
# print("\n--- Diabetes Prediction ---")
# glucose = int(input('Enter Glucose: '))
# bmi = float(input('Enter BMI: '))
# dpf = float(input('Enter DiabetesPedigreeFunction: '))
# age = int(input('Enter Age: '))

# # Add dummy values for missing features (these can be set to zero or any other reasonable default)
# pregnancies = 0
# blood_pressure = 0
# skin_thickness = 0
# insulin = 0

# # Prepare the features for prediction (8 features as expected by the model)
# diabetes_features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]

# scaled_diabetes_features = scaler_diabetes.transform([diabetes_features])
# selected_diabetes_features = selector_diabetes.transform(scaled_diabetes_features)

# # Make prediction for diabetes
# diabetes_prediction = model_diabetes.predict(selected_diabetes_features)
# print(f"Predicted Diabetes Outcome: {diabetes_prediction[0]} (0: No, 1: Yes)")

# Take inputs from the user for heart disease prediction
print("\n--- Heart Disease Prediction ---")
age_heart = int(input('Enter Age: '))
sex = int(input('Enter Sex (1 = Male, 0 = Female): '))
cp = int(input('Enter Chest Pain Type (0-3): '))
trestbps = int(input('Enter Resting Blood Pressure: '))
chol = int(input('Enter Serum Cholesterol: '))
fbs = int(input('Enter Fasting Blood Sugar (1 if > 120 mg/dl, 0 otherwise): '))
restecg = int(input('Enter Resting Electrocardiographic Results (0-2): '))
thalach = int(input('Enter Maximum Heart Rate: '))
exang = int(input('Enter Exercise Induced Angina (1 = Yes, 0 = No): '))
oldpeak = float(input('Enter Depression Induced by Exercise: '))
slope = int(input('Enter Slope of Peak Exercise ST Segment (0-2): '))
ca = int(input('Enter Number of Major Vessels Colored by Fluoroscopy (0-3): '))
thal = int(input('Enter Thalassemia (0 = Normal, 1 = Fixed defect, 2 = Reversable defect): '))

# Prepare the features for prediction
heart_features = [age_heart, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
scaled_heart_features = scaler_heart.transform([heart_features])
selected_heart_features = selector_heart.transform(scaled_heart_features)

# Make prediction for heart disease
heart_prediction = model_heart.predict(selected_heart_features)
print(f"Predicted Heart Disease Outcome: {heart_prediction[0]} (0: No, 1: Yes)")
