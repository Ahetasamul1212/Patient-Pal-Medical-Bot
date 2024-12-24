import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess the dataset
def load_and_preprocess_data():
    # Load dataset (using the provided file directly)
    file_path = r'E:\MY_python_projects\Project_Patient_pal\Models\healthcare-dataset-stroke-data.csv'
    data = pd.read_csv(file_path)

    # Impute missing values in the 'bmi' column with the mean
    data['bmi'].fillna(data['bmi'].mean(), inplace=True)

    # Encode categorical columns
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    label_encoders = {col: LabelEncoder() for col in categorical_columns}
    for col in categorical_columns:
        data[col] = label_encoders[col].fit_transform(data[col])

    # Features and target variable
    X = data.drop(columns=['id', 'stroke'])
    y = data['stroke']

    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, label_encoders

# Train the model
def train_model(X, y):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train Random Forest model
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, class_weight='balanced')
    rf_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return rf_model, accuracy

# Streamlit app
def main():
    st.title("Stroke Prediction App")
    st.write("This app predicts whether a patient is likely to have a stroke based on input features.")

    # Load and preprocess data
    X, y, scaler, label_encoders = load_and_preprocess_data()

    # Train the model
    rf_model, accuracy = train_model(X, y)

    st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")

    # Input features for prediction
    st.write("### Enter Patient Details")
    user_input = {}

    # Input fields for each feature
    columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 
               'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']

    for col in columns:
        if col in label_encoders:
            options = label_encoders[col].classes_
            user_input[col] = st.selectbox(f"{col}", options)
        elif col == 'age':
            user_input[col] = st.slider(f"{col}", min_value=0, max_value=120, step=1)
        elif col == 'avg_glucose_level':
            user_input[col] = st.slider(f"{col}", min_value=40.0, max_value=300.0, step=0.1)
        elif col == 'bmi':
            user_input[col] = st.slider(f"{col}", min_value=10.0, max_value=60.0, step=0.1)
        elif col in ['hypertension', 'heart_disease']:
            user_input[col] = st.selectbox(f"{col}", options=["No", "Yes"])
            user_input[col] = 1 if user_input[col] == "Yes" else 0
        else:
            user_input[col] = st.number_input(f"{col}", step=0.1)

    # Convert user input to DataFrame
    input_data = pd.DataFrame([user_input])

    # Encode categorical features and scale numerical values
    for col in label_encoders:
        input_data[col] = label_encoders[col].transform(input_data[col])
    input_scaled = scaler.transform(input_data)

    if st.button("Predict"):
        prediction_proba = rf_model.predict_proba(input_scaled)[0]
        prediction = rf_model.predict(input_scaled)
        result = "Stroke" if prediction[0] == 1 else "No Stroke"

        st.write(f"### Prediction: {result}")
        st.write(f"### Probability of Stroke: {prediction_proba[1] * 100:.2f}%")
        st.write(f"### Probability of No Stroke: {prediction_proba[0] * 100:.2f}%")

if __name__ == "__main__":
    main()
