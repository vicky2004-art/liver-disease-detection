import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- 1. SETUP PAGE ---
st.set_page_config(page_title="Liver Disease Detector", layout="wide")
st.title("🏥 Liver Disease Detection AI")

# --- 2. ROBUST DATA LOADING ---
# This block tries to find the file. If not found, it asks you to upload it.
data = None
default_file = "Liver Patient Dataset (LPD)_train.csv"

try:
    data = pd.read_csv(default_file, encoding='latin1')
    st.success(f"✅ Automatically loaded '{default_file}'")
except FileNotFoundError:
    st.warning(f"⚠️ Could not find '{default_file}' automatically.")
    st.info("👉 Please drag and drop your 'Liver Patient Dataset (LPD)_train.csv' file below:")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, encoding='latin1')

# --- 3. MAIN APPLICATION LOGIC ---
if data is not None:
    # > DATA CLEANING
    # Standardize column names to prevent KeyErrors
    # Expected order based on your file: Age, Gender, TB, DB, Alkphos, Sgpt, Sgot, TP, ALB, A/G, Result
    if len(data.columns) >= 11:
        data.columns = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
                        'Alkphos', 'Sgpt', 'Sgot', 'Total_Protiens', 
                        'ALB', 'A_G_Ratio', 'Result']
    
    # Handle Missing Values
    data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    
    # Fill numeric NaNs with mean
    for col in data.columns:
        if col != 'Gender' and col != 'Result':
            data[col] = data[col].fillna(data[col].mean())

    # Map Result: 1=Disease, 2=Healthy -> Change 2 to 0 for standard logic
    data['Result'] = data['Result'].map({1: 1, 2: 0})

    # > TRAIN MODEL
    X = data.drop('Result', axis=1)
    y = data['Result']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(max_depth=4, criterion='entropy', random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    
    # > SIDEBAR INPUTS
    st.sidebar.header("Patient Details")
    name = st.sidebar.text_input("Patient Name", "John Doe")
    weight = st.sidebar.number_input("Weight (kg)", 0.0, 200.0, 75.0) # Used for record, not prediction
    
    st.sidebar.header("Clinical Symptoms")
    age = st.sidebar.number_input("Age", 1, 100, 45)
    gender_txt = st.sidebar.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender_txt == "Male" else 0
    
    # Clinical Features
    tb = st.sidebar.number_input("Total Bilirubin", 0.0, 50.0, 0.7)
    db = st.sidebar.number_input("Direct Bilirubin", 0.0, 30.0, 0.1)
    alkphos = st.sidebar.number_input("Alkaline Phosphotase", 0.0, 2000.0, 187.0)
    sgpt = st.sidebar.number_input("Alamine Aminotransferase", 0.0, 2000.0, 16.0)
    sgot = st.sidebar.number_input("Aspartate Aminotransferase", 0.0, 2000.0, 18.0)
    tp = st.sidebar.number_input("Total Proteins", 0.0, 15.0, 6.8)
    alb = st.sidebar.number_input("Albumin", 0.0, 10.0, 3.3)
    ag = st.sidebar.number_input("A/G Ratio", 0.0, 5.0, 0.9)

    # > PREDICTION
    st.divider()
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Patient Record")
        st.write(f"**Name:** {name}")
        st.write(f"**Weight:** {weight} kg")
        st.write(f"**Gender:** {gender_txt}")
        
    with col2:
        if st.button("🔍 Run Diagnosis", type="primary"):
            input_data = [[age, gender, tb, db, alkphos, sgpt, sgot, tp, alb, ag]]
            prediction = model.predict(input_data)[0]
            
            if prediction == 1:
                st.error("### 🔴 Result: LIVER DISEASE DETECTED")
                st.write("The symptoms match patterns associated with liver pathology.")
            else:
                st.success("### 🟢 Result: NO DISEASE DETECTED")
                st.write("The symptoms match patterns of a healthy liver.")

    # > DECISION TREE VISUALIZATION
    st.divider()
    st.subheader("Visualization of Decision Logic")
    if st.checkbox("Show Decision Tree"):
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(model, feature_names=X.columns, class_names=['Healthy', 'Disease'], filled=True, fontsize=10)
        st.pyplot(fig)

else:
    st.info("Waiting for dataset... Upload your CSV file to begin.")