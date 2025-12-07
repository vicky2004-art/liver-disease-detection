import os
import io
import base64
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

# --- MATPLOTLIB SETUP (Must be first) ---
import matplotlib
matplotlib.use('Agg') # Prevents crashes on Windows/Mac
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# --- CONFIGURATION ---
DATASET_NAME = 'Liver Patient Dataset (LPD)_train.csv'

# Global variables
model = None
tree_image_data = None

def init_app():
    global model, tree_image_data
    
    # 1. CHECK FILE EXISTENCE
    if not os.path.exists(DATASET_NAME):
        print(f"\nCRITICAL ERROR: The file '{DATASET_NAME}' was not found.")
        print(f"Current working directory is: {os.getcwd()}")
        print("Please move the CSV file into this folder.\n")
        return # Stop here if no file

    print("Loading dataset...")
    try:
        # 2. LOAD & CLEAN DATA
        df = pd.read_csv(DATASET_NAME, encoding='latin1')
        
        # Rename columns standardly
        df.columns = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
                      'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 
                      'Aspartate_Aminotransferase', 'Total_Protiens', 
                      'Albumin', 'Albumin_and_Globulin_Ratio', 'Result']

        # Fill missing values
        numerical_cols = df.select_dtypes(include=['number']).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
        df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        df['Result'] = df['Result'].map({1: 1, 2: 0}) # 1=Disease, 0=Healthy

        # 3. TRAIN MODEL
        X = df.drop('Result', axis=1)
        y = df['Result']
        feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        print("Model trained successfully.")

        # 4. GENERATE IMAGE (Wrapped in try-catch so it doesn't crash app)
        try:
            print("Generating tree diagram...")
            plt.figure(figsize=(20, 10))
            plot_tree(model, feature_names=feature_names, class_names=['Healthy', 'Disease'], filled=True, fontsize=10)
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight')
            plt.close()
            img_buffer.seek(0)
            tree_image_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            print("Tree diagram ready.")
        except Exception as img_error:
            print(f"Warning: Could not generate image. {img_error}")
            tree_image_data = None # Website will work without image

    except Exception as e:
        print(f"Error processing data: {e}")

# Run initialization
init_app()

@app.route('/')
def home():
    if model is None:
        return "<h1>Error: Model failed to load. Check terminal for details.</h1>"
    return render_template('index.html', tree_image=tree_image_data)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded."
        
    try:
        # Gather inputs
        inputs = [
            float(request.form['age']),
            int(request.form['gender']),
            float(request.form['total_bilirubin']),
            float(request.form['direct_bilirubin']),
            float(request.form['alkphos']),
            float(request.form['sgpt']),
            float(request.form['sgot']),
            float(request.form['total_protiens']),
            float(request.form['albumin']),
            float(request.form['ag_ratio'])
        ]
        
        prediction = model.predict([inputs])[0]
        name = request.form['name']

        if prediction == 1:
            res = "LIVER DISEASE DETECTED"
            col = "#e74c3c"
            adv = "High probability of liver disease."
        else:
            res = "NO LIVER DISEASE DETECTED"
            col = "#27ae60"
            adv = "No signs of disease detected."

        return render_template('index.html', prediction_text=res, name=name, color=col, advice=adv, tree_image=tree_image_data, scroll="result")

    except Exception as e:
        return f"Prediction Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)