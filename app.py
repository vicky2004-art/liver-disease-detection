import os
import io
import base64
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

# --- MATPLOTLIB SETUP ---
# This fixes "RuntimeError: main thread is not in main loop"
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# --- GLOBAL VARIABLES ---
model = None
tree_image_data = None
DATASET_FILE = 'Liver Patient Dataset (LPD)_train.csv'
STATUS_MESSAGE = ""

def load_and_train():
    global model, tree_image_data, STATUS_MESSAGE
    
    print("-" * 50)
    print("STARTING APP INITIALIZATION...")

    # 1. CHECK IF FILE EXISTS
    if not os.path.exists(DATASET_FILE):
        STATUS_MESSAGE = f"ERROR: '{DATASET_FILE}' not found. Using DUMMY data."
        print(f"!!! {STATUS_MESSAGE}")
        df = generate_dummy_data()
    else:
        try:
            print(f"Loading dataset: {DATASET_FILE}")
            df = pd.read_csv(DATASET_FILE, encoding='latin1')
            
            # RENAME COLUMNS (Sanitize names)
            df.columns = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
                          'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 
                          'Aspartate_Aminotransferase', 'Total_Protiens', 
                          'Albumin', 'Albumin_and_Globulin_Ratio', 'Result']
            
            # CLEANING
            # 1. Fill Numerical NaNs with Mean
            num_cols = df.select_dtypes(include=['number']).columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
            
            # 2. Fill Categorical NaNs and Map Gender
            df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
            df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
            
            # 3. Handle Target (1=Disease, 2=No Disease -> 1=Disease, 0=No Disease)
            df['Result'] = df['Result'].map({1: 1, 2: 0})
            
            STATUS_MESSAGE = "Model trained on REAL dataset."
            print("Dataset loaded and cleaned successfully.")

        except Exception as e:
            STATUS_MESSAGE = f"Data Error: {str(e)}. Using DUMMY data."
            print(f"!!! {STATUS_MESSAGE}")
            df = generate_dummy_data()

    # 2. TRAIN MODEL
    try:
        X = df.drop('Result', axis=1)
        y = df['Result']
        
        # Check if X has any remaining NaNs (fail-safe)
        X = X.fillna(0) 

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"Model Training Complete. Accuracy: {acc*100:.2f}%")
        
        # 3. GENERATE IMAGE
        print("Generating Decision Tree Image...")
        feature_names = X.columns.tolist()
        plt.figure(figsize=(20, 10))
        plot_tree(model, feature_names=feature_names, class_names=['Healthy', 'Disease'], filled=True, fontsize=10)
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)
        tree_image_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        print("Image generated.")

    except Exception as e:
        print(f"!!! Training/Plotting Failed: {e}")
        STATUS_MESSAGE += f" | Training Failed: {e}"

def generate_dummy_data():
    # Fallback function to prevent crash if CSV is missing
    print("Generating synthetic backup data...")
    data = []
    for _ in range(200):
        # Create fake healthy/unhealthy people
        age = np.random.randint(20, 80)
        gender = np.random.randint(0, 2)
        tb = np.random.uniform(0.4, 5.0)
        result = 1 if tb > 2.0 else 0 # Simple fake rule
        data.append([age, gender, tb, 0.5, 200, 40, 40, 6.0, 3.0, 1.0, result])
    
    return pd.DataFrame(data, columns=['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
                                       'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 
                                       'Aspartate_Aminotransferase', 'Total_Protiens', 
                                       'Albumin', 'Albumin_and_Globulin_Ratio', 'Result'])

# Initialize App
load_and_train()

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html', tree_image=tree_image_data, status=STATUS_MESSAGE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Error: Model is not loaded. Check server logs."

    try:
        # Get form data
        val_names = ['age', 'gender', 'total_bilirubin', 'direct_bilirubin', 'alkphos', 
                     'sgpt', 'sgot', 'total_protiens', 'albumin', 'ag_ratio']
        
        features = [float(request.form[v]) for v in val_names]
        name = request.form['name']
        
        # Predict
        prediction = model.predict([features])[0]
        
        if prediction == 1:
            res = "LIVER DISEASE DETECTED"
            col = "#e74c3c" # Red
            adv = "High probability of liver disease based on provided symptoms."
        else:
            res = "NO LIVER DISEASE DETECTED"
            col = "#27ae60" # Green
            adv = "Values appear to be within the healthy range."
            
        return render_template('index.html', 
                               prediction_text=res, 
                               name=name, 
                               color=col, 
                               advice=adv, 
                               tree_image=tree_image_data,
                               status=STATUS_MESSAGE,
                               scroll="result")
                               
    except Exception as e:
        return f"Prediction Error: {e}"

if __name__ == '__main__':
    app.run(debug=True, port=5000)