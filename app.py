import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Fix for running on servers/headless
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import io
import base64

app = Flask(__name__)

# --- GLOBAL VARIABLES ---
model = None
tree_image_data = None

def init_app():
    global model, tree_image_data
    print("Loading dataset and training model...")

    try:
        # 1. Load Data
        df = pd.read_csv('Liver Patient Dataset (LPD)_train.csv', encoding='latin1')

        # 2. Clean Columns
        # Rename strictly by index to avoid special character issues
        df.columns = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
                      'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 
                      'Aspartate_Aminotransferase', 'Total_Protiens', 
                      'Albumin', 'Albumin_and_Globulin_Ratio', 'Result']

        # 3. Handle Missing Values
        numerical_cols = df.select_dtypes(include=['number']).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
        
        # Gender handling
        df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

        # 4. Prepare Target (1=Disease, 0=Healthy)
        # Original: 1=Disease, 2=No Disease. Map 2 -> 0.
        df['Result'] = df['Result'].map({1: 1, 2: 0})

        X = df.drop('Result', axis=1)
        y = df['Result']
        feature_names = X.columns.tolist()

        # 5. Train Model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"Model Trained! Accuracy: {acc*100:.2f}%")

        # 6. Generate Decision Tree Image (In-Memory)
        print("Generating Decision Tree visualization...")
        plt.figure(figsize=(25, 12)) # Large size for readability
        plot_tree(model, 
                  feature_names=feature_names, 
                  class_names=['Healthy', 'Disease'], 
                  filled=True, 
                  rounded=True, 
                  fontsize=10)
        
        # Save to a bytes buffer instead of a file
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)
        
        # Encode to base64 string
        tree_image_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        print("Visualization generated successfully.")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        exit()

# Initialize on startup
init_app()

# --- ROUTES ---

@app.route('/')
def home():
    # Pass the image data to the template
    return render_template('index.html', tree_image=tree_image_data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form['name']
        
        # Collect Inputs
        features = [
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
        
        prediction = model.predict([features])[0]

        if prediction == 1:
            result_text = "LIVER DISEASE DETECTED"
            color = "#e74c3c" # Red
            advice = "The system has detected patterns matching liver disease. Please consult a doctor."
        else:
            result_text = "NO LIVER DISEASE DETECTED"
            color = "#27ae60" # Green
            advice = "The system did not detect liver disease patterns."

        return render_template('index.html', 
                               prediction_text=result_text, 
                               name=name,
                               color=color,
                               advice=advice,
                               tree_image=tree_image_data, # Keep showing tree
                               scroll="result")

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)