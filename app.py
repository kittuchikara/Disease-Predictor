from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import os
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load pre-trained models
diabetes_model = pickle.load(open('models/diabetes.pkl', 'rb'))
heart_model = pickle.load(open('models/heart.pkl', 'rb'))

# User database
USER_DB = 'users.json'
if not os.path.exists(USER_DB):
    with open(USER_DB, 'w') as f:
        json.dump({}, f)

# Helper functions for user management
def load_users():
    with open(USER_DB, 'r') as f:
        return json.load(f)

def save_user(username, password):
    users = load_users()
    users[username] = password
    with open(USER_DB, 'w') as f:
        json.dump(users, f)

# ----------------- Routes ------------------

@app.route('/')
def home():
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        if username in users and users[username] == password:
            session['user'] = username
            return redirect('/dashboard')
        else:
            error = 'Invalid Credentials'
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        if username in users:
            error = 'User already exists'
        else:
            save_user(username, password)
            return redirect('/login')
    return render_template('register.html', error=error)

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/login')
    return render_template('dashboard.html')

@app.route('/predict_diabetes', methods=['GET', 'POST'])
def predict_diabetes():
    if 'user' not in session:
        return redirect('/login')
    
    if request.method == 'POST':
        # Ordered input features for diabetes model
        feature_order = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
        features = [float(request.form[f]) for f in feature_order]
        pred = diabetes_model.predict([features])[0]
        prob = diabetes_model.predict_proba([features])[0][1]

        return render_template(
            'result.html',
            disease="Diabetes",
            prediction=pred,
            probability=prob
        )

    return render_template('diabetes.html')

@app.route('/predict_heart', methods=['GET', 'POST'])
def predict_heart():
    if 'user' not in session:
        return redirect('/login')
    
    if request.method == 'POST':
        # Ordered input features for heart model
        feature_order = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]
        features = [float(request.form[f]) for f in feature_order]
        pred = heart_model.predict([features])[0]
        prob = heart_model.predict_proba([features])[0][1]

        return render_template(
            'result.html',
            disease="Heart Disease",
            prediction=pred,
            probability=prob
        )

    return render_template('heart.html')

@app.route('/features')
def features():
    if 'user' not in session:
        return redirect('/login')
    return render_template('features.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

# ----------------- Run Server ------------------

if __name__ == '__main__':
    app.run(debug=True) 