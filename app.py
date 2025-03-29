from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from datetime import datetime
from functools import wraps
from flask_sqlalchemy import SQLAlchemy
import pickle

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# Load the trained model and label encoder
model = joblib.load('model.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Load the data
df = pd.read_csv("cleaned_inventory_data.csv")
# Convert Product ID to string type first
df['Product ID'] = df['Product ID'].astype(str)
# Only transform if not already encoded
if not isinstance(df['Product ID'].iloc[0], (int, np.integer)):
    try:
        df['Product ID'] = label_encoder.transform(df['Product ID'])
    except Exception as e:
        print(f"Warning: Could not transform Product ID: {e}")

# Calculate additional metrics
df['Max Units Required'] = df.apply(lambda x: max(x['Units Sold'], x['Units Ordered'], x['Demand Forecast']), axis=1)

# Simple user database (replace with proper database in production)
users = {}

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    year = db.Column(db.Integer, nullable=False)
    month = db.Column(db.Integer, nullable=False)
    product_id = db.Column(db.String(80), nullable=False)
    prediction = db.Column(db.Float, nullable=False)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@login_required
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Clear any old predictions without timestamps
    if 'predictions' in session:
        session['predictions'] = []
    
    # Get recent predictions
    recent_predictions = Prediction.query.order_by(Prediction.id.desc()).limit(5).all()
    
    # Get unique product IDs for the dropdown
    product_ids = sorted(df['Product ID'].unique().tolist())
    
    # Get latest prediction for the overview section
    latest_prediction = Prediction.query.order_by(Prediction.id.desc()).first()
    latest_value = round(latest_prediction.prediction) if latest_prediction else None
    
    # Prepare data for visualization - get last 12 predictions
    predictions = Prediction.query.order_by(Prediction.id.desc()).limit(12).all()
    viz_data = {
        'labels': [f"{p.month}/{p.year}" for p in predictions][::-1],
        'predictions': [round(p.prediction) for p in predictions][::-1],
        'latest_prediction': latest_value
    }
    
    return render_template('index.html',
                         recent_predictions=recent_predictions,
                         product_ids=product_ids,
                         viz_data=viz_data)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('signup'))
        
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.json
    year = int(data['year'])
    month = int(data['month'])
    product_id = data['product_id']

    try:
        # Transform product_id using label encoder
        encoded_product_id = label_encoder.transform([product_id])[0]
        
        # Make prediction with encoded product ID
        prediction_value = float(model.predict([[year, month, encoded_product_id]])[0])

        # Save prediction to database
        new_prediction = Prediction(
            year=year,
            month=month,
            product_id=product_id,  # Save original product ID
            prediction=prediction_value
        )
        db.session.add(new_prediction)
        db.session.commit()

        return jsonify({'prediction': round(prediction_value, 2)})
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 400

@app.route('/contact')
def contact():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('contact.html')

@app.route('/about')
def about():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('about.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 