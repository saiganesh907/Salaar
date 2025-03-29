# Inventory Demand Predictor

A web application that predicts inventory demand using machine learning. The application features a beautiful and responsive user interface with real-time predictions.

## Features

- User authentication (login/signup)
- Real-time demand prediction
- Beautiful and responsive UI
- Smooth animations and transitions
- Mobile-friendly design

## Setup Instructions

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model and save it:
```bash
python train_model.py
```

4. Run the Flask application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:5000
```

## Project Structure

```
├── app.py                 # Flask application
├── train_model.py         # ML model training script
├── requirements.txt       # Python dependencies
├── static/               # Static files
│   ├── css/
│   │   └── style.css     # Styles
│   └── js/
│       └── main.js       # JavaScript
└── templates/            # HTML templates
    ├── base.html         # Base template
    ├── index.html        # Main dashboard
    ├── login.html        # Login page
    └── signup.html       # Signup page
```

## Usage

1. Sign up for a new account
2. Log in with your credentials
3. Enter the year, month, and product ID
4. Click "Predict Demand" to get the prediction
5. View the predicted demand in units

## Technologies Used

- Python
- Flask
- scikit-learn
- HTML5
- CSS3
- JavaScript
- Pandas
- NumPy 