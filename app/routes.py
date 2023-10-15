from app import App
from flask import render_template, request, redirect
from deploy.recommender import *
import numpy as np



@App.route("/")
def dashboard():
    home = render_template('index.html')
    return home

@App.route("/predict-from", methods=['POST'])
def predict_form():
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphor'])
    K = int(request.form['potassium'])
    Temp = float(request.form['temp'])
    Humidity = float(request.form['humidity'])
    Ph = float(request.form['ph'])
    Rainfall = float(request.form['rainfall'])
    # Create a dictionary with variable names as keys and values
    data = {
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [Temp],
        'humidity': [Humidity],
        'ph': [Ph],
        'rainfall': [Rainfall]
    }

    # Convert the dictionary to a Pandas DataFrame
    data = pd.DataFrame(data)
    prediction = predictData(trained_model, data)

    # Example usage:  # Replace with your nutrient data (only one value)
    img_base64 = nutrient()

    return render_template('result.html', pred_text=f'{prediction}', accuracy_plot=img_base64)

@App.route('/predict-file', methods=['POST'])
def predict_file():
    if 'csv_file' not in request.files:
        return "No file part"

    file = request.files['csv_file']

    if file.filename == '':
        return "No selected file"

    if file:
        # Process the uploaded CSV file using Pandas
        try:
            df = pd.read_csv(file)
        except pd.errors.EmptyDataError:
            return "The uploaded CSV file is empty or invalid."

        # Extract the data from the CSV and reshape it
        try:
            target = df.values.reshape(1, -1)
        except ValueError:
            return "The CSV file should contain only numerical data."

        # Perform prediction based on the data from the CSV file
        prediction = predictData(trained_model, target)

        # Example usage:  # Replace with your nutrient data (only one value)
        img_base64 = nutrient()

    return render_template('result.html', pred_text=f'{prediction}', accuracy_plot=img_base64)

@App.route('/predict', methods=['POST'])
def predict():
    # Check which method (form or file) was used for submission
    if 'csv_file' in request.files:
        return redirect('/predict-file')
    else:
        return redirect('/predict-form')

@App.route("/database")
def database():
    data = pd.read_csv(path_data)
    return render_template('database.html', database=data)


