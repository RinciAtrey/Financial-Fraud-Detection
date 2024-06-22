from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS
import zipfile
import csv
import os
import pyrebase
import numpy as np
import pickle

# Firebase configuration
config = {
    "apiKey": "AIzaSyC_gy00kx7DTIobe791VqHHmqx-XS-yI9A",
    "authDomain": "kriyeta-3e235.firebaseapp.com",
    "projectId": "kriyeta-3e235",
    "databaseURL": "https://kriyeta-3e235-default-rtdb.firebaseio.com",
    "storageBucket": "kriyeta-3e235.appspot.com",
    "messagingSenderId": "575633214148",
    "appId": "1:575633214148:web:27ec985b2db18bc2819791",
    "measurementId": "G-GZEPJNDXFP"
  }


# Initialize Firebase
firebase = pyrebase.initialize_app(config)
db = firebase.database()

# Load the trained model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    extract_dir = "extracted_files"
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    try:
        # Extract the uploaded zip file
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            # List the contents of the zip file
            zip_contents = zip_ref.namelist()
            print("Contents of the zip file:")
            for item in zip_contents:
                print(item)

        # Find the extracted CSV file
        extracted_files = os.listdir(extract_dir)
        csv_file = None
        for file in extracted_files:
            if file.endswith('.csv'):
                csv_file = os.path.join(extract_dir, file)
                break

        if csv_file:
            # Read the CSV file
            with open(csv_file, newline='') as csvfile:
                csv_reader = csv.reader(csvfile)
                header = next(csv_reader)  # Read the header row
                print("Header:", header)

                # Assuming 'isFraud' is one of the header columns
                is_fraud_index = header.index('isFraud')

                for row in csv_reader:
                    # Prepare the row data for prediction
                    row_data = np.array(row[:-1], dtype=float).reshape(1, -1)  # Exclude the 'isFraud' column for prediction
                    row_data_scaled = scaler.transform(row_data)

                    # Make prediction
                    prediction = model.predict(row_data_scaled)

                    if prediction == 0:  # Check if isFraud is 0
                        data_dict = dict(zip(header, row))
                        db.child("fraudulent_transactions").push(data_dict)
                        print("Pushed to Firebase:", data_dict)

            return "Data has been successfully stored in Firebase.", 200
        else:
            return "No CSV file found in the zip archive.", 400

    except Exception as e:
        print(e)
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)



