from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import load_model
import joblib

# Inisialisasi Flask
app = Flask(__name__)

# Load the pre-trained model and preprocessor
model = load_model('model.h5')
preprocessor = joblib.load('pipeline.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()
    print("Received data:", data)  # Debugging line

    # Validate input data
    if 'instances' not in data:
        return jsonify({'error': 'Invalid input data'})

    instances = data['instances']
    print("Instances:", instances)  # Debugging line

    # Convert the input data to DataFrame
    column_names = [
        'Total Household Income', 
        'Household Head Sex', 
        'Household Head Age', 
        'Household Head Marital Status', 
        'Household Head Highest Grade Completed (Simplified)',
        'Region Category', 
        'Type of Building/House', 
        'House Floor Area',
        'Number of bedrooms', 
        'Electricity', 
        'Tenure Status', 
        'Type of Household',
        'Total Number of Family members', 
        'Total number of family members employed', 
        'Number of Kids', 
        'Number of Vehicles',
        'Number of Communication Devices',
        'Number of Electronics'
    ]
    df_instances = pd.DataFrame(instances, columns=column_names)
    print("DataFrame Instances:", df_instances)  # Debugging line

    # Preprocess the input data
    try:
        X = preprocessor.transform(df_instances)
        print("Transformed data:", X)  # Debugging line
    except Exception as e:
        print("Error in transformation:", e)
        return jsonify({'error': str(e)})

    # Make predictions
    try:
        predictions = model.predict(X)
        print("Predictions:", predictions)  # Debugging line
    except Exception as e:
        print("Error in prediction:", e)
        return jsonify({'error': str(e)})

    # Format predictions as JSON and return
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)