
# Import necessary libraries
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app with a name
superkart_api = Flask("SuperKart")

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the serialized Random Forest model
model_path = os.path.join(BASE_DIR, "superkart_model.joblib")
model = joblib.load(model_path)


# Define a route for the home page
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart System"


# Define an endpoint to predict sales for a single product
@superkart_api.post('/v1/predict')
def predict_sales():

    # Get JSON data from the request
    data = request.get_json()

    # Extract the required features from the input data
    sample = {
        'Product_Weight': data['Product_Weight'],
        'Product_Sugar_Content': data['Product_Sugar_Content'],
        'Product_Allocated_Area': data['Product_Allocated_Area'],
        'Product_MRP': data['Product_MRP'],
        'Store_Size': data['Store_Size'],
        'Store_Location_City_Type': data['Store_Location_City_Type'],
        'Store_Type': data['Store_Type'],
        'Product_Id_char': data['Product_Id_char'],
        'Store_Age_Years': data['Store_Age_Years'],
        'Product_Type_Category': data['Product_Type_Category']
    }

    # Convert the input data into a DataFrame
    input_data = pd.DataFrame([sample])

    # Generate the sales prediction
    prediction = model.predict(input_data).tolist()[0]

    # Return the prediction as a JSON response
    return jsonify({'Sales': prediction})


# Define an endpoint to predict sales for multiple products
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():

    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a DataFrame
    input_data = pd.read_csv(file)

    # Generate predictions for all products
    predictions = model.predict(input_data).tolist()

    # Create a dictionary containing the predictions
    output_dict = {
        str(i): round(pred, 2)
        for i, pred in enumerate(predictions)
    }

    return output_dict


# Run the Flask application
if __name__ == '__main__':
    superkart_api.run(debug=True)
