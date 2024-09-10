from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
rf_model = pickle.load(open('random_forest_model1.pkl', 'rb'))
scaler = pickle.load(open('scaler1.pkl', 'rb'))

# Load the preprocessed DataFrame for column reference (exclude 'Price_new')
preprocessed_df = pd.read_csv('lagos_rent_processed.csv')
model_columns = [col for col in preprocessed_df.columns if col != 'Price_new']

# List of cities for dropdown
cities = [
    'Lekki', 'Surulere', 'Ipaja', 'Yaba', 'Ojodu', 'Ikeja', 'Ikoyi', 'Ajah',
    'Bariga', 'Victoria Island', 'Ikosi', 'Shomolu', 'Gbagada', 'Sangotedo',
    'Isolo', 'Ogudu', 'Okota', 'Alimosho', 'Abule Egba', 'Idimu', 'Ogba',
    'Onikan Island', 'Maryland', 'Iju', 'Ojota', 'Odofin', 'Ketu', 'Tbs, Island',
    'Ikorodu', 'Agege', 'Igando', 'Oshodi', 'Ilupeju', 'Orile', 'Mushin',
    'Ejigbo', 'Badagry', 'Apapa', 'Marina Island', 'C.m.s Island', 'Ojo', 'Island Island'
]

# Home route to render form
@app.route('/')
def home():
    return render_template('home.html', cities=cities)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    toilets = int(request.form['toilets'])
    status_list = request.form.getlist('status')  # Retrieve the list of selected statuses
    city = request.form['city']

    # Create the binary features for 'Newly Built', 'Furnished', 'Serviced'
    newly_built = 1 if 'Newly Built' in status_list else 0
    furnished = 1 if 'Furnished' in status_list else 0
    serviced = 1 if 'Serviced' in status_list else 0

    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'Bedroom': [bedrooms],
        'Bathroom': [bathrooms],
        'Toilet': [toilets],
        'Newly Built': [newly_built],
        'Furnished': [furnished],
        'Serviced': [serviced]
    })

    # One-hot encode the 'city' column
    city_dummies = pd.get_dummies([city], prefix='City')

    # Reindex the one-hot encoded cities to match the original columns from the preprocessed DataFrame
    city_dummies = city_dummies.reindex(columns=[col for col in model_columns if col.startswith('City_')], fill_value=0)

    # Combine the input data with the one-hot encoded city
    input_data = pd.concat([input_data, city_dummies], axis=1)

    # Ensure the input_data has all necessary columns that the model expects
    for col in model_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Ensure input_data has no extra columns and matches model expectations
    input_data = input_data[model_columns]

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = rf_model.predict(input_data_scaled)

    # Format the predicted price
    predicted_price = f'₦{int(prediction[0]):,}'

    # Return prediction result
    return jsonify({'price': predicted_price})

if __name__ == "__main__":
    app.run(debug=True)