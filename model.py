from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load your trained model (you may need to serialize it first if not done already)
model = pickle.load(open('models/random_forest_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data from the request
    input_data = [float(x) for x in request.form.values()]
    
    # Convert the input data into the format expected by the model (e.g., a pandas DataFrame)
    # prediction_input = pd.DataFrame([input_data], columns=['column1', 'column2', ...])
    
    # Predict using the model
    # prediction = model.predict(prediction_input)
    
    return render_template('home.html', prediction_text=f'Predicted Result: {prediction}')

if __name__ == "__main__":
    app.run(debug=True)