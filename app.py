from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model = joblib.load('linear_regression_fish_model.joblib')

# Initialize the scaler (it should be the same as used in training)
scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    length1 = float(request.form['length1'])
    length2 = float(request.form['length2'])
    length3 = float(request.form['length3'])
    height = float(request.form['height'])
    width = float(request.form['width'])
    species = request.form['species']
    
    # Create a DataFrame for the input
    input_data = pd.DataFrame([[length1, length2, length3, height, width, species]], 
                              columns=['Length1', 'Length2', 'Length3', 'Height', 'Width', 'Species'])
    
    # One-hot encode the species
    species_dummies = pd.get_dummies(input_data['Species'], prefix='Species')
    input_data = pd.concat([input_data.drop('Species', axis=1), species_dummies], axis=1)
    
    # Ensure all species columns are present and in the correct order
    all_species = ['Species_Bream', 'Species_Parkki', 'Species_Perch', 'Species_Pike', 'Species_Roach', 'Species_Smelt', 'Species_Whitefish']
    for species in all_species:
        if species not in input_data.columns:
            input_data[species] = 0
    input_data = input_data[['Length1', 'Length2', 'Length3', 'Height', 'Width'] + all_species]
    
    # Scale the numerical features
    numerical_features = ['Length1', 'Length2', 'Length3', 'Height', 'Width']
    input_data[numerical_features] = scaler.fit_transform(input_data[numerical_features])
    
    # Predict the weight
    prediction = model.predict(input_data)[0]
    
    # Render the template with the prediction
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
