from flask import Flask, render_template, request
import joblib
import pandas as pd
import logging
import requests

def fetch_heroes():
    url = "https://api.opendota.com/api/heroes"
    try:
        response = requests.get(url)
        response.raise_for_status()
        heroes = response.json()
        hero_dict = {hero['id']: hero['localized_name'] for hero in heroes}
        return hero_dict
    except requests.RequestException as e:
        print(f"Error fetching heroes: {e}")
        return {}

app = Flask(__name__)

# Load the model and columns
model = joblib.load('app/models/model.pkl')
with open('app/models/columns.pkl', 'rb') as f:
    columns = joblib.load(f)

# Fetch and store heroes
heroes = fetch_heroes()

@app.route('/')
def index():
    return render_template('index.html', heroes=heroes)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract selected heroes from the form
    radiant_heroes = [int(request.form[f'radiant_hero_{i}']) for i in range(5)]
    dire_heroes = [int(request.form[f'dire_hero_{i}']) for i in range(5)]
    
    # Create the match data dictionary
    match_data = {f'radiant_hero_{i}': radiant_heroes[i] for i in range(5)}
    match_data.update({f'dire_hero_{i}': dire_heroes[i] for i in range(5)})

    # Ensure all columns are present
    data_df = pd.DataFrame([match_data])
    X = data_df.reindex(columns=columns, fill_value=0)
    
    prediction = model.predict(X)
    result = "Radiant Win" if prediction[0] == 1 else "Radiant Loss"
    
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
