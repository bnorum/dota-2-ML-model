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
        hero_dict = {hero['id']: {
            'name': hero['localized_name'],
            'image': f"https://dotabase.dillerm.io/vpk/panorama/images/heroes/{hero['name']}_png.png"
        } for hero in heroes}
        return hero_dict
    except requests.RequestException as e:
        print(f"Error fetching heroes: {e}")
        return {}

app = Flask(__name__)

# Load the model and columns
model = joblib.load('models/model.pkl')
with open('models/columns.pkl', 'rb') as f:
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
    
    # Get the probability prediction
    prediction_proba = model.predict_proba(X)
    radiant_win_proba = prediction_proba[0][1]  # Probability of radiant win
    radiant_win_percent = radiant_win_proba * 100

    return render_template('result.html', prediction=radiant_win_percent)
#predict
if __name__ == '__main__':
    app.run(debug=True)
