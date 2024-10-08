import requests
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import numpy as np

logging.basicConfig(level=logging.DEBUG)

# Fetch match data from OpenDota API
def fetch_match_data(match_id):
    url = f"https://api.opendota.com/api/matches/{match_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        logging.debug(f"Fetched match data: {response.json()}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching match data: {e}")
        return None

# Preprocess match data
def preprocess_match_data(match_data):
    try:
        logging.debug(f"Preprocessing match data: {match_data}")
        players = match_data.get('players', [])

        # Extract Radiant and Dire heroes
        radiant_heroes = [int(player['hero_id']) for player in players if player['isRadiant']]
        dire_heroes = [int(player['hero_id']) for player in players if not player['isRadiant']]

        # Check if Radiant won the match
        radiant_win = int(match_data.get('radiant_win', 0))  # Ensure it's an integer

        # Construct the result dictionary with hero columns
        features = {f'radiant_hero_{i}': (hero_id if i < len(radiant_heroes) else 0) 
                    for i, hero_id in enumerate(radiant_heroes)}
        features.update({f'dire_hero_{i}': (hero_id if i < len(dire_heroes) else 0) 
                         for i, hero_id in enumerate(dire_heroes)})
        
        features['radiant_win'] = radiant_win

        logging.debug(f"Preprocessed match features: {features}")
        return features
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        return None

# Convert list of match data to DataFrame
def preprocess_detailed_matches(detailed_matches):
    preprocessed_data = [preprocess_match_data(match) for match in detailed_matches if match]

    if not preprocessed_data:
        return pd.DataFrame()  # Return an empty DataFrame if no data

    df = pd.DataFrame(preprocessed_data)
    
    # Identify all possible hero columns to ensure consistency
    radiant_columns = {col for col in df.columns if col.startswith('radiant_hero_')}
    dire_columns = {col for col in df.columns if col.startswith('dire_hero_')}
    
    all_columns = list(radiant_columns | dire_columns)
    
    # Create a DataFrame with all possible columns and fill missing columns with zeros
    final_df = pd.DataFrame(columns=all_columns + ['radiant_win'])
    final_df = pd.concat([final_df, df], axis=0, ignore_index=True).fillna(0)

    logging.debug(f"DataFrame columns: {final_df.columns}")
    logging.debug(f"First few rows:\n{final_df.head()}")
    return final_df

# Train the model and save columns
def train_model(detailed_matches):
    preprocessed_matches = preprocess_detailed_matches(detailed_matches)
    
    if preprocessed_matches.empty:
        logging.error("The DataFrame is empty. No data to train the model.")
        return

    # Separate features and target
    X = preprocessed_matches.drop(columns=['radiant_win'])
    y = preprocessed_matches['radiant_win']

    logging.debug(f"Training DataFrame columns: {X.columns}")
    logging.debug(f"Training DataFrame first few rows:\n{X.head()}")

    model = RandomForestClassifier()
    model.fit(X, y)

    # Save the trained model and the columns
    model_path = 'app/models/model.pkl'
    columns_path = 'app/models/columns.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(X.columns, columns_path)

    logging.info("Model and columns saved successfully.")

# Load the model and make a prediction
def make_prediction(data):
    model_path = 'app/models/model.pkl'
    columns_path = 'app/models/columns.pkl'

    if not (os.path.exists(model_path) and os.path.exists(columns_path)):
        logging.error("Model or columns file does not exist.")
        return None

    model = joblib.load(model_path)
    columns = joblib.load(columns_path)

    data_df = pd.DataFrame([data])
    X = data_df.drop(columns=['radiant_win'], errors='ignore').reindex(columns=columns, fill_value=0)

    logging.debug(f"Prediction DataFrame columns: {X.columns}")
    logging.debug(f"Prediction DataFrame first few rows:\n{X.head()}")

    prediction = model.predict(X)
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()  # Convert ndarray to list for JSON serialization
    return prediction

# Example usage:
# Fetch match data
match_ids = [12345, 67890]  # Example match IDs
detailed_matches = [fetch_match_data(match_id) for match_id in match_ids]
train_model(detailed_matches)

# To make a prediction:
sample_data = {'radiant_hero_0': 1, 'radiant_hero_1': 2, 'radiant_hero_2': 3, 'radiant_hero_3': 4, 'radiant_hero_4': 5, 'dire_hero_0': 6, 'dire_hero_1': 7, 'dire_hero_2': 8, 'dire_hero_3': 9, 'dire_hero_4': 10}
prediction = make_prediction(sample_data)
if prediction is not None:
    print(f"Prediction: {prediction}")
else:
    print("Failed to make a prediction.")
