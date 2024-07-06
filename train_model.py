import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay
import joblib
import os
import logging

logging.basicConfig(level=logging.DEBUG)

# Fetch recent public matches
def fetch_recent_public_matches(num_matches=100):
    url = "https://api.opendota.com/api/proMatches"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        matches = response.json()
        print(f"Fetched {len(matches)} recent public matches")
        return matches
    except requests.RequestException as e:
        print(f"Error fetching recent public matches: {e}")
        return []

# Fetch match data from OpenDota API
def fetch_match_data(match_id):
    url = f"https://api.opendota.com/api/matches/{match_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        match_data = response.json()
        print(f"Fetched data for match ID: {match_id}")
        return match_data
    except requests.RequestException as e:
        print(f"Error fetching match data for match ID {match_id}: {e}")
        return None

# Fetch detailed match data
def fetch_detailed_matches(match_ids):
    matches = []
    for match_id in match_ids:
        match_data = fetch_match_data(match_id)
        if match_data:
            matches.append(match_data)
    print(f"Fetched detailed data for {len(matches)} matches")
    return matches

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
    logging.debug(f"DataFrame columns: {final_df}")
    return final_df

# Fetch recent public matches
recent_public_matches = fetch_recent_public_matches()

# Check if recent_public_matches is empty
if not recent_public_matches:
    print("No recent public matches found or there was an error.")
else:
    # Extract match IDs
    match_ids = [match['match_id'] for match in recent_public_matches]
    
    # Fetch detailed match data for public matches
    detailed_matches = fetch_detailed_matches(match_ids)

    # Check if detailed_matches is empty
    if not detailed_matches:
        print("No detailed match data found or there was an error.")
    else:
        # Preprocess match data
        preprocessed_matches = preprocess_detailed_matches(detailed_matches)
        # Check if the DataFrame is empty
        if preprocessed_matches.empty:
            print("The DataFrame is empty. No data to train the model.")
        else:
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

            # Verify the model is saved correctly
            if os.path.getsize(model_path) == 0:
                print("The model file is empty!")
            else:
                print("The model file is not empty.")
