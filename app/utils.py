import requests
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)

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

def preprocess_match_data(match_data):
    try:
        logging.debug(f"Preprocessing match data: {match_data}")
        features = {
            'duration': int(match_data.get('duration', 0)),
            'radiant_score': int(match_data.get('radiant_score', 0)),
            'dire_score': int(match_data.get('dire_score', 0)),
            # Add more features as needed
        }
        return pd.DataFrame([features])
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        return None
