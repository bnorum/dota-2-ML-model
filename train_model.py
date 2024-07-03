import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay
import joblib
import os
import matplotlib.pyplot as plt

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
    features = {
        'duration': int(match_data.get('duration', 0)),
        'radiant_score': int(match_data.get('radiant_score', 0)),
        'dire_score': int(match_data.get('dire_score', 0)),
        'radiant_win': int(match_data.get('radiant_win', False)),
        # Add more features as needed
    }
    return features

# Convert list of match data to DataFrame
def preprocess_detailed_matches(detailed_matches):
    preprocessed_data = [preprocess_match_data(match) for match in detailed_matches]
    df = pd.DataFrame(preprocessed_data)
    # Ensure 'radiant_win' column is not included in feature columns
    if 'radiant_win' not in df.columns:
        raise KeyError("'radiant_win' column is missing in the preprocessed data")
    print(f"DataFrame columns: {df.columns}")
    print(f"First few rows:\n{df.head()}")
    return df

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

            # Train the model
            model = RandomForestClassifier()
            model.fit(X, y)

            # Save the trained model
            model_path = 'app/models/model.pkl'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)

            # Verify the model is saved correctly
            if os.path.getsize(model_path) == 0:
                print("The model file is empty!")
            else:
                print("The model file is not empty.")

            y_pred = model.predict(X)
            y_pred_prob = model.predict_proba(X)[:, 1]

            # Plot confusion matrix
            cm = confusion_matrix(y, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.title("Confusion Matrix")
            plt.savefig('confusion_matrix.png')  # Save plot as image
            plt.close()

            # Plot ROC curve
            fpr, tpr, _ = roc_curve(y, y_pred_prob)
            roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr)
            roc_disp.plot()
            plt.title("ROC Curve")
            plt.savefig('roc_curve.png')  # Save plot as image
            plt.close()

# To use the model for predictions, make sure you do not include the target variable in the features
def make_prediction(model, data):
    data_df = pd.DataFrame([data])
    X = data_df.drop(columns=['radiant_win'], errors='ignore')  # Ensure 'radiant_win' is not in features
    prediction = model.predict(X)
    return prediction

# Load the model and make a prediction
model_path = 'app/models/model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    sample_data = {
        'duration': 3600,
        'radiant_score': 30,
        'dire_score': 25
        # Add more sample data as needed
    }
    prediction = make_prediction(model, sample_data)
    print(f"Prediction: {prediction}")
else:
    print("Model file does not exist.")
