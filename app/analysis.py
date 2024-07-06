import joblib
import logging
import pandas as pd
from app.utils import preprocess_match_data

model_path = 'app/models/model.pkl'
model = None

try:
    model = joblib.load(model_path)
    logging.info("Model loaded successfully.")
except FileNotFoundError:
    logging.error("Model file not found.")
except EOFError:
    logging.error("EOFError: The model file might be empty or corrupted.")
except ImportError as e:
    logging.error(f"ImportError: {e}")
except Exception as e:
    logging.error(f"An error occurred while loading the model: {e}")

def analyze_match(match_data):
    if model is None:
        return {'error': 'Model could not be loaded'}

    try:
        # Preprocess match data
        preprocessed_data = preprocess_match_data(match_data)
        if preprocessed_data is None:
            return {'error': 'Preprocessing failed'}

        # Convert dictionary to DataFrame
        preprocessed_df = pd.DataFrame([preprocessed_data])

        # Ensure the DataFrame only contains feature columns (no target columns)
        if 'radiant_win' in preprocessed_df.columns:
            preprocessed_df = preprocessed_df.drop(columns=['radiant_win'])

        # Predict using the model
        prediction = model.predict(preprocessed_df)
        return {'radiant_win': int(prediction[0])}
    except Exception as e:
        return {'error': f'Analysis failed: {e}'}
