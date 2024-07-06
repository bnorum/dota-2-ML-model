import pandas as pd
import joblib
import os
import logging

logging.basicConfig(level=logging.DEBUG)

# Load the model and the columns
model_path = 'app/models/model.pkl'
columns_path = 'app/models/columns.pkl'

if os.path.exists(model_path) and os.path.exists(columns_path):
    model = joblib.load(model_path)
    columns = joblib.load(columns_path)
    logging.debug("Model and columns loaded successfully.")
else:
    print("Model or columns file does not exist.")
    model, columns = None, None

# To use the model for predictions, make sure you do not include the target variable in the features
def make_prediction(model, data):
    data_df = pd.DataFrame([data])
    
    # Ensure 'radiant_win' is not in features and use the same columns as during training
    X = data_df.drop(columns=['radiant_win'], errors='ignore').reindex(columns=columns, fill_value=0)

    logging.debug(f"Prediction DataFrame columns: {X.columns}")
    logging.debug(f"Prediction DataFrame first few rows:\n{X.head()}")

    prediction = model.predict(X)
    return prediction

# Example usage:
# data = {...}  # Your input data here
# if model:
#     prediction = make_prediction(model, data)
#     print(f"Prediction: {prediction}")
