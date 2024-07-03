from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load historical match data
data = pd.read_csv('historical_matches.csv')

# Preprocess data
X = data.drop('radiant_win', axis=1)
y = data['radiant_win']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
import joblib
joblib.dump(model, 'model.pkl')
