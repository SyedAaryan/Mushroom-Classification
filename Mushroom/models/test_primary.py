import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the encoded label encoders
label_encoders = joblib.load('../pkl_files/encoders.pkl')

# Load the encoded test dataset
test_data = pd.read_csv('../csv_files/dataset/test_mushroom_encoded.csv')

# Separate features (X) and target variable (y)
X_test = test_data.drop(columns=['class'])  # Features
y_test = test_data['class']  # Target

# Load the trained model
rf_class = joblib.load('../pkl_files/rf_class_model.pkl')

# Predict using the test dataset
y_pred = rf_class.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy:.4f}")
