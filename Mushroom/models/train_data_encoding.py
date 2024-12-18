import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib


# Load your train_data
train_data = pd.read_csv('../csv_files/dataset/train_mushroom.csv')

# Initialize a LabelEncoder for each feature
encoders = {}
for column in train_data.columns:
    le = LabelEncoder()
    train_data[column] = le.fit_transform(train_data[column])  # Fit and transform each column
    encoders[column] = le  # Store the encoder if needed for inverse transformation later

# Save the encoded dataset
train_data.to_csv('../csv_files/dataset/train_mushroom_encoded.csv', index=False)

# Optionally save encoders for future use

joblib.dump(encoders, '../pkl_files/encoders.pkl')  # Save encoders for later
