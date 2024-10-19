# Load the encoded label encoders
import joblib
import pandas as pd

label_encoders = joblib.load('../pkl_files/encoders.pkl')

# Load the test dataset
test_data = pd.read_csv('../csv_files/dataset/test_mushroom.csv')

# Encode categorical features in the test dataset using the same encoders
for column in test_data.columns:
    if column in label_encoders:  # Ensure the column is in the label_encoders
        le = label_encoders[column]
        test_data[column] = le.transform(test_data[column])

# Save the encoded test_data to a CSV file
encoded_test_file_path = '../csv_files/dataset/test_mushroom_encoded.csv'
test_data.to_csv(encoded_test_file_path, index=False)

# Print a message confirming the save
print(f"Encoded test data saved to {encoded_test_file_path}")
