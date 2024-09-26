import pandas as pd
import joblib
import numpy as np

# Example user input list (should be a list of lists where each list is a row)
user_input_list = [
    ['b', 's', 'n', 'i', 'n', 'u', 'n', 't', 'e'],  # Example input
    # Add more rows as needed
]

# Define the column names to match your model's expected input
columns = ['cap-shape', 'Cap-surface', 'cap-color', 'stem-surface', 'stem-color', 'veil-type', 'veil-color', 'has-ring', 'ring-type']

# Check the shape of user_input_list and the columns
print(f"Shape of user_input_list: {len(user_input_list)} x {len(user_input_list[0]) if user_input_list else 0}")
print(f"Number of columns: {len(columns)}")

# Convert the list to a DataFrame
if all(len(row) == len(columns) for row in user_input_list):
    df_input = pd.DataFrame(user_input_list, columns=columns)
else:
    raise ValueError("Mismatch between the number of columns and the length of input rows.")

# Load the trained models and label encoders
clf_family = joblib.load('../pkl_files/family_clf_model.pkl')
clf_name = joblib.load('../pkl_files/name_clf_model.pkl')
clf_class = joblib.load('../pkl_files/class_clf_model.pkl')
label_encoders = joblib.load('../pkl_files/primary_label_encoders.pkl')

# Encode categorical features using the same encoders from the primary dataset
for column in df_input.columns:
    if df_input[column].dtype == object:
        le = label_encoders.get(column)
        if le:
            df_input[column] = le.transform(df_input[column])

# Separate features (X) for predicting family, name, and class
X_input = df_input.copy()  # Copy to ensure no modifications to the original DataFrame

# Predict the family, name, and class
predicted_family = clf_family.predict(X_input)
predicted_name = clf_name.predict(X_input)
predicted_class = clf_class.predict(X_input)

# Decode the predictions back to their original labels
df_input['family'] = label_encoders['family'].inverse_transform(predicted_family)
df_input['name'] = label_encoders['name'].inverse_transform(predicted_name)
df_input['predicted_class'] = label_encoders['class'].inverse_transform(predicted_class)

# Save the predictions to a CSV file
df_input.to_csv('../csv_files/results/predicted_user_input.csv', index=False)

print("Predictions complete and saved.")
