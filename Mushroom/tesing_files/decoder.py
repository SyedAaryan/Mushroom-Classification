import pandas as pd
import joblib
import numpy as np

# Load the label encoders
label_encoders = joblib.load('../pkl_files/primary_label_encoders.pkl')

# Load the trained models
clf_family = joblib.load('../pkl_files/family_clf_model.pkl')
clf_name = joblib.load('../pkl_files/name_clf_model.pkl')
clf_class = joblib.load('../pkl_files/class_clf_model.pkl')

# Load the encoded dataset with predictions
df_encoded = pd.read_csv('../results/mushroom_secondary_with_predictions.csv')

def decode_column(df, column_name, label_encoders):
    """
    Decode a column using the provided label encoders, handling unseen labels.
    """
    encoder = label_encoders.get(column_name)
    if encoder:
        # Map unseen labels to a default value or filter them out
        unseen_labels = set(df[column_name].unique()) - set(encoder.classes_)
        if unseen_labels:
            print(f"Unseen labels in column '{column_name}': {unseen_labels}")
            # Map unseen labels to a default value (e.g., 'unknown')
            df[column_name] = df[column_name].apply(lambda x: x if x in encoder.classes_ else 'unknown')
        # Perform decoding
        df[column_name] = encoder.inverse_transform(df[column_name])
    return df

# Decode all columns
for column in df_encoded.columns:
    if column in label_encoders:
        df_encoded = decode_column(df_encoded, column, label_encoders)

# Ensure that predictions are handled correctly
# Adjust this as needed based on your dataset
if 'predicted_family' in df_encoded.columns:
    df_encoded = decode_column(df_encoded, 'predicted_family', label_encoders)

if 'predicted_name' in df_encoded.columns:
    df_encoded = decode_column(df_encoded, 'predicted_name', label_encoders)

if 'predicted_class' in df_encoded.columns:
    df_encoded = decode_column(df_encoded, 'predicted_class', label_encoders)

# Save the decoded dataset
df_encoded.to_csv('decoded_predictions.csv', index=False)

print("Decoding complete and saved to 'decoded_predictions.csv'.")
