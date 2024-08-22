import joblib

# Load the saved label encoders
label_encoders = joblib.load('../pkl_files/label_encoders.pkl')

# Decode the predictions
decoded_family = label_encoders['family'].inverse_transform([0])[0]
decoded_name = label_encoders['name'].inverse_transform([55])[0]

print(f'Decoded Family: {decoded_family}')
print(f'Decoded Name: {decoded_name}')
