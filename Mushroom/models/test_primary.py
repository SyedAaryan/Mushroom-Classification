import pandas as pd
import joblib

# Load the secondary dataset
df_secondary = pd.read_csv('../csv_files/dataset/test_mushroom.csv')

# Load the trained models and label encoders
clf_family = joblib.load('../pkl_files/family_clf_model.pkl')
clf_name = joblib.load('../pkl_files/name_clf_model.pkl')
clf_class = joblib.load('../pkl_files/class_clf_model.pkl')
label_encoders = joblib.load('../pkl_files/primary_label_encoders.pkl')

# Encode categorical features using the same encoders from the primary dataset
for column in df_secondary.columns:
    if df_secondary[column].dtype == object:
        le = label_encoders.get(column)
        if le:
            df_secondary[column] = le.transform(df_secondary[column])



# Separate features (X) and target variable (y)
X_test = df_secondary.drop(columns=['class'])  # Features for predicting family, name, and class

# Predict the family, name, and class
predicted_family = clf_family.predict(X_test)
predicted_name = clf_name.predict(X_test)
predicted_class = clf_class.predict(X_test)

# Decode other columns back to their original labels
for column in df_secondary.columns:
    if column in label_encoders:
        le = label_encoders[column]
        df_secondary[column] = le.inverse_transform(df_secondary[column])

# Decode the predictions back to their original labels
df_secondary['family'] = label_encoders['family'].inverse_transform(predicted_family)
df_secondary['name'] = label_encoders['name'].inverse_transform(predicted_name)
df_secondary['predicted_class'] = label_encoders['class'].inverse_transform(predicted_class)


# Save the predictions to a CSV file
df_secondary.to_csv('../csv_files/results/predicted_dataset.csv', index=False)

print("Predictions complete and saved.")
