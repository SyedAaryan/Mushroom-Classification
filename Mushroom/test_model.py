import pandas as pd
import joblib

# Load the test data
df_test = pd.read_csv('mushroom_test_minor.csv')  # Replace with your actual testing file name

# Load the saved models and label encoders
clf_family = joblib.load('clf_family.pkl')
clf_name = joblib.load('clf_name.pkl')
clf_class = joblib.load('clf_class.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Preprocess: Fill missing values with 'unknown'
df_test.fillna('unknown', inplace=True)

# Encode categorical variables using the saved label encoders
for column in df_test.columns:
    if df_test[column].dtype == object:
        le = label_encoders.get(column)
        if le:
            df_test[column] = df_test[column].apply(lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([min(le.classes_, key=lambda val: abs(float(val) - float(x)))]))
        else:
            # Handle unseen categories if label encoder for the column is not found
            df_test[column] = df_test[column].apply(lambda x: -1 if x not in label_encoders[column].classes_ else x)

# Define features (X) for testing
X_test = df_test.drop(columns=['class'])  # Assuming 'class' is the column to be predicted

# Predict 'family'
predicted_family = clf_family.predict(X_test)
df_test['family'] = predicted_family

# Predict 'name'
predicted_name = clf_name.predict(X_test)
df_test['name'] = predicted_name

# Predict 'class' (if 'class' is known in the test data)
if 'class' in df_test.columns:
    y_test_class = df_test['class']
    y_pred_class = clf_class.predict(X_test)
    accuracy = (y_test_class == y_pred_class).mean()
    print(f'Accuracy: {accuracy * 100:.2f}%')

# Print the final predictions for 'family' and 'name'
print(df_test[['family', 'name']])
df_test.to_csv('mushroom_test_predictions.csv', index=False)
print("Testing complete and predictions saved.")
