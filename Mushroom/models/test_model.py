import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

# Load the test dataset
df_test = pd.read_csv('../dataset/mushroom_test_split.csv')

# Load the trained model and label encoders
clf = joblib.load('../pkl_files/mushroom_clf_model.pkl')
label_encoders = joblib.load('../pkl_files/label_encoders.pkl')

# Encode categorical features using the same label encoders from training
for column in df_test.columns:
    if df_test[column].dtype == object:
        le = label_encoders.get(column)
        if le:
            df_test[column] = le.transform(df_test[column])

# Separate features (X) and target (y)
X_test = df_test.drop(columns=['class'])  # Features
y_test = df_test['class']  # Target

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the predictions to a CSV file
df_test['predicted_class'] = y_pred
df_test.to_csv('../results/mushroom_test_predictions.csv', index=False)

print("Testing complete and predictions saved.")
