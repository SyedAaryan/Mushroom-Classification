import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the encoded dataset
df_encoded = pd.read_csv('../csv_files/dataset/train_mushroom_encoded.csv')

# Separate features (X) and target variable (y)
X_train = df_encoded.drop(columns=['class', 'family', 'name'])  # Features
y_train_class = df_encoded['class']  # Target: Class

# Train a Random Forest model for the 'class' target
rf_class = RandomForestClassifier(random_state=42)

rf_class.fit(X_train, y_train_class)

# Save the trained model
joblib.dump(rf_class, '../pkl_files/rf_class_model.pkl')

print("Training complete and model saved.")
