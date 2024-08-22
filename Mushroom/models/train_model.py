import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load training data
df_train = pd.read_csv('../dataset/mushroom_train.csv')

# Preprocess: Fill missing values with 'unknown'
df_train.fillna('unknown', inplace=True)

# Convert categorical data to numerical
label_encoders = {}
for column in df_train.columns:
    if df_train[column].dtype == object:
        le = LabelEncoder()
        df_train[column] = le.fit_transform(df_train[column])
        label_encoders[column] = le

# Define features (X) and target variables (y) for training
X_train = df_train.drop(columns=['name', 'class', 'family'])
y_train_family = df_train['family']
y_train_name = df_train['name']
y_train_class = df_train['class']

# Save feature names
feature_names = X_train.columns.tolist()
joblib.dump(feature_names, '../pkl_files/feature_names.pkl')

# Train models
clf_family = DecisionTreeClassifier()
clf_family.fit(X_train, y_train_family)
joblib.dump(clf_family, '../pkl_files/clf_family.pkl')

clf_name = DecisionTreeClassifier()
clf_name.fit(X_train, y_train_name)
joblib.dump(clf_name, '../pkl_files/clf_name.pkl')

clf_class = DecisionTreeClassifier()
clf_class.fit(X_train, y_train_class)
joblib.dump(clf_class, '../pkl_files/clf_class.pkl')

# Save label encoders
joblib.dump(label_encoders, '../pkl_files/label_encoders.pkl')

print("Training complete and models saved.")
