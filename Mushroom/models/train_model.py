import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load training data
df_train = pd.read_csv('../dataset/mushroom_train_old.csv')

# Preprocess: Fill missing values with 'unknown'
df_train.fillna('unknown', inplace=True)

#Converts data other than int into numerical
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

# Train models
clf_family = DecisionTreeClassifier()
clf_family.fit(X_train, y_train_family)
joblib.dump(clf_family, '../clf_family.pkl')

clf_name = DecisionTreeClassifier()
clf_name.fit(X_train, y_train_name)
joblib.dump(clf_name, '../clf_name.pkl')

clf_class = DecisionTreeClassifier()
clf_class.fit(X_train, y_train_class)
joblib.dump(clf_class, '../clf_class.pkl')

# Save label encoders
joblib.dump(label_encoders, '../label_encoders.pkl')

print("Training complete and models saved.")
