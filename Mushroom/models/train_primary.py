import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the primary dataset
df_primary = pd.read_csv('../dataset/mushroom_primary.csv')

# Encode categorical features in the primary dataset
label_encoders = {}
for column in df_primary.columns:
    if df_primary[column].dtype == object:
        le = LabelEncoder()
        df_primary[column] = le.fit_transform(df_primary[column])
        label_encoders[column] = le

# Separate features (X) and target variables (y)
X_train = df_primary.drop(columns=['family', 'name', 'class'])  # Features
y_train_family = df_primary['family']  # Target: Family
y_train_name = df_primary['name']  # Target: Name
y_train_class = df_primary['class']  # Target: Class

# Train a Decision Tree model for each target
clf_family = DecisionTreeClassifier(random_state=42)
clf_name = DecisionTreeClassifier(random_state=42)
clf_class = DecisionTreeClassifier(random_state=42)

clf_family.fit(X_train, y_train_family)
clf_name.fit(X_train, y_train_name)
clf_class.fit(X_train, y_train_class)

# Save the trained models and label encoders
joblib.dump(clf_family, '../pkl_files/family_clf_model.pkl')
joblib.dump(clf_name, '../pkl_files/name_clf_model.pkl')
joblib.dump(clf_class, '../pkl_files/class_clf_model.pkl')
joblib.dump(label_encoders, '../pkl_files/primary_label_encoders.pkl')

print("Training complete and models saved.")
