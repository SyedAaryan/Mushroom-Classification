import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the training dataset
df_train = pd.read_csv('../dataset/mushroom_train_split.csv')

# Encode categorical features
label_encoders = {}
for column in df_train.columns:
    if df_train[column].dtype == object:
        le = LabelEncoder()
        df_train[column] = le.fit_transform(df_train[column])
        label_encoders[column] = le

# Separate features (X) and target (y)
X_train = df_train.drop(columns=['class'])  # Features (drop the target column)
y_train = df_train['class']  # Target column ('class')

# Train the Decision Tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Save the trained model and label encoders
joblib.dump(clf, '../pkl_files/mushroom_clf_model.pkl')
joblib.dump(label_encoders, '../pkl_files/label_encoders.pkl')

print("Training complete and model saved.")
