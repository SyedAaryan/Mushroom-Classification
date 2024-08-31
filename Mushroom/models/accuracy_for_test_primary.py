import pandas as pd
from sklearn.metrics import accuracy_score

# Load the dataset with predictions
df_with_predictions = pd.read_csv('../results/mushroom_secondary_with_predictions.csv')

# Ensure the 'class' and 'predicted_class' columns are present
if 'class' not in df_with_predictions.columns or 'predicted_class' not in df_with_predictions.columns:
    raise ValueError("The dataset must contain 'class' and 'predicted_class' columns.")

# Extract the original and predicted class columns
original_classes = df_with_predictions['class']
predicted_classes = df_with_predictions['predicted_class']

# Define the valid labels
valid_labels = {'e', 'p'}

# Ensure both columns only contain valid labels
original_classes = original_classes.apply(lambda x: x if x in valid_labels else None)
predicted_classes = predicted_classes.apply(lambda x: x if x in valid_labels else None)

# Drop rows where either original or predicted class is invalid (None)
valid_indices = original_classes.notna() & predicted_classes.notna()
original_classes = original_classes[valid_indices]
predicted_classes = predicted_classes[valid_indices]

# Calculate the accuracy
accuracy = accuracy_score(original_classes, predicted_classes)

# Convert accuracy to percentage
accuracy_percentage = accuracy * 100

print(f"Accuracy of predictions: {accuracy_percentage:.2f}%")
