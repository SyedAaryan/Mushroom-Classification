import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv('../dataset/mushroom_primary.csv')

# Perform stratified split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['class', 'family', 'name']),  # Features
    df[['class', 'family', 'name']],  # Target labels
    test_size=0.2,
    random_state=42,
    stratify=df[['class', 'family', 'name']]  # Perform stratified split based on these columns
)

# Save the splits
X_train.to_csv('mushroom_primary_train_split.csv', index=False)
X_test.to_csv('mushroom_primary_test_split.csv', index=False)
