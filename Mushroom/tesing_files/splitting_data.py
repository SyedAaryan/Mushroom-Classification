import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('../original_dataset/Modified/Secondary_Data_Modified.csv')

# Split the dataset into 80% training and 20% testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the training and testing sets as separate CSV files
train_df.to_csv('../dataset/mushroom_train_split.csv', index=False)
test_df.to_csv('../dataset/mushroom_test_split.csv', index=False)

print("Data split complete. Training and testing files saved.")
