import pandas as pd
import joblib
from Mushroom.application.userInput import get_user_input

print(
    "Welcome Stranded Guy who is hungry and is surrounded by mushrooms but doesn't know which one is edible and which "
    "one is poisonous.")
print("I am here to help you identify the mushroom.")
print("Please answer the following questions to identify the mushroom.")

# Define the column names to match your model's expected input for class prediction
columns = ['cap-shape', 'Cap-surface', 'cap-color', 'stem-surface', 'stem-color', 'veil-type', 'veil-color', 'has-ring',
           'ring-type']

# Get user input

user_input_list = [get_user_input()]

# Convert the list to a DataFrame
df_input = pd.DataFrame(user_input_list, columns=columns)

# Load the trained class model and label encoder
clf_class = joblib.load('../pkl_files/rf_class_model.pkl')  # Load only the class model
label_encoders = joblib.load('../pkl_files/encoders.pkl')

# Encode categorical features using the same encoders from the primary dataset
for column in df_input.columns:
    if df_input[column].dtype == object:
        le = label_encoders.get(column)
        if le:
            df_input[column] = le.transform(df_input[column])

# Predict the class
predicted_class = clf_class.predict(df_input)

# Decode the predictions back to their original labels
df_input['predicted_class'] = label_encoders['class'].inverse_transform(predicted_class)

# Print the prediction result
print(f"The predicted class for the input mushroom is: {df_input['predicted_class'].iloc[0]} (Edible or Poisonous)")

# Optionally, print the full input and output for clarity
