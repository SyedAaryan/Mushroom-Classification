import pickle

# Path to your .pkl file
pkl_file_path = '../pkl_files/primary_label_encoders.pkl'

# Open the .pkl file in binary read mode
with open(pkl_file_path, 'rb') as file:
    # Load the data from the file
    data = pickle.load(file)

# Print or inspect the loaded data
print(data)
