import tkinter as tk
from tkinter import Toplevel
import pandas as pd
import joblib
from PIL import Image, ImageTk

# Load the trained model and encoders
clf_class = joblib.load('../pkl_files/rf_class_model.pkl')
label_encoders = joblib.load('../pkl_files/encoders.pkl')

# Define columns and option descriptions
columns = ['cap-shape', 'cap-surface', 'cap-color', 'stem-surface', 'stem-color', 'veil-type', 'veil-color', 'has-ring',
           'ring-type']
options = {
    "cap-shape": {
        "b": "bell", "c": "conical", "x": "convex", "f": "flat", "s": "sunken",
        "p": "spherical", "o": "others"
    },

    "cap-surface": {
        "i": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth", "h": "shiny",
        "l": "leathery", "k": "silky", "w": "wrinkled", "e": "fleshy"
    },

    "cap-color": {
        "n": "brown", "b": "buff", "g": "gray", "r": "green", "p": "pink",
        "u": "purple", "e": "red", "w": "white", "y": "yellow", "l": "blue",
        "o": "orange", "k": "black"
    },

    "stem-surface": {
        "i": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth", "h": "shiny",
        "k": "silky", "f": "none"
    },

    "stem-color": {
        "n": "brown", "b": "buff", "g": "gray", "r": "green", "p": "pink",
        "u": "purple", "e": "red", "w": "white", "y": "yellow", "l": "blue",
        "o": "orange", "k": "black", "f": "none"
    },

    "veil-type": {
        "u": "universal"
    },

    "veil-color": {
        "n": "brown",
        "u": "purple", "e": "red", "w": "white", "y": "yellow", "k": "black"
    },

    "has-ring": {
        "t": "ring", "f": "none"
    },

    "ring-type": {
        "e": "evanescent", "r": "flaring", "g": "grooved",
        "l": "large", "p": "pendant", "z": "zone",
        "m": "movable", "f": "none"
    }
}

# Main UI setup
window = tk.Tk()
window.title("Mushroom Classifier")
window.geometry("900x900")
window.config(bg="#f0f4c3")

# Create header
header = tk.Label(window, text="Mushroom Classifier", font=("Arial", 18, "bold"), bg="#f0f4c3", fg="#4a7c59")
header.pack(pady=10)

# Create a dictionary to store the Tkinter variables for user inputs
entries = {}
for feature in columns:
    frame = tk.Frame(window, bg="#f0f4c3")
    frame.pack(fill="x", padx=10, pady=5)

    label = tk.Label(frame, text=feature, font=("Arial", 12), bg="#f0f4c3", fg="#4a7c59")
    label.pack(anchor="w", padx=10)

    var = tk.StringVar(window)
    choices = [f"{desc} ({code})" for code, desc in options[feature].items()]
    var.set(choices[0])  # Default selection

    # Add dropdown with descriptions
    dropdown = tk.OptionMenu(frame, var, *choices)
    dropdown.config(width=25, font=("Arial", 10), bg="white")
    dropdown.pack(anchor="w", padx=10)

    entries[feature] = var


# Function to classify mushroom and show result in a new window
def classify_mushroom():
    user_input = [[choice.split('(')[-1][0] for choice in [entries[feature].get() for feature in columns]]]
    df_input = pd.DataFrame(user_input, columns=columns)

    # Encode categorical features
    for column in df_input.columns:
        if column in label_encoders:
            le = label_encoders[column]
            df_input[column] = le.transform(df_input[column])

    # Predict class
    predicted_class = clf_class.predict(df_input)
    edible_or_poisonous = label_encoders['class'].inverse_transform(predicted_class)[0]

    # Display result in a new window
    result_window = Toplevel(window)
    result_window.title("Classification Result")
    result_window.geometry("500x300")
    result_window.config(bg="#dcedc8")

    result_label = tk.Label(result_window,
                            text=f"The mushroom is: {edible_or_poisonous} (e for Edible and p for poisonous)",
                            font=("Arial", 12, "bold"),
                            fg="#4a7c59", bg="#dcedc8")
    result_label.pack(pady=10)

    # Load images based on the classification
    if edible_or_poisonous == "e":
        img_path = "../images/edible_image.png"  # Update with your edible mushroom image path
    else:
        img_path = "../images/poisonous_image.png"  # Update with your poisonous mushroom image path

    try:
        # Use PIL to open and display the image
        img = Image.open(img_path)
        img = img.resize((200, 200))  # Resize the image if needed
        img_tk = ImageTk.PhotoImage(img)

        image_label = tk.Label(result_window, image=img_tk, bg="#dcedc8")
        image_label.image = img_tk  # Keep a reference to avoid garbage collection
        image_label.pack(pady=10)
    except Exception as e:
        print(f"Error loading image: {e}")


# Add classify button
classify_button = tk.Button(window, text="Classify Mushroom", command=classify_mushroom, font=("Arial", 12),
                            bg="#4a7c59", fg="white")
classify_button.pack(pady=20)

window.mainloop()
