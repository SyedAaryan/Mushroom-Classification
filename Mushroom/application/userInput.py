import pandas as pd
import joblib


def get_user_input():
    # Define valid options for each parameter with descriptions
    options = {
        "cap-shape": {
            "b": "bell", "c": "conical", "x": "convex", "f": "flat", "s": "sunken",
            "p": "spherical", "o": "others", "null": "If not present"
        },
        "cap-surface": {
            "i": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth", "h": "shiny",
            "l": "leathery", "k": "silky", "t": "sticky", "w": "wrinkled", "e": "fleshy",
            "0": "others", "null": "If not present"
        },
        "cap-color": {
            "n": "brown", "b": "buff", "g": "gray", "r": "green", "p": "pink",
            "u": "purple", "e": "red", "w": "white", "y": "yellow", "l": "blue",
            "o": "orange", "k": "black", "null": "If not present"
        },
        "stem-surface": {
            "i": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth", "h": "shiny",
            "l": "leathery", "k": "silky", "t": "sticky", "w": "wrinkled", "e": "fleshy",
            "0": "others", "f": "none"
        },
        "stem-color": {
            "n": "brown", "b": "buff", "g": "gray", "r": "green", "p": "pink",
            "u": "purple", "e": "red", "w": "white", "y": "yellow", "l": "blue",
            "o": "orange", "k": "black", "f": "none"
        },
        "veil-type": {
            "u": "universal", "null": "If not present"
        },
        "veil-color": {
            "n": "brown", "b": "buff", "g": "gray", "r": "green", "p": "pink",
            "u": "purple", "e": "red", "w": "white", "y": "yellow", "l": "blue",
            "o": "orange", "k": "black", "f": "none"
        },
        "has-ring": {
            "t": "ring", "f": "none", "null": "If not present"
        },
        "ring-type": {
            "c": "cobwebby", "e": "evanescent", "r": "flaring", "g": "grooved",
            "l": "large", "p": "pendant", "s": "sheathing", "z": "zone",
            "y": "scaly", "m": "movable", "f": "none", "?": "unknown"
        }
    }

    user_input = []

    for param, choices in options.items():
        # Display the choices with descriptions
        print(f"\nAvailable options for {param}:")
        for key, description in choices.items():
            print(f"{key}: {description}")

        while True:
            value = input(f"Enter value for {param}: ").strip()
            if value in choices:
                user_input.append(value)
                break
            else:
                print("Invalid input. Please enter a valid value from the list above.")

    return user_input
