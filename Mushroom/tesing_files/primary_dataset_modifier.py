#This function is used to split the data that addhi recommended, refer 23/08/2024 : 21:20 in documentation
import pandas as pd
import re


# Function to generate rows based on bracketed values
def expand_bracketed_values(row):
    # Store all lists of values
    lists = []

    for item in row:
        # Check if the cell contains bracketed values like "[x, f]"
        match = re.findall(r'\[(.*?)\]', str(item))
        if match:
            # Split by comma and strip whitespace
            lists.append([val.strip() for val in match[0].split(',')])
        else:
            lists.append([item])

    # Generate all combinations of rows
    from itertools import product
    return list(product(*lists))


# Read the input CSV file
input_file = '../original_dataset/Modified/Primary_data_Modified.csv'  # Change to your input file
df = pd.read_csv(input_file)

# DataFrame to hold the expanded rows
expanded_rows = []

# Loop through each row of the dataframe
for index, row in df.iterrows():
    expanded_rows.extend(expand_bracketed_values(row))

# Convert the expanded rows back to a DataFrame
expanded_df = pd.DataFrame(expanded_rows, columns=df.columns)

# Save the expanded DataFrame to a new CSV file
output_file = '../dataset/mushroom_primary.csv'  # Change to your desired output file name
expanded_df.to_csv(output_file, index=False)

print(f"Data expansion complete. Expanded data saved to {output_file}")
