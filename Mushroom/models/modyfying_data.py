#This function is used to modify the document, mostly to reduce the dataset by deleting the coplumn
import pandas as pd

def remove_columns_from_csv(input_file, output_file, columns_to_remove):
    # Ensure at least one column is specified
    if not columns_to_remove:
        print("Error: At least one column name must be specified.")
        return

    # Load the dataset
    df = pd.read_csv(input_file)

    # Check which columns exist in the DataFrame
    existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]
    missing_columns = [col for col in columns_to_remove if col not in df.columns]

    if existing_columns_to_remove:
        # Drop the specified columns
        df = df.drop(columns=existing_columns_to_remove)
        print(f"Columns {existing_columns_to_remove} removed successfully.")
    else:
        print("None of the specified columns exist in the file.")

    if missing_columns:
        print(f"Columns {missing_columns} were not found in the file.")

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Updated CSV file saved as '{output_file}'.")


# Example usage
input_file = '../dataset/mushroom_train_old.csv'  #input file name
output_file = '../dataset/mushroom_train.csv'  # Output file name
columns_to_remove = ['cap-diameter', 'does-bruise-or-bleed', 'gill-attachment','gill-spacing','gill-color','stem-height',
                     'stem-width','stem-root','Spore-print-color','habitat','season']  # columns needed to be removed

remove_columns_from_csv(input_file, output_file, columns_to_remove)
