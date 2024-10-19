import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

# Load your encoded train_data
train_data = pd.read_csv('../csv_files/dataset/train_mushroom_encoded.csv')

# Define the target column (class - edible or poisonous)
target_column = 'class'


# K-Fold Cross-Validation with Overall Validation Metrics
def k_fold_split_and_validate(train_data, target_column, n_splits=5):
    X = train_data.drop(target_column, axis=1)  # Features
    y = train_data[target_column]  # Target (class)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize lists to store metrics for all folds
    f1_scores = []
    precision_scores = []
    recall_scores = []

    # Iterate through each fold
    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Initialize and train the model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Predictions on the validation set
        y_pred = model.predict(X_val)

        # Calculate metrics for this fold
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)

        # Store metrics for this fold
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

        # Print metrics for this fold
        print(f"Fold {fold}:")
        print(f"  F1 Score = {f1:.4f}")
        print(f"  Precision = {precision:.4f}")
        print(f"  Recall = {recall:.4f}")

    # Calculate overall validation scores (mean of all folds)
    overall_f1 = sum(f1_scores) / n_splits
    overall_precision = sum(precision_scores) / n_splits
    overall_recall = sum(recall_scores) / n_splits

    print("\nOverall Validation Scores:")
    print(f"  Mean F1 Score = {overall_f1:.4f}")
    print(f"  Mean Precision = {overall_precision:.4f}")
    print(f"  Mean Recall = {overall_recall:.4f}")


# Run the K-Fold validation with the target column 'class'
k_fold_split_and_validate(train_data, target_column)
