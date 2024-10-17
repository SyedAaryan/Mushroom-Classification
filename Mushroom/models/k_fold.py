import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

# Load your encoded train_data
train_data = pd.read_csv('../csv_files/dataset/train_mushroom_encoded.csv')

# Define the target columns
target_columns = ['class', 'family', 'name']


# K-Fold Cross-Validation
def k_fold_split_multi_target(train_data, target_columns, n_splits=5):
    X = train_data.drop(target_columns, axis=1)  # Features
    y = train_data[target_columns]  # Targets

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Model initialization
        model = MultiOutputClassifier(RandomForestClassifier(random_state=42))

        # Training the model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_val)

        # Validation score
        val_score = model.score(X_val, y_val)

        # Initialize lists to hold metrics
        f1_scores = []
        precision_scores = []
        recall_scores = []

        # Calculate metrics for each target
        for i, column in enumerate(target_columns):
            f1 = f1_score(y_val.iloc[:, i], y_pred[:, i], average='weighted', zero_division=0)
            precision = precision_score(y_val.iloc[:, i], y_pred[:, i], average='weighted', zero_division=0)
            recall = recall_score(y_val.iloc[:, i], y_pred[:, i], average='weighted', zero_division=0)

            f1_scores.append(f1)
            precision_scores.append(precision)
            recall_scores.append(recall)

        # Print the scores
        print(f"Fold {fold}:")
        print(f"  Validation Score = {val_score:.4f}")
        for i, column in enumerate(target_columns):
            print(f"  {column}:")
            print(f"    F1 Score = {f1_scores[i]:.4f}")
            print(f"    Precision = {precision_scores[i]:.4f}")
            print(f"    Recall = {recall_scores[i]:.4f}")


# Run the K-Fold validation
k_fold_split_multi_target(train_data, target_columns)
