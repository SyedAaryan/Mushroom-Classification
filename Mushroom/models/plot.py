import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Sample data from your results
folds = np.array([1, 2, 3, 4, 5])
validation_scores = np.array([0.6083, 0.6359, 0.5926, 0.5463, 0.6065])

class_f1_scores = np.array([0.8294, 0.8569, 0.8419, 0.8385, 0.8699])
family_f1_scores = np.array([0.7579, 0.7827, 0.7639, 0.7067, 0.7893])
name_f1_scores = np.array([0.6492, 0.6477, 0.6008, 0.5711, 0.6503])

# Create the plot
plt.figure(figsize=(12, 6))
sns.lineplot(x=folds, y=validation_scores, marker='o', label='Validation Score', color='blue')
sns.lineplot(x=folds, y=class_f1_scores, marker='o', label='Class F1 Score', color='orange')
sns.lineplot(x=folds, y=family_f1_scores, marker='o', label='Family F1 Score', color='green')
sns.lineplot(x=folds, y=name_f1_scores, marker='o', label='Name F1 Score', color='red')

# Adding labels and title
plt.title('Model Performance Across K-Folds')
plt.xlabel('Fold Number')
plt.ylabel('Score')
plt.xticks(folds)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
