# Updating the code to include ResNet18 and ResNet50 models
import matplotlib.pyplot as plt
import numpy as np

# Updated best validation accuracies and corresponding hyperparameters
best_scores = [0.5779, 0.5848, 0.8183, 0.8633, 0.91]  # Added scores for ResNet18 and ResNet50
best_params = [
    {'Learning Rate': '0.001', 'Batch Size': '32'},
    {'Learning Rate': '0.001', 'Batch Size': '32'},
    {'Learning Rate': '0.001', 'Batch Size': '32'},
    {'Learning Rate': '0.001', 'Batch Size': '32'},  # Params for ResNet18
    {'Learning Rate': '0.001', 'Batch Size': '32'}  # Params for ResNet50
]
model_names = ['Baseline CNN', 'Enhanced CNN', 'ResNet Variant (emoNet)', 'ResNet18', 'ResNet50']  # Added model names

# Enhanced plotting with updates
fig, ax = plt.subplots(figsize=(14, 8))

bars = ax.bar(model_names, best_scores, color=['blue', 'green', 'red', 'purple', 'orange'])

# Adding the best score above bars
for bar, score in zip(bars, best_scores):
    y_offset = 0.005
    ax.text(bar.get_x() + bar.get_width() / 2, score + y_offset, f'{score:.2f}',
            ha='center', va='bottom', fontsize=10, color='black')

# Annotating with hyperparameter information
for i, params in enumerate(best_params):
    param_text = f"LR: {params['Learning Rate']}, BS: {params['Batch Size']}"
    ax.text(i, best_scores[i] - 0.05, param_text, ha='center', color='white', fontsize=9,
            bbox=dict(facecolor='black', alpha=0.5))

ax.set_xlabel('Model', fontsize=14)
ax.set_ylabel('Best Validation Accuracy', fontsize=14)
ax.set_title('Model Performance Comparison', fontsize=16)
ax.set_ylim([0.5, 1])  # Adjust based on your accuracy ranges

plt.tight_layout()
plt.show()
