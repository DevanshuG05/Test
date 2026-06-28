import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             classification_report, roc_curve, auc, 
                             roc_auc_score)
import warnings
import os
warnings.filterwarnings('ignore')

output_dir = "titanic_outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✓ Created output directory: {output_dir}")

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("=" * 70)
print("LOADING TITANIC DATASET")
print("=" * 70)

# Using seaborn's built-in Titanic dataset
titanic = sns.load_dataset('titanic')
print(f"\nDataset shape: {titanic.shape}")
print(f"\nFirst few rows:\n{titanic.head()}")
print(f"\nDataset Info:\n{titanic.info()}")

# ============================================================================
# 2. DATA EXPLORATION VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 70)
print("CREATING DATA EXPLORATION VISUALIZATIONS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Data Exploration - Titanic Dataset', fontsize=16, fontweight='bold')

# Plot 1: Survival Distribution
survival_counts = titanic['survived'].value_counts()
axes[0, 0].bar(['Did Not Survive', 'Survived'], survival_counts.values, color=['#d62728', '#2ca02c'])
axes[0, 0].set_title('Survival Distribution', fontweight='bold')
axes[0, 0].set_ylabel('Count')
for i, v in enumerate(survival_counts.values):
    axes[0, 0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# Plot 2: Survival by Sex
survival_by_sex = pd.crosstab(titanic['sex'], titanic['survived'])
survival_by_sex.plot(kind='bar', ax=axes[0, 1], color=['#d62728', '#2ca02c'])
axes[0, 1].set_title('Survival by Sex', fontweight='bold')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_xlabel('Sex')
axes[0, 1].legend(['Did Not Survive', 'Survived'])
axes[0, 1].tick_params(axis='x', rotation=0)

# Plot 3: Age Distribution
axes[1, 0].hist(titanic['age'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='skyblue')
axes[1, 0].set_title('Age Distribution', fontweight='bold')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Frequency')

# Plot 4: Passenger Class Distribution
class_counts = titanic['pclass'].value_counts().sort_index()
axes[1, 1].bar(['Class 1', 'Class 2', 'Class 3'], class_counts.values, color=['gold', 'silver', '#CD7F32'])
axes[1, 1].set_title('Passenger Class Distribution', fontweight='bold')
axes[1, 1].set_ylabel('Count')
for i, v in enumerate(class_counts.values):
    axes[1, 1].text(i, v + 5, str(v), ha='center', fontweight='bold')

plt.tight_layout()
savepath = os.path.join(output_dir, '01_data_exploration.png')
plt.savefig(savepath, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {savepath}")
plt.show()

# ============================================================================
# 3. MISSING VALUES VISUALIZATION
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
missing_data = titanic.isnull().sum()
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
missing_data.plot(kind='barh', ax=ax, color='coral')
ax.set_title('Missing Values in Dataset', fontweight='bold', fontsize=14)
ax.set_xlabel('Count of Missing Values')
for i, v in enumerate(missing_data.values):
    ax.text(v + 2, i, str(v), va='center', fontweight='bold')
plt.tight_layout()
savepath = os.path.join(output_dir, '02_missing_values.png')
plt.savefig(savepath, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {savepath}")
plt.show()

# ============================================================================
# 4. DATA PREPARATION & FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 70)
print("DATA PREPARATION")
print("=" * 70)

# Create a copy for processing
df = titanic.copy()

# Select features and target
X = df[['age', 'pclass', 'sex', 'fare']].copy()
y = df['survived'].copy()

# Handle missing values
X['age'] = X['age'].fillna(X['age'].median())
X['fare'] = X['fare'].fillna(X['fare'].median())

# Remove rows with NaN in sex (very few)
valid_idx = ~X.isnull().any(axis=1)
X = X[valid_idx]
y = y[valid_idx]

# Encode categorical variables
X['sex'] = X['sex'].map({'male': 0, 'female': 1})

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeature columns: {X.columns.tolist()}")
print(f"Feature statistics:\n{X.describe()}")

# ============================================================================
# 5. CORRELATION HEATMAP
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))
correlation_data = X.copy()
correlation_data['survived'] = y
correlation_matrix = correlation_data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
plt.tight_layout()
savepath = os.path.join(output_dir, '03_correlation_heatmap.png')
plt.savefig(savepath, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {savepath}")
plt.show()

# ============================================================================
# 6. SPLIT DATA & TRAIN MODEL
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION MODEL")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                      random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
print(f"Training set survival rate: {y_train.mean():.2%}")
print(f"Testing set survival rate: {y_test.mean():.2%}")

# Train model
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)

print("\n✓ Model trained successfully!")

# ============================================================================
# 7. MAKE PREDICTIONS
# ============================================================================
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2%}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Did Not Survive', 'Survived']))

# ============================================================================
# 8. CONFUSION MATRIX VISUALIZATION
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
            xticklabels=['Did Not Survive', 'Survived'],
            yticklabels=['Did Not Survive', 'Survived'],
            cbar_kws={'label': 'Count'})
ax.set_title(f'Confusion Matrix (Accuracy: {accuracy:.2%})', fontweight='bold', fontsize=14)
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')

# Add metrics text
tn, fp, fn, tp = cm.ravel()
metrics_text = f'TP: {tp}\nTN: {tn}\nFP: {fp}\nFN: {fn}'
ax.text(2.5, 0.5, metrics_text, fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
savepath = os.path.join(output_dir, '04_confusion_matrix.png')
plt.savefig(savepath, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {savepath}")
plt.show()

# ============================================================================
# 9. ROC CURVE
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

ax.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontweight='bold')
ax.set_ylabel('True Positive Rate', fontweight='bold')
ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold', fontsize=14)
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
savepath = os.path.join(output_dir, '05_roc_curve.png')
plt.savefig(savepath, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {savepath}")
plt.show()

# ============================================================================
# 10. PREDICTION PROBABILITY DISTRIBUTION
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.6, label='Did Not Survive', color='#d62728', edgecolor='black')
ax.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.6, label='Survived', color='#2ca02c', edgecolor='black')
ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
ax.set_xlabel('Predicted Probability of Survival', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('Distribution of Predicted Probabilities', fontweight='bold', fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
savepath = os.path.join(output_dir, '06_prediction_probability.png')
plt.savefig(savepath, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {savepath}")
plt.show()

# ============================================================================
# 11. FEATURE IMPORTANCE (Coefficients)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=True)

colors = ['#d62728' if x < 0 else '#2ca02c' for x in feature_importance['Coefficient']]
ax.barh(feature_importance['Feature'], feature_importance['Coefficient'], color=colors)
ax.set_xlabel('Coefficient Value', fontweight='bold')
ax.set_title('Logistic Regression Feature Importance (Coefficients)', fontweight='bold', fontsize=14)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
for i, v in enumerate(feature_importance['Coefficient']):
    ax.text(v + 0.02 if v > 0 else v - 0.02, i, f'{v:.3f}', va='center', 
            ha='left' if v > 0 else 'right', fontweight='bold')
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
savepath = os.path.join(output_dir, '07_feature_importance.png')
plt.savefig(savepath, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {savepath}")
plt.show()

# ============================================================================
# 12. SURVIVAL PREDICTION BY FEATURES
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Survival Predictions by Feature', fontsize=16, fontweight='bold')

# Survival by Age
axes[0, 0].scatter(X_test['age'][y_test == 0], y_pred[y_test == 0], alpha=0.5, label='Actually Did Not Survive', color='#d62728')
axes[0, 0].scatter(X_test['age'][y_test == 1], y_pred[y_test == 1], alpha=0.5, label='Actually Survived', color='#2ca02c')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Predicted Survival (0 or 1)')
axes[0, 0].set_title('Age vs Predicted Survival')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Survival by Class
for survived in [0, 1]:
    label = 'Survived' if survived == 1 else 'Did Not Survive'
    color = '#2ca02c' if survived == 1 else '#d62728'
    mask = y_test == survived
    axes[0, 1].scatter(X_test['pclass'][mask], y_pred[mask], alpha=0.5, label=label, color=color)
axes[0, 1].set_xlabel('Passenger Class')
axes[0, 1].set_ylabel('Predicted Survival (0 or 1)')
axes[0, 1].set_title('Passenger Class vs Predicted Survival')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Survival by Sex
for survived in [0, 1]:
    label = 'Survived' if survived == 1 else 'Did Not Survive'
    color = '#2ca02c' if survived == 1 else '#d62728'
    mask = y_test == survived
    axes[1, 0].scatter(X_test['sex'][mask], y_pred[mask], alpha=0.5, label=label, color=color)
axes[1, 0].set_xlabel('Sex (0=Male, 1=Female)')
axes[1, 0].set_ylabel('Predicted Survival (0 or 1)')
axes[1, 0].set_title('Sex vs Predicted Survival')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Survival by Fare
for survived in [0, 1]:
    label = 'Survived' if survived == 1 else 'Did Not Survive'
    color = '#2ca02c' if survived == 1 else '#d62728'
    mask = y_test == survived
    axes[1, 1].scatter(X_test['fare'][mask], y_pred[mask], alpha=0.5, label=label, color=color)
axes[1, 1].set_xlabel('Fare')
axes[1, 1].set_ylabel('Predicted Survival (0 or 1)')
axes[1, 1].set_title('Fare vs Predicted Survival')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
savepath = os.path.join(output_dir, '08_predictions_by_features.png')
plt.savefig(savepath, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {savepath}")
plt.show()

# ============================================================================
# 13. EXAMPLE PREDICTIONS
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE PREDICTIONS FOR NEW PASSENGERS")
print("=" * 70)

examples = [
    {'age': 25, 'pclass': 2, 'sex': 'female', 'fare': 20},
    {'age': 45, 'pclass': 1, 'sex': 'male', 'fare': 100},
    {'age': 3, 'pclass': 3, 'sex': 'male', 'fare': 5},
    {'age': 60, 'pclass': 3, 'sex': 'female', 'fare': 8},
]

for i, example in enumerate(examples, 1):
    X_new = pd.DataFrame([{
        'age': example['age'],
        'pclass': example['pclass'],
        'sex': 1 if example['sex'] == 'female' else 0,
        'fare': example['fare']
    }])
    
    pred = model.predict(X_new)[0]
    pred_proba = model.predict_proba(X_new)[0]
    
    print(f"\nPassenger {i}:")
    print(f"  Age: {example['age']}, Class: {example['pclass']}, Sex: {example['sex']}, Fare: ${example['fare']}")
    print(f"  Prediction: {'SURVIVED' if pred == 1 else 'DID NOT SURVIVE'}")
    print(f"  Confidence: {pred_proba[pred]:.2%}")

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE!")
print("=" * 70)
print(f"\n✓ All visualizations have been saved to: {os.path.abspath(output_dir)}")
print("\nFiles generated:")
print("  01_data_exploration.png")
print("  02_missing_values.png")
print("  03_correlation_heatmap.png")
print("  04_confusion_matrix.png")
print("  05_roc_curve.png")
print("  06_prediction_probability.png")
print("  07_feature_importance.png")
print("  08_predictions_by_features.png")