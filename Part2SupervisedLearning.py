# =============================================================================
# ASSIGNMENT PART E - MACHINE LEARNING
# Part 2: Supervised Learning - Classification
# Algorithms: K-Nearest Neighbors (KNN) and Decision Tree
# Dataset: gym_cleaned.csv (output from Part 1)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

# =============================================================================
# STEP 1: LOAD THE CLEANED DATASET FROM PART 1
# =============================================================================

df = pd.read_csv('gym_cleaned.csv')

print("=== Dataset loaded ===")
print("Shape:", df.shape)
print(df.head())

# =============================================================================
# STEP 2: DEFINE FEATURES (X) AND TARGET (y)
# Target: Experience_Level (1=Beginner, 2=Intermediate, 3=Expert)
# Features: all other columns
# =============================================================================

X = df.drop(columns=['Experience_Level'])
y = df['Experience_Level']

print("\n=== Features used for classification ===")
print(X.columns.tolist())
print("\n=== Target class distribution ===")
print(y.value_counts().sort_index())

# =============================================================================
# STEP 3: SPLIT INTO TRAINING AND TEST SETS (70% train, 30% test)
# stratify=y ensures each class is proportionally represented in both sets
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print("\n=== Train/Test split ===")
print(f"Total data objects:     {len(df)}")
print(f"Training set:           {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
print(f"Test set:               {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")

print("\n=== Class distribution in training set ===")
print(y_train.value_counts().sort_index())
print("\n=== Class distribution in test set ===")
print(y_test.value_counts().sort_index())

# =============================================================================
# ALGORITHM 1: K-NEAREST NEIGHBORS (KNN)
#
# How it works: For a new data point, KNN looks at the K closest
# training examples and assigns the most common class among them.
#
# Key hyperparameter: n_neighbors (K)
#   - Small K (e.g. 1-3): very sensitive to noise, risk of overfitting
#   - Large K (e.g. 20+): smoother boundary, risk of underfitting
#   - We test K = 3, 7, 15 to find the best balance
# =============================================================================

print("\n" + "="*60)
print("ALGORITHM 1: K-NEAREST NEIGHBORS (KNN)")
print("="*60)

knn_results = []
knn_k_values = [3, 7, 15]

for k in knn_k_values:
    # --- Train ---
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # --- Evaluate on training set (to check overfitting) ---
    train_acc = knn.score(X_train, y_train)

    # --- Evaluate on test set ---
    y_pred = knn.predict(X_test)
    test_acc = knn.score(X_test, y_test)

    knn_results.append({
        'K': k,
        'Train Accuracy': round(train_acc, 4),
        'Test Accuracy': round(test_acc, 4),
        'Gap (overfit indicator)': round(train_acc - test_acc, 4)
    })

    print(f"\n--- KNN with K={k} ---")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Beginner', 'Intermediate', 'Expert']))

# Print summary table
print("\n=== KNN Experiment Summary ===")
knn_summary = pd.DataFrame(knn_results)
print(knn_summary.to_string(index=False))

# --- Figure 7: KNN Accuracy vs K ---
plt.figure(figsize=(7, 4))
plt.plot(knn_k_values,
         [r['Train Accuracy'] for r in knn_results],
         marker='o', label='Train Accuracy', color='steelblue')
plt.plot(knn_k_values,
         [r['Test Accuracy'] for r in knn_results],
         marker='s', label='Test Accuracy', color='tomato')
plt.xticks(knn_k_values)
plt.xlabel('K (number of neighbors)')
plt.ylabel('Accuracy')
plt.title('Figure 7: KNN - Train vs Test Accuracy for Different K Values')
plt.legend()
plt.tight_layout()
plt.savefig('fig7_knn_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Figure 8: Confusion matrix for best KNN model ---
best_k = knn_results[np.argmax([r['Test Accuracy'] for r in knn_results])]['K']
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred_knn_best = knn_best.predict(X_test)

cm_knn = confusion_matrix(y_test, y_pred_knn_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_knn,
                              display_labels=['Beginner', 'Intermediate', 'Expert'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title(f'Figure 8: KNN Confusion Matrix (K={best_k}, best model)')
plt.tight_layout()
plt.savefig('fig8_knn_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n=== Best KNN model: K={best_k} ===")
print(f"Test Accuracy: {knn_best.score(X_test, y_test):.4f}")

# =============================================================================
# ALGORITHM 2: DECISION TREE
#
# How it works: The algorithm splits the data into branches based on
# feature thresholds, building a tree of if/else rules.
# Each leaf node represents a predicted class.
#
# Key hyperparameter: max_depth
#   - Shallow tree (depth 2-3): simple, may underfit
#   - Deep tree (no limit): complex, may overfit (memorizes training data)
#   - We test max_depth = 3, 6, None (unlimited) to find best balance
# =============================================================================

print("\n" + "="*60)
print("ALGORITHM 2: DECISION TREE")
print("="*60)

dt_results = []
dt_depths = [3, 6, None]
dt_depth_labels = ['3', '6', 'None (unlimited)']

for depth, label in zip(dt_depths, dt_depth_labels):
    # --- Train ---
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)

    # --- Evaluate on training set (to check overfitting) ---
    train_acc = dt.score(X_train, y_train)

    # --- Evaluate on test set ---
    y_pred = dt.predict(X_test)
    test_acc = dt.score(X_test, y_test)

    dt_results.append({
        'max_depth': label,
        'Train Accuracy': round(train_acc, 4),
        'Test Accuracy': round(test_acc, 4),
        'Gap (overfit indicator)': round(train_acc - test_acc, 4)
    })

    print(f"\n--- Decision Tree with max_depth={label} ---")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Beginner', 'Intermediate', 'Expert']))

# Print summary table
print("\n=== Decision Tree Experiment Summary ===")
dt_summary = pd.DataFrame(dt_results)
print(dt_summary.to_string(index=False))

# --- Figure 9: Decision Tree Accuracy comparison ---
x_pos = [0, 1, 2]
plt.figure(figsize=(7, 4))
plt.plot(x_pos,
         [r['Train Accuracy'] for r in dt_results],
         marker='o', label='Train Accuracy', color='steelblue')
plt.plot(x_pos,
         [r['Test Accuracy'] for r in dt_results],
         marker='s', label='Test Accuracy', color='tomato')
plt.xticks(x_pos, dt_depth_labels)
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Figure 9: Decision Tree - Train vs Test Accuracy for Different Depths')
plt.legend()
plt.tight_layout()
plt.savefig('fig9_dt_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Figure 10: Decision tree visualized (best model, depth=3 for readability) ---
dt_viz = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_viz.fit(X_train, y_train)
plt.figure(figsize=(20, 8))
plot_tree(dt_viz,
          feature_names=X.columns.tolist(),
          class_names=['Beginner', 'Intermediate', 'Expert'],
          filled=True, rounded=True, fontsize=9)
plt.title('Figure 10: Decision Tree Visualization (max_depth=3)', fontsize=13)
plt.tight_layout()
plt.savefig('fig10_dt_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Figure 11: Confusion matrix for best Decision Tree model ---
best_depth_idx = np.argmax([r['Test Accuracy'] for r in dt_results])
best_depth = dt_depths[best_depth_idx]
dt_best = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_best.fit(X_train, y_train)
y_pred_dt_best = dt_best.predict(X_test)

cm_dt = confusion_matrix(y_test, y_pred_dt_best)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_dt,
                               display_labels=['Beginner', 'Intermediate', 'Expert'])
fig, ax = plt.subplots(figsize=(6, 5))
disp2.plot(ax=ax, cmap='Greens', colorbar=False)
ax.set_title(f'Figure 11: Decision Tree Confusion Matrix (max_depth={dt_depth_labels[best_depth_idx]}, best model)')
plt.tight_layout()
plt.savefig('fig11_dt_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Figure 12: Feature importances (Decision Tree depth=6) ---
dt_feat = DecisionTreeClassifier(max_depth=6, random_state=42)
dt_feat.fit(X_train, y_train)
importances = pd.Series(dt_feat.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(8, 6))
importances.plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('Figure 12: Feature Importances (Decision Tree, max_depth=6)', fontsize=12)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('fig12_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# STEP 4: FINAL COMPARISON - KNN vs DECISION TREE
# =============================================================================

print("\n" + "="*60)
print("FINAL COMPARISON: KNN vs DECISION TREE")
print("="*60)

knn_best_acc = knn_best.score(X_test, y_test)
dt_best_acc = dt_best.score(X_test, y_test)

print(f"\nBest KNN (K={best_k}) Test Accuracy:                    {knn_best_acc:.4f}")
print(f"Best Decision Tree (depth={dt_depth_labels[best_depth_idx]}) Test Accuracy:  {dt_best_acc:.4f}")

if knn_best_acc > dt_best_acc:
    print("\n→ KNN performed better on this dataset.")
else:
    print("\n→ Decision Tree performed better on this dataset.")

print("\n=== Overfitting/Underfitting check ===")
print("KNN Results:")
print(knn_summary[['K', 'Train Accuracy', 'Test Accuracy', 'Gap (overfit indicator)']].to_string(index=False))
print("\nDecision Tree Results:")
print(dt_summary[['max_depth', 'Train Accuracy', 'Test Accuracy', 'Gap (overfit indicator)']].to_string(index=False))
print("\nNote: A large gap between train and test accuracy suggests overfitting.")
print("A low accuracy on both suggests underfitting.")