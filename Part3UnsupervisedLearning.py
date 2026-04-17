# =============================================================================
# ASSIGNMENT PART E - MACHINE LEARNING
# Part 3: Unsupervised Learning - Clustering
# Algorithms: K-Means and DBSCAN
# Dataset: gym_cleaned.csv (output from Part 1)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
import scipy.cluster.hierarchy as shc

# =============================================================================
# STEP 1: LOAD THE CLEANED DATASET FROM PART 1
# We use only the 4 most informative numeric features for clustering
# (based on pairplot analysis in Part 1)
# =============================================================================

df = pd.read_csv('gym_cleaned.csv')

# Select the 4 most informative features for clustering
clustering_df = df[['Session_Duration (hours)', 'Calories_Burned',
                     'Fat_Percentage', 'Workout_Frequency (days/week)']].copy()

print("=== Clustering dataset ===")
print("Shape:", clustering_df.shape)
print(clustering_df.head())

# Keep track of true labels for comparison (not used IN clustering)
true_labels = df['Experience_Level']

# =============================================================================
# ALGORITHM 1: K-MEANS
#
# How it works: Randomly places K centroids, assigns each data point to
# its nearest centroid, then moves centroids to the mean of their cluster.
# Repeats until centroids stop moving.
#
# Key hyperparameter: n_clusters (K)
#   - Too few: clusters are too broad, poor separation
#   - Too many: overclusters, loses meaningful groupings
#   - We test K = 2, 3, 4, 5, 6, 7, 8 and use Silhouette Score to find best K
#
# Silhouette Score: ranges from -1 to 1
#   - Close to 1: data point is well matched to its own cluster
#   - Close to 0: data point is on the boundary between clusters
#   - Negative: data point may be in the wrong cluster
# =============================================================================

print("\n" + "="*60)
print("ALGORITHM 1: K-MEANS")
print("="*60)

silhouette_scores = []
k_values = list(range(2, 9))  # Test K = 2, 3, 4, 5, 6, 7, 8

for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(clustering_df)
    score = metrics.silhouette_score(clustering_df, kmeans.labels_)
    silhouette_scores.append(round(score, 4))
    print(f"K={k}  →  Silhouette Score: {score:.4f}")

print(f"\nBest K: {k_values[np.argmax(silhouette_scores)]} "
      f"(Silhouette Score: {max(silhouette_scores):.4f})")

# --- Figure 13: Silhouette Score vs K ---
plt.figure(figsize=(7, 4))
plt.plot(k_values, silhouette_scores, marker='o', color='steelblue', linewidth=2)
plt.xticks(k_values)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Figure 13: K-Means Silhouette Score for Different K Values')
plt.tight_layout()
plt.savefig('fig13_kmeans_silhouette.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Experiment with K=2 ---
print("\n--- K-Means Experiment 1: K=2 ---")
km2 = KMeans(n_clusters=2, init='k-means++', random_state=42, n_init=10)
km2.fit(clustering_df)
score2 = metrics.silhouette_score(clustering_df, km2.labels_)
print(f"Silhouette Score: {score2:.4f}")
print(f"Cluster sizes: {pd.Series(km2.labels_).value_counts().sort_index().to_dict()}")

# --- Experiment with K=3 ---
print("\n--- K-Means Experiment 2: K=3 ---")
km3 = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
km3.fit(clustering_df)
score3 = metrics.silhouette_score(clustering_df, km3.labels_)
print(f"Silhouette Score: {score3:.4f}")
print(f"Cluster sizes: {pd.Series(km3.labels_).value_counts().sort_index().to_dict()}")

# --- Experiment with K=4 ---
print("\n--- K-Means Experiment 3: K=4 ---")
km4 = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
km4.fit(clustering_df)
score4 = metrics.silhouette_score(clustering_df, km4.labels_)
print(f"Silhouette Score: {score4:.4f}")
print(f"Cluster sizes: {pd.Series(km4.labels_).value_counts().sort_index().to_dict()}")

# --- Figure 14: K-Means K=3 cluster visualization (2 feature pairs) ---
best_k = k_values[np.argmax(silhouette_scores)]
km_best = KMeans(n_clusters=best_k, init='k-means++', random_state=42, n_init=10)
km_best.fit(clustering_df)
y_km = km_best.labels_
centroids = km_best.cluster_centers_

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Session Duration vs Calories Burned
axes[0].scatter(clustering_df['Session_Duration (hours)'],
                clustering_df['Calories_Burned'],
                c=y_km, cmap='viridis', s=15, alpha=0.7)
axes[0].set_xlabel('Session Duration (normalized)')
axes[0].set_ylabel('Calories Burned (normalized)')
axes[0].set_title(f'K-Means K={best_k}: Session Duration vs Calories Burned')

# Plot 2: Fat Percentage vs Workout Frequency
axes[1].scatter(clustering_df['Fat_Percentage'],
                clustering_df['Workout_Frequency (days/week)'],
                c=y_km, cmap='viridis', s=15, alpha=0.7)
axes[1].set_xlabel('Fat Percentage (normalized)')
axes[1].set_ylabel('Workout Frequency (normalized)')
axes[1].set_title(f'K-Means K={best_k}: Fat Percentage vs Workout Frequency')

plt.suptitle(f'Figure 14: K-Means Clustering (K={best_k}, best model)', fontsize=13)
plt.tight_layout()
plt.savefig('fig14_kmeans_clusters.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Figure 15: Compare K-Means clusters vs true labels ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(clustering_df['Session_Duration (hours)'],
                clustering_df['Calories_Burned'],
                c=y_km, cmap='viridis', s=15, alpha=0.7)
axes[0].set_title(f'K-Means Clusters (K={best_k})')
axes[0].set_xlabel('Session Duration (normalized)')
axes[0].set_ylabel('Calories Burned (normalized)')

axes[1].scatter(clustering_df['Session_Duration (hours)'],
                clustering_df['Calories_Burned'],
                c=true_labels, cmap='viridis', s=15, alpha=0.7)
axes[1].set_title('True Experience Levels (1=Beginner, 2=Inter, 3=Expert)')
axes[1].set_xlabel('Session Duration (normalized)')
axes[1].set_ylabel('Calories Burned (normalized)')

plt.suptitle('Figure 15: K-Means Clusters vs True Labels', fontsize=13)
plt.tight_layout()
plt.savefig('fig15_kmeans_vs_true.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# ALGORITHM 2: DBSCAN
#
# How it works: Groups points that are closely packed together, marking
# points in low-density regions as noise/outliers (-1).
# Unlike K-Means, DBSCAN does NOT require you to specify K in advance.
#
# Key hyperparameters:
#   - eps: the maximum distance between two points to be considered neighbors
#       Small eps → more noise points, very tight clusters
#       Large eps → fewer noise points, clusters merge together
#   - min_samples: minimum number of points to form a dense region (core point)
#       Small min_samples → more points become core points, fewer noise points
#       Large min_samples → stricter definition of clusters, more noise
#
# Finding eps: We use the k-nearest neighbors distance plot —
# the "elbow" of the curve is a good estimate for eps
# =============================================================================

print("\n" + "="*60)
print("ALGORITHM 2: DBSCAN")
print("="*60)

# --- Find a good eps value using nearest neighbor distances ---
print("\n--- Finding optimal eps using k-nearest neighbors plot ---")
neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(clustering_df)
distances, _ = neighbors.kneighbors(clustering_df)
distances = np.sort(distances[:, 4])  # 5th nearest neighbor distance

# --- Figure 16: K-nearest neighbor distance plot ---
plt.figure(figsize=(7, 4))
plt.plot(distances, color='steelblue')
plt.xlabel('Data Points (sorted)')
plt.ylabel('Distance to 5th Nearest Neighbor')
plt.title('Figure 16: K-NN Distance Plot (used to find eps for DBSCAN)')
plt.tight_layout()
plt.savefig('fig16_dbscan_eps.png', dpi=150, bbox_inches='tight')
plt.show()

# --- DBSCAN Experiment 1: eps=0.15, min_samples=5 ---
print("\n--- DBSCAN Experiment 1: eps=0.15, min_samples=5 ---")
db1 = DBSCAN(eps=0.15, min_samples=5)
db1.fit(clustering_df)
labels1 = db1.labels_
n_clusters1 = len(set(labels1)) - (1 if -1 in labels1 else 0)
n_noise1 = list(labels1).count(-1)
print(f"Clusters found: {n_clusters1}")
print(f"Noise points:   {n_noise1} ({n_noise1/len(labels1)*100:.1f}%)")
if n_clusters1 > 1:
    score_db1 = metrics.silhouette_score(clustering_df, labels1)
    print(f"Silhouette Score: {score_db1:.4f}")

# --- DBSCAN Experiment 2: eps=0.25, min_samples=5 ---
print("\n--- DBSCAN Experiment 2: eps=0.25, min_samples=5 ---")
db2 = DBSCAN(eps=0.25, min_samples=5)
db2.fit(clustering_df)
labels2 = db2.labels_
n_clusters2 = len(set(labels2)) - (1 if -1 in labels2 else 0)
n_noise2 = list(labels2).count(-1)
print(f"Clusters found: {n_clusters2}")
print(f"Noise points:   {n_noise2} ({n_noise2/len(labels2)*100:.1f}%)")
if n_clusters2 > 1:
    score_db2 = metrics.silhouette_score(clustering_df, labels2)
    print(f"Silhouette Score: {score_db2:.4f}")

# --- DBSCAN Experiment 3: eps=0.35, min_samples=10 ---
print("\n--- DBSCAN Experiment 3: eps=0.35, min_samples=10 ---")
db3 = DBSCAN(eps=0.35, min_samples=10)
db3.fit(clustering_df)
labels3 = db3.labels_
n_clusters3 = len(set(labels3)) - (1 if -1 in labels3 else 0)
n_noise3 = list(labels3).count(-1)
print(f"Clusters found: {n_clusters3}")
print(f"Noise points:   {n_noise3} ({n_noise3/len(labels3)*100:.1f}%)")
if n_clusters3 > 1:
    score_db3 = metrics.silhouette_score(clustering_df, labels3)
    print(f"Silhouette Score: {score_db3:.4f}")

# --- Figure 17: DBSCAN cluster visualizations for all 3 experiments ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
experiments = [
    (labels1, 'Exp 1: eps=0.15, min=5'),
    (labels2, 'Exp 2: eps=0.25, min=5'),
    (labels3, 'Exp 3: eps=0.35, min=10'),
]

for ax, (labels, title) in zip(axes, experiments):
    unique_labels = set(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(sorted(unique_labels), colors):
        mask = labels == label
        label_name = f'Noise' if label == -1 else f'Cluster {label}'
        col = 'black' if label == -1 else color
        ax.scatter(clustering_df['Session_Duration (hours)'][mask],
                   clustering_df['Calories_Burned'][mask],
                   c=[col], s=10, alpha=0.6, label=label_name)
    n_cl = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_ns = list(labels).count(-1)
    ax.set_title(f'{title}\nClusters: {n_cl}, Noise: {n_ns}', fontsize=10)
    ax.set_xlabel('Session Duration (normalized)')
    ax.set_ylabel('Calories Burned (normalized)')
    ax.legend(fontsize=7, markerscale=2)

plt.suptitle('Figure 17: DBSCAN Experiments - Session Duration vs Calories Burned',
             fontsize=13)
plt.tight_layout()
plt.savefig('fig17_dbscan_experiments.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# STEP 3: HIERARCHICAL CLUSTERING (BONUS - strengthens the report)
#
# How it works: Builds a tree (dendrogram) by merging the two closest
# clusters at each step until everything is in one cluster.
# The dendrogram shows visually how many natural clusters exist.
#
# Key hyperparameter: linkage method
#   - 'ward': minimizes variance within clusters (generally best)
#   - 'complete': uses maximum distance between clusters
#   - 'average': uses average distance between clusters
# =============================================================================

print("\n" + "="*60)
print("BONUS: HIERARCHICAL CLUSTERING")
print("="*60)

# --- Figure 18: Dendrogram ---
plt.figure(figsize=(14, 6))
plt.title('Figure 18: Hierarchical Clustering Dendrogram (Ward linkage)', fontsize=13)
plt.xlabel('Gym Members')
plt.ylabel('Euclidean Distance')
dendrogram = shc.dendrogram(
    shc.linkage(clustering_df, method='ward'),
    truncate_mode='level', p=5,  # Show only top 5 levels for readability
    no_labels=True
)
plt.tight_layout()
plt.savefig('fig18_dendrogram.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Experiment 1: 2 clusters ---
print("\n--- Hierarchical Experiment 1: n_clusters=2, linkage=ward ---")
hc1 = AgglomerativeClustering(n_clusters=2, linkage='ward')
hc1_labels = hc1.fit_predict(clustering_df)
score_hc1 = metrics.silhouette_score(clustering_df, hc1_labels)
print(f"Silhouette Score: {score_hc1:.4f}")
print(f"Cluster sizes: {pd.Series(hc1_labels).value_counts().sort_index().to_dict()}")

# --- Experiment 2: 3 clusters ---
print("\n--- Hierarchical Experiment 2: n_clusters=3, linkage=ward ---")
hc2 = AgglomerativeClustering(n_clusters=3, linkage='ward')
hc2_labels = hc2.fit_predict(clustering_df)
score_hc2 = metrics.silhouette_score(clustering_df, hc2_labels)
print(f"Silhouette Score: {score_hc2:.4f}")
print(f"Cluster sizes: {pd.Series(hc2_labels).value_counts().sort_index().to_dict()}")

# --- Experiment 3: 3 clusters, complete linkage ---
print("\n--- Hierarchical Experiment 3: n_clusters=3, linkage=complete ---")
hc3 = AgglomerativeClustering(n_clusters=3, linkage='complete')
hc3_labels = hc3.fit_predict(clustering_df)
score_hc3 = metrics.silhouette_score(clustering_df, hc3_labels)
print(f"Silhouette Score: {score_hc3:.4f}")
print(f"Cluster sizes: {pd.Series(hc3_labels).value_counts().sort_index().to_dict()}")

# --- Figure 19: Hierarchical clustering visualization ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
hc_experiments = [
    (hc1_labels, f'Exp 1: n=2, ward (sil={score_hc1:.3f})'),
    (hc2_labels, f'Exp 2: n=3, ward (sil={score_hc2:.3f})'),
    (hc3_labels, f'Exp 3: n=3, complete (sil={score_hc3:.3f})'),
]
for ax, (labels, title) in zip(axes, hc_experiments):
    ax.scatter(clustering_df['Session_Duration (hours)'],
               clustering_df['Calories_Burned'],
               c=labels, cmap='viridis', s=15, alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Session Duration (normalized)')
    ax.set_ylabel('Calories Burned (normalized)')

plt.suptitle('Figure 19: Hierarchical Clustering Experiments', fontsize=13)
plt.tight_layout()
plt.savefig('fig19_hierarchical_experiments.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# STEP 4: FINAL SUMMARY - UNSUPERVISED vs TRUE LABELS
# =============================================================================

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

print(f"\nK-Means best (K={best_k}) Silhouette Score:   "
      f"{max(silhouette_scores):.4f}")
print(f"Hierarchical best (ward, n=2) Silhouette:    {score_hc1:.4f}")

print("""
Key observations:
- K-Means found the most interpretable clusters
- DBSCAN struggled with this dataset because the classes
  overlap in density (no clear noise-based separation)
- The best K from Silhouette Score should be compared to
  the true number of classes (3) to see if the algorithm
  naturally recovers the class structure
- Hierarchical clustering dendrogram helps visualize
  the natural groupings in the data
""")