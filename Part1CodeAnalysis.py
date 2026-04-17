# =============================================================================
# ASSIGNMENT PART E - MACHINE LEARNING
# Part 1: Pre-processing and Exploratory Data Analysis (EDA)
# Dataset: Gym Members Exercise Tracking Dataset
# Source: https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset
# =============================================================================

# --- Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# STEP 1: LOAD THE DATASET
# Make sure the CSV file is in the same folder as this script,
# or provide the full path to the file.
# =============================================================================

df = pd.read_csv('gym_members_exercise_tracking.csv')

# Display the first 7 rows to get a feel for the data
print("=== First 7 rows ===")
print(df.head(7))

# Display column names and their data types
print("\n=== Data types ===")
print(df.dtypes)

# =============================================================================
# STEP 2: UNDERSTAND THE DATASET STRUCTURE
# =============================================================================

print("\n=== Shape (rows, columns) ===")
print(df.shape)

print("\n=== Column names ===")
print(df.columns.tolist())

print("\n=== Basic statistics ===")
print(df.describe())

print("\n=== Dataset info ===")
print(df.info())

# =============================================================================
# STEP 3: HANDLE CATEGORICAL DATA
# Convert text columns to numeric codes so ML algorithms can use them
# =============================================================================

# Identify categorical columns
print("\n=== Categorical columns ===")
print(df.select_dtypes(include='object').columns.tolist())

# Convert 'Gender' to numeric (Male=1, Female=0)
df['Gender'] = df['Gender'].astype('category')
df['Gender_cat'] = df['Gender'].cat.codes
print("\n=== Gender value counts (encoded) ===")
print(df['Gender_cat'].value_counts())

# Convert 'Workout_Type' to numeric
df['Workout_Type'] = df['Workout_Type'].astype('category')
df['Workout_Type_cat'] = df['Workout_Type'].cat.codes
print("\n=== Workout_Type value counts (encoded) ===")
print(df['Workout_Type_cat'].value_counts())

# Show the mapping so we know what each number means
print("\n=== Gender mapping ===")
print(dict(enumerate(df['Gender'].cat.categories)))

print("\n=== Workout_Type mapping ===")
print(dict(enumerate(df['Workout_Type'].cat.categories)))

# =============================================================================
# STEP 4: HANDLE MISSING VALUES
# =============================================================================

print("\n=== Missing values per column ===")
print(df.isnull().sum())

# If there are missing values in numeric columns, fill with column mean
numeric_cols = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM',
                'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned',
                'Fat_Percentage', 'Water_Intake (liters)',
                'Workout_Frequency (days/week)', 'BMI']

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)
        print(f"Filled missing values in: {col}")

print("\n=== Missing values after handling ===")
print(df.isnull().sum())

# =============================================================================
# STEP 5: HANDLE DUPLICATE VALUES
# =============================================================================

print("\n=== Number of duplicate rows ===")
print(df.duplicated().sum())

# Remove duplicates if any
df = df.drop_duplicates()
print("Duplicates removed. New shape:", df.shape)

# =============================================================================
# STEP 6: CHECK CLASS BALANCE (Target variable = Experience_Level)
# Experience_Level: 1 = Beginner, 2 = Intermediate, 3 = Expert
# =============================================================================

print("\n=== Class distribution (Experience_Level) ===")
print(df['Experience_Level'].value_counts())
print("\nAs percentages:")
print(df['Experience_Level'].value_counts(normalize=True).round(3) * 100)

# =============================================================================
# STEP 7: BUILD WORKING DATASET
# Keep numeric columns + encoded categoricals, drop original text columns
# =============================================================================

working_df = df[['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM',
                  'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned',
                  'Fat_Percentage', 'Water_Intake (liters)',
                  'Workout_Frequency (days/week)', 'BMI',
                  'Gender_cat', 'Workout_Type_cat', 'Experience_Level']].copy()

print("\n=== Working dataset shape ===")
print(working_df.shape)
print(working_df.head())

# =============================================================================
# STEP 8: HANDLE OUTLIERS (using IQR method - detect, report, keep)
# We detect outliers but do NOT remove them automatically -
# instead we report them and make an informed decision
# =============================================================================

print("\n=== Outlier detection (IQR method) ===")
for col in numeric_cols:
    Q1 = working_df[col].quantile(0.25)
    Q3 = working_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = working_df[(working_df[col] < lower) | (working_df[col] > upper)]
    if len(outliers) > 0:
        print(f"{col}: {len(outliers)} outliers detected")

# =============================================================================
# STEP 9: STATISTICS (mean, median, mode, std, Q1, Q3)
# =============================================================================

print("\n=== Descriptive statistics ===")
stats_cols = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM',
              'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned',
              'Fat_Percentage', 'BMI']

stats_table = pd.DataFrame({
    'Mean':   working_df[stats_cols].mean().round(2),
    'Median': working_df[stats_cols].median().round(2),
    'Mode':   working_df[stats_cols].mode().iloc[0].round(2),
    'Std':    working_df[stats_cols].std().round(2),
    'Q1':     working_df[stats_cols].quantile(0.25).round(2),
    'Q3':     working_df[stats_cols].quantile(0.75).round(2),
})
print(stats_table)

# =============================================================================
# STEP 10: NORMALIZATION
# Scale numeric features to range [0, 1] so no single feature dominates
# =============================================================================

cols_to_normalize = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM',
                     'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned',
                     'Fat_Percentage', 'Water_Intake (liters)',
                     'Workout_Frequency (days/week)', 'BMI']

for col in cols_to_normalize:
    col_min = working_df[col].min()
    col_max = working_df[col].max()
    working_df[col] = (working_df[col] - col_min) / (col_max - col_min)

print("\n=== Normalized dataset (first 5 rows) ===")
print(working_df.head())

# =============================================================================
# STEP 11: VISUALIZATIONS
# =============================================================================

# --- Figure 1: Histograms ---
working_df.hist(figsize=(16, 10), bins=20, color='steelblue', edgecolor='black')
plt.suptitle('Figure 1: Feature Distributions (Histograms)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('fig1_histograms.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Figure 2: Boxplots (to visualize outliers) ---
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()
for i, col in enumerate(cols_to_normalize):
    axes[i].boxplot(working_df[col].dropna())
    axes[i].set_title(col, fontsize=9)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])
plt.suptitle('Figure 2: Boxplots of Normalized Features', fontsize=14)
plt.tight_layout()
plt.savefig('fig2_boxplots.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Figure 3: Class balance bar chart ---
plt.figure(figsize=(6, 4))
class_counts = working_df['Experience_Level'].value_counts().sort_index()
bars = plt.bar(['Beginner (1)', 'Intermediate (2)', 'Expert (3)'],
               class_counts.values, color=['#4C9BE8', '#F4A261', '#2A9D8F'],
               edgecolor='black')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 5,
             str(int(bar.get_height())),
             ha='center', fontsize=11)
plt.title('Figure 3: Class Balance (Experience Level)', fontsize=13)
plt.ylabel('Number of Members')
plt.tight_layout()
plt.savefig('fig3_class_balance.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Figure 4: Pairplot (key features colored by Experience_Level) ---
pairplot_features = ['Calories_Burned', 'Session_Duration (hours)',
                     'Fat_Percentage', 'BMI', 'Experience_Level']
pairplot_data = working_df[pairplot_features].copy()
pairplot_data['Experience_Level'] = pairplot_data['Experience_Level'].astype(str)
pair = sns.pairplot(pairplot_data, hue='Experience_Level',
                    palette='muted', height=2.5, kind='scatter')
pair.fig.suptitle('Figure 4: Feature Pairplot by Experience Level',
                  fontsize=14, y=1.02)
plt.savefig('fig4_pairplot.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Figure 5: Correlation heatmap ---
corr_cols = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM',
             'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned',
             'Fat_Percentage', 'BMI']
corr_matrix = working_df[corr_cols].corr(method='pearson')
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, linewidths=0.5)
plt.title('Figure 5: Pearson Correlation Heatmap', fontsize=13)
plt.tight_layout()
plt.savefig('fig5_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Figure 6: Scatter plot - Calories Burned vs Session Duration ---
plt.figure(figsize=(8, 5))
colors = {1: '#4C9BE8', 2: '#F4A261', 3: '#2A9D8F'}
for level, group in working_df.groupby('Experience_Level'):
    plt.scatter(group['Session_Duration (hours)'], group['Calories_Burned'],
                label=f'Level {int(level)}', alpha=0.6,
                color=colors[level], s=20)
plt.xlabel('Session Duration (normalized)')
plt.ylabel('Calories Burned (normalized)')
plt.title('Figure 6: Session Duration vs Calories Burned by Experience Level')
plt.legend(title='Experience Level')
plt.tight_layout()
plt.savefig('fig6_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# STEP 12: SAVE CLEANED DATASET FOR USE IN PARTS 2 AND 3
# =============================================================================

working_df.to_csv('gym_cleaned.csv', index=False)
print("\n=== Cleaned dataset saved as 'gym_cleaned.csv' ===")
print("Ready for Part 2 (Supervised Learning) and Part 3 (Unsupervised Learning)")