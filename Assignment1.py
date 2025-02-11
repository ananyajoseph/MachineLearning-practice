# 1. Import the dataset from the module using required packages. 
#%%
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# %%
df = pd.read_csv('/Users/anany/OneDrive/Desktop/SEM2/6302-machine learning/diabetes.csv',sep=',')
# %%
#2. Display the first and last 10 rows of the dataset. Additionally, check the data types of all features and identify any missing values. Explain your findings in detail.
# first 10 rows
df.head(10)
# %%
# last 10 rows - tail function is used
df.tail(10)

# %%
# checking data types
df.dtypes
# %%
# checking the missing values
df.isnull().sum()
# There are no null values in the dataset
#%%
def introduce_missing_values(df, columns, missing_percent=0.2, random_state= 21):
    np.random.seed(random_state)
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns:
            n_missing = int(len(df_copy) * missing_percent)
            missing_indices = np.random.choice(df_copy.index, n_missing, replace=False)
            df_copy.loc[missing_indices, col] = np.nan
    return df_copy

#%%
np.random.seed(21)
df = pd.read_csv('/Users/anany/OneDrive/Desktop/SEM2/6302-machine learning/diabetes.csv',sep=',')

dfm = introduce_missing_values(df, columns=['Pregnancies','BloodPressure','Glucose'], missing_percent=0.3)
dfm.head(10)

# %%
# Random Sampling - randomly selects 150 samples , no grouping
random_sample = dfm.sample(n=150, random_state=21)
random_sample.head()
# unbiased method
# may not ensure representation of subgroups
# %%
#Stratified sampling - divides subgroups based on categorical feature. Randomly selects from each stratum in proportion of their occurence.
dfm.columns = dfm.columns.str.strip()
stratified_sample, _ = train_test_split(dfm, test_size=(150/ len(dfm)), stratify=dfm['Class'], random_state=21)
stratified_sample.head()
# %%
# systematic sampling - ensures even distribution across the dataset
#calculating size
k=len(dfm)//150
systematic_sample = dfm.iloc[::k, :]
systematic_sample.head()

# %%
#Median Imputation
imputer = SimpleImputer(strategy="median")
df_imputed = pd.DataFrame(imputer.fit_transform(dfm),columns=dfm.columns)
print(df_imputed)
# %%
y = df_imputed.pop('Class')
#correlation matrix
corr_matrix = df_imputed.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# %%
#top 3 features
# so the top 3 pairs of features with the highest correlation are Insulin and SkinThickness (0.44), Age and Pregnancies(0.43), BMI and SkinThickness(0.39)

# Pregnancies and Age - indicates a moderate positive correlation, meaning as age increases the number of pregnancies tends to increase. this correlation shows that age and pregnancies might have a combined effect on health outcomes, such as diabetes risk.

# skin thickness and Insulin - shows a moderate positive correlation, implying that individuals with higher sking thickness measurements tend to have higher insulin levels. This may be because skin thickness is linked to body fat, which in turn is related to insulin regulation.
# # Insulin resistance is a key factor for diabetes development so this above features should be taken into account.

# BMI and skin thickness - has a moderate correlation, which suggests that individuals with higher BMI tent to have higher skin thickness. Since BMI is a measure of body fat and skin thickness can indicate subcutaneous fat.
# # This correlation highlights the role of body composition in metabolism of an individual and how it is linked with diabetes risk.

#%%
# Normalize using Standardization
standard_scaler = StandardScaler()
df_standard = pd.DataFrame(standard_scaler.fit_transform(df_imputed),columns=df_imputed.columns)
df_standard
# %%
# Normalize using Min-Max Scaling
minmax_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(minmax_scaler.fit_transform(df_imputed), columns=df_imputed.columns)
df_minmax
#%%
# Normalize using Robust Scaling
robust_scaler = RobustScaler()
df_scaled= pd.DataFrame(robust_scaler.fit_transform(df_imputed), columns=df_imputed.columns)
df_scaled
# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

sns.boxplot(data=df_standard, ax=axes[0])
axes[0].set_title("Standardized Data")

sns.boxplot(data=df_minmax, ax=axes[1])
axes[1].set_title("Min-Max Scaled Data")

sns.boxplot(data=df_scaled, ax=axes[2])
axes[2].set_title("Robust Scaled Data")

plt.show()

# Comparison of Normalization Methods:
# Standardization (Z-score) works well for normally distributed data, not a good method if we have many outliers.
# Min-Max Scaling preserves relationships and keeps values in a fixed range, sensitive to outliers. Not Good if there is a huge difference between the minimum value and maximum value as it compresses the inner values and stretches out the outside values.
# Robust Scaling handles outliers well using median and IQR. Doesn't maintain exact distribution shape.
# Here we can see from the visualization that the Robust Scaling is better. And also since in this case, there are outliers too Robust Scaling is the better Normalization method here.
#%%
# checking the missing values
df_scaled.isnull().sum()
# PCA does not accept missing values encoded as NaN natively. 

# %%
pca=PCA()
df_pca = pd.DataFrame(pca.fit_transform(df_scaled))

# %%
#decide how many pricipal components we need to pick
# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Find the number of components needed for 85% variance
num_components = np.argmax(cumulative_variance >= 0.85) + 1

# Plot explained variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.axhline(y=0.85, color='r', linestyle='-')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Cumulative Explained Variance')
plt.grid()
plt.show()

# Print results
print(f"Explained Variance Ratio: {explained_variance}")
print(f"Cumulative Explained Variance: {cumulative_variance}")
print(f"Number of Principal Components needed to explain at least 85% variance: {num_components}")

# %%
# Create Scree Plot
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, color='b', label="Explained Variance")
plt.plot(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), marker='o', linestyle='--', color='r', label="Cumulative Explained Variance")
plt.axhline(y=0.85, color='g', linestyle='-', label="85% Threshold")
plt.xlabel('Principal Component Number')
plt.ylabel('Explained Variance')
plt.title('Scree Plot - Explained Variance by Principal Components')
plt.legend()
plt.grid()
plt.show()
# %%
# PCA helps in reducing the dimensions while maintaing most of the data's information.
# thus by choosing fewer components improves coputational efficiency.
# elbow point in the plot, where the explained varaiance starts flattening, suggests an optimal number of components - in this case 5
# 85% variance is explained by 5 components which reduces the dimensionality while preserving the information from the data.
# scree plot visualizes how much variance each principal component explains.
# bar graph shows the variance explained by each component.
# scree plot helps us for deciding the optimal number of principal components to be considered.

#%% t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(df_scaled)
tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
tsne_df['Outcome'] = y.values

#%% plotting
plt.figure(figsize=(8, 5))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='Outcome', data=tsne_df, palette='coolwarm')
plt.title('t-SNE Visualization')
plt.show()

# %%
# PCA is effective at reducing dimensionality while preserving global variance in the dataset.
# PCA efficiently identifies dominant patterns in the dataset by focusing on variance
# Only a few components are necessary to capture most information.

# PCA does not necessarily preserve local structures or cluster separation, as it focuses only on variance.
# Since PCA does not consider class separation, it may not be the best choice for classification tasks where complex, nonlinear relationships exist.

#Unlike PCA, t-SNE is nonlinear
# The plot shows a clear separation between clusters, meaning t-SNE has successfully identified patterns in the data.
# There is a distinct cluster on the left, indicating a subgroup that is clearly different from the others.
# t-SNE is better at visualizing clusters, making it useful for classification and identifying hidden patterns.
# The results do not retain variance information, making t-SNE unsuitable for feature reduction in predictive modeling.
# t-SNE is slow for high-dimensional data whereas PCA is Fast and scalable for large datasets

# PCA is good for dimensionality reduction for machine learning models.
# t-SNE is better for visualizing class separability- how well the features distinguish diabetics from non-diabetics.
# The t-SNE plot shows two clear groups based on the Outcome (diabetic or not).
# One area has a strong separation, meaning the dataset can distinguish between the two groups well.
# However, some overlap exists, which could mean borderline cases or noisy data.
# The dataset mainly has two groups: diabetic (1) and non-diabetic (0).
# Some overlap suggests that diagnosing diabetes isn't always straightforward.
# The features help separate the two groups