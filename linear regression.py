#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
# %%
df_auto = pd.read_csv('/Users/anany/OneDrive/Desktop/SEM2/6302-machine learning/auto.csv')
df_auto.head()
# %%
print("Shape of dataset:", df_auto.shape)
# %%
df_auto.dtypes
# %%
# preprocessing
df_auto['horsepower'] = pd.to_numeric(df_auto['horsepower'], errors='coerce')
# check missing values
df_auto.isnull().sum()
# %%
# so drop the 6 rows with null values
df_auto.dropna(subset=['horsepower'], inplace=True)
# %%
# dropping car name
df_auto.drop('car name', axis=1, inplace=True)
# %%
print(df_auto.dtypes)
# %%
# One-hot encoding - 'origin' categorical variable 
df = pd.get_dummies(df_auto, columns=['origin'], prefix='origin', drop_first=True)
print(df.head())
# %%
# splitting target variable and rest of the features

# target variable is mpg
y=df['mpg']

# features are assigned
X = df.drop('mpg',axis=1)
# %%
# splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Standardizing the numerical features in the dataset
num_cols = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year']
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
# %%
# Training the linear regression model using the sklearn 
lr = LinearRegression()
lr.fit(X_train, y_train)
# %%
# Predictions
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)
# %%
# Linearity check by scatter plot
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for idx, col in enumerate(num_cols):
    sns.scatterplot(x=df[col], y=df['mpg'], ax=axes[idx])
    axes[idx].set_title(f"{col} vs. mpg")
plt.tight_layout()
plt.show()
# %%
# Homoscedasticity: Plot residuals vs. fitted values ( training data)
residuals = y_train - y_train_pred
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_train_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values')
plt.show()
# %%
# Normality of Residuals: Histogram and Q-Q plot
plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True)
plt.title('Histogram of Residuals')
plt.show()

# Q-Q Plot
sm.qqplot(residuals, line='45', fit=True)
plt.title('Q-Q Plot of Residuals')
plt.show()
# %%
# converting all columns to a numeric type before computing the VIF.
X_train_const = X_train.astype(float)
print(X_train_const.head())

print(X_train_const.dtypes)
# %%
# to identify Multicollinearity issues if any - Calculating VIF for each feature
# add a constant column for statsmodels
X_train_const = sm.add_constant(X_train_const)
# We add a constant column (intercept) using sm.add_constant because it is necessary for statistical modeling for tests like VIF to identify multicollinearity.
# Create a DataFrame to store VIF values
vif = pd.DataFrame()
vif["Feature"] = X_train_const.columns
vif["VIF"] = [variance_inflation_factor(X_train_const.values, i) 
              for i in range(X_train_const.shape[1])]
print(vif)
# %%
#  Evaluating the model
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

print("Training MSE:", train_mse)
print("Test MSE:", test_mse)
print("Training RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("Training R-squared:", train_r2)
print("Test R-squared:", test_r2)

