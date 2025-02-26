#%%
## Importing the modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
#%%
df_boston = pd.read_csv('/Users/anany/OneDrive/Desktop/SEM2/6302-machine learning/boston.csv')


# %%
df_boston.drop('Unnamed: 0', inplace=True, axis=1)

#%%
df_boston.head()
#%%
df_boston.tail()
#%%
# checking data types
df_boston.dtypes
# %%
# check  missing values
df_boston.isnull().sum()
# no missing values

# %% outlier detection using IQR method
def remove_outliers(df, cols, threshold=1.5):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Removing outliers from numeric columns
numeric_cols = df_boston.select_dtypes(include=[np.number]).columns
df_boston = remove_outliers(df_boston, numeric_cols)

#%% One-hot encoding for categorical variables - 'CHAS'
if 'CHAS' in df_boston.columns:
    df_boston['CHAS'] = df_boston['CHAS'].astype('category')
    df_boston = pd.get_dummies(df_boston, columns=['CHAS'], drop_first=True)

#%%
# HEATMAP FOR CORRELATION
plt.figure(figsize=(10,10))
sns.heatmap(df_boston.corr(), annot=True)
plt.show()

# %%
# Dropping highly correlated features
df_boston.drop(columns=['TAX', 'NOX', 'DIS'], inplace=True)
# %%
plt.figure(figsize=(10,10))
corr=df_boston.corr()
matrix=np.triu(corr)
sns.heatmap(df_boston.corr(), annot=True, mask=matrix)
# %%
sns.pairplot(df_boston)
# %%
features = df_boston.columns[0:10]
target = df_boston.columns[-1]
# %%
X = df_boston[features].values
y = df_boston[target].values
# %%
# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=15)
# %%
# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# %%
# Function to compute Adjusted R²
def adjusted_r2(r2, n, p):
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))

# %%
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
# Predictions
y_pred_lr = lr.predict(X_test)
# R² and Adjusted R²
r2_lr = r2_score(y_test, y_pred_lr)
adj_r2_lr = adjusted_r2(r2_lr, X_test.shape[0], X_test.shape[1])
print(f"Linear Regression R²: {r2_lr}")
print(f"Linear Regression Adjusted R²: {adj_r2_lr}")
# %%
# Ridge Regression - Hyperparameter Tuning using GridSearchCV
folds = KFold(n_splits=10, shuffle=True, random_state=15)#cross validation
params = {'alpha' : np.arange(0.0001, 10, 0.01)}

ridge_cv = GridSearchCV(Ridge(), param_grid=params, scoring='r2', cv=folds,
return_train_score=True, verbose=1)
ridge_cv.fit(X_train, y_train)

print(f"Best alpha for Ridge: {ridge_cv.best_params_['alpha']}")

# Train Ridge with best alpha
ridge_reg = Ridge(alpha=ridge_cv.best_params_['alpha'])
ridge_reg.fit(X_train, y_train)

# Predictions
y_pred_ridge = ridge_reg.predict(X_test)

# R² and Adjusted R²
r2_ridge = r2_score(y_test, y_pred_ridge)
adj_r2_ridge = adjusted_r2(r2_ridge, X_test.shape[0], X_test.shape[1])
print(f"Ridge Regression R²: {r2_ridge}")
print(f"Ridge Regression Adjusted R²: {adj_r2_ridge}")
# %%
# Lasso Regression - Hyperparameter Tuning using GridSearchCV
lasso_cv = GridSearchCV(Lasso(), param_grid=params, scoring='r2', cv=folds,
return_train_score=True, verbose=1)
lasso_cv.fit(X_train, y_train)
print(f"Best alpha for Lasso: {lasso_cv.best_params_['alpha']}")

# Train Lasso with best alpha
lasso_reg = Lasso(alpha=lasso_cv.best_params_['alpha'])
lasso_reg.fit(X_train, y_train)

# Predictions
y_pred_lasso = lasso_reg.predict(X_test)

# R² and Adjusted R²
r2_lasso = r2_score(y_test, y_pred_lasso)
adj_r2_lasso = adjusted_r2(r2_lasso, X_test.shape[0], X_test.shape[1])
print(f"Lasso Regression R²: {r2_lasso}")
print(f"Lasso Regression Adjusted R²: {adj_r2_lasso}")

# %%
# elastic net regression - Hyperparameter Tunning using GridSearchCV
elastic_params = {
    'alpha': np.arange(0.0001, 10, 0.01),
    'l1_ratio': [0.1, 0.5, 0.7, 0.9, 1]  # l1_ratio=1 is equivalent to Lasso
}
elastic_cv = GridSearchCV(ElasticNet(), param_grid=elastic_params, scoring='r2', cv=folds,
                          return_train_score=True, verbose=1)
elastic_cv.fit(X_train, y_train)
print(f"Best parameters for Elastic Net: {elastic_cv.best_params_}")

elastic_reg = ElasticNet(alpha=elastic_cv.best_params_['alpha'], 
                         l1_ratio=elastic_cv.best_params_['l1_ratio'])
elastic_reg.fit(X_train, y_train)
y_pred_elastic = elastic_reg.predict(X_test)
r2_elastic = r2_score(y_test, y_pred_elastic)
adj_r2_elastic = adjusted_r2(r2_elastic, X_test.shape[0], X_test.shape[1])
print(f"Elastic Net Regression R²: {r2_elastic}")
print(f"Elastic Net Regression Adjusted R²: {adj_r2_elastic}")

# %%
