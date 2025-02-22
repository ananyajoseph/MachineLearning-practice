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
#%%
# loading the data set
df_boston = pd.read_csv('/Users/anany/OneDrive/Desktop/SEM2/6302-machine learning/Boston.csv')
df_boston.head()
# %%
df_boston.dtypes # to check the datatypes
# %%
# Computing the Pearson correlation coefficient matrix for the dataset
correlation_matrix = df_boston.corr()
print(correlation_matrix)
# %%
# creating heatmap
plt.figure(figsize=(12,10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Boston")
plt.show()
# %%
### c
## Strong positive correlation
#### 1. RAD (radial highways) and TAX (Tax Rate) have a very strong positive correlation of 0.91
#### It indicates that areas with higher accessibility to highways tend to have higher property tax rates. 
#### Shows that as connectivity increases the tax for the property also increases.

#### 2. Positive Correlation: RM and Price (0.70)
#### The number of rooms per dwelling (RM) is positively correlated with house prices (Price)
#### Houses with more rooms tend to have higher prices. This makes sense because larger homes generally have higher market values

#### 3. INDUS and NOX (0.76) 
#### The proportion of non-retail business acres per town (INDUS) is strongly positively correlated with nitric oxide concentration (NOX).
#### As the proportion of industrial land increases in an area, air pollution (NOX) also increases. This makes sense since industrial zones tend to have more emissions from factories and vehicles.

## Strong negative correlation
#### 1. LSTAT (Lower status population) and housing price shares a strong negative correlation. (-0.74)
#### This indicates that as the percentage of lower-income residents in a neighborhood increases, property prices tend to decrease significantly. This is a common trend in real estate where wealthier neighborhoods tend to have higher home values

#### 2. Nitric oxide concentration (NOX) has a strong negative correlation of -0.77 with the weighted distance to employment centers (DIS)
#### Areas farther from employment centers tend to have lower pollution levels. This suggests that urban areas (closer to employment centers) experience more pollution than suburban or rural areas.

#### 3.DIS and NOX (-0.77)
####  The weighted distance to employment centers (DIS) is strongly negatively correlated with nitric oxide concentration (NOX).
#### As the distance from employment hubs increases, pollution levels decrease. This suggests that urban areas (closer to jobs) experience higher pollution, while suburban or rural areas have cleaner air.

#### From the heatmap, if two features have a correlation greater than 0.7, we should consider keeping only one of them  for modelling to avoid redundancy.
