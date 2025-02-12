#%%
!pip install imblearn
#%%
# Importing necessary libraries
import pandas as pd
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
# %%
df_credit = pd.read_csv('/Users/anany/OneDrive/Desktop/SEM2/6302-machine learning/winequality-red.csv')
#%%
class_0 = df_credit[df_credit['Class'] == 0]
class_1 = df_credit[df_credit['Class'] == 1]
print('class 0:', class_0.shape)
print('class_1:', class_1.shape )
# how to calculate the % from this? [class_1.shape[0]/df_credit.shape[0]]