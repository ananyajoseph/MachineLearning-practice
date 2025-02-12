
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
df_credit = pd.read_csv('/Users/anany/OneDrive/Desktop/SEM2/6302-machine learning/creditcard.csv')

#%%
df_credit.head()
#%%
class_0 = df_credit[df_credit['Class'] == 0]
class_1 = df_credit[df_credit['Class'] == 1]
print('class 0:', class_0.shape)
print('class_1:', class_1.shape )
# how to calculate the % from this? [class_1.shape[0]/df_credit.shape[0]]

# %%
# Seperate my target and variables(independent)
x = df_credit.drop('Class', axis = 1)
y = df_credit['Class']

# %%
# Apply undersampling to the majority class
under_sampler = RandomUnderSampler(random_state=10,
replacement=True)

x_under, y_under = under_sampler.fit_resample(x, y)

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y,
test_size = 0.2,
random_state = 10)

#%%
x_train, x_test, y_train, y_test = train_test_split(x_under, y_under,
test_size = 0.2,
random_state = 10)

# %%
clf = RandomForestClassifier(random_state = 10)
clf.fit(x_train, y_train)
# %%
y_pred = clf.predict(x_test)

#%%
#Evaluation of the model
print(confusion_matrix(y_test, y_pred))

#%%
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
# %%
print(classification_report(y_test,y_pred))

# %%
# SMOTE

smote = SMOTE(random_state = 10)
x_smote, y_smote = smote.fit_resample(x, y)
# %%
# %%
x_train, x_test, y_train, y_test = train_test_split(x_smote, y_smote,
test_size = 0.2,
random_state = 10)
# %%
clf = RandomForestClassifier(random_state = 10)
clf.fit(x_train, y_train)
# %%
y_pred = clf.predict(x_test)

#%%
#Evaluation of the model
print(confusion_matrix(y_test, y_pred))
# %%
print(classification_report(y_test,y_pred))
