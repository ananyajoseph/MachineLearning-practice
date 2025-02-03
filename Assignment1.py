# 1. Import the dataset from the module using required packages. 
#%%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# %%
df = pd.read_csv('/Users/anany/OneDrive/Desktop/SEM2/6302-machine learning/winequality-red.csv',sep=';')
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

# %%
scalar = StandardScaler()
df_scaled = pd.DataFrame(scalar.fit_transform(df),
columns=df.columns) #to normalise - range between 0 and 1
# %%
# Random Sampling - randomly selects 150 samples , no grouping
random_sample = df.sample(n=150, random_state=21)
random_sample.head()
# unbiased method
# may not ensure representation of subgroups
# %%
#Stratified sampling - divides subgroups based on categorical feature. Randomly selects from each stratum in proportion of their occurence.
df.columns = df.columns.str.strip()
stratified_sample, _ = train_test_split(df, test_size=(150/ len(df)), stratify=df['quality'], random_state=21)
stratified_sample.head()
# %%
# systematic sampling - ensures even distribution across the dataset
#calculating ste size
k=len(df)//150
systematic_sample = df.iloc[::k, :]
systematic_sample.head()
# %%
