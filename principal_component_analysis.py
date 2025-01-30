#%%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# %%
df_wine = pd.read_csv('/Users/anany/OneDrive/Desktop/SEM2/6302-machine learning/winequality-red.csv',sep=';')
# %%
df_wine.head()
# %%
y=df_wine.pop('quality')
# %%
scalar = StandardScaler()
df_scaled = pd.DataFrame(scalar.fit_transform(df_wine),
columns=df_wine.columns) #to normalise - range between 0 and 1
# %%
pca=PCA()
df_pca = pd.DataFrame(pca.fit_transform(df_scaled))
# %%
#decide how many pricipal components we need to pick
pd.DataFrame(np.cumsum(pca.explained_variance_ratio_)).plot()
plt.legend('')
plt.xlabel('principal components')
plt.ylabel('explained variability')
plt.title('PCA')
# %%
