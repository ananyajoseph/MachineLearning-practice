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
df_adult = pd.read_csv('/Users/anany/OneDrive/Desktop/SEM2/6302-machine learning/adult.csv')
print(df_adult.head())
# %%
df_adult.dtypes # to check the datatypes
# %%
print(df_adult.isnull().sum())
# %%
# Checking the distribution of the target variable 'income'
class_low = df_adult[df_adult['income'] == '<=50K']  
class_high = df_adult[df_adult['income'] == '>50K']  

# Print the class distribution
print("Income <=50K:", class_low.shape)
print("Income >50K:", class_high.shape)
# %%
# Encoding the categorical features
categorical_columns = df_adult.select_dtypes(include=["object"]).columns
encoder = LabelEncoder()
for col in categorical_columns:
    df_adult[col] = encoder.fit_transform(df_adult[col])
# %%
# Split features and target variable
X = df_adult.drop(columns=["income"])
y = df_adult["income"]
# %%
# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %%
# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# %%
# function to train and evaluate models
def train_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, zero_division=1), recall_score(y_test, y_pred), f1_score(y_test, y_pred)
# %%
# b) evaluating model performance of imbalanced data
acc_lr, prec_lr, rec_lr, f1_lr = train_evaluate(LogisticRegression(), X_train, y_train, X_test, y_test) # logistic regression
acc_rf, prec_rf, rec_rf, f1_rf = train_evaluate(RandomForestClassifier(), X_train, y_train, X_test, y_test) # randomforest


# %%
# c) Handling the class imbalance using SMOTE - Oversampling Technique
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Tomek links - Undersampling technique
tomek = TomekLinks()
X_train_tomek, y_train_tomek = tomek.fit_resample(X_train, y_train)
# %%
# training and evaluating models after handling imbalance
# following is the model performance after SMOTE Oversampling.
acc_lr_smote, prec_lr_smote, rec_lr_smote, f1_lr_smote = train_evaluate(LogisticRegression(), X_train_smote, y_train_smote, X_test, y_test)
acc_rf_smote, prec_rf_smote, rec_rf_smote, f1_rf_smote = train_evaluate(RandomForestClassifier(), X_train_smote, y_train_smote, X_test, y_test)

# %%
# logistic regression and random forest after Tomek Undersampling
acc_lr_tomek, prec_lr_tomek, rec_lr_tomek, f1_lr_tomek = train_evaluate(LogisticRegression(), X_train_tomek, y_train_tomek, X_test, y_test)
acc_rf_tomek, prec_rf_tomek, rec_rf_tomek, f1_rf_tomek = train_evaluate(RandomForestClassifier(), X_train_tomek, y_train_tomek, X_test, y_test)

# %%
metrics_df = pd.DataFrame({
    "Model": ["Logistic Regression (Imbalanced)", "Random Forest (Imbalanced)",
              "Logistic Regression (SMOTE)", "Random Forest (SMOTE)",
              "Logistic Regression (Tomek Links)", "Random Forest (Tomek Links)"],
    "Accuracy": [acc_lr, acc_rf, acc_lr_smote, acc_rf_smote, acc_lr_tomek, acc_rf_tomek],
    "Precision": [prec_lr, prec_rf, prec_lr_smote, prec_rf_smote, prec_lr_tomek, prec_rf_tomek],
    "Recall": [rec_lr, rec_rf, rec_lr_smote, rec_rf_smote, rec_lr_tomek, rec_rf_tomek],
    "F1 Score": [f1_lr, f1_rf, f1_lr_smote, f1_rf_smote, f1_lr_tomek, f1_rf_tomek]
})
print(metrics_df)
# %%
# bargraph for visual representation.

metrics_df.set_index("Model").plot(kind="bar", figsize=(12, 6), colormap="viridis", edgecolor="black")
plt.title("Comparison of Model Performance Before & After Handling Imbalance")
plt.xticks(rotation=45)
plt.ylabel("Score")
plt.legend(loc="lower right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
# %%
## Model Performance before and after handling imbalance
#### Before Handling Imbalance the accuracy of logistic regression was 0.826808 and the accuracy of random forest was 0.854906 whcih is high but the recall values are low (0.455357 for logistic regression and 0.621173 for random forest) - recall is very low for logistic regression. 
#### precision(0.722672 for logistic regression and 0.735094 for random forest) is compratively good but the F1 scores are too low due to the class imbalance.
#### This shows that model is biased towards the majority class.

## Results after applying SMOTE
#### Recall has significantly improved to a good extent particularly for logistic regression (0.776786)
#### The F1-score has improved.(0.625899,0.684639)
#### But the accuracy has slightly decreased.(0.776447,0.845540)

## Tomek Links Results
#### Overall performance after applying tomek results is in between imbalanced dataset's and SMOTE's 
#### Recall is better than the imbalanced models but lower than SMOTE (0.509566, 0.679847)
#### F1-Score has improved (0.587284,0.690191), but precision is slightly lower than in the imbalanced models.

#### Random Forest generally outperforms Logistic Regression across all metrics.
#### SMOTE leads to significant increase in recall, which is useful when detecting the minority class.
#### Tomek improves recall in comparison with imbalanced dataset but not as effective as SMOTE
#### The best balanced model here is Random Forest with SMOTE as it has high recall while maintaining good precision and F1 score

#### Tomek lonks is not as effective as SMOTE in addressing class imbalance.
#### Random forest performs consistently good for handling class imbalance.
#### Smote is the best technique for getting good recall and overall balance of metrics.

####  Class Imabalance need to be handled because models trained on imbalanced datasets favours the majority class.