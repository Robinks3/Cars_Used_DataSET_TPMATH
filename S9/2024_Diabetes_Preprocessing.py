import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from scipy import stats
import category_encoders as ce  # Pour Binary Encoding et Leave-One-Out Encoding
##############################################
### 1. Importation des données et exploration
##############################################
Data = pd.read_csv("diabetes.csv", sep=',')

# Liste des features
target_name = "Outcome"
explanatory_columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# Extraction des variables prédictives et de la cible
target = Data[target_name]
df = Data[explanatory_columns]
df=Data

# Observation des données
print("Résumé statistique des données :")
print(df.describe())

##############################################
### 2. Visualisation des distributions (Histogrammes + Boxplots)
##############################################

# a) Histogrammes
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
for i in range(2):
    for j in range(4):
        sns.histplot(
            data=df, x=explanatory_columns[i * 4 + j], kde=False,
            ax=axs[i, j], color=["skyblue", "olive", "gold", "teal", "blue", "yellow", "green", "pink"][i * 4 + j]
        )
plt.suptitle("Histograms des variables", fontsize=16)
plt.show()

# b) Boxplots
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
for i in range(2):
    for j in range(4):
        sns.boxplot(
            data=df, x=explanatory_columns[i * 4 + j],
            ax=axs[i, j], color=["skyblue", "olive", "gold", "teal", "blue", "yellow", "green", "pink"][i * 4 + j]
        )
plt.suptitle("Boxplots des variables", fontsize=16)
plt.show()

##############################################
### 3. Détection des outliers avec plusieurs méthodes
##############################################

# a) Méthode par l'Interquartile Range (IQR)
def find_outliers_IQR(dataf):
    Q1 = dataf.quantile(0.25)
    Q3 = dataf.quantile(0.75)
    IQR = Q3 - Q1
    outliers = dataf[((dataf < (Q1 - 1.5 * IQR)) | (dataf > (Q3 + 1.5 * IQR)))]
    return outliers

outliers_IQR = find_outliers_IQR(df)
print("\nOutliers détectés avec IQR :")
print(outliers_IQR)

# b) Méthode Z-score
z_scores = np.abs(stats.zscore(df))
threshold = 3
outliers_z = df[(z_scores > threshold).any(axis=1)]
print("\nOutliers détectés avec Z-score :")
print(outliers_z)

# c) Isolation Forest
iso_forest = IsolationForest(contamination=0.05)
outlier_labels = iso_forest.fit_predict(df)
outliers_iso_forest = df[outlier_labels == -1]
print("\nOutliers détectés avec Isolation Forest :")
print(outliers_iso_forest)

# d) Elliptic Envelope
elliptic = EllipticEnvelope(contamination=0.05)
outlier_labels = elliptic.fit_predict(df)
outliers_elliptic = df[outlier_labels == -1]
print("\nOutliers détectés avec Elliptic Envelope :")
print(outliers_elliptic)

##############################################
### 4. Traitement des outliers
##############################################

# a) Suppression des outliers
def drop_outliers(dataf):
    Q1 = dataf.quantile(0.25)
    Q3 = dataf.quantile(0.75)
    IQR = Q3 - Q1
    filtered = dataf[~((dataf < (Q1 - 1.5 * IQR)) | (dataf > (Q3 + 1.5 * IQR))).any(axis=1)]
    return filtered

df_filtered = drop_outliers(df)

# b) Capping des outliers à 3 écarts-types de la moyenne
def cap_outliers(dataf, feature):
    upper_limit = dataf[feature].mean() + 3 * dataf[feature].std()
    lower_limit = dataf[feature].mean() - 3 * dataf[feature].std()
    dataf[feature] = np.where(
        dataf[feature] > upper_limit, upper_limit,
        np.where(dataf[feature] < lower_limit, lower_limit, dataf[feature])
    )
    return dataf

df_capped = df.copy()
for feature in explanatory_columns:
    cap_outliers(df_capped, feature)

##############################################
### 5. Imputation des valeurs manquantes
##############################################

# a) Imputation simple par la moyenne
imputer_mean = SimpleImputer(strategy='mean')
df_imputed_mean = pd.DataFrame(imputer_mean.fit_transform(df), columns=df.columns)

# b) Imputation par KNN
imputer_knn = KNNImputer(n_neighbors=5)
df_imputed_knn = pd.DataFrame(imputer_knn.fit_transform(df), columns=df.columns)

# c) Imputation par régression (Iterative Imputer)
imputer_iter = IterativeImputer()
df_imputed_iter = pd.DataFrame(imputer_iter.fit_transform(df), columns=df.columns)

##############################################
### 6. Normalisation et Standardisation des données
##############################################

# a) Standardisation des données
scaler_standard = StandardScaler()
df_scaled_standard = pd.DataFrame(scaler_standard.fit_transform(df), columns=df.columns)

# b) Normalisation des données (MinMaxScaler)
scaler_minmax = MinMaxScaler()
df_scaled_minmax = pd.DataFrame(scaler_minmax.fit_transform(df), columns=df.columns)

##############################################
### 7. Encodage des variables catégorielles
##############################################

# Exemple d'encodage d'une variable catégorique (ici on crée une variable fictive)
df['Category'] = pd.cut(df['Age'], bins=3, labels=['Young', 'Middle', 'Old'])

# a) One-Hot Encoding
encoder_ohe = OneHotEncoder(sparse=False)
df_encoded_ohe = pd.DataFrame(encoder_ohe.fit_transform(df[['Category']]), columns=encoder_ohe.get_feature_names_out())

# b) Label Encoding
encoder_le = LabelEncoder()
df['Category_encoded'] = encoder_le.fit_transform(df['Category'])

# c) Ordinal Encoding (pour des catégories ordonnées)
encoder_ordinal = OrdinalEncoder(categories=[['Young', 'Middle', 'Old']])
df['Category_ordinal'] = encoder_ordinal.fit_transform(df[['Category']])

# d) Target Encoding
df['Category_target_encoded'] = df.groupby('Category')['Outcome'].transform('mean')

# e) Frequency Encoding
freq_encoding = df['Category'].value_counts(normalize=True)
df['Category_freq_encoded'] = df['Category'].map(freq_encoding)

# f) Binary Encoding
encoder_bin = ce.BinaryEncoder()
df_encoded_bin = encoder_bin.fit_transform(df[['Category']])

# g) Leave-One-Out Encoding
encoder_loo = ce.LeaveOneOutEncoder(cols=['Category'])
df_encoded_loo = encoder_loo.fit_transform(df['Category'], df['Outcome'])

##############################################
### 8. Visualisation des transformations finales
##############################################

# a) Histogrammes après transformation
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
for i in range(2):
    for j in range(4):
        sns.histplot(
            data=df_scaled_standard, x=explanatory_columns[i * 4 + j], kde=False,
            ax=axs[i, j], color=["skyblue", "olive", "gold", "teal", "blue", "yellow", "green", "pink"][i * 4 + j]
        )
plt.suptitle("Histograms après standardisation", fontsize=16)
plt.show()

# b) Boxplots après transformation
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
for i in range(2):
    for j in range(4):
        sns.boxplot(
            data=df_scaled_standard, x=explanatory_columns[i * 4 + j],
            ax=axs[i, j], color=["skyblue", "olive", "gold", "teal", "blue", "yellow", "green", "pink"][i * 4 + j]
        )
plt.suptitle("Boxplots après standardisation", fontsize=16)
plt.show()
