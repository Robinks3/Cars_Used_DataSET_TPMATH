import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.formula.api import ols
from sklearn.decomposition import PCA

# Charger le fichier CSV
file_path = 'C:\\Users\\rfhba\\Documents\\ETUDE\\ENSIBS\\S6\\Math\\Cars_Used_DataSET_TPMATH\\DataSetUserCarData\\UserCarData.csv'
Data = pd.read_csv(file_path, sep=',')
Data.columns = [col.strip() for col in Data.columns]  # Supprimer les espaces autour des noms de colonnes

# Afficher les premières lignes pour vérifier le chargement
print(Data.head())

# Sélectionner des colonnes pour l'analyse
columns = ["year", "selling_price", "km_driven", "mileage", "engine", "max_power", "seats"]
Data = Data[columns]

# Analyse descriptive
print(Data.describe())

# Histogrammes
Data.hist(bins=30, figsize=(15, 10))
plt.show()

# Boîtes à moustaches
Data.boxplot(figsize=(15, 10))
plt.show()

# Relation entre le prix de vente et les kilomètres parcourus
plt.figure(figsize=(10, 6))
sns.scatterplot(x='km_driven', y='selling_price', data=Data)
plt.xlabel('Kilomètres parcourus')
plt.ylabel('Prix de vente')
plt.title('Relation entre le prix de vente et les kilomètres parcourus')
plt.show()

# Matrice de corrélation
correlation_matrix = Data.corr()
print(correlation_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de Corrélation')
plt.show()

# Régression linéaire simple : selling_price ~ km_driven
X = sm.add_constant(Data['km_driven'])
model = sm.OLS(Data['selling_price'], X).fit()
print(model.summary())

# Régression linéaire simple : selling_price ~ year
X = sm.add_constant(Data['year'])
model = sm.OLS(Data['selling_price'], X).fit()
print(model.summary())

# Régression linéaire multiple
X = Data[['km_driven', 'year', 'mileage', 'max_power', 'engine', 'seats']]
Y = Data['selling_price']
X = sm.add_constant(X)  # Ajouter une constante pour le terme d'interception

model = sm.OLS(Y, X).fit()
predictions = model.predict(X)

print(model.summary())

# PCA (Analyse en Composantes Principales)
pca = PCA(n_components=2)
components = pca.fit_transform(Data[['km_driven', 'year', 'mileage', 'max_power', 'engine']].dropna())
plt.figure(figsize=(10, 6))
plt.scatter(components[:, 0], components[:, 1], c=Data['selling_price'], cmap='viridis')
plt.colorbar(label='Prix de vente')
plt.xlabel('Composante Principale 1')
plt.ylabel('Composante Principale 2')
plt.title('PCA des caractéristiques de la voiture')
plt.show()