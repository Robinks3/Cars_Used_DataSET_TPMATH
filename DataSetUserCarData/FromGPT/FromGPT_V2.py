import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Charger le fichier CSV
file_path = 'C:\\Users\\rfhba\\Documents\\ETUDE\\ENSIBS\\S6\\Math\\Cars_Used_DataSET_TPMATH\\DataSetUserCarData\\UserCarData.csv'
Data = pd.read_csv(file_path, sep=',')
Data.columns = [col.strip() for col in Data.columns]  # Supprimer les espaces autour des noms de colonnes

# Afficher les premières lignes pour vérifier le chargement
print(Data.head())

# Sélectionner des colonnes pour l'analyse
columns = ["year", "selling_price", "km_driven", "mileage", "engine", "max_power", "seats", "name"]
Data = Data[columns]

# Convertir les variables catégorielles (nom/marque) en variables indicatrices
Data = pd.get_dummies(Data, columns=["name"], drop_first=True)

# Afficher les premières lignes après transformation
print(Data.head())

# Matrice de corrélation pour les variables numériques
correlation_matrix = Data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de Corrélation')
plt.show()

# Régression linéaire multiple
X = Data.drop(columns=['selling_price'])
Y = Data['selling_price']
X = sm.add_constant(X)  # Ajouter une constante pour le terme d'interception

model = sm.OLS(Y, X).fit()
predictions = model.predict(X)

print(model.summary())

# Visualisation des résidus pour vérifier l'ajustement du modèle
plt.figure(figsize=(10, 6))
sns.residplot(x=predictions, y=model.resid, lowess=True, line_kws={'color': 'red', 'lw': 2})
plt.xlabel('Valeurs Prédictes')
plt.ylabel('Résidus')
plt.title('Diagramme des Résidus')
plt.show()
