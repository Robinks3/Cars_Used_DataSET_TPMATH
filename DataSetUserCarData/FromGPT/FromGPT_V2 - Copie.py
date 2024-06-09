import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

# Charger les données
file_path = 'C:\\Users\\rfhba\\Documents\\ETUDE\\ENSIBS\\S6\\Math\\Cars_Used_DataSET_TPMATH\\DataSetUserCarData\\UserCarData.csv'
data = pd.read_csv(file_path)

# Préparer les données
# Assurons-nous qu'il n'y a pas de valeurs manquantes dans les colonnes importantes
data = data.dropna(subset=['kilometrage', 'annee', 'puissance_max', 'prix'])

# Séparer les features et la cible
X = data[['kilometrage', 'annee', 'puissance_max']]
y = data['prix']

# Créer et ajuster le modèle
model = LinearRegression()
model.fit(X, y)

# Fonction pour prédire le prix et donner un intervalle de confiance à 95%
def predict_price_interval(kilometrage, annee, puissance_max, model, confidence=0.95):
    # Préparer les données d'entrée
    X_new = np.array([[kilometrage, annee, puissance_max]])
    
    # Faire la prédiction
    pred = model.predict(X_new)[0]
    
    # Obtenir les erreurs de prédiction
    preds = model.predict(X)
    residuals = y - preds
    s_err = np.sqrt(np.sum(residuals**2) / (len(y) - 2))
    
    # Calculer l'intervalle de confiance
    t_val = stats.t.ppf((1 + confidence) / 2, len(y) - 1)
    margin_of_error = t_val * s_err * np.sqrt(1 + 1 / len(y) + (X_new - np.mean(X, axis=0)).dot(np.linalg.inv(np.dot(X.T, X))).dot((X_new - np.mean(X, axis=0)).T))
    
    lower_bound = pred - margin_of_error
    upper_bound = pred + margin_of_error
    
    return pred, lower_bound, upper_bound

# Exemple d'utilisation
kilometrage = float(input("Entrez le kilométrage de la voiture: "))
annee = int(input("Entrez l'année de la voiture: "))
puissance_max = float(input("Entrez la puissance max de la voiture: "))

prediction, lower_bound, upper_bound = predict_price_interval(kilometrage, annee, puissance_max, model)

print(f"Le prix estimé de la voiture est de {prediction:.2f} € avec un intervalle de confiance de 95% allant de {lower_bound:.2f} € à {upper_bound:.2f} €")
