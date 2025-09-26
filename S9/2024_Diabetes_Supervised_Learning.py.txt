# -*- coding: utf-8 -*-
"""
Démonstration des techniques SVM sur le dataset diabetes.

@author: Farida
"""

# Importation des bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Chargement du dataset
try:
    Data = pd.read_csv("diabetes.csv", sep=',')
    print("Données chargées avec succès.")
except FileNotFoundError:
    print("Erreur : Fichier non trouvé. Veuillez vérifier le chemin du fichier.")
    exit()

# Vérification de la présence de valeurs manquantes
if Data.isnull().values.any():
    print("Attention : Le dataset contient des valeurs manquantes. Pensez à nettoyer les données.")
else:
    print("Aucune valeur manquante détectée.")

# Définition des noms des variables explicatives et de la variable cible
target_name = "Outcome"
explanatory_columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# Extraction des caractéristiques (X) et de la cible (y)
X = Data[explanatory_columns]
y = Data[target_name]

# Séparation des données en un jeu d'entraînement et un jeu de test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisation des caractéristiques (étape essentielle pour SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

############
## Classificateur SVM avec noyau linéaire (paramètres par défaut)
###########
svm_clf = SVC(kernel='linear', random_state=42, probability=True)
svm_clf.fit(X_train, y_train)

# Prédictions sur le jeu de test
y_pred = svm_clf.predict(X_test)

# Évaluation du modèle SVM linéaire
print("\n--- Évaluation du modèle SVM linéaire ---")
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# Calcul du score ROC-AUC pour le SVM linéaire
auc = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC : {auc:.2f}")

# Tracé de la courbe ROC pour le SVM linéaire
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure()
plt.plot(fpr, tpr, label=f'Courbe ROC (AUC = {auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='Classificateur aléatoire')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbe ROC - SVM linéaire')
plt.legend(loc="lower right")
plt.show()

# Optimisation des hyperparamètres avec Grid Search et validation croisée (noyaux RBF et linéaire)
param_grid = [
    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000], 'probability': [True]},
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000], 'probability': [True]}
]

# Grid Search avec validation croisée (5 plis)
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Affichage des meilleurs paramètres trouvés par Grid Search
print("\n--- Meilleurs paramètres trouvés avec Grid Search ---")
print(grid_search.best_params_)

# Utilisation du meilleur modèle pour prédire sur le jeu de test
best_clf = grid_search.best_estimator_
y_pred_best = best_clf.predict(X_test)

# Évaluation du meilleur modèle
print("\n--- Évaluation du meilleur modèle ---")
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred_best))

print("\nRapport de classification :")
print(classification_report(y_test, y_pred_best))

# Calcul du score ROC-AUC pour le meilleur modèle
auc_best = roc_auc_score(y_test, y_pred_best)
print(f"ROC-AUC (Meilleur modèle) : {auc_best:.2f}")

# Tracé de la courbe ROC pour le meilleur modèle
fpr_best, tpr_best, thresholds_best = roc_curve(y_test, y_pred_best)
plt.figure()
plt.plot(fpr_best, tpr_best, label=f'Courbe ROC (AUC = {auc_best:.2f})', color='green')
plt.plot([0, 1], [0, 1], 'k--', label='Classificateur aléatoire')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbe ROC - Meilleur modèle SVM')
plt.legend(loc="lower right")
plt.show()

# Données de prédiction (un exemple fictif pour un patient)
# Attention, les nouvelles données doivent avoir exactement les mêmes caractéristiques que celles utilisées pour entraîner le modèle
new_data = pd.DataFrame({
    "Pregnancies": [2],
    "Glucose": [120],
    "BloodPressure": [80],
    "SkinThickness": [20],
    "Insulin": [85],
    "BMI": [32.0],
    "DiabetesPedigreeFunction": [0.5],
    "Age": [35]
})

# Standardisation des nouvelles données
new_data_scaled = scaler.transform(new_data)

# Prédiction avec le modèle entraîné (meilleur modèle trouvé avec Grid Search)
predicted_class = best_clf.predict(new_data_scaled)

# Affichage du résultat de la prédiction
print(f"Prédiction pour le patient : {'Diabète' if predicted_class[0] == 1 else 'Pas de diabète'}")

# Si on veut aussi obtenir la probabilité associée à chaque classe
if hasattr(best_clf, "predict_proba"):
    predicted_proba = best_clf.predict_proba(new_data_scaled)
    print(f"Probabilité de développer le diabète : {predicted_proba[0][1]:.2f}")
else:
    print("Le modèle ne supporte pas la méthode 'predict_proba'.")

##########################
### Régression logistique binaire 
##########################

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, log_loss, cohen_kappa_score, matthews_corrcoef
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import chi2

# Supposons que vous ayez déjà nettoyé le dataset Diabetes

# Chargement du dataset Diabetes
Data = pd.read_csv("diabetes.csv")

# Définir les variables explicatives et la variable cible
X = Data.drop(columns=['Outcome'])  # Features
y = Data['Outcome']  # Cible (0 = pas de diabète, 1 = diabète)

# Séparation des données en un jeu d'entraînement et un jeu de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Standardisation des caractéristiques (très important pour la régression logistique)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Modèle de régression logistique binaire
model = LogisticRegression()

# Entraînement du modèle
model.fit(X_train_scaled, y_train)

# 3. Vérifications des conditions de validité
# -------------------------------------------
# Multicolinéarité avec le VIF (Facteur d'inflation de la variance)
X_train_vif = pd.DataFrame(X_train_scaled, columns=X.columns)
X_train_vif = add_constant(X_train_vif)  # Ajout de la constante pour statsmodels

vif_data = pd.DataFrame()
vif_data["Feature"] = X_train_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_vif.values, i) for i in range(X_train_vif.shape[1])]
print("\nFacteurs d'inflation de la variance (VIF) :")
print(vif_data)

# Interprétation :
# Un VIF supérieur à 10 indique une multicolinéarité élevée, ce qui peut affecter les coefficients du modèle.

# Analyse des corrélations entre les variables explicatives
plt.figure(figsize=(10, 8))
sns.heatmap(X_train_vif.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5,vmin=-1,vmax=1)
plt.title("Matrice de corrélation des variables explicatives")
plt.show()

# Interprétation :
# Une corrélation élevée (> 0.7 ou < -0.7) indique une forte redondance entre deux variables explicatives,
# ce qui peut aggraver la multicolinéarité. Envisagez de combiner ou supprimer ces variables.

# Relation linéaire entre les variables explicatives et le logit
X_train_logit = pd.DataFrame(X_train_scaled, columns=X.columns)
X_train_logit['logit'] = model.predict_proba(X_train_scaled)[:, 1]

# Tracé des graphes pour chaque variable vs logit
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for i, col in enumerate(X_train_logit.columns[:-1]):
    ax = axes[i//3, i%3]
    ax.scatter(X_train_logit[col], X_train_logit['logit'], alpha=0.5)
    ax.set_title(f'Relation {col} et logit')

plt.tight_layout()
plt.show()

# Interprétation :
# Si les points suivent une tendance linéaire, la relation est valide.
# Sinon, des transformations (logarithmiques, quadratiques) peuvent être nécessaires.

# Vérification des valeurs aberrantes (outliers) avec la distance de Cook
X_train_const = add_constant(X_train_scaled)
logit_model = sm.Logit(y_train, X_train_const)
logit_result = logit_model.fit()

# Résumé du modèle
print(logit_result.summary())

# Calcul des probabilités prédites
y_hat = logit_result.predict(X_train_const)

# Calcul des résidus de Pearson
residuals_pearson = (y_train - y_hat) / np.sqrt(y_hat * (1 - y_hat))

# Calcul des valeurs de levier (hat values)
hat_matrix_diag = np.diag(X_train_const @ np.linalg.inv(X_train_const.T @ X_train_const) @ X_train_const.T)

# Calcul de la distance de Cook
n = X_train_const.shape[0]  # Nombre d'observations
p = X_train_const.shape[1]  # Nombre de paramètres (y compris la constante)

cook_distance = residuals_pearson**2 / p * hat_matrix_diag / (1 - hat_matrix_diag)

# Tracé de la distance de Cook
plt.figure(figsize=(10, 6))
plt.scatter(np.arange(len(cook_distance)), cook_distance, color='blue', label="Distance de Cook")
plt.axhline(y=4 / n, color='red', linestyle='--', label='Seuil de 4/n')
plt.title("Distance de Cook pour détecter les points influents")
plt.xlabel("Observation")
plt.ylabel("Distance de Cook")
plt.legend(loc="upper right")
plt.show()

# Tracé des résidus binaires pour vérifier l'ajustement du modèle
plt.figure(figsize=(10, 6))
plt.scatter(np.arange(len(residuals_pearson)), residuals_pearson, alpha=0.6, color='purple')
plt.axhline(0, color='red', linestyle='--', label='Résidus = 0')
plt.title("Résidus de Pearson")
plt.xlabel("Observation")
plt.ylabel("Résidus de Pearson")
plt.legend(loc="upper right")
plt.show()

# Interprétation :
# Les points avec des résidus significativement éloignés de 0 (au-delà de ±3) peuvent indiquer des observations mal ajustées.

# 4. Évaluation des performances du modèle
# -----------------------------------------
# Prédictions sur le jeu de test
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probabilité pour la classe positive (classe 1)

# Précision globale
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Matrice de confusion
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

# Rapport de classification (precision, recall, f1-score)
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# Calcul du score ROC-AUC
auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC : {auc:.2f}")

# Tracé de la courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'Courbe ROC (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Classificateur aléatoire')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbe ROC - Régression Logistique Binaire')
plt.legend(loc="lower right")
plt.show()

# Explication de la courbe ROC-AUC :
# La courbe ROC montre la performance du modèle en termes de compromis entre le "taux de vrais positifs" (TPR, ou sensibilité)
# et le "taux de faux positifs" (FPR). 
# - L'axe des Y correspond au TPR (taux de vrais positifs) : il indique à quelle fréquence le modèle prédit correctement
#   une classe positive (détecter un diabète).
# - L'axe des X correspond au FPR (taux de faux positifs) : il montre à quelle fréquence le modèle prédit à tort une classe
#   positive (prédit diabète alors qu'il n'y en a pas).

# 5. Prédiction pour un nouvel exemple de patient
# --------------------------------------------
new_data = pd.DataFrame({
    "Pregnancies": [2],
    "Glucose": [120],
    "BloodPressure": [80],
    "SkinThickness": [20],
    "Insulin": [85],
    "BMI": [32.0],
    "DiabetesPedigreeFunction": [0.5],
    "Age": [35]
})

# Standardisation des nouvelles données avec le même scaler utilisé pour l'entraînement
new_data_scaled = scaler.transform(new_data)

# Prédiction de la classe avec le modèle de régression logistique
predicted_class = model.predict(new_data_scaled)
print(f"Prédiction pour le patient : {'Diabète' if predicted_class[0] == 1 else 'Pas de diabète'}")

# Affichage des probabilités associées aux classes
predicted_proba = model.predict_proba(new_data_scaled)
print(f"Probabilités associées aux classes : {predicted_proba[0]}")

# 6. Vérifications supplémentaires
# ---------------------------------
# 6.1. Log-Loss
log_loss_value = log_loss(y_test, y_proba)
print(f"Log-Loss: {log_loss_value:.2f}")

# Interprétation :
# Le log-loss mesure la différence entre les probabilités prédites et les classes observées. 
# Un log-loss plus bas indique de meilleures prédictions probabilistes. Contrairement à la précision, 
# il punit davantage les prédictions confidentes qui sont incorrectes.


# 6.2. Test de Hosmer-Lemeshow
# Prédictions sur les probabilités
predicted_proba = model.predict_proba(X_test_scaled)[:, 1]
# Diviser les probabilités prédites en 10 groupes égaux
groups = pd.qcut(predicted_proba, 10, duplicates='drop')

# Créer un tableau croisé des groupes et des observations
observed_counts = pd.crosstab(groups, y_test)

# Calculer les taux observés et attendus
observed_events = observed_counts[1]
observed_non_events = observed_counts[0]
total_counts = observed_events + observed_non_events

# Probabilités moyennes pour chaque groupe
predicted_group_means = pd.Series(predicted_proba).groupby(groups).mean()

# Calcul des événements attendus pour chaque groupe
expected_events = predicted_group_means * total_counts
expected_non_events = total_counts - expected_events

# Calculer le chi2 pour comparer les taux observés et attendus
hosmer_lemeshow_stat = np.sum((observed_events - expected_events)**2 / expected_events + 
                              (observed_non_events - expected_non_events)**2 / expected_non_events)

# Le test de Hosmer-Lemeshow suit une distribution du chi2 avec 8 degrés de liberté (10 groupes - 2)
p_value = 1 - chi2.cdf(hosmer_lemeshow_stat, df=8)

# Affichage des résultats
print(f"Hosmer-Lemeshow Test: Chi2 = {hosmer_lemeshow_stat:.2f}, p-value = {p_value:.2f}")

# Interprétation :
# Un p-value élevé (> 0.05) suggère que le modèle s'ajuste bien aux données observées.
# Si le p-value est faible (< 0.05), cela signifie que le modèle ne correspond pas bien aux données.

# 6.3. Résidus standardisés vs valeurs prédites
plt.figure(figsize=(10, 6))
plt.scatter(y_hat, residuals_pearson, alpha=0.6, color='green')
plt.axhline(0, color='red', linestyle='--')
plt.title("Résidus de Pearson vs Valeurs prédites")
plt.xlabel("Valeurs prédites (logit)")
plt.ylabel("Résidus de Pearson")
plt.show()

# Interprétation :
# Ce graphe permet de vérifier si les résidus sont répartis de manière aléatoire autour de 0. 
# Si une tendance systématique apparaît (par exemple, des résidus plus grands pour certaines valeurs prédites), 
# cela pourrait indiquer que le modèle n'est pas bien ajusté et des transformations supplémentaires pourraient être nécessaires.

# 6.4. Note sur l'indépendance des observations
# --------------------------------------------
# En régression logistique, il est supposé que les observations sont indépendantes les unes des autres.
# Cela signifie qu'il ne doit pas y avoir de relation ou de dépendance entre les observations dans les données.
# Cette condition est souvent garantie par la conception de l'étude ou la manière dont les données ont été collectées.

# Si les données proviennent d'une étude où les observations sont groupées (par exemple, patients dans différents hôpitaux ou classes d'élèves),
# il serait nécessaire d'utiliser des **modèles de régression logistique hiérarchiques** ou **des modèles mixtes** pour prendre en compte ces dépendances.

# Si les observations sont indépendantes, il n'est pas nécessaire d'ajuster le modèle pour cela.


#########################
### Régression logistique multinomiale avec le dataset Iris
#########################
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, cohen_kappa_score, matthews_corrcoef
from statsmodels.tools import add_constant
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Chargement du dataset Iris pour une régression logistique multinomiale
from sklearn.datasets import load_iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Séparation en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Standardisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modèle de régression logistique multinomiale
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train_scaled, y_train)

# 2. Prédictions et évaluation du modèle
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

# 3. Vérification des conditions de validité

# 3.1 Multicolinéarité avec le VIF
# ---------------------------------
X_train_vif = pd.DataFrame(X_train_scaled, columns=iris.feature_names)
X_train_vif = add_constant(X_train_vif)  # Ajout de la constante pour statsmodels

vif_data = pd.DataFrame()
vif_data["Feature"] = X_train_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_vif.values, i) for i in range(X_train_vif.shape[1])]
print("\nFacteurs d'inflation de la variance (VIF) :")
print(vif_data)

# Interprétation des VIF :
# Un VIF supérieur à 10 indique une multicolinéarité élevée entre certaines variables,
# ce qui peut rendre difficile l'estimation précise des coefficients dans le modèle.

# 3.2 Relation linéaire entre les variables explicatives et le logit
# -----------------------------------------------------------------
X_train_logit = pd.DataFrame(X_train_scaled, columns=iris.feature_names)
X_train_logit['logit_class_0'] = model.predict_proba(X_train_scaled)[:, 0]  # Pour Setosa
X_train_logit['logit_class_1'] = model.predict_proba(X_train_scaled)[:, 1]  # Pour Versicolor
X_train_logit['logit_class_2'] = model.predict_proba(X_train_scaled)[:, 2]  # Pour Virginica

# Tracé des graphes pour chaque variable vs logit pour chaque classe
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for i, col in enumerate(X_train_logit.columns[:-3]):
    for j in range(3):  # Pour chaque classe
        ax = axes[j, i]
        ax.scatter(X_train_logit[col], X_train_logit[f'logit_class_{j}'])
        ax.set_title(f'Relation {col} et logit (classe {iris.target_names[j]})')

plt.tight_layout()
plt.show()

# Interprétation :
# Si les points suivent une tendance linéaire, la relation est valide.
# Sinon, des transformations (logarithmes, puissances) peuvent être nécessaires.

# 3.3 Valeurs aberrantes (outliers) avec la distance de Cook
# ----------------------------------------------------------
X_train_const = add_constant(X_train_scaled)
logit_model = Logit(y_train, X_train_const)
logit_result = logit_model.fit()

# Calcul des résidus standardisés et distance de Cook
influence = logit_result.get_influence()
summary_frame = influence.summary_frame()

# Tracé de la distance de Cook
plt.figure(figsize=(10, 6))
plt.scatter(np.arange(len(summary_frame)), summary_frame['cooks_d'], color='blue')
plt.axhline(y=4/len(X_train), color='red', linestyle='--')  # Seuil pour la distance de Cook
plt.title("Distance de Cook pour détecter les valeurs aberrantes")
plt.xlabel("Observation")
plt.ylabel("Distance de Cook")
plt.show()

# Interprétation :
# Les points au-dessus de la ligne rouge sont des outliers ayant un effet disproportionné sur le modèle.
# Ces points doivent être examinés de plus près et potentiellement retirés.

# 3.4 Vérification de la taille de l'échantillon et événements par classe
# -----------------------------------------------------------------------
n_total = len(y_train)
n_events_class_0 = np.sum(y_train == 0)  # Setosa
n_events_class_1 = np.sum(y_train == 1)  # Versicolor
n_events_class_2 = np.sum(y_train == 2)  # Virginica
n_features = X_train_scaled.shape[1]

print(f"\nTaille totale de l'échantillon : {n_total}")
print(f"Nombre d'événements (Setosa) : {n_events_class_0}")
print(f"Nombre d'événements (Versicolor) : {n_events_class_1}")
print(f"Nombre d'événements (Virginica) : {n_events_class_2}")
print(f"Nombre de variables explicatives : {n_features}")

if n_events_class_0 >= 10 * n_features and n_events_class_1 >= 10 * n_features and n_events_class_2 >= 10 * n_features:
    print("Le nombre d'événements dans chaque classe est suffisant pour la régression logistique multinomiale.")
else:
    print("Le nombre d'événements dans une ou plusieurs classes est insuffisant. Essayez d'obtenir plus de données.")

# 4. Mesures de performance
# -------------------------
# Précision globale (Accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Interprétation :
# La précision mesure la proportion d'observations correctement classées par rapport au total.
# Par exemple, une précision de 0.95 signifie que 95% des prédictions étaient correctes.
# Cependant, la précision seule ne suffit pas si les classes sont déséquilibrées (peu de données dans certaines classes).

# Matrice de confusion
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

# Interprétation :
# La matrice de confusion indique combien de fois chaque classe a été prédite correctement (diagonale),
# et combien de fois une classe a été confondue avec une autre (hors-diagonale).
# Elle aide à comprendre où le modèle se trompe et si certaines classes sont particulièrement mal prédites.

# Rapport de classification (precision, recall, f1-score)
print("Rapport de classification :")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Interprétation :
# - **Précision** (Precision) : Parmi les prédictions faites pour une classe donnée, combien sont correctes ?
# - **Rappel** (Recall) : Parmi toutes les observations réelles d'une classe donnée, combien ont été bien prédictes ?
# - **F1-score** : Moyenne harmonique de la précision et du rappel. C'est une mesure globale utile lorsque les données sont déséquilibrées (importance égale au rappel et à la précision).
# Ces trois scores sont donnés pour chaque classe (Setosa, Versicolor, Virginica).

# Log-loss (Logarithmic Loss)
logloss = log_loss(y_test, y_proba)
print(f"Log-Loss: {logloss:.2f}")

# Interprétation :
# Le log-loss mesure la performance d'un modèle probabiliste. Il compare les probabilités prédites pour chaque classe
# avec les vraies étiquettes. Plus le log-loss est bas, meilleure est la qualité du modèle.
# Une valeur de log-loss proche de 0 signifie que les probabilités prédites sont très proches des vraies valeurs.

# Cohen's Kappa (Cohen’s Kappa Score)
kappa = cohen_kappa_score(y_test, y_pred)
print(f"Cohen's Kappa: {kappa:.2f}")

# Interprétation :
# Le Kappa de Cohen mesure l'accord entre les prédictions du modèle et les valeurs réelles, tout en tenant compte
# de l'accord attendu par hasard. Un score de 1 signifie un accord parfait, et un score proche de 0 signifie
# que le modèle ne fait pas mieux qu'un tirage au sort. Si Kappa > 0.8, le modèle a une bonne performance.

# Matthews Correlation Coefficient (MCC)
mcc = matthews_corrcoef(y_test, y_pred)
print(f"Matthews Correlation Coefficient (MCC): {mcc:.2f}")

# Interprétation :
# Le MCC est une mesure qui tient compte de toutes les valeurs dans la matrice de confusion (vrais positifs, faux positifs,
# vrais négatifs, faux négatifs). C'est une mesure globale plus équilibrée, particulièrement utile si les classes
# sont déséquilibrées. Un score de 1 signifie une bonne corrélation (toutes les classes bien prédites), tandis qu'un score
# de 0 signifie que les prédictions ne sont pas meilleures qu'un choix aléatoire.

# 5. Prédiction pour un nouvel exemple de fleur
# --------------------------------------------
new_data = pd.DataFrame({
    "sepal length (cm)": [5.1],
    "sepal width (cm)": [3.5],
    "petal length (cm)": [1.4],
    "petal width (cm)": [0.2]
})

# Standardisation des nouvelles données avec le même scaler utilisé pour l'entraînement
new_data_scaled = scaler.transform(new_data)

# Prédiction de la classe avec le modèle de régression logistique
predicted_class = model.predict(new_data_scaled)
print(f"Prédiction pour la fleur : {iris.target_names[predicted_class[0]]}")

# Interprétation :
# Ici, on prédit la classe de la fleur (Setosa, Versicolor, ou Virginica) en fonction de ses caractéristiques
# (longueur/largeur des sépales et pétales). Le modèle renvoie la classe la plus probable pour cette nouvelle observation.

# Affichage des probabilités associées aux classes
predicted_proba = model.predict_proba(new_data_scaled)
print(f"Probabilités associées aux classes : {predicted_proba[0]}")

# Interprétation :
# Le modèle prédit également les probabilités pour chaque classe. Par exemple, il pourrait prédire que
# la probabilité que la fleur soit Setosa est de 97%, Versicolor de 2%, et Virginica de 1%.
# Cela peut être utile pour des décisions plus nuancées où une classe a une probabilité proche des autres.
