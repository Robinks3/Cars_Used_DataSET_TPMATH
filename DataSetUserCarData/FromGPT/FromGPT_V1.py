import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.decomposition import PCA

# Charger les données
file_path = 'C:\\Users\\rfhba\\Documents\\ETUDE\\ENSIBS\\S6\\Math\\Cars_Used_DataSET_TPMATH\\DataSetUserCarData\\UserCarData.csv'
car_data = pd.read_csv(file_path)

# Diviser les prix par 10
car_data['selling_price'] = car_data['selling_price'] / 10

# Analyse descriptive
descriptive_stats = car_data.describe()
print(descriptive_stats)

# Visualiser les distributions
sns.histplot(car_data['selling_price'], kde=True)
plt.title('Distribution des prix de vente')
plt.show()

sns.histplot(car_data['km_driven'], kde=True)
plt.title('Distribution des kilomètres parcourus')
plt.show()

# Test de Shapiro-Wilk pour la normalité
shapiro_test_price = stats.shapiro(car_data['selling_price'])
shapiro_test_km = stats.shapiro(car_data['km_driven'])
print(f'Test de Shapiro-Wilk pour le prix de vente: {shapiro_test_price}')
print(f'Test de Shapiro-Wilk pour les kilomètres parcourus: {shapiro_test_km}')

# Test du Chi-carré pour les variables catégorielles
contingency_table = pd.crosstab(car_data['fuel'], car_data['Region'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Test du chi-carré: chi2={chi2}, p-value={p}")

# ANOVA pour le prix de vente par région
anova_model = ols('selling_price ~ C(Region)', data=car_data).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print(f'ANOVA pour les régions: {anova_table}')

# Régression linéaire multiple
X = car_data[['year', 'km_driven', 'mileage', 'max_power']]
y = car_data['selling_price']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# Régression logistique
y_logistic = car_data['sold'].apply(lambda x: 1 if x == 'Y' else 0)
logistic_model = sm.Logit(y_logistic, X).fit()
print(logistic_model.summary())

# PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(car_data[['year', 'km_driven', 'mileage', 'max_power']])
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['selling_price'] = car_data['selling_price']
sns.scatterplot(x='PC1', y='PC2', hue='selling_price', data=pca_df, palette='viridis')
plt.title('PCA des caractéristiques techniques')
plt.show()

# Matrice de corrélation
correlation_matrix = car_data[['selling_price', 'km_driven', 'year', 'mileage', 'max_power']].corr()
print(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matrice de Corrélation')
plt.show()

# Test de Levene pour l'égalité des variances
levene_test = stats.levene(car_data[car_data['fuel'] == 'Petrol']['selling_price'],
                           car_data[car_data['fuel'] == 'Diesel']['selling_price'])
print(f'Test de Levene pour l\'égalité des variances entre types de carburant: {levene_test}')