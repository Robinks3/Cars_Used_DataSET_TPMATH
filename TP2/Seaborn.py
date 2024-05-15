import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats

Data = pd.read_csv("Abena_Data.csv", sep='\;')
colums=["urbaine", "age", "couple", "enfants", "scolaire", "situation", "repas", "duree", "assurance", "imc"]
urbaine_data, age_data, couple_data, enfants_data, scolaire_data, situation_data, repas_data, duree_data, assurance_data, imc_data = [Data[col] for col in colums]


"""
plt.figure(figsize=(10,5)) 
plt.subplot(1,2,1) 
sns.boxplot(x=data_imc, color='lightblue', notch=True, flierprops={'marker':'o', 'markersize':6, 'markerfacecolor':'red'})
plt.subplot(1,2,2) 
sns.histplot(data_age)
"""


def plot_qqplot(data, title):
    plt.figure()
    sm.qqplot(data, line='45')
    plt.title(f'0Q Plot of {title}')
    plt.show()
    
#plot_qqplot(data_age, 'Age')

def plot_categorical_distribution(data,category_labels,plot_title, figsize=(12,6)):
    frequencies = data.value_counts(normalize=True)
    fig,ax = plt.subplots(2,1,figsize=figsize)
    ax[0].pie(frequencies, labels=category_labels, autopct='%1.1f%%')
    ax[0].set_title(f'Pie Chart of {plot_title}')
    sns.barplot(x=frequencies.index, y=frequencies.values, ax=ax[1])
    ax[1].set_title(f'Bar Chart of {plot_title}')
    plt.show()

#plot_qqplot(age_data, 'Age')
urban_categories = ['Val-de-Marne','Seine-Saint-Denis','Paris','Marseille','Hauts-de-France','Dijon']
plot_categorical_distribution(urbaine_data, urban_categories, 'Urban distribution')




def shapiro_wilk_test(data):
    # Remove NaN values which can't be handled by the Shapiro-Wilk test
    data_clean = data.dropna()

    # Performing the Shapiro-Wilk test
    stat, p_value = stats.shapiro(data_clean)

    # Interpreting the results
    alpha = 0.05
    if p_value > alpha:
        print("X looks Gaussian (fail to reject H0)")
    else:
        print("X does not look Gaussian (reject H0)")

    return stat, p_value

# Call the function
test_statistic, p_value = shapiro_wilk_test(age_data)
print(f"Shapiro-Wilk Test Statistic: {test_statistic}, P-Value: {p_value}")




#TestdeStudent = stats.ttest_1samp(data_age, mu)





######## TEST sur une et eux population

#One sample t-test
#testing if the mean of the group1 is signifanctly different form the population mean of 25
t_statistic, p_value = stats.ttest_1samp(age_data, popmean =45, alternative = 'two-sided')
#alternative ='two sided', 'less', 'greater'
print("One sample t-test statistic :", t_statistic, "P value :", p_value)

#two sample t test
#filtrer les ages en 2 groupes : seule vs couple
#filter to get ages of women living alone
age_seule = Data[Data['couple']== 'seule']['age']

#filter to get ages of women living in couple
age_couple = Data[Data['couple']=='couple']['age']

# Test de Levene d'égalité des variances (moins sensible à l'écart de normalité que Bartlett par exemple)
# Perform Levene's test for equal variances
stat, p_value = stats.levene(age_seule, age_couple)
print(f"Levene's Test Statistic: {stat:.3f}, P-value: {p_value:.3f}")

# Testing if there's a significant difference between the means of two independent groups
t_statistic, p_value = stats.ttest_ind(age_seule, age_couple, equal_var=True, alternative='two-sided') # Assumes equal variance, can be set to False
print("Two-sample t-test statistic:", t_statistic, "P-value:", p_value)


from scipy.stats import chi2_contingency

# Contingency Table
contingency_table = pd.crosstab(scolaire_data, situation_data)
print("Contingency Table between 'scolaire' and 'situation':")
print(contingency_table)

# Chi-square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square Test: \nChi2 Statistic: {chi2}, P-value: {p}")

# Check if the result is statistically significant
if p < 0.05:
    # Calculate Cramer's V
    n = contingency_table.sum().sum() # Total sample size
    phi2 = chi2 / n
    v = np.sqrt(phi2 / min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1))

    # Adjusted interpretation based on degrees of freedom
    def interpret_cramers_v(v, dof, n):
        adjusted_v = v / np.sqrt(dof / n)
        if adjusted_v < 0.10:
            return 'Très faible'
        elif adjusted_v < 0.20:
            return 'Faible'
        elif adjusted_v < 0.40:
            return 'Modérée'
        elif adjusted_v < 0.60:
            return 'Forte'
        elif adjusted_v < 0.80:
            return 'Très forte'
        else:
            return 'Extremement forte'
    print(f"Cramer's V : {v:.3f}")
    print(f"Adjusted interpretation based on dof({dof}) : {interpret_cramers_v(v, dof,n)}")


# plt.title('Heatmap of Contingency Table between 'urbaine' and 'couple'')
# plt.ylabel('Urbaine')
# plt.xlabel('Couple')
# plt.show()

# Calculate Standardized Residuals
standardized_residuals = (contingency_table - expected) / np.sqrt(expected)
print(standardized_residuals)

# Plot heatmap of standardized residuals
plt.figure(figsize=(10, 5))
sns.heatmap(standardized_residuals, annot=True, cmap="coolwarm", center=0)
plt.title('Heatmap of Standardized Residuals')
plt.ylabel('Urbaine')
plt.xlabel('Couple')
plt.show()

# Determine significant attractions and repulsions
print('Significant interactions.')
for index, row in standardized_residuals.iterrows():
    for col, value in row.items():
        if value > 1.96:
            print(f'Attraction between {index} and {col} (Residual: {value:.2f})')
        elif value < -1.96:
            print(f'Repulsion between {index} and {col} (Residual: {value:.2f})')
        else:
            print('No significant association found between the variables.')
            
            
            
########Test de comparaison de plusieurs moyennes#########ANOVA
from statsmodels.formula.api import ols
from scipy.stats import levene
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Setting the aesthetic style for better visualization
sns.set(style="whitegrid")

# Creating boxplots
plt.figure(figsize=(10, 6))  # Taille de la figure
sns.boxplot(x='situation', y='imc', data=Data, notch=True)
plt.title('Distribution de l\'IMC par situation professionnelle')
plt.xlabel('Situation')
plt.ylabel('Indice de Masse Corporelle (IMC)')
plt.show()

# Réalisation de l'ANOVA
model = ols('imc ~ C(situation)', data=Data).fit()
anova_results = sm.stats.anova_lm(model, typ=2)  # typ=2 pour ANOVA de type II

print(anova_results)



# Vérification de la significativité de l'ANOVA
if anova_results["PR(>F)"][0] < 0.05:
    print("L'ANOVA est significative, procédons au test de Tukey.")
    # Préparation des données pour le test de Tukey
    tukey_results = pairwise_tukeyhsd(endog=Data['imc'], groups=Data['situation'], alpha=0.05)
    print(tukey_results)
    
    # Affichage des résultats sous forme de tableau
    tukey_summary = tukey_results.summary()
    print(tukey_summary)
    
    # Affichage graphique des résultats du test de Tukey
    fig = tukey_results.plot_simultaneous(comparison_name=Data['situation'].iloc[0], figsize=(8, 6))
    plt.title('Comparaisons multiples - Test de Tukey HSD')
    plt.show()
else:
    print("L'ANOVA n'est pas significative, pas de besoin de test post-hoc.")