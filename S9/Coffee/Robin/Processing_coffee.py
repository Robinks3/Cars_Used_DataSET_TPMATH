# -*- coding: utf-8 -*-
"""
Prétraitement des données du projet café
Créé le 7 octobre 2025
@author: rfhba
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger le dataset avec le bon séparateur
df = pd.read_csv("synthetic_coffee_health_10000.csv", sep=';')

# Sélectionner uniquement les colonnes numériques
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
print("\nColonnes numériques détectées :", num_cols)

# Exclure les colonnes binaires (représentant Vrai/Faux par 0/1)
binary_cols = []
plot_cols = []
for col in num_cols:
    vals = set(df[col].dropna().unique())
    # considérer comme binaire si les valeurs distinctes non nulles ne sont que 0 et 1
    if vals <= {0, 1}:
        binary_cols.append(col)
    else:
        plot_cols.append(col)

print("Colonnes binaires exclues (0/1) :", binary_cols)

# Tracer 8 boxplots par figure (2 lignes x 4 colonnes), verticalement, avec informations
per_fig = 8
ncols = 4
nrows = 2

for i in range(0, len(plot_cols), per_fig):
    cols = plot_cols[i:i+per_fig]
    n = len(cols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        # boxplot vertical : on passe la série en y
        sns.boxplot(y=df[col], ax=ax, color="skyblue", showfliers=True, whis=1.5)

        # Calculs pour annotations
        series = df[col]
        mean = series.mean()
        median = series.median()
        count = int(series.count())
        missing = int(series.isna().sum())

        # Calcul des moustaches (whis=1.5 par défaut comme utilisé dans le boxplot)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        whis = 1.5
        lower_bound = q1 - whis * iqr
        upper_bound = q3 + whis * iqr

        # Nombre de points en dehors des moustaches (outliers selon la définition de whis)
        outliers = int(((series < lower_bound) | (series > upper_bound)).sum())

        # Ligne montrant la moyenne
        ax.axhline(mean, color='red', linestyle='--', linewidth=1)

        # Titre et label
        ax.set_title(f"Boxplot de {col}")
        ax.set_ylabel(col)

        # Annotation contextuelle en haut à droite de la sous-figure
        info = (
            f"n={count}\nmissing={missing}\noutliers={outliers}\n"
            f"mean={mean:.2f}\nmedian={median:.2f}"
        )
        ax.text(
            0.98,
            0.98,
            info,
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6),
        )

    # Masquer les axes inutilisés si on a moins de 8 colonnes dans la dernière figure
    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

# Afficher toutes les figures en même temps
plt.show()
