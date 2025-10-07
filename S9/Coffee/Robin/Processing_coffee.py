# -*- coding: utf-8 -*-
"""
Processing_coffee.py

Script de prétraitement léger pour le jeu de données synthétique
"synthetic_coffee_health_10000.csv".

Ce fichier réalise :
 - Partie 1 : génération de boxplots (identification des valeurs aberrantes)
     pour les colonnes numériques (hors colonnes binaires 0/1).
 - Partie 2 : calcul et affichage d'une matrice de corrélation entre toutes les
     colonnes numériques.

Les commentaires en français expliquent étapes et choix. Le script affiche
toutes les figures à la fin (un seul plt.show()) pour les visualiser ensemble.

Créé le 7 octobre 2025
@author: rfhba
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
try:
    from scipy.stats import pearsonr
except Exception:
    pearsonr = None


# ---------------------------
# Configuration / Chargement
# ---------------------------
# Charger le dataset avec le bon séparateur (';' dans ce fichier CSV)
df = pd.read_csv("synthetic_coffee_health_10000.csv", sep=';')


# ---------------------------
# Identification des colonnes
# ---------------------------
# Récupérer la liste des colonnes numériques
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
print("\nColonnes numériques détectées :", num_cols)


# ---------------------------
# Exclure les colonnes binaires
# (colonnes qui représentent Vrai/Faux encodé en 0/1)
# ---------------------------
binary_cols = []  # colonnes à exclure du tracé des boxplots
plot_cols = []    # colonnes numériques restantes à tracer
for col in num_cols:
    # prendre les valeurs distinctes non-null
    vals = set(df[col].dropna().unique())
    # si l'ensemble est inclus dans {0,1} on considère la colonne binaire
    if vals <= {0, 1}:
        binary_cols.append(col)
    else:
        plot_cols.append(col)

print("Colonnes binaires exclues (0/1) :", binary_cols)


# ---------------------------
# Partie 0 : Normalisation des données
# ---------------------------
print("\n=== Partie 0 : Normalisation (z-score) des colonnes numériques non-binaires ===")
# On normalise uniquement les colonnes numériques qui ne sont pas binaires (plot_cols)
# Création d'une copie normalisée pour l'affichage (on conserve `df` intact pour les stats)
norm_df = df.copy()
scalers = {}
for col in plot_cols:
    col_mean = df[col].mean()
    col_std = df[col].std()
    # éviter division par zéro si std == 0
    if pd.isna(col_std) or col_std == 0:
        norm_df[col] = df[col]
        scalers[col] = (col_mean, col_std)
    else:
        norm_df[col] = (df[col] - col_mean) / col_std
        scalers[col] = (col_mean, col_std)

print(f"Colonnes normalisées (z-score) : {plot_cols}")

print("Note: les boxplots de la Partie 1 afficheront les valeurs normalisées (z-score).\n")


# ---------------------------
# Partie 1 : Boxplots — identification des valeurs aberrantes
# ---------------------------
print("\n=== Partie 1 : Boxplot — identification des valeurs aberrantes ===")

# Paramètres d'affichage : 8 plots par figure (2 lignes x 4 colonnes)
per_fig = 8
ncols = 4
nrows = 2

# Boucle sur les colonnes à tracer par groupes de `per_fig`
for i in range(0, len(plot_cols), per_fig):
    cols = plot_cols[i:i+per_fig]
    n = len(cols)

    # Créer la figure et les axes (tableau d'axes)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()

    # Pour chaque sous-figure, tracer le boxplot et annoter avec des stats utiles
    for ax, col in zip(axes, cols):
        # boxplot vertical : la série (normalisée) est placée en y
        sns.boxplot(y=norm_df[col], ax=ax, color="skyblue", showfliers=True, whis=1.5)

        # --- Calculs statistiques pour l'annotation sur les données normalisées ---
        series = norm_df[col]
        mean = series.mean()
        median = series.median()
        std = series.std()
        count = int(series.count())            # nombre de valeurs non-NA
        missing = int(series.isna().sum())    # nombre de valeurs manquantes

        # Calcul des moustaches selon la règle IQR * whis (whis=1.5) sur données normalisées
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        whis = 1.5
        lower_bound = q1 - whis * iqr
        upper_bound = q3 + whis * iqr

        # Nombre de points en dehors des moustaches (considérés outliers ici)
        outliers = int(((series < lower_bound) | (series > upper_bound)).sum())

        # Tracer une ligne horizontale pour la moyenne afin de la visualiser
        ax.axhline(mean, color='red', linestyle='--', linewidth=1)

        # Titre et label de l'axe
        ax.set_title(f"Boxplot de {col}")
        ax.set_ylabel(col)

        # Annotation contextuelle : n, missing, outliers, mean, median
        info = (
            f"n={count}\nmissing={missing}\noutliers={outliers}\n"
            f"mean={mean:.2f}\nstd={std:.2f}\nmedian={median:.2f}"
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

    # ---------------------------
    # Résumé statistique pour les colonnes tracées dans cette figure
    # (moyenne, médiane, Q1, Q3, min, max, count, missing, outliers)
    # Nous construisons ce résumé au niveau global plus bas aussi, mais on
    # calcule ici un petit extrait pour information par groupe si nécessaire.
    
# Après la génération de tous les boxplots, construire un tableau récapitulatif
print("\n--- Résumé statistique global des colonnes analysées ---")
stats = []
for col in plot_cols:
    s = df[col]
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    whis = 1.5
    lower_bound = q1 - whis * iqr
    upper_bound = q3 + whis * iqr
    outliers = int(((s < lower_bound) | (s > upper_bound)).sum())
    stats.append({
        'column': col,
        'mean': s.mean(),
            'std': s.std(),
        'median': s.median(),
        'q1': q1,
        'q3': q3,
        'min': s.min(),
        'max': s.max(),
        'count': int(s.count()),
        'missing': int(s.isna().sum()),
        'outliers': outliers,
    })

stats_df = pd.DataFrame(stats).set_index('column')
# Arrondir les valeurs flottantes pour affichage
for c in ['mean', 'std', 'median', 'q1', 'q3', 'min', 'max']:
    stats_df[c] = stats_df[c].round(6)

print(stats_df)



# ---------------------------
# Partie 2 : Matrice de corrélation
# ---------------------------
print("\n=== Partie 2 : Matrice de corrélation entre toutes les colonnes numériques ===")

# Calculer la matrice de corrélation sur toutes les colonnes numériques d'origine
corr_df_orig = df[num_cols].corr()

# Construire une version entièrement normalisée des colonnes numériques
norm_all = df[num_cols].copy()
scalers_all = {}
for col in num_cols:
    m = df[col].mean()
    s = df[col].std()
    scalers_all[col] = (m, s)
    if pd.isna(s) or s == 0:
        norm_all[col] = df[col]
    else:
        norm_all[col] = (df[col] - m) / s

# Matrice de corrélation sur données normalisées
corr_df_norm = norm_all.corr()

# Si SciPy est disponible, calculer p-values de Pearson pour chaque paire
def pearson_pvals_matrix(df_numeric):
    cols = df_numeric.columns.tolist()
    r_mat = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    p_mat = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    if pearsonr is None:
        # SciPy manquant : on retourne NaN pour p-values
        return r_mat * df_numeric.corr(), p_mat.replace(0, np.nan)

    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            a = df_numeric[cols[i]]
            b = df_numeric[cols[j]]
            common = df_numeric[[cols[i], cols[j]]].dropna()
            if common.shape[0] < 3:
                r, p = np.nan, np.nan
            else:
                r, p = pearsonr(common.iloc[:, 0], common.iloc[:, 1])
            r_mat.iat[i, j] = r_mat.iat[j, i] = r
            p_mat.iat[i, j] = p_mat.iat[j, i] = p
    return r_mat, p_mat

# Calcul p-values pour original et normalized
r_orig, p_orig = pearson_pvals_matrix(df[num_cols])
r_norm, p_norm = pearson_pvals_matrix(norm_all)

# Fonction utilitaire pour transformer p-value en étoiles de significance
def sig_stars(p):
    try:
        if np.isnan(p):
            return ''
    except Exception:
        return ''
    if p <= 0.001:
        return '***'
    if p <= 0.01:
        return '**'
    if p <= 0.05:
        return '*'
    return ''

# Construire annotations texte (r rounded + stars)
def build_annot(r_df, p_df):
    cols = r_df.columns
    annot = pd.DataFrame('', index=cols, columns=cols)
    for i in cols:
        for j in cols:
            r = r_df.at[i, j]
            p = p_df.at[i, j]
            if pd.isna(r):
                annot.at[i, j] = ''
            else:
                annot.at[i, j] = f"{r:.2f}{sig_stars(p)}"
    return annot

annot_orig = build_annot(r_orig, p_orig)
annot_norm = build_annot(r_norm, p_norm)

# Taille de la figure : deux heatmaps côte à côte
fig_w = max(12, 0.6 * len(num_cols))
fig_h = max(6, 0.5 * len(num_cols) * 0.4)
fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))

# Dessiner heatmaps annotées (utiliser annot DataFrame directement)
sns.heatmap(r_orig, annot=annot_orig.values if annot_orig is not None else None, fmt='', cmap='vlag', center=0, ax=axes[0], cbar=True)
axes[0].set_title('Heatmap des corrélations de Pearson (original)')

sns.heatmap(r_norm, annot=annot_norm.values if annot_norm is not None else None, fmt='', cmap='vlag', center=0, ax=axes[1], cbar=True)
axes[1].set_title('Heatmap des corrélations de Pearson (normalisée)')

plt.suptitle('Heatmap des corrélations de Pearson avec niveaux de significativité', y=1.02)
plt.tight_layout()


# ---------------------------
# Affichage final
# ---------------------------
# Afficher toutes les figures générées (boxplots + heatmap)
plt.show()
