# -*- coding: utf-8 -*-
"""
Demonstration of SVM techniques on the diabetes dataset.

@author: Farida
"""

# Importing the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Loading the dataset
try:
    Data = pd.read_csv("diabetes.csv", sep=',')
    print("Data successfully loaded.")
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")
    exit()

# Checking for missing values
if Data.isnull().values.any():
    print("Warning: The dataset contains missing values. Consider cleaning the data.")
else:
    print("No missing values detected.")

# Defining the names of the explanatory variables and the target variable
target_name = "Outcome"
explanatory_columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# Extracting features (X) and the target (y)
X = Data[explanatory_columns]
y = Data[target_name]

# Splitting the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing features (an essential step for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

############
## SVM classifier with a linear kernel (default parameters)
###########
svm_clf = SVC(kernel='linear', random_state=42, probability=True)
svm_clf.fit(X_train, y_train)

# Predictions on the test set
y_pred = svm_clf.predict(X_test)

# Evaluating the linear SVM model
print("\n--- Evaluation of the linear SVM model ---")
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred))

# Calculating the ROC-AUC score for the linear SVM
auc = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC: {auc:.2f}")

# Plotting the ROC curve for the linear SVM
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Linear SVM')
plt.legend(loc="lower right")
plt.show()

# Hyperparameter optimization with Grid Search and cross-validation (RBF and linear kernels)
param_grid = [
    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000], 'probability': [True]},
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000], 'probability': [True]}
]

# Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Displaying the best parameters found by Grid Search
print("\n--- Best parameters found with Grid Search ---")
print(grid_search.best_params_)

# Using the best model to make predictions on the test set
best_clf = grid_search.best_estimator_
y_pred_best = best_clf.predict(X_test)

# Evaluating the best model
print("\n--- Evaluation of the best model ---")
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred_best))

print("\nClassification report:")
print(classification_report(y_test, y_pred_best))

# Calculating the ROC-AUC score for the best model
auc_best = roc_auc_score(y_test, y_pred_best)
print(f"ROC-AUC (Best model): {auc_best:.2f}")

# Plotting the ROC curve for the best model
fpr_best, tpr_best, thresholds_best = roc_curve(y_test, y_pred_best)
plt.figure()
plt.plot(fpr_best, tpr_best, label=f'ROC curve (AUC = {auc_best:.2f})', color='green')
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Best SVM Model')
plt.legend(loc="lower right")
plt.show()

# Prediction data (a fictive example for a patient)
# Note: The new data must have exactly the same features as those used to train the model
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

# Standardizing the new data
new_data_scaled = scaler.transform(new_data)

# Predicting with the trained model (best model found with Grid Search)
predicted_class = best_clf.predict(new_data_scaled)

# Displaying the prediction result
print(f"Prediction for the patient: {'Diabetes' if predicted_class[0] == 1 else 'No Diabetes'}")

# If we also want to obtain the probability associated with each class
if hasattr(best_clf, "predict_proba"):
    predicted_proba = best_clf.predict_proba(new_data_scaled)
    print(f"Probability of developing diabetes: {predicted_proba[0][1]:.2f}")
else:
    print("The model does not support the 'predict_proba' method.")
    

##########################
### Binary Logistic Regression
##########################

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, roc_auc_score, log_loss, cohen_kappa_score, matthews_corrcoef
)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import chi2

# Assuming that the Diabetes dataset has already been cleaned

# Loading the Diabetes dataset
Data = pd.read_csv("diabetes.csv", sep=',')

# Defining the explanatory variables and the target variable
X = Data.drop(columns=['Outcome'])  # Features
y = Data['Outcome']  # Target (0 = No Diabetes, 1 = Diabetes)

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Standardizing features (critical for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Binary Logistic Regression Model
model = LogisticRegression()

# Training the model
model.fit(X_train_scaled, y_train)

# 3. Validity checks
# --------------------
# Multicollinearity with VIF (Variance Inflation Factor)
X_train_vif = pd.DataFrame(X_train_scaled, columns=X.columns)
X_train_vif = add_constant(X_train_vif)  # Adding a constant for statsmodels

vif_data = pd.DataFrame()
vif_data["Feature"] = X_train_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_vif.values, i) for i in range(X_train_vif.shape[1])]
print("\nVariance Inflation Factors (VIF):")
print(vif_data)

# Interpretation:
# A VIF greater than 10 indicates high multicollinearity, which can affect model coefficients.

# Analyzing correlations among explanatory variables
plt.figure(figsize=(10, 8))
sns.heatmap(X_train_vif.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title("Correlation Matrix of Explanatory Variables")
plt.show()

# Interpretation:
# High correlation (> 0.7 or < -0.7) indicates strong redundancy between two variables,
# which may exacerbate multicollinearity. Consider combining (PCA for instance) or removing such variables.

# Linear relationship between explanatory variables and the logit
X_train_logit = pd.DataFrame(X_train_scaled, columns=X.columns)
X_train_logit['logit'] = model.predict_proba(X_train_scaled)[:, 1]

# Plotting graphs for each variable vs logit
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for i, col in enumerate(X_train_logit.columns[:-1]):
    ax = axes[i // 3, i % 3]
    ax.scatter(X_train_logit[col], X_train_logit['logit'], alpha=0.5)
    ax.set_title(f'Relationship between {col} and logit')

plt.tight_layout()
plt.show()

# Interpretation:
# If the points follow a linear trend, the relationship is valid.
# Otherwise, transformations (logarithmic, quadratic) may be necessary.

### Detecting outliers with Cook's distance
X_train_const = add_constant(X_train_scaled)  # Adding a constant term for the regression model
logit_model = sm.Logit(y_train, X_train_const)  # Building the logistic regression model
logit_result = logit_model.fit()  # Fitting the model to the data

# Model summary
print(logit_result.summary())  # Displays model statistics, including coefficients and significance levels

# Calculating predicted probabilities
y_hat = logit_result.predict(X_train_const)  # Predictions for the training set

# Calculating Pearson residuals
residuals_pearson = (y_train - y_hat) / np.sqrt(y_hat * (1 - y_hat))  # Measures deviations between observed and predicted values

# Calculating leverage values (hat values)
hat_matrix_diag = np.diag(X_train_const @ np.linalg.inv(X_train_const.T @ X_train_const) @ X_train_const.T)  
# Leverage indicates the influence of each observation on the model

# Calculating Cook's distance
n = X_train_const.shape[0]  # Number of observations
p = X_train_const.shape[1]  # Number of parameters (including the constant)

cook_distance = residuals_pearson**2 / p * hat_matrix_diag / (1 - hat_matrix_diag)  
# Combines residuals and leverage to measure the overall influence of each observation on the model fit

# Plotting Cook's distance
plt.figure(figsize=(10, 6))
plt.scatter(np.arange(len(cook_distance)), cook_distance, color='blue', label="Cook's Distance")  
plt.axhline(y=4 / n, color='red', linestyle='--', label='Threshold of 4/n')  # Common threshold for identifying influential points
plt.title("Cook's Distance for Detecting Influential Points")
plt.xlabel("Observation")
plt.ylabel("Cook's Distance")
plt.legend(loc="upper right")
plt.show()

# Plotting binary residuals to check model fit
plt.figure(figsize=(10, 6))
plt.scatter(np.arange(len(residuals_pearson)), residuals_pearson, alpha=0.6, color='purple')  # Visualizes residuals
plt.axhline(0, color='red', linestyle='--', label='Residuals = 0')  # Indicates perfect fit
plt.title("Pearson Residuals")
plt.xlabel("Observation")
plt.ylabel("Pearson Residuals")
plt.legend(loc="upper right")
plt.show()

# Interpretation:
# - Cook's distance helps identify influential points that have a disproportionate effect on the regression model.
# - Observations with Cook's distance above the threshold (4/n) should be further examined as potential outliers.
# - Pearson residuals significantly distant from 0 (e.g., beyond ±3) may indicate poorly fitted observations.
# - Influential or poorly fitted points might require further investigation or removal to improve model reliability.


# 4. Evaluating model performance
# ---------------------------------
# Predictions on the test set
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability for the positive class (class 1)

# Overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion matrix
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification report (precision, recall, f1-score)
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# Calculating the ROC-AUC score
auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {auc:.2f}")

# Plotting the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Binary Logistic Regression')
plt.legend(loc="lower right")
plt.show()

# Explanation of the ROC-AUC curve:
# The ROC curve evaluates the trade-off between the "True Positive Rate" (TPR or sensitivity) and the "False Positive Rate" (FPR).
# - The Y-axis (TPR) represents the model's ability to correctly identify positive cases (e.g., detecting diabetes when present).
# - The X-axis (FPR) shows the rate of false alarms (e.g., predicting diabetes when it is not present).
# - A curve closer to the top-left corner indicates better performance, as it maximizes TPR while minimizing FPR.
# - The AUC (Area Under Curve) score summarizes the overall quality of the classifier; a value closer to 1 indicates high accuracy.


# 5. Prediction for a new patient example
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

# Standardizing the new data using the same scaler used for training
new_data_scaled = scaler.transform(new_data)

# Predicting the class with the logistic regression model
predicted_class = model.predict(new_data_scaled)
print(f"Prediction for the patient: {'Diabetes' if predicted_class[0] == 1 else 'No Diabetes'}")

# Displaying probabilities associated with each class
predicted_proba = model.predict_proba(new_data_scaled)
print(f"Probabilities associated with the classes: {predicted_proba[0]}")

# 6. Additional validations
# -------------------------
# 6.1. Log-Loss
log_loss_value = log_loss(y_test, y_proba)
print(f"Log-Loss: {log_loss_value:.2f}")

# Interpretation:
# Log-loss measures the difference between predicted probabilities and observed classes.
# Lower log-loss indicates better probabilistic predictions. Unlike accuracy,
# it penalizes more for confident predictions that are incorrect.

# 6.2. Hosmer-Lemeshow Test
# Predictions on probabilities
predicted_proba = model.predict_proba(X_test_scaled)[:, 1]
# Dividing predicted probabilities into 10 equal groups
groups = pd.qcut(predicted_proba, 10, duplicates='drop')

# Creating a crosstab of groups and observations
observed_counts = pd.crosstab(groups, y_test)

# Calculating observed and expected rates
observed_events = observed_counts[1]
observed_non_events = observed_counts[0]
total_counts = observed_events + observed_non_events

# Mean probabilities for each group
predicted_group_means = pd.Series(predicted_proba).groupby(groups).mean()

# Calculating expected events for each group
expected_events = predicted_group_means * total_counts
expected_non_events = total_counts - expected_events

# Calculating chi-squared statistic to compare observed and expected rates
hosmer_lemeshow_stat = np.sum(
    (observed_events - expected_events)**2 / expected_events +
    (observed_non_events - expected_non_events)**2 / expected_non_events
)

# The Hosmer-Lemeshow test follows a chi-squared distribution with 8 degrees of freedom (10 groups - 2)
p_value = 1 - chi2.cdf(hosmer_lemeshow_stat, df=8)

# Displaying results
print(f"Hosmer-Lemeshow Test: Chi2 = {hosmer_lemeshow_stat:.2f}, p-value = {p_value:.2f}")

# Interpretation:
# A high p-value (> 0.05) suggests that the model fits the observed data well.
# If the p-value is low (< 0.05), it indicates that the model does not fit the data well.

# 6.3. Standardized residuals vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_hat, residuals_pearson, alpha=0.6, color='green')
plt.axhline(0, color='red', linestyle='--')
plt.title("Pearson Residuals vs Predicted Values")
plt.xlabel("Predicted Values (logit)")
plt.ylabel("Pearson Residuals")
plt.show()

# Interpretation:
# This graph helps check if residuals are randomly distributed around 0.
# If a systematic trend appears (e.g., larger residuals for certain predicted values),
# it could indicate that the model is not well-fitted and additional transformations may be needed.

# 6.4. Note on observation independence
# --------------------------------------------
# Logistic regression assumes that observations are independent of each other.
# This means there should be no relationship or dependency between observations in the data.

# If the data comes from a study where observations are grouped (e.g., patients in different hospitals or students in classes),
# hierarchical logistic regression models or mixed models should be used to account for these dependencies.

# If observations are independent, there is no need to adjust the model for this.

#########################
### Multinomial Logistic Regression with the Iris Dataset
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

# Loading the Iris dataset for multinomial logistic regression
from sklearn.datasets import load_iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Multinomial Logistic Regression Model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train_scaled, y_train)

# 2. Predictions and model evaluation
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

# 3. Checking validity conditions

# 3.1 Multicollinearity with VIF (Variance Inflation Factor)
# ---------------------------------
X_train_vif = pd.DataFrame(X_train_scaled, columns=iris.feature_names)
X_train_vif = add_constant(X_train_vif)  # Adding constant for statsmodels

vif_data = pd.DataFrame()
vif_data["Feature"] = X_train_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_vif.values, i) for i in range(X_train_vif.shape[1])]
print("\nVariance Inflation Factors (VIF):")
print(vif_data)

# Interpretation of VIF:
# A VIF greater than 10 indicates high multicollinearity between some variables,
# which may make it difficult to estimate precise coefficients in the model.

# 3.2 Linear relationship between predictors and logit
# -----------------------------------------------------------------
X_train_logit = pd.DataFrame(X_train_scaled, columns=iris.feature_names)
X_train_logit['logit_class_0'] = model.predict_proba(X_train_scaled)[:, 0]  # For Setosa
X_train_logit['logit_class_1'] = model.predict_proba(X_train_scaled)[:, 1]  # For Versicolor
X_train_logit['logit_class_2'] = model.predict_proba(X_train_scaled)[:, 2]  # For Virginica

# Plotting graphs for each variable vs logit for each class
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for i, col in enumerate(X_train_logit.columns[:-3]):
    for j in range(3):  # For each class
        ax = axes[j, i]
        ax.scatter(X_train_logit[col], X_train_logit[f'logit_class_{j}'])
        ax.set_title(f'Relationship {col} and logit (class {iris.target_names[j]})')

plt.tight_layout()
plt.show()

# Interpretation:
# If the points follow a linear trend, the relationship is valid.
# Otherwise, transformations (logarithms, powers) may be needed.

# 3.3 Detecting outliers with Cook's Distance
# ----------------------------------------------------------
X_train_const = add_constant(X_train_scaled)
logit_model = Logit(y_train, X_train_const)
logit_result = logit_model.fit()

# Computing standardized residuals and Cook's distance
influence = logit_result.get_influence()
summary_frame = influence.summary_frame()

# Plotting Cook's Distance
plt.figure(figsize=(10, 6))
plt.scatter(np.arange(len(summary_frame)), summary_frame['cooks_d'], color='blue')
plt.axhline(y=4/len(X_train), color='red', linestyle='--')  # Threshold for Cook's distance
plt.title("Cook's Distance for Detecting Outliers")
plt.xlabel("Observation")
plt.ylabel("Cook's Distance")
plt.show()

# Interpretation:
# Points above the red line are outliers with a disproportionate influence on the model.
# These points should be reviewed and potentially removed.

# 3.4 Checking sample size and events per class
# -----------------------------------------------------------------------
n_total = len(y_train)
n_events_class_0 = np.sum(y_train == 0)  # Setosa
n_events_class_1 = np.sum(y_train == 1)  # Versicolor
n_events_class_2 = np.sum(y_train == 2)  # Virginica
n_features = X_train_scaled.shape[1]

print(f"\nTotal sample size: {n_total}")
print(f"Number of events (Setosa): {n_events_class_0}")
print(f"Number of events (Versicolor): {n_events_class_1}")
print(f"Number of events (Virginica): {n_events_class_2}")
print(f"Number of predictors: {n_features}")

if n_events_class_0 >= 10 * n_features and n_events_class_1 >= 10 * n_features and n_events_class_2 >= 10 * n_features:
    print("The number of events in each class is sufficient for multinomial logistic regression.")
else:
    print("The number of events in one or more classes is insufficient. Consider collecting more data.")

# 4. Performance Metrics
# -------------------------
# Overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Interpretation:
# Accuracy measures the proportion of observations correctly classified.
# For example, an accuracy of 0.95 means 95% of predictions were correct.
# However, accuracy alone is insufficient if classes are imbalanced (few observations in some classes).

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Interpretation:
# The confusion matrix shows how often each class was correctly predicted (diagonal)
# and how often one class was confused with another (off-diagonal).
# It helps identify where the model makes errors and if specific classes are poorly predicted.

# Classification report (precision, recall, F1-score)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Interpretation:
# - **Precision**: Of the predictions made for a given class, how many were correct?
# - **Recall**: Of all actual observations of a given class, how many were correctly predicted?
# - **F1-score**: Harmonic mean of precision and recall, useful for imbalanced data (equal weight to precision and recall).
# These metrics are provided for each class (Setosa, Versicolor, Virginica).

# Log-loss (Logarithmic Loss)
logloss = log_loss(y_test, y_proba)
print(f"Log-Loss: {logloss:.2f}")

# Interpretation:
# Log-loss measures the performance of a probabilistic model. It compares predicted probabilities for each class
# with the actual labels. Lower log-loss indicates a better model.
# A log-loss value near 0 means the predicted probabilities are very close to the true labels.

# Cohen's Kappa (Cohen’s Kappa Score)
kappa = cohen_kappa_score(y_test, y_pred)
print(f"Cohen's Kappa: {kappa:.2f}")

# Interpretation:
# Cohen's Kappa measures agreement between the model's predictions and actual values,
# accounting for chance agreement. A score of 1 indicates perfect agreement, while a score near 0 indicates
# the model performs no better than random guessing. If Kappa > 0.8, the model performs well.

# Matthews Correlation Coefficient (MCC)
mcc = matthews_corrcoef(y_test, y_pred)
print(f"Matthews Correlation Coefficient (MCC): {mcc:.2f}")

# Interpretation:
# MCC considers all values in the confusion matrix (true positives, false positives,
# true negatives, false negatives). It is a balanced measure, particularly useful for imbalanced classes.
# A score of 1 means perfect correlation (all classes correctly predicted), while a score of 0 means
# predictions are no better than random guessing.

# 5. Prediction for a new flower example
# --------------------------------------------
new_data = pd.DataFrame({
    "sepal length (cm)": [5.1],
    "sepal width (cm)": [3.5],
    "petal length (cm)": [1.4],
    "petal width (cm)": [0.2]
})

# Standardizing the new data using the same scaler used for training
new_data_scaled = scaler.transform(new_data)

# Predicting the class with the multinomial logistic regression model
predicted_class = model.predict(new_data_scaled)
print(f"Prediction for the flower: {iris.target_names[predicted_class[0]]}")

# Interpretation:
# Here, we predict the class of the flower (Setosa, Versicolor, or Virginica) based on its features
# (sepal and petal length/width). The model returns the most probable class for this new observation.

# Displaying probabilities associated with each class
predicted_proba = model.predict_proba(new_data_scaled)
print(f"Probabilities associated with the classes: {predicted_proba[0]}")

# Interpretation:
# The model also predicts probabilities for each class. For example, it might predict that
# the probability of the flower being Setosa is 97%, Versicolor 2%, and Virginica 1%.
# This is useful for nuanced decisions where one class has a probability close to others.

