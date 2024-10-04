<div style="font-size: 36px; font-weight: bold; color: blue; border: 2px solid #000; padding: 15px; background-color: #fff; text-align: center;">
    Détection de Transactions Frauduleuses
</div>


# FR

![img](img.png)

Ce projet se concentre sur la détection des fraudes dans les transactions bancaires. Il s’appuie sur un ensemble de données provenant de Kaggle, qui comprend des informations détaillées sur des transactions antérieures. Étant donné que ce jeu de données présente un déséquilibre significatif entre les transactions non frauduleuses et les transactions frauduleuses, l’utilisation de techniques de Machine Learning devient essentielle pour identifier les transactions suspectes avec précision.

L’objectif principal est de développer des modèles capables de détecter efficacement les fraudes tout en garantissant la prédiction correcte des transactions non frauduleuses, afin de ne pas nuire à l’expérience utilisateur. Pour cela, il est important de choisir les bonnes métriques d’évaluation, telles que le F1-score et l’AUC-PR, qui permettent de mieux appréhender les performances des modèles dans des contextes de données déséquilibrées.

---

## Les données
**Données** : "Credit Card Fraud Detection"<br> 
**Dimensions des données** : L'ensemble de données comprend 284,807 lignes et 31 colonnes. Les variables sont les suivantes : 

- **Time** : Nombre de secondes écoulées depuis la première transaction dans l'ensemble de données.
- **V1, V2, ..., V28** : Caractéristiques techniques résultant d'une ACP (Analyse en Composantes Principales) pour protéger la confidentialité des données.
- **Amount** : Montant de la transaction.
- **Class** : Classe cible où 1 représente une transaction frauduleuse et 0 une transaction non frauduleuse.

---

## Méthodologie

Les données, résultant d'une analyse en composantes principales, ont requis peu de traitement. Seul le montant des transactions, présentant des valeurs aberrantes significatives, a été normalisé à l'aide de RobustScaler. 

Le premier modèle utilisé est une régression logistique, avec l'option `class_weight = balanced`, permettant de gérer le déséquilibre des données. Cette méthode ajuste automatiquement les poids des classes pour mieux traiter les transactions frauduleuses, minoritaires dans le jeu de données.

Des variations des poids de la classe minoritaire ont été explorées. Cela a permis d'évaluer l'impact de différentes pondérations sur les résultats de prédiction, en particulier sur la capacité du modèle à détecter les transactions frauduleuses.

Ensuite, une recherche des meilleurs hyperparamètres via **grid search** a été menée sur différents modèles. Cette approche a permis de sélectionner les modèles les plus robustes face au déséquilibre des classes. Les meilleurs modèles identifiés ont été combinés à l'aide de techniques d'ensemble, telles que le **Stacking** et le **Voting**, afin de renforcer la performance prédictive.

Enfin, des techniques de rééchantillonnage ont été appliquées pour traiter le déséquilibre des classes. L'**undersampling** a permis de réduire la taille de la classe majoritaire (transactions non frauduleuses), tandis que le **SMOTE** a artificiellement augmenté la classe minoritaire (transactions frauduleuses). Ces méthodes ont été respectivement testées avec plusieurs modèles pour identifier ceux offrant les meilleurs résultats.

Les performances des modèles ont été évaluées à l'aide des métriques suivantes :

- **F1-score** : Il permet de trouver un équilibre entre la **précision** et le **rappel**, deux aspects essentiels pour la détection de fraudes. 
  - Le **rappel** est crucial pour identifier les transactions frauduleuses.
  - La **précision** garantit que les transactions non frauduleuses sont correctement classées, ce qui améliore l'expérience utilisateur.

- **AUC-PR (Aire sous la courbe de Précision-Rappel)** : Cette métrique est pertinente car elle évalue la performance du modèle en tenant compte du déséquilibre des classes.

L'**accuracy** n'a pas été retenue, car elle peut être trompeuse en cas de données fortement déséquilibrées. 

Cette approche garantit une détection efficace des fraudes en optimisant la robustesse des modèles via des techniques adaptées aux données déséquilibrées et des métriques spécifiques comme le F1-score et l'AUC-PR.

---

## Resultats
Les résultats de ces différentes méthodes sont les suivants : 

| Modèles                                           | F1-score | AUC-PR  |
|--------------------------------------------------|----------|---------|
| Régression Logistique avec class weight = balanced | 0.11     | 0.76    |
| Régression Logistique avec variation de poids     |          |         |
| - {0:1, 1:3}                                     | 0.76     | 0.747   |
| - {0:1, 1:7}                                     | 0.77     | 0.747   |
| - {0:1, 1:20}                                    | 0.71     | 0.748   |
| - {0:1, 1:100}                                   | 0.40     | 0.743   |
|                                                  |          |         |
| Random Forest                                     | 0.88     | 0.876   |
| XGBoost                                          | 0.87     | 0.876   |
| K-Nearest Neighbors                               | 0.86     | 0.881   |
| CatBoost                                         | 0.87     | 0.880   |
| LightGBM                                         | 0.86     | 0.869   |
| Support Vector Machine                            | 0.85     | 0.865   |
| Easy Ensemble Classifier                          | 0.09     | 0.757   |
| Balanced Random Forest                            | 0.11     | 0.772   |
|                                                  |          |         |
| Voting Classifier                                 | 0.88     | 0.880   |
| Stacking Classifier                               | 0.86     | 0.855   |
|                                                  |          |         |
| Undersampling 0.1 + Random Forest                | 0.97     | 0.982   |
| Undersampling 0.1 + Support Vector Machine       | 0.92     | 0.937   |
| Undersampling 0.1 + K-Nearest Neighbors          | 0.93     | 0.959   |
| Undersampling 0.1 + XGBoost                      | 0.96     | 0.975   |
| Undersampling 0.1 + CatBoost                     | 0.96     | 0.974   |
| Undersampling 0.1 + LightGBM                     | 0.92     | 0.937   |
|                                                  |          |         |
| SMOTE Strategy 0.1 + XGBoost                     | 0.83     | 0.872   |
| SMOTE Strategy 0.1 + K-Nearest Neighbors         | 0.73     | 0.806   |
| SMOTE Strategy 0.1 + Random Forest                | 0.88     | 0.883   |
| SMOTE Strategy 0.1 + CatBoost                     | 0.81     | 0.817   |


L’évaluation des modèles de détection des fraudes a montré des performances variées. Bien que le modèle de régression logistique ait initialement affiché une précision de 97 %, son efficacité à détecter les fraudes était limitée par un déséquilibre des classes. L’optimisation des poids a permis d'atteindre un F1-score de 0.77, soulignant l'importance de l'équilibre entre précision et rappel.

L'exploration de paramètres via Grid Search a révélé des résultats prometteurs avec des modèles comme KNN, RandomForest, XGBoost, et CatBoost, tous ayant des F1-scores supérieurs à 0.85. Les techniques d'ensemble telles que le voting et le stacking ont amélioré ces performances.

Les techniques de rééchantillonnage, notamment RandomUnderSampling et SMOTE, ont été bénéfiques, avec RandomForest atteignant un AUC de 0.982 et un F1-score de 0.97. Cependant, SMOTE a montré des résultats moins convaincants.

Pour optimiser davantage les performances, des pistes comme la variation du seuil de classification et l'utilisation de réseaux de neurones pourraient être explorées pour mieux capturer les relations complexes dans les données.

## Structure des Dossiers

Les dossiers sont organisés de la manière suivante :
```bash
fraud-detection/
│
├── data/
│   └── dataset.csv.gz          # Le jeu de données utilisé pour entraîner et tester les modèles
│
├── models/
│   ├── checkpoints/               # Les checkpoints des différents modèles (versions entraînées non sauvegardées)
│   └── models_fitted/             # Modèles entraînés sauvegardés et prêts à être utilisés
│
├── notebooks/                     # Dossier qui contient les notebooks 
│
└── utils/
│   ├── __init__.py                # Indique que ce répertoire est un package Python
│   └── functions.py               # Contient des fonctions utilitaires pour la manipulation des données et les modèles
│
└── requirements.txt               # Liste des bibliothèques nécessaires pour le projet

```

 Les résultats des modélisations sont organisés et sauvegardés au format `dict_python.pkl` dans le répertoire `models/models_fitted/`. Par ailleurs, le répertoire `models/checkpoints/` contient des dictionnaires des modèles enregistrés durant leur entraînement.


---


## Instructions
Clonez ce repertoire dans un nouvel environnement virtuel python et installer les modules Python indispensables pour reproduire ce travail avec `pip install -r requirements.txt`


# EN

This project focuses on detecting fraud in banking transactions. It utilizes a dataset from Kaggle, which contains detailed information about past transactions. Given that this dataset exhibits a significant imbalance between non-fraudulent and fraudulent transactions, employing Machine Learning techniques becomes essential for accurately identifying suspicious transactions.

The primary goal is to develop models capable of effectively detecting fraud while ensuring the correct prediction of non-fraudulent transactions to avoid harming the user experience. Choosing the right evaluation metrics, such as F1-score and AUC-PR, is crucial for understanding model performance in the context of imbalanced data.

---

## Data
**Dataset**: "Credit Card Fraud Detection"  
**Dimensions**: The dataset comprises 284,807 rows and 31 columns. The variables are as follows:

- **Time**: The number of seconds elapsed since the first transaction in the dataset.
- **V1, V2, ..., V28**: Technical features resulting from Principal Component Analysis (PCA) to protect data privacy.
- **Amount**: The transaction amount.
- **Class**: Target class where 1 represents a fraudulent transaction and 0 represents a non-fraudulent transaction.

---

## Methodology

The data, resulting from Principal Component Analysis, required minimal preprocessing. Only the transaction amounts, which exhibited significant outliers, were normalized using RobustScaler.

The first model used was logistic regression, with the option `class_weight = balanced`, which helps address the data imbalance. This method automatically adjusts class weights to better handle fraudulent transactions, which are minority in the dataset.

Variations of the minority class weights were explored to assess the impact of different weightings on prediction outcomes, particularly regarding the model's ability to detect fraudulent transactions.

Next, a search for the best hyperparameters was conducted using **grid search** across various models. This approach allowed for the selection of the most robust models against class imbalance. The best identified models were then combined using ensemble techniques, such as **Stacking** and **Voting**, to enhance predictive performance.

Finally, resampling techniques were applied to address class imbalance. **Undersampling** was used to reduce the size of the majority class (non-fraudulent transactions), while **SMOTE** artificially increased the minority class (fraudulent transactions). These methods were tested with several models to identify those providing the best results.

Model performance was evaluated using the following metrics:

- **F1-score**: This metric helps find a balance between **precision** and **recall**, both essential aspects of fraud detection.
  - **Recall** is crucial for identifying fraudulent transactions.
  - **Precision** ensures that non-fraudulent transactions are correctly classified, thereby improving the user experience.

- **AUC-PR (Area Under the Precision-Recall Curve)**: This metric is relevant as it evaluates model performance while considering class imbalance.

**Accuracy** was not retained, as it can be misleading in cases of heavily imbalanced data.

This approach ensures effective fraud detection by optimizing model robustness through techniques suitable for imbalanced data and specific metrics like F1-score and AUC-PR.

---

## Results
The results of the various methods are as follows:

| Models                                           | F1-score | AUC-PR  |
|--------------------------------------------------|----------|---------|
| Logistic Regression with class weight = balanced | 0.11     | 0.76    |
| Logistic Regression with weight variation         |          |         |
| - {0:1, 1:3}                                     | 0.76     | 0.747   |
| - {0:1, 1:7}                                     | 0.77     | 0.747   |
| - {0:1, 1:20}                                    | 0.71     | 0.748   |
| - {0:1, 1:100}                                   | 0.40     | 0.743   |
|                                                  |          |         |
| Random Forest                                     | 0.88     | 0.876   |
| XGBoost                                          | 0.87     | 0.876   |
| K-Nearest Neighbors                               | 0.86     | 0.881   |
| CatBoost                                         | 0.87     | 0.880   |
| LightGBM                                         | 0.86     | 0.869   |
| Support Vector Machine                            | 0.85     | 0.865   |
| Easy Ensemble Classifier                          | 0.09     | 0.757   |
| Balanced Random Forest                            | 0.11     | 0.772   |
|                                                  |          |         |
| Voting Classifier                                 | 0.88     | 0.880   |
| Stacking Classifier                               | 0.86     | 0.855   |
|                                                  |          |         |
| Undersampling 0.1 + Random Forest                | 0.97     | 0.982   |
| Undersampling 0.1 + Support Vector Machine       | 0.92     | 0.937   |
| Undersampling 0.1 + K-Nearest Neighbors          | 0.93     | 0.959   |
| Undersampling 0.1 + XGBoost                      | 0.96     | 0.975   |
| Undersampling 0.1 + CatBoost                     | 0.96     | 0.974   |
| Undersampling 0.1 + LightGBM                     | 0.92     | 0.937   |
|                                                  |          |         |
| SMOTE Strategy 0.1 + XGBoost                     | 0.83     | 0.872   |
| SMOTE Strategy 0.1 + K-Nearest Neighbors         | 0.73     | 0.806   |
| SMOTE Strategy 0.1 + Random Forest                | 0.88     | 0.883   |
| SMOTE Strategy 0.1 + CatBoost                     | 0.81     | 0.817   |

The evaluation of the fraud detection models showed varied performances. Although the logistic regression model initially displayed an accuracy of 97%, its effectiveness in detecting fraud was limited by class imbalance. Weight optimization achieved an F1-score of 0.77, highlighting the importance of balancing precision and recall.

Exploring parameters through Grid Search revealed promising results with models such as KNN, RandomForest, XGBoost, and CatBoost, all yielding F1-scores above 0.85. Ensemble techniques such as voting and stacking enhanced these performances.

Resampling techniques, including RandomUnderSampling and SMOTE, proved beneficial, with RandomForest achieving an AUC of 0.982 and an F1-score of 0.97. However, SMOTE showed less convincing results.

To further optimize performance, avenues such as varying the classification threshold and utilizing neural networks could be explored to better capture complex relationships within the data.

## Directory Structure

The directories are organized as follows:

```bash
fraud-detection/
│
├── data/
│   └── dataset.csv.gz          # The dataset used for training and testing the models
│
├── models/
│   ├── checkpoints/               # Checkpoints of different models (non-saved trained versions)
│   └── models_fitted/             # Saved trained models ready for use
│
├── notebooks/                     # Folders of notebooks
│
└── utils/
│   ├── __init__.py                # Indicates that this directory is a Python package
│   └── functions.py               # Contains utility functions for data manipulation and modeling
│
└── requirements.txt               # List of libraries required for the project

```

The modeling results are organized and saved in `dict_python.pkl` format within the `models/models_fitted/` directory. Additionally, the `models/checkpoints/` directory contains dictionaries of models saved during training.

## Instructions
Clone this repository into a new Python virtual environment and install the required Python modules to reproduce this work using `pip install -r requirements.txt`.
