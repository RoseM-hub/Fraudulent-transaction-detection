# Importation des librairies nécessaires
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix, recall_score, classification_report
import plotly.graph_objs as go
import plotly.subplots as sp
import os
import joblib

def calculate_metrics(model, X_test, y_test):
    """
    Calcule les courbes Precision-Recall, AUC PR et le F1 score pour un modèle donné.

    Args:
        model: Le modèle entraîné.
        X_test: Les données de test.
        y_test: Les vraies étiquettes.

    Returns:
        results: Un dictionnaire contenant précision, rappel, seuils, AUC PR et F1 score.
    """
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    auc_pr = auc(recall, precision)

    # Calculer les étiquettes prédites
    y_pred = model.predict(X_test)
    
    # Calculer le F1 score
    f1 = f1_score(y_test, y_pred)

    # Retourner un dictionnaire avec tous les résultats
    results = {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'auc_pr': auc_pr,
        'f1': f1
    }
    return results

def plot_auc_pr(models_pr_dict: dict, subtitle: str = ""):
    """
    Crée des courbes Precision-Recall interactives pour plusieurs modèles,
    triées d'abord par AUC PR, puis par F1 score.

    Args:
        models_pr_dict (dict): Dictionnaire contenant les noms des modèles et les données associées (precision, recall, thresholds, auc_pr, f1).
        subtitle (str): Sous-titre optionnel pour le graphique.
    """

    # Trier les modèles par AUC PR d'abord, puis par F1 score décroissant
    sorted_models_pr_dict = dict(sorted(
        models_pr_dict.items(),
        key=lambda item: (-item[1]['auc_pr'], -item[1]['f1'])  # Trier par AUC PR en premier, puis F1
    ))

    # Créer le graphique principal Precision-Recall
    fig = go.Figure()
    for model_name, pr_data in sorted_models_pr_dict.items():
        precision = pr_data['precision']
        recall = pr_data['recall']
        auc_pr = pr_data['auc_pr']  # AUC PR
        f1 = pr_data['f1']  # F1 Score

        # Ajouter une trace avec le label contenant AUC et F1
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'{model_name} (AUC-PR = {auc_pr:.3f}, F1 = {f1:.2f})'
        ))

    # Mise à jour de la disposition pour avoir des axes égaux (carrés)
    fig.update_layout(
        title={
            'text': f"Courbe AUC - Précision Rappel<br><span style='font-size:12px;color:blue'>{subtitle}</span>",
            'x': 0.5,
            'y': 0.95,  # Positionner le titre
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Rappel',
        yaxis_title='Précision',
        xaxis=dict(
            range=[0, 1.03],
            scaleanchor="y",
            constrain="domain",
        ),
        yaxis=dict(
            range=[0, 1.03],
            scaleanchor="x",
            constrain="domain",
        ),
        margin=dict(b=10, t=70, r=150),  # Ajout d'une marge à droite pour laisser de l'espace pour la légende
        width=900,
        height=500,
        legend=dict(
            x=1,  # Positionner la légende à droite (1 correspond à l'extrémité droite)
            y=0.5,  # Centrer la légende verticalement
            xanchor='left',
            yanchor='middle',  # Aligner verticalement au centre
            orientation='v',
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)',
            borderwidth=0,
        )
    )

    # Afficher le graphique
    fig.show()

def plot_confusion_matrix(models_dict, X_test, y_test, title=""):
    """
    Crée des heatmaps de matrices de confusion pour plusieurs modèles.

    Args:
        models_dict (dict): Dictionnaire avec les modèles.
        X_test: Données de test.
        y_test: Vraies étiquettes de test.
        title (str): Titre du graphique.
    """
    n_models = len(models_dict)
    
    # Calculer le nombre de colonnes (maximum 3) et de lignes (maximum 2)
    n_cols = min(3, n_models)  # Maximum de 3 colonnes
    n_rows = (n_models + n_cols - 1) // n_cols  # Arrondi vers le haut
    n_rows = min(n_rows, 2)  # Limiter à 2 lignes

    # Ajuster la taille de la figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    
    # Aplatir les axes si c'est un tableau numpy
    if n_models > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Si un seul modèle, faire une liste pour uniformiser l'itération

    for ax, (model_name, model) in zip(axes, models_dict.items()):
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Calculer la matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculer le F1 score et le rappel
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Créer une heatmap de la matrice de confusion
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False, annot_kws={"size": 14})
        
        # Mettre à jour le titre
        ax.set_title(f'{model_name}\nF1 = {f1:.2f}, Recall = {recall:.2f}', fontsize=12)
        ax.set_xlabel('Predictions', fontsize=10)
        ax.set_ylabel('True Labels', fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

        # Masquer la graduation
        ax.tick_params(left=False, bottom=False)

    # Masquer les axes restants si n_models < n_cols * n_rows
    for i in range(n_models, len(axes)):
        fig.delaxes(fig.axes[i])
        
    # Ajouter un titre général à la figure
    fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()

def generate_color_palette(n):
    # Utiliser la palette de couleurs de Matplotlib
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, n))  # Colormap 'tab10' pour des couleurs distinctes
    return [f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 1)' for r, g, b, _ in colors]

def plot_model_performance(RUS_models_results):
    # Liste de modèles
    models = list(next(iter(RUS_models_results.values())).keys())
    
    # Générer une palette de couleurs
    colors = generate_color_palette(len(models))

    # Création du graphique avec subplots
    fig = sp.make_subplots(rows=1, cols=3, subplot_titles=("F1 Score", "Recall", "AUC PR"))

    # Boucle sur les taux de sous-échantillonnage
    for i, model_name in enumerate(models):  # Pour parcourir les modèles
        f1_scores = []
        recalls = []
        auc_prs = []
        sampling_strategies = []

        for sampling_strategy, models in RUS_models_results.items():
            sampling_strategies.append(sampling_strategy)
            metrics = models[model_name]
            f1_scores.append(metrics['F1 Score'])
            recalls.append(metrics['Recall'])
            auc_prs.append(metrics['AUC PR'])

        # F1 Score
        fig.add_trace(go.Scatter(
            x=sampling_strategies,
            y=f1_scores,
            mode='markers+lines',
            name=model_name,
            marker=dict(symbol='circle', size=10),
            line=dict(color=colors[i]),
            legendgroup=model_name  # Groupes de légendes pour synchroniser les interactions
        ), row=1, col=1)

        # Recall
        fig.add_trace(go.Scatter(
            x=sampling_strategies,
            y=recalls,
            mode='markers+lines',
            name=model_name,
            showlegend=False,  # On ne répète pas la légende
            marker=dict(symbol='cross', size=10),
            line=dict(color=colors[i]),
            legendgroup=model_name  # Même groupe de légendes
        ), row=1, col=2)

        # AUC PR
        fig.add_trace(go.Scatter(
            x=sampling_strategies,
            y=auc_prs,
            mode='markers+lines',
            name=model_name,
            showlegend=False,  # On ne répète pas la légende
            marker=dict(symbol='triangle-up', size=10),
            line=dict(color=colors[i]),
            legendgroup=model_name  # Même groupe de légendes
        ), row=1, col=3)

    # Mise en forme de la figure
    fig.update_layout(
        title={
            'text': 'Performance des Modèles selon le Taux de Sous-Échantillonnage',
            'x': 0.5,  # Centrer le titre principal
            'xanchor': 'center'
        },
        width=1000,  # Largeur de la figure
        height=400,  # Hauteur de la figure
        template='plotly_white'
    )

    # Mettre à jour les titres des axes X pour chaque sous-graphe
    for col in range(1, 4):
        fig.update_xaxes(title_text="Taux de Sous-Échantillonnage", row=1, col=col)

    # Affichage de la figure
    fig.show()

def load_checkpoint_model(checkpoint_dir: str, model_name_in_folder: str, force_retrain: bool = False):
    """
    Charge un modèle et ses résultats à partir des fichiers de checkpoint, ou retourne des dictionnaires vides si `force_retrain=True`.

    Args:
        checkpoint_dir (str): Dossier contenant les checkpoints.
        model_name_in_folder (str): Nom du modèle à charger.
        force_retrain (bool): Si True, retourne des dictionnaires vides pour forcer le réentraînement.
        
    Returns:
        (dict, dict): Tuple contenant le modèle chargé et les résultats. 
                      Si aucun checkpoint n'est trouvé ou si `force_retrain=True`, retourne des dictionnaires vides.
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name_in_folder}_checkpoint.pkl")
    results_path = os.path.join(checkpoint_dir, f"{model_name_in_folder}_results.pkl")

    # Si force_retrain est True, on retourne des dictionnaires vides pour forcer le réentraînement
    if force_retrain:
        print(f"force_retrain is True. Returning empty dictionaries for {model_name_in_folder} to force retraining.")
        return {}, {}

    # Sinon, on tente de charger les checkpoints
    if os.path.exists(checkpoint_path) and os.path.exists(results_path):
        try:
            model = joblib.load(checkpoint_path)
            results = joblib.load(results_path)
            print(f"Checkpoint loaded from {checkpoint_path} and {results_path}")
            return model, results
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return {}, {}
    else:
        print(f"No checkpoint found for {model_name_in_folder}. Returning empty dictionaries.")
        return {}, {}

def save_checkpoint(checkpoint_dir: str, model_name_in_folder: str, model: dict, results: dict):
    """
    Sauvegarde un modèle et ses résultats dans des fichiers de checkpoint.

    Parameters:
    - checkpoint_dir : Le dossier des checkpoints.
    - model_name_in_folder (str): Le nom du modèle sous lequel il sera enregistré dans le dossier de checkpoints.
    - model (dict): Le modèle à sauvegarder.
    - results (dict): Les résultats à sauvegarder associés au modèle.
    
    Example:
    >>> save_checkpoint("my_model_name", trained_model, results_dict)
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name_in_folder}_checkpoint.pkl")
    results_path = os.path.join(checkpoint_dir, f"{model_name_in_folder}_results.pkl")
    
    try:
        joblib.dump(model, checkpoint_path)
        joblib.dump(results, results_path)
        print(f"Checkpoint saved successfully at {checkpoint_path} and {results_path}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def save_final_model(models_fitted_dir: str, model_name_in_folder: str, trained_models: dict):
    """
    Sauvegarde le modèle entraîné dans un fichier .pkl.

    Args:
        models_fitted_dir (str): Chemin du répertoire où le modèle sera sauvegardé.
        model_name_in_folder (str): Nom du modèle pour la sauvegarde.
        trained_models (dict): Dictionnaire des modèles entraînés à sauvegarder.

    Returns:
        str : Chemin du fichier de modèle sauvegardé.
    """
    final_model_path = os.path.join(models_fitted_dir, f"{model_name_in_folder}.pkl")
    
    try:
        joblib.dump(trained_models, final_model_path)
        print(f"Models saved in {final_model_path}")
    except Exception as e:
        print(f"Error saving final models: {e}")

def load_final_model(models_fitted_dir: str, model_name_in_folder: str):
    """
    Charge un modèle final enregistré à partir du fichier correspondant.

    Args:
        models_fitted_dir (str): Dossier contenant le modèle final enregistré.
        model_name_in_folder (str): Nom du fichier du modèle final à charger (sans l'extension).

    Returns:
        dict: Dictionnaire des modèles sauvegardés ou un dictionnaire vide s'il n'y a pas de modèle.
    """
    model_path = os.path.join(models_fitted_dir, f"{model_name_in_folder}.pkl")

    # Vérifie si le fichier de modèle final existe
    if os.path.exists(model_path):
        try:
            # Charge le modèle final enregistré
            models = joblib.load(model_path)
            print(f"Final models loaded from {model_path} \n\n")
            return models
        except Exception as e:
            print(f"Error loading final model from {model_path}: {e} \n")
            return {}
    else:
        print(f"No final model found at {model_path}. Returning empty dictionary. \n")
        return {}

def load_results(results_dir: str, results_file_name: str):
    """
    Charge un modèle final enregistré à partir du fichier correspondant.

    Args:
        results_dir (str): Dossier contenant les résultats du modèle entrainé.
        results_file_name (str): Nom du fichier du dictionnaire des résultas (sans l'extension).

    Returns:
        dict: Dictionnaire des modèles sauvegardés ou un dictionnaire vide s'il n'y a pas de modèle.
    """
    model_path = os.path.join(results_dir, f"{results_file_name}_results.pkl")

    # Vérifie si le fichier de modèle final existe
    if os.path.exists(model_path):
        try:
            # Charge le modèle final enregistré
            models = joblib.load(model_path)
            print(f"Final results loaded from {model_path} \n\n")
            return models
        except Exception as e:
            print(f"Error loading final results from {model_path}: {e} \n")
            return {}
    else:
        print(f"No final results found at {model_path}. Returning empty dictionary. \n")
        return {}