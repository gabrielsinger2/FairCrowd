

from faircrowd.datasets import load_crowd_judgment, load_jigsaw_toxicity, load_synthetic
from faircrowd.metrics import (
  accuracy, precision, recall, f1_score, false_positive_rate, false_negative_rate,
  DemographicParity_, EqualOpportunities_, PredictiveParity_
)
from faircrowd.truth_inference import MajorityVoting, DawidSkene, LearningFromCrowd
from faircrowd.utils.exploratory import *


from faircrowd.competitors.fair_td import *
from faircrowd.competitors.geom_model import *
from faircrowd.benchmark_new.comparison import *
from faircrowd.competitors import FairTD,Optimal_reg, Geometric,GLAD_op,DS_op,DawidSkeneSharedConfusion,Majority_gold_op

import numpy as np
import matplotlib.pyplot as plt

def plot_annotations_per_expert(X):
    """
    Trace l'histogramme du nombre d'annotations par expert.

    Parameters
    ----------
    X : np.ndarray of shape (N, R)
        Tableau des réponses (np.nan si pas de réponse)
    """
    # Nombre d'annotations par expert (par colonne)
    annotations_per_expert = np.sum(~np.isnan(X), axis=0)
    print("nb d'expert à moins de 3 réponses", len(annotations_per_expert[annotations_per_expert<2]))

    plt.figure()
    plt.hist(annotations_per_expert, bins='auto')
    plt.xlabel("Nombre d'annotations")
    plt.ylabel("Nombre d'experts")
    plt.title("Histogramme du nombre d'annotations par expert")
    plt.show()

def plot_experts_per_response(X):
    """
    Trace l'histogramme du nombre d'experts par réponse.

    Parameters
    ----------
    X : np.ndarray of shape (N, R)
        Tableau des réponses (np.nan si pas de réponse)
    """
    # On enlève les NaN et on aplatit
    responses = np.sum(~np.isnan(X), axis=1)

    print("mean", np.median(responses),"std",np.std(responses))

    # Comptage par réponse
    #unique_responses, counts = np.unique(responses, return_counts=True)

    plt.figure()
    plt.hist(responses, bins='auto')
    plt.xlabel("Réponse")
    plt.ylabel("Nombre d'experts")
    plt.title("Nombre d'experts par réponse")
    plt.show()

import numpy as np

def filter_experts_min_annotations(X, min_annotations=3):
    """
    Retire les experts ayant moins de `min_annotations` réponses.

    Parameters
    ----------
    X : np.ndarray of shape (N, R)
        Tableau des réponses (np.nan si pas de réponse)
    min_annotations : int, default=3
        Nombre minimum de réponses requises par expert

    Returns
    -------
    X_filtered : np.ndarray
        Tableau avec uniquement les experts conservés
    kept_experts : np.ndarray
        Indices des experts conservés
    """
    # Nombre de réponses par expert (par colonne)
    annotations_per_expert = np.sum(~np.isnan(X), axis=0)

    # Masque des experts à conserver
    kept_experts = np.where(annotations_per_expert >= min_annotations)[0]

    # Filtrage des colonnes
    X_filtered = X[:, kept_experts]

    return X_filtered, kept_experts



seed=0
min_annotations_per_expert=3
df = load_jigsaw_toxicity(min_annotations_per_expert=min_annotations_per_expert)

#print(df.answers.columns)

print("ratio S", np.sum(df["s"].values)/len(df))

print("ratio Y", np.sum(df["y"].values)/len(df))


Y_noise_S=np.hstack((df.answers.values,df["s"].values[:,None]))
df.head()
print("datset size", len(df.df))
print("number annotators", len(df.answers.values[0]))





new,kep=filter_experts_min_annotations(df.answers.values)
print(new.shape)


plot_annotations_per_expert(new)
plot_experts_per_response(new)
