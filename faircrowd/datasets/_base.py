import numpy as np
import pandas as pd
from typing import Dict, Any
from ..core.fair_crowd_dataset import FairCrowdDataset
import os


def load_file(name: str, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    current_dir = os.path.dirname(__file__)
    absolute_path = os.path.join(current_dir, name)
    return pd.read_csv(absolute_path, **kwargs)


def load_crowd_judgment():
    """
    Load the Crowd Judgment dataset
    """
    compas = load_file("./CrowdJudgement/BROWARD_CLEAN_SUBSET.csv")
    compas["black"] = np.where(compas["race"] == 2, 1, 0)
    compas.drop("race", axis=1, inplace=True)
    compas.drop("charge_id", axis=1, inplace=True)
    compas.drop("block_num", axis=1, inplace=True)
    compas.drop("compas_decile_score", axis=1, inplace=True)
    compas.set_index("id", inplace=True)
    s = compas[["black"]]
    x = compas[
        [
            "sex",
            "age",
            "juv_fel_count",
            "juv_misd_count",
            "priors_count",
            "charge_degree",
        ]
    ]
    y = compas[["two_year_recid"]]
    mturk = load_file("./CrowdJudgement/MTURK_RACE.csv")
    mturk.drop(0, axis=0, inplace=True)
    mturk["id"] = np.array(mturk["mTurk_code"], dtype="int64")
    mturk.drop("mTurk_code", axis=1, inplace=True)
    mturk.set_index("id", inplace=True)
    mturk.replace(-1, 0, inplace=True)
    for col in mturk.columns:
        mturk[col] = np.where(np.isnan(mturk[col]), np.nan, 1) * np.where(
            mturk[col] == 1, y["two_year_recid"], 1 - y["two_year_recid"]
        )
    return FairCrowdDataset(answers=mturk, s=s, x=x, y=y)


def filter_experts_min_annotations_df(df, min_annotations=3):
    """
    Retire les experts (colonnes) ayant moins de `min_annotations` réponses dans un DataFrame.

    Parameters
    ----------
    df : pd.DataFrame of shape (N, R)
        Tableau des réponses (NaN si pas de réponse)
    min_annotations : int, default=3
        Nombre minimum de réponses requises par expert

    Returns
    -------
    df_filtered : pd.DataFrame
        DataFrame avec uniquement les experts conservés
    kept_experts : list
        Liste des noms de colonnes conservées
    """
    # Nombre de réponses par expert
    annotations_per_expert = df.notna().sum(axis=0)

    # Colonnes à conserver
    kept_experts = annotations_per_expert[annotations_per_expert >= min_annotations].index.tolist()

    # Filtrage
    df_filtered = df[kept_experts].copy()

    return df_filtered, kept_experts

def load_jigsaw_toxicity(min_annotations_per_expert=3):
    """
    Load the Jigsaw Toxicity dataset
    """
    features = load_file("./JigsawToxicity/features_for_ml.csv")
    features.set_index("id", inplace=True)
    x = features[
        [
            "obscene",
            "identity_attack",
            "insult",
            "threat",
            "funny",
            "wow",
            "sad",
            "likes",
            "disagree",
            "sexual_explicit",
        ]
    ].copy()
    all_s = features[
        [
            "asian",
            "atheist",
            "bisexual",
            "black",
            "christian",
            "female",
            "heterosexual",
            "hindu",
            "homosexual_gay_or_lesbian",
            "intellectual_or_learning_disability",
            "jewish",
            "latino",
            "male",
            "muslim",
            "physical_disability",
            "psychiatric_or_mental_illness",
            "transgender",
            "white",
        ]
    ]
    responses = load_file("./JigsawToxicity/responses_td_format.csv")
    mturk = pd.DataFrame(
        index=x.index,
        columns=pd.unique(responses.sort_values("worker")["worker"]),
        dtype=np.float64,
    )
    for _, resp in responses.iterrows():
        mturk.loc[resp["question"], resp["worker"]] = resp["answer"]
    y = load_file("./JigsawToxicity/ground_truth_label.csv")
    y.set_index("question", inplace=True)
    s = all_s.iloc[:, 0] > 0
    for col in all_s.columns:
        s = s | (features[col] > 0)
    s = 0 + s
    s = pd.DataFrame(s.values, index=all_s.index, columns=["s"])

    #filter ici

    print("keep annotators with at least 3 annotations")
    mturk_new,ke=filter_experts_min_annotations_df(mturk , min_annotations=min_annotations_per_expert)



    return FairCrowdDataset(answers=mturk_new, s=s, x=x, y=y)


def load_synthetic():
    """
    Load the Synthetic dataset
    """
    synthetic = load_file("./Synthetic/dataset.csv", index_col=0)
    mturk = load_file("./Synthetic/annotations.csv", index_col=0)
    x = synthetic[["x"]]
    s = synthetic[["s"]]
    y = synthetic[["result"]]
    return FairCrowdDataset(answers=mturk, s=s, x=x, y=y)
