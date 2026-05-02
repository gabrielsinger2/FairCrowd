import pandas as pd
import numpy as np
from faircrowd.utils.string_utils import title_case, name
import matplotlib.pyplot as plt
from ..core import FairCrowdDataset, FairnessMetric, AccuracyMetric, TDAlgorithm
from typing import Sequence
from sklearn.model_selection import train_test_split
from faircrowd.utils.exploratory import maj_gold_prior_extension
import time

from faircrowd.competitors.geom_model import Majority_vote
def evaluate_algorithms(
    df,
    td_algorithms,
    eps_list,
    accuracy_metrics,
    fairness_metrics: Sequence[FairnessMetric],n_repet=10,train_sizes_list=[0.4],filename_prefix="exp",save_dir="faircrowd/benchmark_new/exp_save",full_test=False
):
    """
    Génère un DataFrame pandas où chaque ligne correspond à
    (algorithme, epsilon) et les colonnes aux métriques d'accuracy et de fairness.
    """

    rows = []
    
    
    for eps in eps_list:
        for algo in td_algorithms:
            print("Exp "+algo.name+" eps={:.2f}".format(eps))
            for train_size in train_sizes_list:
                if algo.name in ["FairTD"]:
                    print("FAIRTD_run")
                    predictions = algo.run(df, eps=eps).labels
                for seed in range(n_repet):
                    print('seed',seed)
                    train_i, test_i = train_test_split(
                    df._df.index, train_size=train_size, shuffle=True, random_state=seed)

                    df_index_man=df._df
                    df_index_man["easy_index"]=np.arange(len(df._df))
                    test_index_eq_i=df_index_man["easy_index"].loc[test_i].values

                    df_train_answers=df.answers.loc[train_i].values
                    df_train_s=df.s.loc[train_i].values.flatten()
                    df_train_y=df.y.loc[train_i].values.flatten()

                    df_test_answers_=df.answers.loc[test_i]
                    df_test_s=df.s.loc[test_i]
                    df_test_x=df.x.loc[test_i]
                    df_test_y=df.y.loc[test_i]

                    annotations_per_expert_train = np.sum(~np.isnan(df_train_answers), axis=0)
                    # expets that contributes
                    kept_experts_train = np.where(annotations_per_expert_train >= 1)[0]

                    set_train_experts=set(np.arange(len(df_train_answers[0]))[kept_experts_train])

                    dt1=time.time()

                    if algo.name in ["FC_Majority_Gold"]:
                        print("fit")
                        df_test_answers=df.answers.loc[test_i].values
                        # take annotators that are not in train
                        annotations_per_expert_test = np.sum(~np.isnan(df_test_answers), axis=0)

                        # expets that contributes
                        kept_experts_test = np.where(annotations_per_expert_test >= 1)[0]
                        set_test_experts=set(np.arange(len(df_test_answers[0]))[kept_experts_test])
                        experts_that_should_use_a_prior=set_test_experts.difference(set_train_experts)
                        if len(experts_that_should_use_a_prior)>0:
                            print("experts not present in the train set",experts_that_should_use_a_prior)
                            prior_answers,y_prior=maj_gold_prior_extension(experts_that_should_use_a_prior,len(df_train_answers),len(df_train_answers[0]))
                            algo.ovr.fit(np.concatenate((df_train_answers,prior_answers),axis=0),df_train_s,np.concatenate((df_train_y,y_prior),axis=0))
                        else:
                            algo.ovr.fit(df_train_answers,df_train_s,df_train_y)
 
                    if algo.name in ["FC_Bayes_Count","FC_Bayes_Count_Glad","FC_Bayes_Count_Res","Post_TD_Bayes_Count","Post_TD_Bayes_Count_Glad"]:
                        print("fit")
                        #maj=Majority_vote(2,len(df_train_answers[0]))
                        #df_train_answers_extend=np.concatenate((df_train_answers,df_test_answers_.values),axis=0)
                        #df_train_s_extend=np.concatenate((df_train_s,df_test_s.values),axis=0)
                        
                        #X_S=np.hstack((df_test_answers_.values,df_test_s.values.flatten()[:,None]))
                        #df_train_y_extend=np.concatenate((df_train_y,maj.predict(X_S)),axis=0)



                        algo.ovr.fit(df_train_answers,df_train_s,df_train_y)
                        #algo.ovr.fit(df_train_answers_extend,df_train_s_extend,df_train_y_extend)

                    
                    df_test=FairCrowdDataset(df_test_answers_,df_test_s,df_test_x,df_test_y)

                    #Fair TD can consider allthedataset without looking at the groundtruth
                    if algo.name in ["FairTD"]:
                        print("FAIRTD_run")
                        
                        
            
                        #predictions = pd.DataFrame(predictions).reindex(df._df.index)
                        if not(full_test):
                            predictions_ = predictions[test_index_eq_i]
                            true_y=df_test["y"]
                            true_s=df_test["s"]
                        else:
                            true_y=df["y"]
                            true_s=df["s"]
                    else:
                        if not(full_test):
                            predictions_ = algo.run(df_test, eps=eps).labels
                            true_y=df_test["y"]
                            true_s=df_test["s"]
                        else:
                            predictions_ = algo.run(df, eps=eps).labels
                            true_y=df["y"]
                            true_s=df["s"]

                    dt2=time.time()
                    delta=dt2-dt1

                    
                    row = {
                        "algorithm": algo.name,
                        "epsilon": eps,"seed":seed,"train_size":train_size,"execution_time":delta
                    }

                    # Accuracy metrics
                    for metric in accuracy_metrics:
                        #print(predictions)
                        #print(df_test["y"])
                        row[name(metric)] = metric(true_y, predictions_)

                    # Fairness metrics
                    for metric in fairness_metrics:
                        row[name(metric)] = metric.compute(predictions_, true_s, true_y)

                    rows.append(row)
            

    result = pd.DataFrame(rows)
    algo_col="algorithm"

    trains = sorted(result["train_size"].unique())
    trains=[str(el) for el in trains]
    train_tag = "-".join(trains)

    algos = sorted(result[algo_col].unique())
    algo_tag = "-".join(algos)
    seed_col="seed"
    n_seeds = result[seed_col].nunique()
    filename = (
        f"{filename_prefix}_"
        f"algos-{algo_tag}_"
        f"seeds-{n_seeds}"
        f"seeds-{train_tag}"
    )

    # Sauvegarde
    csv_path = os.path.join(save_dir, f"{filename}.csv")
    result.to_csv(csv_path)
    return result



def plot_f1_vs_demographic_parity(
    df,
    algo_col="algorithm",
    eps_col="epsilon",
    dp_col="DemographicParity_",
    f1_col="f1_score",
):
    """
    Trace F1-score vs Demographic Parity.
    Une couleur par algorithme, chaque point annoté avec epsilon.
    """
    #df.group
    dg=df.groupby([algo_col, eps_col])[[dp_col,f1_col]].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(7, 5))

    for algo, subdf in dg.groupby(algo_col):
        
        ax.scatter(
            subdf[dp_col],
            subdf[f1_col],
            label=algo,
            alpha=1
        )

        # Annotation epsilon pour chaque point
        for _, row in subdf.iterrows():
            ax.annotate(
                f"{row[eps_col]:.2f}",
                (row[dp_col], row[f1_col]),
                fontsize=8,
                xytext=(3, 3),
                textcoords="offset points",
            )

    ax.set_xlabel("Demographic Parity")
    ax.set_ylabel("F1 score")
    ax.set_title("F1 vs Demographic Parity")
    ax.legend(title="Algorithm")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_f1_vs_demographic_parity_with_variance(
    df,
    algo_col="algorithm",
    eps_col="epsilon",
    seed_col="seed",
    dp_col="DemographicParity_",
    f1_col="f1_score",
    save_dir="figures_crowd_judgement",
    filename_prefix="f1_vs_dp",
):
    """
    - Courbe moyenne (sur les seeds)
    - Frange = ± écart-type
    - Sauvegarde automatique avec métadonnées dans le nom
    """

    # Nombre de seeds distinctes
    n_seeds = df[seed_col].nunique()

    trains = sorted(df["train_size"].unique())
    trains=[str(el) for el in trains]
    train_tag = "-".join(trains)
    # Algorithmes utilisés (ordre déterministe)
    algos = sorted(df[algo_col].unique())
    algo_tag = "-".join(algos)

    # Agrégation stats
    agg = (
        df
        .groupby([algo_col, eps_col])
        .agg(
            dp_mean=(dp_col, "mean"),
            dp_std=(dp_col, "std"),
            f1_mean=(f1_col, "mean"),
            f1_std=(f1_col, "std"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))  # ICML-friendly ratio

    for algo, subdf in agg.groupby(algo_col):
        subdf = subdf.sort_values(eps_col)

        x = subdf["dp_mean"].values
        y = subdf["f1_mean"].values
        y_std = subdf["f1_std"].values
        if algo[:2]=="FC":
            bonus=' (Ours)'
        else:
            bonus=''

        if algo[:2]=="FC":
            ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=algo+bonus)
            
        else:
            
            ax.plot(
            x,
            y,'--',
            marker="o",
            linewidth=2,
            label=algo+bonus,
        )
        
        

        ax.fill_between(
            x,
            y - y_std,
            y + y_std,
            alpha=0.2,
        )
        #Check if annotations work
        k=0
        for _, row in subdf.iterrows():
            if k==0:
                leg=r"$\epsilon:$"+f"{row[eps_col]:.2f}"
            else:
                leg=f"{row[eps_col]:.2f}"
            k=k+1
            ax.annotate(
                    leg,
                    (row["dp_mean"], row["f1_mean"]),
                    fontsize=8,
                    xytext=(3, 3),
                    textcoords="offset points",
                )

    ax.set_xlabel("Demographic Parity")
    ax.set_ylabel("F1 score")
    ax.set_title("Fairness–Accuracy Trade-off")
    ax.legend(title="Algorithm")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    # Création du dossier si nécessaire
    os.makedirs(save_dir, exist_ok=True)

    # Nom de fichier informatif
    filename = (
        f"{filename_prefix}_"
        f"algos-{algo_tag}_"
        f"seeds-{n_seeds}"
        f"seeds-{train_tag}"
    )

    # Sauvegarde
    pdf_path = os.path.join(save_dir, f"{filename}.pdf")
    png_path = os.path.join(save_dir, f"{filename}.png")

    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to:\n- {pdf_path}\n- {png_path}")


def plot_f1_vs_train_size_with_variance(
    df,
    algo_col="algorithm",
    seed_col="seed",
    dp_col="train_size",
    f1_col="f1_score",
    save_dir="figures_crowd_judgement",
    filename_prefix="f1_vs_dp",
):
    """
    - Courbe moyenne (sur les seeds)
    - Frange = ± écart-type
    - Sauvegarde automatique avec métadonnées dans le nom
    """

    # Nombre de seeds distinctes
    n_seeds = df[seed_col].nunique()
    eps_col=dp_col
    # Algorithmes utilisés (ordre déterministe)
    algos = sorted(df[algo_col].unique())
    algo_tag = "-".join(algos)

    trains = sorted(df["train_size"].unique())
    trains=[str(el) for el in trains]
    train_tag = "-".join(trains)

    # Agrégation stats
    agg = (
        df
        .groupby([algo_col, eps_col])
        .agg(
            dp_mean=(dp_col, "mean"),
            f1_mean=(f1_col, "mean"),
            f1_std=(f1_col, "std"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))  # ICML-friendly ratio

    for algo, subdf in agg.groupby(algo_col):
        subdf = subdf.sort_values(eps_col)

        x = subdf["dp_mean"].values
        y = subdf["f1_mean"].values
        y_std = subdf["f1_std"].values

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=algo,
        )
        

        ax.fill_between(
            x,
            y - y_std,
            y + y_std,
            alpha=0.2,
        )
        #Check if annotations work
        # for _, row in subdf.iterrows():
        #     ax.annotate(
        #             f"{row[eps_col]:.2f}",
        #             (row["dp_mean"], row["f1_mean"]),
        #             fontsize=8,
        #             xytext=(3, 3),
        #             textcoords="offset points",
        #         )

    ax.set_xlabel("Train size ratio")
    ax.set_ylabel("F1 score")
    ax.set_title("Fairness–Accuracy Trade-off")
    ax.legend(title="Algorithm")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    # Création du dossier si nécessaire
    os.makedirs(save_dir, exist_ok=True)

    # Nom de fichier informatif
    filename = (
        f"{filename_prefix}_"
        f"algos-{algo_tag}_"
        f"seeds-{n_seeds}"
        f"seeds-{train_tag}"
    )

    # Sauvegarde
    pdf_path = os.path.join(save_dir, f"{filename}.pdf")
    png_path = os.path.join(save_dir, f"{filename}.png")

    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to:\n- {pdf_path}\n- {png_path}")


