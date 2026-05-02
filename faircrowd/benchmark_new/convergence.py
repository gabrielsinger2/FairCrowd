import pandas as pd
import numpy as np
from faircrowd.utils.string_utils import title_case, name
import matplotlib.pyplot as plt
from ..core import FairCrowdDataset, FairnessMetric, AccuracyMetric, TDAlgorithm
from typing import Sequence
from sklearn.model_selection import train_test_split
from faircrowd.utils.exploratory import maj_gold_prior_extension
import time

from faircrowd.competitors.geom_model import Majority_vote, Geometric
from faircrowd.truth_inference import DawidSkene
from faircrowd.datasets import *
from generate_synthetic_data import *



from faircrowd.metrics.fairness import DemographicParity_
def evaluate_convergence_algorithms(
    nb_exp_list,param,n_repet=10,filename_prefix="exp",save_dir="convergence_figures/"):
    N=param["N"]
    #R=param["R"]
    bias_toward_1=param["bias_toward_1"]
    prop_s_1=param["prop_s_1"] # proportion sensible attrivute 
    rho_0=param["rho_0"] # typical scaling of annotor errors for sensitive attribute =0
    rho_1=param["rho_1"]# typical scaling of annotor errors for sensitive attribute =1

    DP=DemographicParity_()
    rows=[]
    for R_exp in nb_exp_list:
        print("R_exp",R_exp)

        R_annots_max=R_exp
        
    
        for seed in range(n_repet):
            print('seed',seed)
            Y_annotators,S,Y,confusion,prop= generate_synhtetic_binary_restricted(N=N,R=R_exp,R_anot_max=R_annots_max,bias_toward_1=bias_toward_1,prop_s_1=prop_s_1,rho_0=rho_0,rho_1=rho_1,seed=seed)
   
            df= FairCrowdDataset(pd.DataFrame(Y_annotators),pd.DataFrame(S[:,None]),pd.DataFrame(np.zeros(len(Y_annotators))),pd.DataFrame(Y[:,None]))

            Y_noise_S=np.hstack((df.answers.values,df["s"].values.flatten()[:,None]))
            

            n_classes_=max(df["y"].values)+1
            n_annotators=len(df.answers.values[0])
            geo=Geometric(n_classes_,n_annotators)
            geo.fit(Y_annotators,S,Y)

            Y_pred=geo.predict(Y_noise_S)

            DP_pred_geo=DP.compute(Y_pred,S,Y)
            DP_true=DP.compute(Y,S,Y)

            sum_ind_DP=0
            for i in range(R_exp):
                Y_i=Y_annotators[:,i]
                Ind_i=~np.isnan(Y_i)
                DP_i=DP.compute(Y_i[Ind_i],S[Ind_i],Y[Ind_i])
                sum_ind_DP+=DP_i

            

            row1 = {
                    "algorithm": "Bayes",
                    "seed":seed,"Nb_annotators":R_exp,"DP_pred_DP_true_diff":DP_pred_geo-DP_true,"DP_true":DP_true,"sum_ind_DP":sum_ind_DP
                    }
            rows.append(row1)

            maj=Majority_vote(n_classes_,n_annotators)
            

            Y_pred=maj.predict(Y_noise_S)

            DP_pred_maj=DP.compute(Y_pred,S,Y)
            DP_true=DP.compute(Y,S,Y)

            row2 = {
                    "algorithm": "Majority_vote",
                    "seed":seed,"Nb_annotators":R_exp,"DP_pred_DP_true_diff":DP_pred_maj-DP_true,"DP_true":DP_true,"sum_ind_DP":sum_ind_DP
                    }
            rows.append(row2)


            dwm=DawidSkene()
            out=dwm.run(df).labels

            DP_pred_DS=DP.compute(out,S,Y)
            DP_true=DP.compute(Y,S,Y)

            row3 = {
                    "algorithm": "DS",
                    "seed":seed,"Nb_annotators":R_exp,"DP_pred_DP_true_diff":DP_pred_DS-DP_true,"DP_true":DP_true,"sum_ind_DP":sum_ind_DP
                    }
            rows.append(row3)

    result = pd.DataFrame(rows)
    

    ans = sorted(result["Nb_annotators"].unique())
    ans=[str(el) for el in ans]
    train_tag = "-".join(ans)

    
    n_seeds = result["seed"].nunique()
    filename = (
        f"{filename_prefix}_"
        f"seeds-{n_seeds}"
        f"seeds-{train_tag}"
        f"{str(param)}"
    )

    # Sauvegarde
    csv_path = os.path.join(save_dir, f"{filename}.csv")
    result.to_csv(csv_path)
    return result

def evaluate_convergence_algorithms_fairtd_synth(
    nb_exp_list,n_repet=10,filename_prefix="exp",save_dir="convergence_figures/"):
   
    DP=DemographicParity_()
    rows=[]
    df= load_synthetic()
    Y_annotators_=df.answers.values
    S_=df.s.values.flatten()
    Y_=df.y.values.flatten()
    for R_exp in nb_exp_list:
        print("R_exp",R_exp)

        R_annots_max=R_exp
        Y_annotators=Y_annotators_[:,:R_exp]
        
    
        for seed in range(n_repet):
            print('seed',seed)
            Y_annotators,S,Y,confusion,prop= generate_synhtetic_binary_restricted(N=N,R=R_exp,R_anot_max=R_annots_max,bias_toward_1=bias_toward_1,prop_s_1=prop_s_1,rho_0=rho_0,rho_1=rho_1,seed=seed)
   
            df= FairCrowdDataset(pd.DataFrame(Y_annotators),pd.DataFrame(S[:,None]),pd.DataFrame(np.zeros(len(Y_annotators))),pd.DataFrame(Y[:,None]))

            Y_noise_S=np.hstack((df.answers.values,df["s"].values.flatten()[:,None]))
            

            n_classes_=max(df["y"].values)+1
            n_annotators=len(df.answers.values[0])
            geo=Geometric(n_classes_,n_annotators)
            geo.fit(Y_annotators,S,Y)

            Y_pred=geo.predict(Y_noise_S)

            DP_pred_geo=DP.compute(Y_pred,S,Y)
            DP_true=DP.compute(Y,S,Y)

            sum_ind_DP=0
            for i in range(R_exp):
                Y_i=Y_annotators[:,i]
                Ind_i=~np.isnan(Y_i)
                DP_i=DP.compute(Y_i[Ind_i],S[Ind_i],Y[Ind_i])
                sum_ind_DP+=DP_i

            

            row1 = {
                    "algorithm": "Bayes",
                    "seed":seed,"Nb_annotators":R_exp,"DP_pred_DP_true_diff":DP_pred_geo-DP_true,"DP_true":DP_true,"sum_ind_DP":sum_ind_DP
                    }
            rows.append(row1)

            maj=Majority_vote(n_classes_,n_annotators)
            

            Y_pred=maj.predict(Y_noise_S)

            DP_pred_maj=DP.compute(Y_pred,S,Y)
            DP_true=DP.compute(Y,S,Y)

            row2 = {
                    "algorithm": "Majority_vote",
                    "seed":seed,"Nb_annotators":R_exp,"DP_pred_DP_true_diff":DP_pred_maj-DP_true,"DP_true":DP_true,"sum_ind_DP":sum_ind_DP
                    }
            rows.append(row2)


            dwm=DawidSkene()
            out=dwm.run(df).labels

            DP_pred_DS=DP.compute(out,S,Y)
            DP_true=DP.compute(Y,S,Y)

            row3 = {
                    "algorithm": "DS",
                    "seed":seed,"Nb_annotators":R_exp,"DP_pred_DP_true_diff":DP_pred_DS-DP_true,"DP_true":DP_true,"sum_ind_DP":sum_ind_DP
                    }
            rows.append(row3)

    result = pd.DataFrame(rows)
    

    ans = sorted(result["Nb_annotators"].unique())
    ans=[str(el) for el in ans]
    train_tag = "-".join(ans)

    
    n_seeds = result["seed"].nunique()
    filename = (
        f"{filename_prefix}_"
        f"seeds-{n_seeds}"
        f"seeds-{train_tag}"
        f"{str(param)}"
    )

    # Sauvegarde
    csv_path = os.path.join(save_dir, f"{filename}.csv")
    result.to_csv(csv_path)
    return result



import matplotlib.pyplot as plt
import numpy as np
import os

def plot_convergence(
    df,param,
    an_col="Nb_annotators",
    seed_col="seed",
    DP_col="DP_pred_DP_true_diff",
    save_dir="convergence_figures",
    filename_prefix="conv",
):
    """
    - Courbe moyenne (sur les seeds)
    - Frange = ± écart-type
    - Sauvegarde automatique avec métadonnées dans le nom
    """
    algo_col="algorithm"
    # Nombre de seeds distinctes
    ans = sorted(df["Nb_annotators"].unique())
    ans=[str(el) for el in ans]
    train_tag = "-".join(ans)

    
    n_seeds = df["seed"].nunique()
    

    # Agrégation stats
    agg = (
        df
        .groupby([algo_col,an_col])
        .agg(
            dp_mean=(DP_col, "mean"),
            dp_std=(DP_col, "std"),
            an_mean=(an_col, "mean"),
            f1_std=(an_col, "std"),
            ind_DP_mean=("sum_ind_DP","mean")
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))  # ICML-friendly ratio

    for algo, subdf in agg.groupby(algo_col):
        subdf = subdf.sort_values(an_col)

        x = subdf["an_mean"].values
        y = subdf["dp_mean"].values
        y_std = subdf["dp_std"].values
        print(algo,y)
        ax.plot(
        x,
        y,
        marker="o",
        linewidth=2,
        label=algo
        )
        

        ax.fill_between(
            x,
            y - y_std,
            y + y_std,
            alpha=0.2,
        )
        #Check if annotations work
        k=0
        # ind_dp = subdf["ind_DP_mean"].values
        # plt.plot(x[:3] ,ind_dp[:3]*np.sqrt(x[:3]))
        

    ax.set_xlabel("Number of annotators")
    ax.set_ylabel(r"DP Gap $\hat{Y}$ - DP gap $Y$")
    ax.set_title("Convergence of DP gap")
    ax.legend(title="Algorithm")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    # Création du dossier si nécessaire
    os.makedirs(save_dir, exist_ok=True)

    # Nom de fichier informatif
    filename = (
        f"{filename_prefix}_"
        f"seeds-{n_seeds}"
        f"seeds-{train_tag}"
        f"{str(param)}"
    )

    # Sauvegarde
    pdf_path = os.path.join(save_dir, f"{filename}.pdf")
    png_path = os.path.join(save_dir, f"{filename}.png")

    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to:\n- {pdf_path}\n- {png_path}")
