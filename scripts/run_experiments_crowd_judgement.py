
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from faircrowd.datasets import load_crowd_judgment, load_jigsaw_toxicity, load_synthetic
from faircrowd.metrics import (
  accuracy, precision, recall, f1_score, false_positive_rate, false_negative_rate,
  DemographicParity_, EqualOpportunities_, PredictiveParity_
)

from faircrowd.utils.exploratory import *

from faircrowd.competitors.fair_td import *
from faircrowd.competitors.geom_model import *
from faircrowd.benchmark_new.comparison import *
from faircrowd.competitors import FairTD,Optimal_reg, Geometric,Majority_gold_op

from faircrowd.truth_inference.dawid_skene_multiple import DawidSkeneMultiple

import os
from sklearn.model_selection import train_test_split
path_save="faircrowd/benchmark_new/exp_save"


seed=0
df = load_crowd_judgment()
Y_noise_S=np.hstack((df.answers.values,df["s"].values[:,None]))
df.head()

n_classes_=max(df["y"].values)+1
n_annotators=len(df.answers.values[0])

train_i, test_i = train_test_split(
                    df._df.index, test_size=0.4, shuffle=True, random_state=0)#, random_state=seed
                #)

df_test_answers=df.answers.loc[test_i].values
df_test_s=df.s.loc[test_i].values.flatten()
df_test_y=df.y.loc[test_i].values.flatten()

df_answers=df.answers.values
df_s=df.s.values.flatten()
df_y=df.y.values.flatten()


# DS method, called DSMultiple because confusion probabilities depends on A=a
dwm=DawidSkeneMultiple()
out=dwm.run(df)
reg_DSM=Optimal_reg(dwm)
reg_DSM.name="FC_DS"



post_td_DSM=FairTD_Post(dwm)
post_td_DSM.name="Post_TD_DS"


# Geometric is Bayes 
geo=Geometric(n_classes_,n_annotators)
geo.fit(df_test_answers,df_test_s,df_test_y)

#Optimal reg is FairCrowd
reg_count=Optimal_reg(geo)
reg_count.name="FC_Bayes_Count"

geo=Geometric(n_classes_,n_annotators)
geo.fit(df_answers,df_s,df_y)


reg_bayes=Optimal_reg(geo)
reg_bayes.name="FC_Bayes"

post_td_bayes=FairTD_Post(geo)
post_td_bayes.name="Post_TD_Bayes"








#Majority vote
maj=Majority_vote(n_classes_,n_annotators)
reg_maj=Optimal_reg(maj)
reg_maj.name="FC_Maj"

post_td_maj=FairTD_Post(maj)
post_td_maj.name="Post_TD_Maj"

# maj_gold=Majority_gold_op(n_classes_)
# maj_gold.fit(df_test_answers,df_test_s,df_test_y)
# reg_maj_gold=Optimal_reg(maj_gold)
# reg_maj_gold.name="FC_Majority_Gold"



td_algorithms=[FairTD_ref(),reg_bayes,post_td_bayes,reg_maj,post_td_maj,post_td_DSM,reg_DSM]
eps_list=[0.01,0.05,0.1,0.2]
accuracy_metrics=[accuracy,f1_score]
fairness_metrics=[DemographicParity_(), EqualOpportunities_(), PredictiveParity_()]

train_sizes_list=[0.4]

print("RUN")
d_data=evaluate_algorithms(
    df,
    td_algorithms,
    eps_list,
    accuracy_metrics,
    fairness_metrics,train_sizes_list=train_sizes_list,n_repet=1,filename_prefix="reg_crowd_jugement_exp",full_test=False
)


print("Visual")
plot_f1_vs_demographic_parity_with_variance(
    d_data,save_dir="figures_crowd_judgement",filename_prefix="test_f1_vs_dp"
)

print("datset size", len(df.df))
print("number annotators", len(df.answers.values[0]))







