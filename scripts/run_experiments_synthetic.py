
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
from faircrowd.truth_inference import MajorityVoting, DawidSkene, LearningFromCrowd
from faircrowd.utils.exploratory import *

from faircrowd.competitors.fair_td import *
from faircrowd.competitors.geom_model import *
from faircrowd.benchmark_new.comparison import *
from faircrowd.competitors import FairTD,FairTD_ref,Optimal_reg, Geometric,Majority_gold_op

from faircrowd.truth_inference.dawid_skene_multiple import DawidSkeneMultiple

import os
from sklearn.model_selection import train_test_split
path_save="faircrowd/benchmark_new/exp_save"
from data_generators.generate_synthetic_data import *



N=2000
R=100


R_annots_max=5
bias_toward_1=0.2
prop_s_1=0.6 # proportion sensible attrivute 
rho_0=(0,0.5) # typical scaling of annotor errors for sensitive attribute =0
rho_1=(0,0.4)# typical scaling of annotor errors for sensitive attribute =0

seed=0

Y_annotators,S,Y,confusion,prop= generate_synhtetic_binary_restricted(N=N,R=R,R_anot_max=R_annots_max,bias_toward_1=bias_toward_1,prop_s_1=prop_s_1,rho_0=rho_0,rho_1=rho_1,seed=seed)

df= FairCrowdDataset(pd.DataFrame(Y_annotators),pd.DataFrame(S[:,None]),pd.DataFrame(np.zeros(len(Y_annotators))),pd.DataFrame(Y[:,None]))


Y_noise_S=np.hstack((df.answers.values,df["s"].values.flatten()[:,None]))
df.head()

n_classes_=max(df["y"].values)+1
n_annotators=len(df.answers.values[0])

train_i, test_i = train_test_split(
                    df._df.index, test_size=0.4, shuffle=True, random_state=0)#, random_state=seed
                #)

df_test_answers=df.answers.loc[test_i].values
df_test_s=df.s.loc[test_i].values.flatten
df_test_y=df.y.loc[test_i].values.flatten()

df_answers=df.answers.values
df_s=df.s.values.flatten()
df_y=df.y.values.flatten()



dwm=DawidSkeneMultiple()
out=dwm.run(df)
reg_DS=Optimal_reg(dwm)
reg_DS.name="FC_DS"


post_td_DS=FairTD_Post(dwm)
post_td_DS.name="Post_TD_DS"



geo=Geometric(n_classes_,n_annotators)
geo.fit(df_answers,df_s,df_y)

reg_bayes=Optimal_reg(geo)
reg_bayes.name="FC_Bayes"

post_td_bayes=FairTD_Post(geo)
post_td_bayes.name="Post_TD_Bayes"





#Majoritu vote
maj=Majority_vote(n_classes_,n_annotators)
reg_maj=Optimal_reg(maj)
reg_maj.name="FC_Maj"

post_td_maj=FairTD_Post(maj)
post_td_maj.name="Post_TD_Maj"

# maj_gold=Majority_gold_op(n_classes_)
# maj_gold.fit(df_test_answers,df_test_s,df_test_y)
# reg_maj_gold=Optimal_reg(maj_gold)
# reg_maj_gold.name="FC_Majority_Gold"






td_algorithms=[FairTD_ref(),post_td_bayes,reg_bayes,reg_maj,post_td_maj,reg_DS,post_td_DS]

eps_list=[0.01,0.05,0.1,0.2]

accuracy_metrics=[accuracy,f1_score]
fairness_metrics=[DemographicParity_(), EqualOpportunities_(), PredictiveParity_()]

train_sizes_list=[0.4]
d_data=evaluate_algorithms(
    df,
    td_algorithms,
    eps_list,
    accuracy_metrics,
    fairness_metrics,train_sizes_list=train_sizes_list,n_repet=10,filename_prefix="1_synhtetic_exp",full_test=False
)



plot_f1_vs_demographic_parity_with_variance(
    d_data,save_dir="figures_synthetic"
)

print("datset size", len(df.df))
print("number annotators", len(df.answers.values[0]))






# fairtd=FairTD(eps=0.03)

# output=fairtd.run(df)
# demo=DemographicParity()
# print(output.labels)
# print(f1_score(df["y"],output.labels))
# print(demo.compute(output.labels,df["s"],df["y"]))
