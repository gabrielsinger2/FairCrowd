
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from faircrowd.benchmark_new.convergence import *

nb_exp_list=[3,5,8,10,15,20,30,40]

N=2000
R=100#8 donne des beau plot mais apprentissage trop facile

# Semble que le problème devient plus facile avec plus d'annotateur
R_annots_max=5
bias_toward_1=0
prop_s_1=0.5 

# Figure 1-(a) Globally competent
rho_0=(0,0.5) # 1-P_0 typical scaling of annotor errors for sensitive attribute =0
rho_1=(0,0.4)# 1-P_1 typical scaling of annotor errors for sensitive attribute =0

# Figure 1-(b) Globally adversarial 
rho_0=(0.4,0.8) # 
rho_1=(0.4,0.9)# 


## Figure 1-(c) No information
rho_0=(0.49,0.51) 
rho_1=(0.49,0.51) 

seed=0 

param={"N":N,"R":R,"bias_toward_1":bias_toward_1,"prop_s_1":prop_s_1,"rho_0":rho_0,"rho_1":rho_1}

d_data=evaluate_convergence_algorithms(
    nb_exp_list,param,n_repet=30,filename_prefix="exp",save_dir="convergence_figures/")



plot_convergence(
    d_data,param,
    an_col="Nb_annotators",
    seed_col="seed",
    DP_col="DP_pred_DP_true_diff",
    save_dir="convergence_figures",
    filename_prefix="conv",
)








# fairtd=FairTD(eps=0.03)

# output=fairtd.run(df)
# demo=DemographicParity()
# print(output.labels)
# print(f1_score(df["y"],output.labels))
# print(demo.compute(output.labels,df["s"],df["y"]))
