import pandas as pd
import matplotlib as plt
import numpy as np
import os

from faircrowd.benchmark_new.comparison import *
path_save="faircrowd/benchmark_new/exp_save"

name_="0.40_crowd_jugement_exp_algos-FairTD-Reg_Bayes_Count-Reg_Majority-Reg_Majority_Gold_seeds-10.csv"
#name_='1_synhtetic_exp_algos-FairTD_seeds-10seeds-0.4.csv'
path_in_td=os.path.join(path_save,name_)


d_data=pd.read_csv(path_in_td)

d_fairtd=d_data.loc[d_data["algorithm"]=="FairTD"]

name_all='new_crowd_jugement_exp_algos-FC_Bayes-FC_DS-FC_Maj-Post_TD_Bayes-Post_TD_DS-Post_TD_Maj_seeds-10seeds-0.4.csv'
#name_all='1_synhtetic_exp_algos-FC_Bayes-FC_DS-FC_Maj-Post_TD_Bayes-Post_TD_DS-Post_TD_Maj_seeds-10seeds-0.4.csv'

path_in_all=os.path.join(path_save,name_all)
d_data_all=pd.read_csv(path_in_all)

d_data=pd.concat((d_fairtd,d_data_all),axis=0)

d_data.to_csv(os.path.join(path_save,'final_crowd.csv'))
#d_data.to_csv(os.path.join(path_save,'final_synhtetic.csv'))


plot_f1_vs_demographic_parity_with_variance(
    d_data,save_dir="with_fairTD",filename_prefix="crowd2_f1_vs_dp"
)

