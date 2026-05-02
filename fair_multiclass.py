
import numpy as np

from sklearn.svm import SVC
from utils_optimization import optCE, optSAGD, optSCIPY, optSCIPY_bivar, optSCIPY_bivar_bis #,beta_optSCIPY_bivar_bis
from scipy.special import softmax

import matplotlib.pyplot as plt

#from synthetic_data import make_unfair_poolclassif, data_viz_tsne
import time



def prepare_fairness(X_pool, ovr):
    """
    compute the inference of the data 'X_pool' by the model 'ovr' and retrieve the inference and the contamination rate
    """
    X0 = X_pool[X_pool[:,-1] == 0]
    X1 = X_pool[X_pool[:,-1] == 1]
    ps = np.array([len(X0), len(X1)])/len(X_pool)
    y_prob_dict = dict()
    y_prob_dict[0] = ovr.predict_proba(X0)#+0.02*np.random.random(size=ovr.predict_proba(X0).shape)#ovr.predict_proba(X0[:,:-1]), we use the sensitive attribute in the geometric model
    y_prob_dict[1] = ovr.predict_proba(X1)#+0.02*np.random.random(size=ovr.predict_proba(X1).shape)#ovr.predict_proba(X1[:,:-1])
    return y_prob_dict, ps
def log_prepare_fairness(X_pool, ovr):
    """
    compute the inference of the data 'X_pool' by the model 'ovr' and retrieve the inference and the contamination rate
    """
    X0 = X_pool[X_pool[:,-1] == 0]
    X1 = X_pool[X_pool[:,-1] == 1]
    ps = np.array([len(X0), len(X1)])/len(X_pool)
    log_y_prob_dict = dict()
    log_y_prob_dict[0] = ovr.predict_log_proba(X0)#ovr.predict_proba(X0[:,:-1]), we use the sensitive attribute in the geometric model
    log_y_prob_dict[1] = ovr.predict_log_proba(X1)#ovr.predict_proba(X1[:,:-1])
    return log_y_prob_dict, ps,np.log(ps)


def concentrate_prob(y,fac=60):

    y=y**fac 
    y=y/y.sum(axis=1)[:,None]
    return y


def reordonate(y,ind):
    z=np.zeros(y.shape)
    for i in range(len(y)):
        z[ind[i]]=i
    return z
def unconcentrate_prob(y,th=0.96):

    ind_0=(y[:,0]>th)
    
    nb_extreme0=np.sum(ind_0*1)

    indices_sort_0 = np.argsort(y[ind_0,0])

    ind_1=(y[:,1]>th)

    nb_extreme1=np.sum(ind_1*1)
    indices_sort_1 = np.argsort(y[ind_1,1])
    #print(max(nb_extreme0,nb_extreme1))
    #print(indices_sort_0[:5])
    #print(np.arange(nb_extreme0)[indices_sort_0][:5])

    y[ind_0,0]=th+(1-th)*reordonate(np.arange(nb_extreme0),indices_sort_0)/nb_extreme0
    y[ind_1,1]=th+(1-th)*reordonate(np.arange(nb_extreme1),indices_sort_1)/nb_extreme1

    return y

#USE
def fair_soft_max_test(X_test, X_pool, ovr, c = 0.001, opt = "SAGD", sigma = 10**(-4), epsilon_fair = 0.1, print_lambda=False):
    """
    for the optimization technique we have "CE" (cross-entropy) or "SAGD" (smoothed accelerated GD)
    """
    start_time = time.time()

    # computation of lambda (soft and hard)
    y_prob_dict, ps = prepare_fairness(X_pool, ovr)
    test=True
    # pre_processing step

    if test:
        
        y_prob_dict[0]=unconcentrate_prob(y_prob_dict[0])
        
        y_prob_dict[1]=unconcentrate_prob(y_prob_dict[1])
        
       
    try:    
        n_classes = ovr.n_classes_
    except:
        n_classes = len(ovr.classes_)

    


    def lam_fairness_soft(lam, c = c):
        res = 0
        for s in [0,1]:
            val = y_prob_dict[s]*ps[s] - (2*s-1)*lam
            res += np.mean(np.sum(softmax(val/c, axis=1)*val, axis=1)) # Smooth arg max
        return(res)

    def bivar_fairness_soft(lam, n_classes = n_classes, c = c):
        res = 0
        lamb = lam[:n_classes]
        beta = lam[n_classes:]
        for s in [0,1]:
            val = y_prob_dict[s]*ps[s] - (2*s-1)*(lamb-beta)
            res += np.mean(np.sum(softmax(val/c, axis=1)*val, axis=1)) # Smooth arg max
        res += epsilon_fair * np.sum(lamb+beta)
        #print(res)
        return(res)


    def nablaG(lam, c = c):
        res = 0
        for s in [0,1]:
            val = y_prob_dict[s]*ps[s] - (2*s-1)*lam
            softmax_val = softmax(val/c, axis=1)
            res -= (2*s-1) * np.mean( softmax_val, axis = 0) # Smooth arg max
        return(res)


    
    if opt == "CE":
        lam_soft = optCE(fun = lam_fairness_soft, n = 2000, d = n_classes, eps = 0.001, max_iter = 100, print_results = False)
        beta_soft = np.zeros(len(lam_soft))
    elif opt == "SAGD":
        lam_soft = optSAGD(nablaG, n_classes, c = c, T = 1000)
        beta_soft = np.zeros(lam_soft.shape)
    #elif opt == "SAGD_bivar":
    #    lam_soft, beta_soft = optSAGD_bivar(nablaGlam, nablaGbeta, n_classes, epsilon = epsilon_fair, c = c, T = 1000)
    #    #print(lam_soft)
    #    #print(beta_soft)
    elif opt == "optim":
        lam_soft = optSCIPY(fun = lam_fairness_soft, n_classes = n_classes)
        beta_soft = np.zeros(len(lam_soft))
    elif opt == "optim_bivar":
        lam_soft, beta_soft = optSCIPY_bivar(fun = bivar_fairness_soft, n_classes = n_classes)
    elif opt == "optim_bivar_bis":
        lam_soft, beta_soft = optSCIPY_bivar_bis(fun = bivar_fairness_soft, n_classes = n_classes)
    # inference with and without fairness

    
    index_0 = np.where(X_test[:,-1] == 0)[0]
    index_1 = np.where(X_test[:,-1] == 1)[0]
    


    y_probs = ovr.predict_proba(X_test)
    ### test
    if test:
        
        y_probs=unconcentrate_prob(y_probs)
     
    
    y_preds = ovr.predict(X_test)
    #Tricks from Denis
    eps = np.random.uniform(0, sigma, (y_probs.shape))
    y_prob_fair_soft = np.zeros(y_probs.shape)
    y_prob_fair_soft[index_0] = ps[0]*(y_probs[index_0]+eps[index_0]) - (-1)*(lam_soft-beta_soft)
    y_prob_fair_soft[index_1] = ps[1]*(y_probs[index_1]+eps[index_1]) - 1*(lam_soft-beta_soft)
    y_pred_fair_soft = np.argmax(y_prob_fair_soft, axis = 1)

    if print_lambda:
        print("lamb:", lam_soft.round(5))
        print("beta:", beta_soft.round(5))

    # track the time
    time_soft = time.time() - start_time

    return y_pred_fair_soft, y_prob_fair_soft, index_0, index_1, y_preds, y_probs, time_soft


