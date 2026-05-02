import numpy as np

from fair_multiclass import *
from faircrowd.core.algorithms import TDAlgorithm, TDOutput
from faircrowd.core.fair_crowd_dataset import FairCrowdDataset
import pandas as pd

class Optimal_reg(TDAlgorithm):

    eps:float 
    def __init__(self,ovr,eps=0.01,sigma=10**(-(5)),c = 0.001):
        #confusion_matrix of shape (n_annotator,n_classes_,n_classes_,2=n_sensitive_attribute)
        self.n_classes_=ovr.n_classes_
        self.ovr=ovr
        self.eps=eps
        self.sigma=sigma
        self.c=c
        self.name="Reg"

    def run(self, df: FairCrowdDataset,eps=0.01, print_lambda=False) -> TDOutput:
        self.eps=eps
        
        
        X_S=np.hstack((df.answers.values,df["s"].values[:,None]))
        X_test=X_S
        X_pool=X_S
        
        #plug ici fair_soft_max
        ovr=self.ovr
        
        # X_test (N)
        # The model ovr is fit on X_pool
        # the fairness correction is applied on X_test
        # y_pred_fair_soft, y_prob_fair_soft are the corrected predicitons, while y_preds, y_probs is not corrected
        #y_pred_fair_soft, y_prob_fair_soft, index_0, index_1, y_preds, y_probs, time_soft= fair_hard_max(X_test, X_pool, ovr, c =c, opt = "SAGD", sigma = sigma, epsilon_fair =epsilon_fair, print_lambda=print_lambda)
        y_pred_fair_soft, y_prob_fair_soft, index_0, index_1, y_preds, y_probs, time_soft= fair_soft_max_test(X_test, X_pool, ovr, c =self.c, opt = "optim_bivar_bis", sigma = self.sigma, epsilon_fair =self.eps, print_lambda=print_lambda)
        

        return TDOutput(probabilities=y_pred_fair_soft)




class Majority_vote():

   
    def __init__(self,n_classes_,n_annotators):

        self.n_classes_=n_classes_
        self.n_annotators=n_annotators
       
        self.name="majority_vote"
    

    def predict(self,Y_noise_S):
        prob=self.predict_proba(Y_noise_S)
       
        return np.argmax(prob,axis=1)

    def predict_proba(self,Y_noise_S):
        finite_elements = np.isfinite(Y_noise_S[:,:-1])
        prob_es=np.array([np.mean(Y_noise_S[:,:-1][i,finite_elements[i]]) for i in range(len(Y_noise_S))])
        
        prob=np.hstack((1-prob_es[:,None],prob_es[:,None]))
        return prob



class Geometric():

 

    def __init__(self,n_classes_,n_annotators,confusion_matrix=None,prior=None,fit_method="basic"):

        self.n_classes_=n_classes_
        self.n_annotators=n_annotators
        self.confusion_matrix=confusion_matrix # P(hat_Y^R=k'|Y=k,S=s) =confusion_matrix [R,k',k,s]
        self.prior=prior # prior s\to P(Y|S=s)
        self.name="geo_count"
        self.fit_method=fit_method
    

    def fit(self,answers,s,y,prior_activate=True):

        #Y_noise_S=np.hstack((df.answers.values,df["s"].values[:,None]))
        Y_noise_S=np.hstack((answers,s[:,None]))

        Y_true=y

        

        # By counting
        N=len(Y_true)
        prior_val=np.zeros((2,self.n_classes_))+10**(-2)
        confusion_matrix=np.zeros((self.n_annotators,self.n_classes_,self.n_classes_,2))+10**(-1)
        # prior matrix per annotator (true 2/3, 1/3 error)
        if prior_activate:
            Mat=np.zeros((self.n_classes_,self.n_classes_))
            for i in range(self.n_classes_):
                Mat[i,i]=2/3
                if i>=1:
                    Mat[i,:i]=1/(3*(self.n_classes_-1))
                if i<self.n_classes_-1:
                    Mat[i,i+1:]=1/(3*(self.n_classes_-1))
            for r in range(self.n_annotators):
                for at in range(2):
                    confusion_matrix[r,:,:,at]=Mat

        if self.fit_method=="basic":
            for i in range(N):
                prior_val[int(Y_noise_S[i,-1]),int(Y_true[i])]+=1
                for r in range(self.n_annotators):
                    if not(np.isnan(Y_noise_S[i,r])):
                        confusion_matrix[r,int(Y_noise_S[i,r]),int(Y_true[i]),int(Y_noise_S[i,-1])]+=1
        elif self.fit_method=="restricted":
            for i in range(N):
                prior_val[int(Y_noise_S[i,-1]),int(Y_true[i])]+=1
                for r in range(self.n_annotators):
                    if not(np.isnan(Y_noise_S[i,r])):
                        confusion_matrix[r,int(Y_noise_S[i,r]),int(Y_true[i]),0]+=1
                        confusion_matrix[r,int(Y_noise_S[i,r]),int(Y_true[i]),1]+=1
        elif self.fit_method=="type_glad":#work only in binary cas as it is
            # to finish
            for i in range(N):
                prior_val[int(Y_noise_S[i,-1]),int(Y_true[i])]+=1
                for r in range(self.n_annotators):
                    if not(np.isnan(Y_noise_S[i,r])):
                        if int(Y_true[i])==int(Y_noise_S[i,r]):
                            for k in range(self.n_classes_):
                                confusion_matrix[r,k,k,int(Y_noise_S[i,-1])]+=1
                        else: 
                            for k in range(self.n_classes_):
                                for j in range(self.n_classes_):
                                    if j!=k:
                                        confusion_matrix[r,j,k,int(Y_noise_S[i,-1])]+=1

                        #confusion_matrix[r,int(Y_noise_S[i,r]),int(Y_true[i]),1]+=1



        prior_val_normalized=prior_val/prior_val.sum(axis=1)[:,None]
        self.prior=lambda s:prior_val_normalized[s.astype(int)]

        self.confusion_matrix=confusion_matrix/confusion_matrix.sum(axis=1)[:,None,:,:]

        print("everything positive", (self.confusion_matrix>0).all())

        

    def predict_log_proba(self,Y_noise_S): #see eq 6 in the review of ibrahim
        N=len(Y_noise_S)
        log_prob=np.zeros((N,self.n_classes_))+np.log(self.prior(Y_noise_S[:,-1]))# we add directly prior values
        #prior_val=self.prior(Y_noise_S[:,-1])
        #print(Y_noise_S.shape)
        for i in range(N):
            indic_annotators=np.zeros((self.n_classes_,self.n_annotators))
            for k_prime in range(self.n_classes_):
                #print(len(Y_noise_S[i,:-1]))
                #print(self.n_annotators)
                indic_annotators[k_prime,Y_noise_S[i,:-1]==k_prime]=1
            for k in range(self.n_classes_):
                for k_prime in range(self.n_classes_):
                    log_prob[i,k]+= np.dot(indic_annotators[k_prime],np.log(self.confusion_matrix[:,k_prime,k,int(Y_noise_S[i,-1])]))

            
        return log_prob#-log_prob.sum(axis=1)[:,None]

    def predict(self,Y_noise_S):
        prob=self.predict_proba(Y_noise_S)
       
        return np.argmax(prob,axis=1)

    def predict_proba(self,Y_noise_S):
        log_prob=self.predict_log_proba(Y_noise_S)
        prob=np.exp(log_prob/10)
        prob=prob/prob.sum(axis=1)[:,None]
        return prob




