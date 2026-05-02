import numpy as np


def generate_synhtetic_binary(N=20000,R=5,bias_toward_1=0.2,prop_s_1=0.5,rho_0=0.2,rho_1=0.1,seed=0):#proba de se tromper
    
    np.random.seed(seed)
    Y=np.zeros(N)
    S=np.random.binomial(1, prop_s_1, N).flatten()
    #print(S)
    
    prop=np.array([1/2+bias_toward_1/2,1/2-bias_toward_1/2]) # P(Y=1|S=0), P(Y=1|S=1)
    Y_annotators=np.zeros((N,R))
    #print( np.array([prop[int(el)] for el in S]))
    Y = np.random.binomial(np.ones(N, dtype=int), np.array([prop[int(el)] for el in S]))

    confusion=np.zeros((R,2,2,2))

    
    for i in range(R):
        mat=np.zeros((2,2,2))
        rho_0_r=np.random.rand()*rho_0
        rho_1_r=np.random.rand()*rho_1
        for k in range(2):
            mat[1,k,0]=(1==k)*(1-rho_0_r)+(0==k)*rho_0_r
            mat[0,k,0]=(0==k)*(1-rho_0_r)+(1==k)*rho_0_r
            mat[1,k,1]=(1==k)*(1-rho_1_r)+(0==k)*rho_1_r
            mat[0,k,1]=(0==k)*(1-rho_1_r)+(1==k)*rho_1_r

        confusion[i]=mat

    for i in range(R):
        Y_annotators[:,i]=np.random.binomial(np.ones(N, dtype=int), confusion[i,1,Y,S])

    return Y_annotators,S,Y,confusion,prop


def generate_synhtetic_binary_restricted(rho_0,rho_1,N=20000,R=8,R_anot_max=5,bias_toward_1=0.2,prop_s_1=0.5,seed=0):#proba de se tromper
    np.random.seed(seed)
    Y=np.zeros(N)
    S=np.random.binomial(1, prop_s_1, N).flatten()
    #print(S)
    
    prop=np.array([1/2+bias_toward_1/2,1/2-bias_toward_1/2]) # P(Y=1|S=0), P(Y=1|S=1)
    Y_annotators=np.zeros((N,R))
    #print( np.array([prop[int(el)] for el in S]))
    Y = np.random.binomial(np.ones(N, dtype=int), np.array([prop[int(el)] for el in S]))

    confusion=np.zeros((R,2,2,2))

    
    for i in range(R):
        mat=np.zeros((2,2,2))
        rho_0_r=np.random.rand()*(rho_0[1]-rho_0[0])+rho_0[0]
        rho_1_r=np.random.rand()*(rho_1[1]-rho_1[0])+rho_1[0]
        for k in range(2):
            mat[1,k,0]=(1==k)*(1-rho_0_r)+(0==k)*rho_0_r
            mat[0,k,0]=(0==k)*(1-rho_0_r)+(1==k)*rho_0_r
            mat[1,k,1]=(1==k)*(1-rho_1_r)+(0==k)*rho_1_r
            mat[0,k,1]=(0==k)*(1-rho_1_r)+(1==k)*rho_1_r

        confusion[i]=mat

    for i in range(R):
        Y_annotators[:,i]=np.random.binomial(np.ones(N, dtype=int), confusion[i,1,Y,S])

    #We remove values randomly by inserting nan
    for j in range(len(Y_annotators)):
        index_i=np.random.choice(R, R-R_anot_max,replace=False)
        Y_annotators[j,index_i]=np.nan

        

    return Y_annotators,S,Y,confusion,prop

#Y_annotators,S,Y,confusion,prop= generate_synhtetic_binary(N=20000,R=5,bias_toward_1=0.2,prop_s_1=0.5,rho_0=0.2,rho_1=0.1)
