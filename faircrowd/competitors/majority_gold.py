from crowdkit.aggregation import GLAD,DawidSkene,OneCoinDawidSkene,GoldMajorityVote

from faircrowd.utils.exploratory import convert_data_to_glad

import numpy as np 
class Majority_gold_op():
    def __init__(self,n_classes_):
        self.n_classes_=n_classes_
        self.model=GoldMajorityVote()

    def fit(self,X,s,y):
        # to complete
        # We add a prior about the unsen worker accuracy about 2/3
        dg=convert_data_to_glad(X)
        self.model.fit(data=dg,true_labels=y)
        
    def predict_proba(self,X):
        
        dg=convert_data_to_glad(X[:,:-1])
        
        pred_prob=self.model.predict_proba(dg).values #should be of the same size of X
        # print(min(pred_prob[:,1]))
        # c=-np.log(min(pred_prob[:,1]))
        # print("c",c)
        #pred_prob=pred_prob+0.5*np.random.random(size=pred_prob.shape)
        #pred_prob=pred_prob/pred_prob.sum(axis=1)[:,None]
        return pred_prob

    def predict_log_proba(self,X):
        return np.log(self.predict_proba(X))

    def predict(self,X):
        dg=convert_data_to_glad(X[:,:-1])
        return self.model.predict(dg).values.astype(int)




import numpy as np
from scipy.special import logsumexp
class DawidSkeneSharedConfusion():
    """
    Dawid–Skene model with a *single shared confusion matrix* for all annotators.

    Parameters
    ----------
    n_classes : int
        Number of true classes.
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance on the log-likelihood.
    smoothing : float
        Dirichlet-like smoothing to avoid zeros.
    """

    def __init__(self, n_classes, max_iter=100, tol=1e-6, smoothing=1e-3):
        self.n_classes_ = n_classes
        self.max_iter = max_iter
        self.tol = tol
        self.smoothing = smoothing

        # learned parameters
        self.pi_ = None            # class prior (K,)
        self.confusion_ = None     # shared confusion matrix (K, K)
        self.log_likelihood_ = []

    def _initialize(self, annotations):
        """Simple initialization using majority vote."""
        N = annotations.shape[0]
        K = self.n_classes_

        # majority vote (ignoring -1 = missing)
        counts = np.zeros((N, K))
        for k in range(K):
            counts[:, k] = np.sum(annotations == k, axis=1)

        post = counts + 1e-6
        post /= post.sum(axis=1, keepdims=True)
        print(post)
        self.pi_ = post.mean(axis=0)

        # initialize confusion close to identity
        self.confusion_ = np.eye(K) * 0.9 + 0.1 / K
        self.confusion_ /= self.confusion_.sum(axis=1, keepdims=True)

        return post

    def fit(self, annotations):
        """
        Fit the model by EM.

        Parameters
        ----------
        annotations : array, shape (N, M)
            annotations[i, j] in {0, ..., K-1} or -1 if missing
        """
        annotations = np.asarray(annotations)
        N, M = annotations.shape
        K = self.n_classes_

        post = self._initialize(annotations)
        prev_ll = -np.inf

        for it in range(self.max_iter):
            # ---------- E-step ----------
            log_post = np.log(self.pi_ + 1e-12)[None, :]

            for i in range(N):
                for j in range(M):
                    y = annotations[i, j]
                    if y >= 0:
                        log_post[i] += np.log(self.confusion_[:, y] + 1e-12)

            # normalize
            log_post -= log_post.max(axis=1, keepdims=True)
            post = np.exp(log_post)
            post /= post.sum(axis=1, keepdims=True)

            # ---------- M-step ----------
            # class prior
            self.pi_ = post.mean(axis=0)

            # shared confusion matrix
            conf = np.zeros((K, K))
            for i in range(N):
                for j in range(M):
                    y = annotations[i, j]
                    if y >= 0:
                        conf[:, y] += post[i]

            conf += self.smoothing
            conf /= conf.sum(axis=1, keepdims=True)
            self.confusion_ = conf

            # ---------- log-likelihood ----------
            ll = 0.0
            for i in range(N):
                tmp = self.pi_.copy()
                for j in range(M):
                    y = annotations[i, j]
                    if y >= 0:
                        tmp *= self.confusion_[:, y]
                ll += np.log(tmp.sum() + 1e-12)

            self.log_likelihood_.append(ll)

            if np.abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self

    def predict_log_proba(self, annotations):
        """Posterior over true labels."""
        annotations = np.asarray(annotations)
        N, M = annotations.shape
        K = self.n_classes_

        log_post = np.log(self.pi_ + 1e-12)[None, :]
        for i in range(N):
            for j in range(M):
                y = annotations[i, j]
                if y >= 0:
                    log_post[i] += np.log(self.confusion_[:, y] + 1e-12)

        log_post -= log_post.max(axis=1, keepdims=True)
        #post = np.exp(log_post)
        log_post=log_post- logsumexp(log_post,axis=1, keepdims=True)
        return log_post

    def predict(self, annotations):
        """MAP estimate of true labels."""
        return self.predict_proba(annotations).argmax(axis=1)
    def predict_proba(self, annotations):
        """MAP estimate of true labels."""
        return np.exp(self.predict_log_proba(annotations))

