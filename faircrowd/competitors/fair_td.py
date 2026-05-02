from random import uniform
from copy import deepcopy
from collections import OrderedDict
import numpy as np
import pandas as pd
from faircrowd.core.algorithms import TDAlgorithm, TDOutput
from faircrowd.core.fair_crowd_dataset import FairCrowdDataset


class FairTD(TDAlgorithm):
    """
    Description & reference
    """

    theta: float

    def __init__(
        self,
        eps: float = 0.01,
    ):
        self.theta = eps
        self.name="FairTD"

    def run(self, df: FairCrowdDataset, eps=0.01) -> TDOutput:
        self.theta = eps
        
        truth, bias, quality, sigma, repeat_times = optimized_fairtd_loop(
            df, theta=eps, max_iterations=100000, convergence_threshold=1e-6
        )
        
        # CORRECTION: truth[0] = groupe sensible, truth[1] = autres
        # Créer les prédictions pour chaque groupe
        predictions_sensitive = pd.DataFrame(
            truth[0],  # Vérités pour le groupe sensible (s=1)
            index=df.answers[df["s"] == 1].index,
            columns=["probabilities"]
        )
        
        predictions_others = pd.DataFrame(
            truth[1],  # Vérités pour les autres (s=0)
            index=df.answers[df["s"] == 0].index,
            columns=["probabilities"]
        )
        
        # Combiner et réordonner selon l'index original
        predictions = pd.concat([predictions_sensitive, predictions_others]).reindex(
            df.answers.index
        )
        
        print(f"repeat_times: {repeat_times}")
        print(f"Demographic parity gap final: {abs(np.mean(truth[0]) - np.mean(truth[1])):.6f}")
        return TDOutput(probabilities=predictions["probabilities"])


class FairTD_Post(TDAlgorithm):
    """
    Description & reference
    """

    theta: float

    def __init__(
        self,model,
        eps: float = 0.01,
    ):
        self.theta = eps
        self.name="FairTD_Post"
        self.ovr=model

    def run(self, df: FairCrowdDataset,eps=0.01) -> TDOutput:
        self.theta=eps
        S=df["s"].values
        
        X_S=np.hstack((df.answers.values,df["s"].values[:,None]))
        pred=self.ovr.predict(X_S)
        
        N_S0=len(S[S==0])
        N_S1=len(S[S==1])
        pred_S0=pred[S==0]
        pred_S1=pred[S==1]

        truth=[pred_S0,pred_S1]
        new_truth=massaging(truth=truth,theta=int(eps/0.01))
        #print(new_truth)
        pred[S==0]=np.array(new_truth[0])
        pred[S==1]=np.array(new_truth[1])

        

        
        return TDOutput(probabilities=pred)
from random import *
from copy import deepcopy
import math
def massaging(theta,truth,r=1,cat=2):

    th=theta

    def point(x,y):
        list = []
        for i in range(th):
            list.append((x[i],y[i]))
        return list

    def disparityd(t):
        p0=0
        p1=0
        for i in range(len(t[0])):
            if t[0][i] < 0.5:
                p0+=1
        for i in range(len(t[1])):
            if t[1][i] < 0.5:
                p1+=1
        return p0, p1, (p1) / (len(t[1])) - (p0) / (len(t[0]))

    def disparityi(t):
        p0=0
        p1=0
        for i in range(len(t[0])):
            if t[0][i] == 0:
                p0+=1
        for i in range(len(t[1])):
            if t[1][i] == 0:
                p1+=1
        return p0, p1, (p1) / (len(t[1])) - (p0) / (len(t[0]))

    def disparitys(t,c):
        p0 = 0
        p1 = 0
        for i in range(cat):
            if i == c:
                for j in range(len(t[i])):
                    if t[i][j] < 0.5:
                        p0 += 1
            else:
                for j in range(len(t[i])):
                    if t[i][j] < 0.5:
                        p1 += 1

        return p0, p1, (p0) / (len(t[0])) - (p1) / (len(t[1]))

    def dis(m,p0,p1,len0,len1):
        return (p1 - m) / len1 - (p0 + m) / len0

    def mdis(dis,p0,p1,len0,len1):
        return (len0*p1 - len1*p0 - len0*len1*dis) / (len0 + len1)

    dlist = []
    for c in range(cat):
        p0,p1,d = disparitys(truth,c)
        dlist.append(d)
    p = dlist.index(min(dlist))

    pr = []
    de = []
    for i in range(cat):
        if i == p:
            for j in range(len(truth[i])):
                if truth[i][j] >= 0.5:
                    pr.append(j)
        else:
            for j in range(len(truth[i])):
                if truth[i][j] < 0.5:
                    de.append(j)

    disD = abs(dlist[p])
    len0 = len(truth[p])
    len1 = 0
    for i in range(cat):
        if i != p:
            len1 += len(truth[i])
    M = (disD * len0 * len1) / (len0 + len1)
    M = math.ceil(M)
    p0,p1,d = disparitys(truth,p)

    mlist = []
    thetas = [0.01*i for i in range(th,th+1)]
    for t in thetas:
        mlist.append(max(0,math.ceil(mdis(t,p0,p1,len0,len1))))

    newtruth = [[], []]
    for i in range(len(truth)):
        for j in truth[i]:
            if j >= 0.5:
                newtruth[i].append(1)
            else:
                newtruth[i].append(0)

    def mass(m):
        lenp=len(pr)
        lend=len(de)
        plist=[i for i in range(lenp)]
        shuffle(plist)
        dlist=[i for i in range(lend)]
        shuffle(dlist)
        newt=deepcopy(newtruth)
        count=0
        for i in plist:
            if newt[p][i] == 1:
                if count == m:
                    break
                else:
                    newt[p][i]=0
                    count+=1
        count=0
        for i in dlist:
            if newt[abs(1 - p)][i] == 0:
                if count == m:
                    break
                else:
                    newt[abs(1 - p)][i]=1
                    count+=1

        return newt

    displist = [[] for i in range(th)]
    acclist=[[] for i in range(th)]

    newt = mass(mlist[0])

    return newt


class FairTD_ref(TDAlgorithm):
    """
    Description & reference
    """

    theta: float

    def __init__(
        self,
        eps: float = 0.01,
    ):
        self.theta = eps
        self.name="FairTD"

    def run(self, df: FairCrowdDataset,eps=0.01) -> TDOutput:
        raw_answers = df.answers
        self.theta=eps

        # Convert to the format used by FairTD
        raw_answers_none = (
            raw_answers.dropna(how="all", axis=0)
            .dropna(how="all", axis=1)
            .astype(object)
            .where(pd.notnull(df.answers), None)
        )
        sensitive = raw_answers_none[df["s"] == 1]
        others = raw_answers_none[df["s"] == 0]
        answer = []
        for worker in raw_answers.columns:
            if (
                len(sensitive[worker].dropna(how="all")) > 0
                and len(others[worker].dropna(how="all")) > 0
            ):
                answer.append([sensitive[worker].values, others[worker].values])
        del raw_answers_none
        del sensitive
        del others

        # Original FairTD code
        n = len(answer)
        m = len(answer[0])
        r = [len(answer[0][j]) for j in range(0, m)]

        truth = [[uniform(0.1, 0.9) for k in range(0, r[j])] for j in range(0, m)]
        bias = [[uniform(-0.1, 0.1) for j in range(0, m)] for i in range(0, n)]
        sigma = [0.1 for i in range(0, n)]
        quality = [1.0 / n for i in range(0, n)]

        last_truth = np.array([])

        disparity = np.array([0.0 for i in range(0, m)])

        maxdiff = -1

        repeat_times = -1

        while True and repeat_times<10000:
            repeat_times += 1
            if repeat_times%100==0:
                print(repeat_times)

            if self.theta != None:
                disparity_self = [[0.0, 0.0] for i in range(0, m)]
                for j in range(0, m):
                    for k in range(0, len(truth[j])):
                        if truth[j][k] == None:
                            continue
                        disparity_self[j][1] += 1
                        if truth[j][k] > 0.5:
                            disparity_self[j][0] += 1
                for j in range(0, m):
                    disparity_self[j] = disparity_self[j][0] / disparity_self[j][1]

                disparity_other = [[0.0, 0.0] for i in range(0, m)]
                for j_ in range(0, m):
                    for j in range(0, m):
                        if j == j_:
                            continue
                        for k in range(0, len(truth[j])):
                            if truth[j][k] == None:
                                continue
                            disparity_other[j_][1] += 1
                            if truth[j][k] > 0.5:
                                disparity_other[j_][0] += 1
                for j in range(0, m):
                    disparity_other[j] = disparity_other[j][0] / disparity_other[j][1]

                disparity = []
                for j in range(0, m):
                    disparity.append(disparity_self[j] - disparity_other[j])
                disparity = np.array(disparity)

            if len(last_truth) != 0:
                maxdiff = -1
                for j in range(0, m):
                    for k in range(0, r[j]):
                        if last_truth[j][k] != None and truth[j][k] != None:
                            maxdiff = max(maxdiff, abs(last_truth[j][k] - truth[j][k]))
                if self.theta == None or (
                    self.theta != None and max(abs(disparity)) < self.theta
                ):
                    break
                else:
                    pass
            if self.theta != None:
                rejection_list = OrderedDict()
                sum_score = 0.0
                for i in range(0, n):
                    for j in range(0, m):
                        z = disparity[j] * bias[i][j]
                        if z < 0:
                            rejection_list[(i, j)] = -z
                            sum_score += -z
                p = uniform(0, 1)
                acc = 0.0
                selected = None
                for item in rejection_list:
                    rejection_list[item] /= sum_score
                    acc += rejection_list[item]
                    if acc >= p:
                        selected = item
                        break

            last_truth = deepcopy(truth)
            for j in range(0, m):
                for k in range(0, r[j]):
                    truth[j][k] = 0.0
                    acc_q = 0.0
                    for i in range(0, n):
                        if answer[i][j][k] != None:
                            if self.theta != None:
                                # if bias[i][j]*disparity[j]>0 or uniform(0,1)>abs(disparity[j]):# or max(abs(disparity))<theta:
                                if (i, j) != selected or max(
                                    abs(disparity)
                                ) < self.theta:
                                    truth[j][k] += (
                                        answer[i][j][k] - bias[i][j]
                                    ) * quality[i]
                                else:
                                    truth[j][k] += answer[i][j][k] * quality[i]
                            else:
                                truth[j][k] += (answer[i][j][k] - bias[i][j]) * quality[
                                    i
                                ]
                            acc_q += quality[i]
                    if acc_q == 0:
                        truth[j][k] = None
                    else:
                        truth[j][k] /= acc_q

            for i in range(0, n):
                for j in range(0, m):
                    bias[i][j] = 0.0
                    acc_n = 0
                    for k in range(0, r[j]):
                        if answer[i][j][k] != None:
                            bias[i][j] += answer[i][j][k] - truth[j][k]
                            acc_n += 1
                    if acc_n == 0:
                        bias[i][j] = None
                    else:
                        bias[i][j] /= acc_n

            for i in range(0, n):
                sigma[i] = 0.1
                acc_n = 0
                for j in range(0, m):
                    for k in range(0, r[j]):
                        if answer[i][j][k] != None:
                            sigma[i] += (
                                answer[i][j][k] - truth[j][k] - bias[i][j]
                            ) ** 2
                            acc_n += 1
                sigma[i] = np.sqrt(sigma[i] / acc_n)

            sum_q = 0.0
            for i in range(0, len(quality)):
                quality[i] = 1.0 / sigma[i] ** 2
                sum_q += quality[i]
            for i in range(0, len(quality)):
                quality[i] = quality[i] / sum_q + 1e-10

        # Convert back to the format used by TDAlgorithm
        predictions_sensitive = pd.DataFrame(
            truth[0], index=df.answers[df["s"] == 1].index, columns=["probabilities"]
        )
        predictions_others = pd.DataFrame(
            truth[1], index=df.answers[df["s"] == 0].index, columns=["probabilities"]
        )
        predictions = pd.concat([predictions_sensitive, predictions_others]).reindex(
            df.answers.index
        )
        return TDOutput(probabilities=predictions["probabilities"])


import numpy as np
import pandas as pd
from random import uniform
def convert_answer_to_tensor(answer, n, m, r):
    """
    Convertit la structure answer[i][j][k] en tenseurs NumPy 3D.
    
    Returns:
        answer_array: np.array de shape (n, m, r_max) avec les valeurs
        valid_mask: np.array booléen de shape (n, m, r_max) indiquant les non-None
    """
    r_max = max(r)
    answer_array = np.full((n, m, r_max), np.nan, dtype=np.float64)
    valid_mask = np.zeros((n, m, r_max), dtype=bool)
    
    for i in range(n):
        for j in range(m):
            for k in range(r[j]):
                if answer[i][j][k] is not None:
                    answer_array[i, j, k] = answer[i][j][k]
                    valid_mask[i, j, k] = True
    
    return answer_array, valid_mask


def compute_disparity_vectorized(truth_array, valid_mask, m):
    """
    Calcul vectorisé du demographic parity gap.
    
    Demographic Parity Gap = P(Y=1 | S=s) - P(Y=1 | S≠s)
    
    Pour chaque groupe s:
    - P(Y=1 | S=s) = proportion de prédictions positives dans le groupe s
    - P(Y=1 | S≠s) = proportion de prédictions positives dans tous les autres groupes
    
    Args:
        truth_array: array de shape (m, r_max) avec les vérités estimées
        valid_mask: array booléen de shape (m, r_max) indiquant les valeurs valides
        m: nombre de groupes démographiques
    
    Returns:
        disparity: array de shape (m,) avec le demographic parity gap pour chaque groupe
    """
    # Masque des prédictions positives (vérités > 0.5) valides uniquement
    truth_high = (truth_array > 0.5) & valid_mask  # shape: (m, r_max)
    
    # Pour chaque groupe: nombre de positifs et total
    count_high_self = truth_high.sum(axis=1)  # (m,) - nombre de Y=1 dans groupe s
    count_total_self = valid_mask.sum(axis=1)  # (m,) - taille du groupe s
    
    # P(Y=1 | S=s) pour chaque groupe
    positive_rate_self = np.divide(
        count_high_self, 
        count_total_self,
        out=np.zeros_like(count_high_self, dtype=float),
        where=count_total_self > 0
    )
    
    # Totaux globaux (tous groupes confondus)
    total_high = count_high_self.sum()  # total de Y=1 partout
    total_count = count_total_self.sum()  # taille totale
    
    # Pour chaque groupe s: statistiques de TOUS LES AUTRES groupes (S≠s)
    count_high_other = total_high - count_high_self  # (m,) - nombre de Y=1 hors groupe s
    count_total_other = total_count - count_total_self  # (m,) - taille hors groupe s
    
    # P(Y=1 | S≠s) pour chaque groupe
    positive_rate_other = np.divide(
        count_high_other,
        count_total_other,
        out=np.zeros_like(count_high_other, dtype=float),
        where=count_total_other > 0
    )
    
    # Demographic Parity Gap = P(Y=1 | S=s) - P(Y=1 | S≠s)
    disparity = positive_rate_self - positive_rate_other
    
    return disparity


def optimized_fairtd_loop(df, theta=None, max_iterations=1000, convergence_threshold=1e-6):
    """
    Version optimisée de l'algorithme FairTD avec correction du rejection sampling.
    
    Args:
        df: FairCrowdDataset avec colonnes 'answers' et 's'
        theta: seuil de demographic parity (None = pas de contrainte de fairness)
        max_iterations: nombre max d'itérations
        convergence_threshold: seuil de convergence pour truth
    
    Returns:
        truth_result: liste de listes avec vérités par groupe et item
        bias: array (n, m) des biais par worker et groupe
        quality: array (n,) de qualité par worker
        sigma: array (n,) d'écart-type par worker
        repeat_times: nombre d'itérations effectuées
    """
    
    # ===== PRÉPARATION DES DONNÉES =====
    raw_answers = df.answers
    
    raw_answers_none = (
        raw_answers.dropna(how="all", axis=0)
        .dropna(how="all", axis=1)
        .astype(object)
        .where(pd.notnull(df.answers), None)
    )
    sensitive = raw_answers_none[df["s"] == 1]
    others = raw_answers_none[df["s"] == 0]
    answer = []
    
    for worker in raw_answers.columns:
        if (
            len(sensitive[worker].dropna(how="all")) > 0
            and len(others[worker].dropna(how="all")) > 0
        ):
            answer.append([sensitive[worker].values, others[worker].values])
    
    del raw_answers_none, sensitive, others
    
    # ===== PARAMÈTRES =====
    n = len(answer)  # nombre de workers
    m = len(answer[0])  # nombre de groupes (2: sensitive, others)
    r = [len(answer[0][j]) for j in range(m)]
    r_max = max(r)
    
    print(f"Paramètres: n={n} workers, m={m} groupes, r={r}")
    
    # ===== CONVERSION EN TENSEURS NUMPY =====
    answer_array, valid_mask = convert_answer_to_tensor(answer, n, m, r)
    
    # ===== INITIALISATION =====
    truth = np.random.uniform(0.1, 0.9, (m, r_max))
    # Masquer les positions invalides (où il n'y a pas de données)
    for j in range(m):
        for k in range(r[j], r_max):
            truth[j, k] = np.nan
    
    bias = np.random.uniform(-0.1, 0.1, (n, m))
    sigma = np.full(n, 1e-9, dtype=np.float64)
    quality = np.full(n, 1.0 / n, dtype=np.float64)
    
    last_truth = None
    disparity = np.zeros(m)
    
    # ===== BOUCLE PRINCIPALE =====
    repeat_times = 0
    
    while repeat_times < max_iterations:
        if repeat_times % 100 == 0:
            disp_str = f", disparity={disparity}" if theta is not None else ""
            print(f"Itération {repeat_times}{disp_str}")
        repeat_times += 1
        
        # ===== CALCUL DE DISPARITY =====
        if theta is not None:
            disparity = compute_disparity_vectorized(truth, valid_mask[0], m)
        
        # ===== TEST DE CONVERGENCE =====
        if last_truth is not None:
            # Calculer la différence maximale
            valid_comparison = ~np.isnan(last_truth) & ~np.isnan(truth)
            if valid_comparison.any():
                maxdiff = np.abs(truth[valid_comparison] - last_truth[valid_comparison]).max()
            else:
                maxdiff = 0
            
            # Vérifier convergence
            should_stop = False
            if theta is None:
                # Sans contrainte de fairness: converger quand truth stable
                if maxdiff < convergence_threshold:
                    should_stop = True
            else:
                # Avec contrainte: converger quand truth stable ET disparity < theta
                if maxdiff < convergence_threshold and np.abs(disparity).max() < theta:
                    should_stop = True
            
            if should_stop:
                print(f"Convergence atteinte à l'itération {repeat_times}")
                print(f"  maxdiff={maxdiff:.6f}, max_disparity={np.abs(disparity).max():.6f}")
                break
        
        # ===== REJECTION SAMPLING (si theta activé) =====
        selected = None
        if theta is not None and np.abs(disparity).max() >= theta:
            # Calculer les scores de rejet: on rejette (i,j) si bias[i][j] et disparity[j] 
            # ont le MÊME signe (ils se renforcent mutuellement)
            z = disparity[np.newaxis, :] * bias  # Broadcasting: (n, m)
            
            # On veut rejeter les paires (i,j) où z > 0 (même signe)
            # Le score de rejet est proportionnel à z
            rejection_scores = np.where(z < 0, z, 0)
            sum_score = rejection_scores.sum()
            
            if sum_score > 0:
                # Sélection aléatoire pondérée
                flat_scores = rejection_scores.flatten()
                flat_scores /= sum_score
                cumsum = np.cumsum(flat_scores)
                p = uniform(0, 1)
                selected_idx = np.searchsorted(cumsum, p)
                selected = (selected_idx // m, selected_idx % m)
                
                if repeat_times % 100 == 0:
                    print(f"  Rejection: worker {selected[0]}, groupe {selected[1]}")
        
        # ===== UPDATE TRUTH =====
        last_truth = truth.copy()
        
        # Créer un tenseur de biais étendu: (n, m, 1)
        bias_expanded = bias[:, :, np.newaxis]
        quality_expanded = quality[:, np.newaxis, np.newaxis]
        
        # Calcul vectorisé
        if selected is None:
            # Cas normal: soustraire le biais partout
            numerator = np.where(
                valid_mask,
                (answer_array - bias_expanded) * quality_expanded,
                0
            )
        else:
            # Cas avec rejection: ne pas soustraire biais pour (i,j) sélectionné
            # On garde answer[i][j][k] tel quel (sans correction de biais)
            bias_adjusted = bias_expanded.copy()
            bias_adjusted[selected[0], selected[1], :] = 0
            
            numerator = np.where(
                valid_mask,
                (answer_array - bias_adjusted) * quality_expanded,
                0
            )
        
        # Somme sur les workers (axe 0)
        sum_weighted = numerator.sum(axis=0)  # (m, r_max)
        
        # Somme des qualités pour chaque position
        quality_sum = np.where(
            valid_mask,
            quality_expanded,
            0
        ).sum(axis=0)  # (m, r_max)
        
        # Division finale
        truth = np.divide(
            sum_weighted,
            quality_sum,
            out=np.full_like(sum_weighted, np.nan),
            where=quality_sum > 0
        )
        
        # ===== UPDATE BIAS =====
        # bias[i][j] = mean_k(answer[i][j][k] - truth[j][k])
        truth_expanded = truth[np.newaxis, :, :]  # (1, m, r_max)
        diff = np.where(valid_mask, answer_array - truth_expanded, 0)
        
        counts = valid_mask.sum(axis=2)
        bias = np.divide(
            diff.sum(axis=2),
            counts,
            out=np.zeros_like(bias),
            where=counts > 0
        )
        
        # ===== UPDATE SIGMA =====
        residuals = np.where(
            valid_mask,
            (answer_array - truth_expanded - bias_expanded) ** 2,
            0
        )
        
        counts_total = valid_mask.sum(axis=(1, 2))
        sigma = np.sqrt(
            (residuals.sum(axis=(1, 2)) + 1e-9) / np.maximum(counts_total, 1)
        )
        
        # ===== UPDATE QUALITY =====
        quality = 1.0 / (sigma ** 2)
        sum_q = quality.sum()
        quality = quality / sum_q + 1e-10
    
    # ===== CONVERSION DU RÉSULTAT =====
    # Reconvertir truth en liste de listes (format original)
    truth_result = []
    for j in range(m):
        truth_j = []
        for k in range(r[j]):
            val = truth[j, k]
            truth_j.append(None if np.isnan(val) else float(val))
        truth_result.append(truth_j)
    
    print(f"\nRésultats finaux:")
    print(f"  Itérations: {repeat_times}")
    print(f"  Disparity finale: {disparity}")
    print(f"  Max disparity: {np.abs(disparity).max():.6f}")
    if theta is not None:
        print(f"  Contrainte theta: {theta}")
        print(f"  Satisfaite: {np.abs(disparity).max() < theta}")
    
    return truth_result, bias, quality, sigma, repeat_times