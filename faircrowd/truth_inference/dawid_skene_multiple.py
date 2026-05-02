import numpy as np
import numpy.typing as npt
from typing import Dict, Any
from ..core.algorithms import TDAlgorithm, TDOutput
from ..core.fair_crowd_dataset import FairCrowdDataset


class DawidSkeneMultiple(TDAlgorithm):
    """
    Vairation of the Dawid-Skene algorithm that uses two (or more) confusion matrices, one for sensitive and one for non-sensitive groups.
    """

    iterations: int
    initial_quality: float
    predicted_probabilities: npt.NDArray[np.float64]
    positive_probability: npt.NDArray[np.float64]
    confusion_matrices: Dict[Any, npt.NDArray]
    groups: npt.NDArray

    def __init__(self, iterations: int = 20, initial_quality: float = 0.7) -> None:
        self.iterations = iterations
        self.initial_quality = initial_quality
        self.n_classes_=None

    def e_step(self, answers: npt.NDArray, sensitive: npt.NDArray) -> None:
        """
        Update the predicted probabilities for each sample.
        """
        finite_elements = np.isfinite(answers)

        for sample in range(answers.shape[0]):
            # Get the subset of workers that labelled that sample
            workers = finite_elements[sample]
            responses = answers[sample, workers].astype(np.int32)

            weight_neg = (1 - self.positive_probability) * self.confusion_matrices[
                sensitive[sample]
            ][workers, 0, responses].prod()
            weight_pos = (
                self.positive_probability
                * self.confusion_matrices[sensitive[sample]][
                    workers, 1, responses
                ].prod()
            )
            total_weight = weight_neg + weight_pos

            self.predicted_probabilities[sample] = (
                np.array([0.5, 0.5])
                if total_weight == 0
                else np.array([weight_neg, weight_pos]) / total_weight
            )

    def m_step(self, answers: npt.NDArray, sensitive: npt.NDArray) -> None:
        """
        Update the confusion matrices and the label probabilities.
        """
        # Update label probabilities
        self.positive_probability = self.predicted_probabilities.mean(axis=0)[1]

        # Update confusion matrices
        finite_elements = np.isfinite(answers)
        self.confusion_matrices = {
            group: np.tile([[0.0, 0.0], [0.0, 0.0]], (answers.shape[1], 1, 1))
            for group in self.groups
        }
        for worker in range(answers.shape[1]):
            for group in self.groups:
                samples = finite_elements[:, worker] & (sensitive == group)
                weights = self.predicted_probabilities[samples].sum(axis=0)
                responses = answers[samples, worker].astype(np.int32)
                for tlabel in [0, 1]:
                    if weights[tlabel] == 0:
                        if tlabel == 0:
                            self.confusion_matrices[group][worker, tlabel] = np.array(
                                [self.initial_quality, 1 - self.initial_quality]
                            )
                        else:
                            self.confusion_matrices[group][worker, tlabel] = np.array(
                                [1 - self.initial_quality, self.initial_quality]
                            )
                    else:
                        np.add.at(
                            self.confusion_matrices[group][worker, tlabel],
                            responses,
                            self.predicted_probabilities[samples, tlabel]
                            / weights[tlabel],
                        )

    def initialize_parameters(self, df: FairCrowdDataset) -> None:
        """
        Initialize the parameters of the algorithm.
        """
        self.n_classes_=max(df["y"].values)+1
        self.groups = np.unique(df["s"].values)
        self.predicted_probabilities = np.tile(
            [np.nan, np.nan], (df.answers.shape[0], 1)
        )
        self.positive_probability = 0.5
        self.confusion_matrices = {
            group: np.tile(
                [
                    [self.initial_quality, 1 - self.initial_quality],
                    [1 - self.initial_quality, self.initial_quality],
                ],
                (df.answers.shape[1], 1, 1),
            )
            for group in self.groups
        }
        maj_prob=np.nanmean(df.answers.values,axis=1)[:,None]
        self.predicted_probabilities=np.hstack((1-maj_prob,maj_prob))
        self.m_step(df.answers.values,df["s"].values)
    def run(self, df: FairCrowdDataset) -> TDOutput:
        """
        Run the Dawid-Skene algorithm.
        """
        self.initialize_parameters(df)

        iterations = self.iterations
        while iterations > 0:
            self.e_step(df.answers.values, df["s"].values)
            self.m_step(df.answers.values, df["s"].values)
            iterations -= 1

        # Normalize confusion matrices
        confusion_matrices = {
            group: self.confusion_matrices[group] / 2 for group in self.groups
        }

        return TDOutput(
            probabilities=self.predicted_probabilities[:, 1],
            confusion_matrices=confusion_matrices,positive_probabilities=self.positive_probability
        )
    # Faire les calculs en log?
    def predict_proba_old(self,X):
        answers=X[:,:-1]
        sensitive=X[:,-1]
        finite_elements = np.isfinite(answers)
        predicted_proba=np.ones((X.shape[0],self.n_classes_))/self.n_classes_
        for sample in range(answers.shape[0]):
            # Get the subset of workers that labelled that sample
            workers = finite_elements[sample]
            responses = answers[sample, workers].astype(np.int32)

            weight_neg = (1 - self.positive_probability) * self.confusion_matrices[
                sensitive[sample]
            ][workers, 0, responses].prod()
            weight_pos = (
                self.positive_probability
                * self.confusion_matrices[sensitive[sample]][
                    workers, 1, responses
                ].prod()
            )
            total_weight = weight_neg + weight_pos
            #print(total_weight)
            predicted_proba[sample] = (
                np.array([0.5, 0.5])
                if total_weight == 0
                else np.array([weight_neg, weight_pos]) / total_weight
            )
            
        
        return predicted_proba

    def predict_proba(self,X):
        answers=X[:,:-1]
        sensitive=X[:,-1]
        finite_elements = np.isfinite(answers)
        predicted_proba=np.ones((X.shape[0],self.n_classes_))/self.n_classes_
        for sample in range(answers.shape[0]):

            workers = finite_elements[sample]
            responses = answers[sample, workers].astype(np.int32)

            weight_neg = (1 - self.positive_probability) * self.confusion_matrices[
                sensitive[sample]
            ][workers, 0, responses].prod()
            weight_pos = (
                self.positive_probability
                * self.confusion_matrices[sensitive[sample]][
                    workers, 1, responses
                ].prod()
            )
            total_weight = weight_neg + weight_pos
            #print(total_weight)
            predicted_proba[sample] = (
                np.array([0.5, 0.5])
                if total_weight == 0
                else np.array([weight_neg, weight_pos]) #/ total_weight
            )

       
        smooth_proba=predicted_proba
        predicted_proba=smooth_proba/np.sum(smooth_proba,axis=1)[:,None]
        return predicted_proba

    def predict(self,Y_noise_S):
        prob=self.predict_proba(Y_noise_S)
        return np.argmax(prob,axis=1)

    def predict_log_proba(self,Y_noise_S):
        log_prob=self.predict_log_proba(Y_noise_S)
        return log_prob
