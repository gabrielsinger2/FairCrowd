import numpy as np
import numpy.typing as npt
from typing import Tuple, Union
from .dawid_skene_multiple import DawidSkeneMultiple
from ..core.fair_crowd_dataset import FairCrowdDataset


class LearningFromCrowdMultiple(DawidSkeneMultiple):
    """
    Vairation of the LFC algorithm that uses two (or more) confusion matrices, one for sensitive and one for non-sensitive groups.
    """

    beta_param: Union[Tuple[float, float], None]
    sensitivity: float
    specificity: float

    def __init__(
        self,
        iterations: int = 20,
        beta_param: Union[Tuple[float, float], None] = None,
        sensitivity: float = 0.7,
        specificity: float = 0.7,
    ) -> None:
        # The initial quality is unused as we have sensitivity and specificity instead in the overriden m_step
        super().__init__(iterations, np.nan)
        self.beta_param = beta_param
        self.sensitivity = sensitivity
        self.specificity = specificity

    def m_step(self, answers: npt.NDArray, sensitive: npt.NDArray) -> None:
        """
        Override the m-step to use the beta prior, sensitivity, and specificity instead of initial quality.
        """
        # Update label probabilities either like D&S or using the beta prior if available
        if self.beta_param is not None:
            alpha, beta = self.beta_param
            # Sum of positive probabilities
            positive_count = self.predicted_probabilities.sum(axis=0)[1]
            total_count = len(self.predicted_probabilities)
            self.positive_probability = (alpha - 1 + positive_count) / (
                alpha + beta - 2 + total_count
            )
        else:
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
                        # Differently from D&S, use sensitivity and specificity instead of initial quality
                        if tlabel == 0:
                            self.confusion_matrices[group][worker, tlabel] = np.array(
                                [self.specificity, 1 - self.specificity]
                            )
                        else:
                            self.confusion_matrices[group][worker, tlabel] = np.array(
                                [1 - self.sensitivity, self.sensitivity]
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
        Override the initialization of the predicted probabilities to use sensitivity and specificity.
        """
        self.groups = np.unique(df["s"].values)
        self.predicted_probabilities = np.tile(
            [np.nan, np.nan], (df.answers.shape[0], 1)
        )
        self.positive_probability = 0.5
        self.confusion_matrices = {
            group: np.tile(
                [
                    [self.specificity, 1 - self.specificity],
                    [1 - self.sensitivity, self.sensitivity],
                ],
                (df.answers.shape[1], 1, 1),
            )
            for group in self.groups
        }
