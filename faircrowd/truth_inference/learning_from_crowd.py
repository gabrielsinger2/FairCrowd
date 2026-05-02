import numpy as np
import numpy.typing as npt
from typing import Tuple, Union
from .dawid_skene import DawidSkene
from ..core.fair_crowd_dataset import FairCrowdDataset


class LearningFromCrowd(DawidSkene):
    """
    Learning from Crowd is a variant of Dawid & Skene that uses a beta prior, with parameters (alpha, beta),
    for the label probabilities, and distinct sensitivity and specificity instead of the initial quality
    for the confusion matrices.

    N.B.: Differently from the original implementation used for the SDM paper, this version uses 0.7 as a
    default parameter for both sensitivity and specificity (intead of 0.684 and 0.73). Also the
    initialization of the predicted probabilities is different, using the e-step as in the D&S algorithm
    instead of the average of the labels.
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

    def m_step(self, answers: npt.NDArray) -> None:
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
        self.confusion_matrices = np.tile(
            [[0.0, 0.0], [0.0, 0.0]], (answers.shape[1], 1, 1)
        )
        for worker in range(answers.shape[1]):
            samples = finite_elements[:, worker]
            weights = self.predicted_probabilities[samples].sum(axis=0)
            responses = answers[samples, worker].astype(np.int32)

            for tlabel in [0, 1]:
                if weights[tlabel] == 0:
                    # Differently from D&S, use sensitivity and specificity instead of initial quality
                    if tlabel == 0:
                        self.confusion_matrices[worker, tlabel] = np.array(
                            [self.specificity, 1 - self.specificity]
                        )
                    else:
                        self.confusion_matrices[worker, tlabel] = np.array(
                            [1 - self.sensitivity, self.sensitivity]
                        )
                else:
                    np.add.at(
                        self.confusion_matrices[worker, tlabel],
                        responses,
                        self.predicted_probabilities[samples, tlabel] / weights[tlabel],
                    )

    def initialize_parameters(self, df: FairCrowdDataset) -> None:
        """
        Override the initialization of the predicted probabilities to use sensitivity and specificity.
        """
        self.predicted_probabilities = np.tile(
            [np.nan, np.nan], (df.answers.shape[0], 1)
        )
        self.positive_probability = 0.5
        self.confusion_matrices = np.tile(
            [
                [self.specificity, 1 - self.specificity],
                [1 - self.sensitivity, self.sensitivity],
            ],
            (df.answers.shape[1], 1, 1),
        )
