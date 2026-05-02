from abc import ABC, abstractmethod
from typing import Union, Dict, Any
import numpy as np
import numpy.typing as npt
from .fair_crowd_dataset import FairCrowdDataset


class TDOutput:
    """
    Output of a truth discovery algorithm.
    """

    probabilities: Union[npt.NDArray, None]
    labels: Union[npt.NDArray, None]
    confusion_matrices: Union[Dict[Any, npt.NDArray], npt.NDArray, None]

    def __init__(
        self,
        probabilities: Union[npt.NDArray, None] = None,
        confusion_matrices: Union[npt.NDArray, None] = None,positive_probabilities=None
    ) -> None:
        self.probabilities = probabilities
        self.confusion_matrices = confusion_matrices
        self.positive_probabilities= positive_probabilities
        # Compute labels from probabilities
        if probabilities is not None:
            self.labels = np.where(probabilities > 0.5, 1, 0)


class TDAlgorithm(ABC):
    """
    Abstract class for all truth discovery algorithms and in-processing
    mitigation techniques.
    """

    @abstractmethod
    def run(self, df: FairCrowdDataset) -> TDOutput:
        pass


class PreProcessingAlgorithm(ABC):
    """
    Abstract class for all pre-processing mitigation techinques.
    """

    @abstractmethod
    def run(self, df: FairCrowdDataset) -> FairCrowdDataset:
        pass


class PostProcessingAlgorithm(ABC):
    """
    Abstract class for all post-processing mitigation techinques.
    """

    @abstractmethod
    def run(self, df: FairCrowdDataset, td_output: TDOutput) -> TDOutput:
        pass
