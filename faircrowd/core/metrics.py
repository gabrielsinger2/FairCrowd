from abc import ABC, abstractmethod
from typing import Union, Callable
import numpy.typing as npt
import pandas as pd


MatrixLike = Union[pd.DataFrame, pd.Series, npt.NDArray]
AccuracyMetric = Callable[[MatrixLike, MatrixLike], float]


class FairnessMetric(ABC):
    """
    Abstract class for all group fairness metrics that require ground truth.
    """

    @abstractmethod
    def compute(
        self,
        predictions: npt.NDArray,
        sensitive: npt.NDArray,
        truth: npt.NDArray,
    ) -> float:
        pass


class CrowdFairnessMetric(ABC):
    """
    Abstract class for all crowdsourcing fairness metrics that do not require
    ground truth and work on user answers.
    """

    @abstractmethod
    def compute(
        self,
        answers: npt.NDArray,
        sensitive: npt.NDArray,
    ) -> float:
        pass
