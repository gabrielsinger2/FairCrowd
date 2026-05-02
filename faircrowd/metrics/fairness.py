import numpy as np
import numpy.typing as npt
from ..core.metrics import FairnessMetric


class DemographicParity_(FairnessMetric):
    """
    Computes Demographic Parity
    """

    def compute(
        self,
        predictions: npt.NDArray,
        sensitive: npt.NDArray,
        truth: npt.NDArray,
    ) -> float:
        groups = np.unique(sensitive)
        positive_rates = [
            (predictions[sensitive == group] > 0.5).mean() for group in groups
        ]
        return (
            np.nanmax(positive_rates) - np.nanmin(positive_rates)
            if len(groups) > 1
            else np.nan
        )


class EqualOpportunities_(FairnessMetric):
    """
    Computes Equal Opportunities
    """

    def compute(
        self, predictions: npt.NDArray, sensitive: npt.NDArray, truth: npt.NDArray
    ) -> float:
        div = lambda num, den: np.nan if den == 0 else num / den
        groups = np.unique(sensitive)

        fprs = [
            div(
                len(
                    predictions[
                        (sensitive == group) & (predictions > 0.5) & (truth == 1)
                    ]
                ),
                len(truth[(sensitive == group) & (truth == 1)]),
            )
            for group in groups
        ]
        return np.nanmax(fprs) - np.nanmin(fprs) if len(groups) > 1 else np.nan


class PredictiveParity_(FairnessMetric):
    """
    Computes Predictive Parity
    """

    def compute(
        self, predictions: npt.NDArray, sensitive: npt.NDArray, truth: npt.NDArray
    ) -> float:
        div = lambda num, den: np.nan if den == 0 else num / den
        groups = np.unique(sensitive)

        ppvs = [
            div(
                len(
                    predictions[
                        (sensitive == group) & (predictions > 0.5) & (truth == 1)
                    ]
                ),
                len(predictions[(sensitive == group) & (predictions > 0.5)]),
            )
            for group in groups
        ]
        return np.nanmax(ppvs) - np.nanmin(ppvs) if len(groups) > 1 else np.nan
