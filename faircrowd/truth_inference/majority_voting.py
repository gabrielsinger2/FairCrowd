from ..core.algorithms import TDAlgorithm, TDOutput
from ..core.fair_crowd_dataset import FairCrowdDataset
import numpy as np


class MajorityVoting(TDAlgorithm):
    """
    A truth-discovery algorithm that uses majority voting to infer the true labels.
    """

    def __init__(self):
        pass

    def run(self, df: FairCrowdDataset) -> TDOutput:
        """
        Run the majority voting algorithm.
        """
        return TDOutput(probabilities=np.nanmean(df.answers, axis=1) > 0.5)
