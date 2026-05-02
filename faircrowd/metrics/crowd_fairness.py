import numpy as np
from numpy import typing as npt
from typing import Union
from ..core.metrics import CrowdFairnessMetric


class SimilarityFairness(CrowdFairnessMetric):
    """
    Computes the similarity-based unfairness
    """

    distance_matrix: npt.NDArray
    k: Union[int, float, None]
    dist: Union[int, float, None]

    def __init__(
        self,
        distance_matrix: npt.NDArray,
        k: Union[int, float, None] = None,
        dist: Union[int, float, None] = None,
    ) -> None:
        if k is None and dist is None or k is not None and dist is not None:
            raise ValueError("Exactly one between k and dist must be defined")
        self.k = k
        self.dist = dist
        self.distance_matrix = distance_matrix

    def compute(
        self,
        answers: npt.NDArray,
        sensitive: npt.NDArray,
    ) -> float:
        mean = lambda x: x.mean() if len(x) > 0 else np.nan
        score = 0
        count = 0
        predictions = np.nanmean(answers, axis=1) > 0.5
        distance_matrix = np.ma.array(self.distance_matrix, mask=False)
        for sample in range(answers.shape[0]):
            distances = distance_matrix[sample]
            distances.mask[sample] = True
            if self.k is None:
                neighbors = np.append(np.argwhere(distances <= self.dist)[:, 0], sample)
            else:
                dist_idxs = distances.argsort()

                if isinstance(self.k, float):
                    n = int(self.k * len(dist_idxs) / 2)
                else:
                    n = self.k
                sensit_idxs = dist_idxs[sensitive[dist_idxs] == 1][:n]
                others_idxs = dist_idxs[sensitive[dist_idxs] == 0][:n]
                neighbors = np.concatenate([sensit_idxs, others_idxs, [sample]])

            distances.mask[sample] = False
            labels = predictions[neighbors]
            sensit_labels = mean(labels[sensitive[neighbors] == 1])
            others_labels = mean(labels[sensitive[neighbors] == 0])
            if np.isfinite(sensit_labels) and np.isfinite(others_labels):
                score += np.abs(sensit_labels - others_labels)
            count += 1

        return score / count
