import numpy as np
import pandas as pd
from pandas.core.indexing import (
    IndexingMixin,
    _iLocIndexer,
    _LocIndexer,
    _AtIndexer,
    _iAtIndexer,
)
from typing import Union


class FairCrowdDataset(IndexingMixin):
    ...


class FairCrowdDataset(IndexingMixin):
    """
    Base class for fair crowdsourcing datasets.
    """

    _answers: pd.DataFrame
    _s: pd.DataFrame
    _x: pd.DataFrame
    _y: Union[pd.DataFrame, None] = None
    _df: pd.DataFrame

    def __init__(
        self,
        answers: pd.DataFrame,
        s: pd.DataFrame,
        x: pd.DataFrame,
        y: Union[pd.DataFrame, None],
    ) -> None:
        self._answers = answers
        self._s = s
        self._x = x
        self._y = y
        if self.y is not None:
            self._df = pd.concat([self.s, self.x, self.answers, self.y], axis=1)
            
        else:
            self._df = pd.concat([self.s, self.x, self.answers], axis=1)

    # Encapsulated datasets
    @property
    def answers(self) -> pd.DataFrame:
        return self._answers

    @property
    def s(self) -> pd.DataFrame:
        return self._s

    @property
    def x(self) -> pd.DataFrame:
        return self._x

    @property
    def y(self) -> pd.DataFrame:
        return self._y

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def __len__(self) -> int:
        return len(self.answers)

    def __contains__(self, key) -> bool:
        return (
            key in self.df
            or key == "s"
            and len(self.s.columns) == 1
            or key == "x"
            and len(self.x.columns) == 1
            or key == "y"
            and len(self.y.columns) == 1
        )

    def __getitem__(self, key) -> pd.Series:
        # Shortcuts to get a column if there's only one in the respective dataset
        if str(key) == "s" and len(self.s.columns) == 1:
            return self.s[self.s.columns[0]]
        if str(key) == "x" and len(self.x.columns) == 1:
            return self.x[self.x.columns[0]]
        if str(key) == "y" and len(self.y.columns) == 1:
            return self.y[self.y.columns[0]]

        # Otherwise get it from the datasets
        return self.df[key]

    # Shortcuts to index the concatenated dataset
    @property
    def iloc(self) -> _iLocIndexer:
        return self.df.iloc

    @property
    def loc(self) -> _LocIndexer:
        return self.df.loc

    @property
    def at(self) -> _AtIndexer:
        return self.df.loc

    @property
    def iat(self) -> _iAtIndexer:
        return self.df.loc

    def head(self) -> pd.DataFrame:
        """
        Call pandas' "head" function on the concatenated datasets.
        """
        return self.df.head()

    def describe(self) -> pd.DataFrame:
        """
        Print basic statistics about the dataset, returns pandas' describe for s, x and (if it's present) y.
        """
        print(
            f"Answers: {self.answers.shape[0]} tasks and {self.answers.shape[1]} workers"
        )
        counts = np.isfinite(self.answers).sum(axis=1)
        print("Number of answers per worker:")
        df = pd.DataFrame(
            {}, index=[""], columns=["Mean", "Std", "Min", "Median", "Max"]
        )
        df["Mean"] = np.mean(counts)
        df["Std"] = np.var(counts)
        df["Min"] = np.min(counts)
        df["Median"] = np.median(counts)
        df["Max"] = np.max(counts)
        print(df.to_string())
        print(f'{len(self.s.columns)} Sensitive Features: {", ".join(self.s.columns)}')
        print(
            f'{len(self.x.columns)} Non-Sensitive Features: {", ".join(self.x.columns)}'
        )
        if self.y is not None:
            print(f'{len(self.y.columns)} Response: {", ".join(self.y.columns)}')
        datasets = [self.s, self.x, self.y] if self.y is not None else [self.s, self.x]
        return pd.concat(datasets, axis=1).describe()

    def copy(self) -> FairCrowdDataset:
        """
        Copy the dataset.
        """
        return FairCrowdDataset(
            self.answers.copy(), self.s.copy(), self.x.copy(), self.y.copy()
        )
