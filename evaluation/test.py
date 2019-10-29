"""
===========================
Test base classes.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2019
---------------------------
"""

from abc import ABCMeta, abstractmethod
from os import path

from pandas import DataFrame, read_csv


class Test:
    def __init__(self, name: str):
        self.name: str = name


class Tester(metaclass=ABCMeta):
    def __init__(self, save_progress: bool = True, force_reload: bool = False):
        self._save_progress: bool = save_progress

        # Load existing data if any, else start afresh
        self._data: DataFrame
        if self._could_load_data() and not force_reload:
            self._data = self._load_data()
        else:
            self._data = self._fresh_data()

        if self._save_progress:
            self._save_data()

    def _could_load_data(self) -> bool:
        return path.isfile(self._save_path)

    def _load_data(self) -> DataFrame:
        return read_csv(self._save_path, index_col=None)

    @abstractmethod
    def _fresh_data(self) -> DataFrame:
        raise NotImplementedError()

    def _save_data(self):
        with open(self._save_path, mode="w", encoding="utf-8") as save_file:
            self._data.to_csv(save_file, index=False)

    @property
    @abstractmethod
    def _save_path(self) -> str:
        raise NotImplementedError()
