"""
===========================
Base classes for results of tests.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2017
---------------------------
"""

from __future__ import annotations

import os
import pickle
import logging

from abc import ABCMeta
from typing import List, Optional

import pandas
from numpy import nan

from ..model.base import DistributionalSemanticModel
from ..model.predict import PredictVectorModel
from ..utils.maths import DistanceType


logger = logging.getLogger(__name__)


class EvaluationResults(metaclass=ABCMeta):
    """
    The results of a model evaluation.
    """

    model_index_column_names = [
       "Test name",
       "Model type",
       "Embedding size",
       "Window radius",
       "Distance type",
       "Corpus"
    ]

    def __init__(self,
                 results_column_names: List[str],
                 save_dir: str):

        self._save_path = os.path.join(save_dir, "evaluation_results")
        self._pickle_path = self._save_path + ".pickle"
        self._csv_path = self._save_path + ".csv"

        column_names = self.model_index_column_names + results_column_names

        self.data: pandas.DataFrame = pandas.DataFrame(columns=column_names)

    @property
    def column_names(self):
        """
        The column names in the results table.
        """
        return self.data.columns.values

    def add_result(self,
                   test_name: str,
                   model: DistributionalSemanticModel,
                   distance_type: Optional[DistanceType],
                   # a dictionary whose keys are the same as the results_column_names
                   result: dict,
                   append_to_model_name: str = None):
        """
        Add a single result.
        """
        # Add model keys to result row
        result["Test name"] = test_name
        # TODO: this is also gross
        result["Model type"] = model.model_type.name + (append_to_model_name if append_to_model_name is not None else "")
        result["Embedding size"] = model.embedding_size if isinstance(model, PredictVectorModel) else None
        result["Window radius"] = model.window_radius
        result["Distance type"] = distance_type.name if distance_type is not None else ""
        result["Corpus"] = model.corpus_meta.name

        assert set(result.keys()) == set(self.column_names)

        # TODO: This is possibly inefficient
        self.data = self.data.append(result, ignore_index=True)

    def extend_with_results(self, other: 'EvaluationResults'):
        """
        Adds a batch of results.
        """
        assert set(self.column_names) == set(other.column_names)
        self.data = self.data.append(other.data, ignore_index=True)

    def results_exist_for(self,
                          test_name: str,
                          model: DistributionalSemanticModel,
                          distance_type: Optional[DistanceType],
                          truncate_vectors_at_length: int = None) -> bool:
        """
        Do results exist for this model?
        """
        return self.data[
                   (self.data["Test name"] == test_name) &
                   # TODO: this is gross
                   (self.data["Model type"] == (model.model_type.name + (f" ({truncate_vectors_at_length})" if truncate_vectors_at_length is not None else ""))) &
                   ((self.data["Embedding size"] == model.embedding_size) if isinstance(model, PredictVectorModel) else pandas.isnull(self.data["Embedding size"])) &
                   (self.data["Window radius"] == model.window_radius) &
                   ((self.data["Distance type"] == distance_type.name) if distance_type is not None else pandas.isnull(self.data["Distance type"])) &
                   (self.data["Corpus"] == model.corpus_meta.name)
               ].shape[0] > 0

    def save(self):
        """Save (and overwrite) data."""
        assert self.data is not None
        with open(self._pickle_path, mode="wb") as data_file:
            pickle.dump(self.data, data_file)
        self._export_csv()

    def _previously_saved_pickle(self) -> bool:
        return os.path.isfile(self._pickle_path)

    def _previously_saved_csv(self) -> bool:
        return os.path.isfile(self._csv_path)

    def load(self) -> EvaluationResults:
        """
        Load previously saved data.
        Returns self for method chaining.
        """
        old_column_names = self.data.columns.values
        if self._previously_saved_pickle():
            with open(self._pickle_path, mode="rb") as data_file:
                self.data = pickle.load(data_file)
            assert set(self.column_names) == set(old_column_names)
        elif self._previously_saved_csv():
            logger.warning(f"Previous binary {self._pickle_path} not found.")
            logger.info(f"Importing from csv.")
            self.import_csv()
            assert set(self.column_names) == set(old_column_names)
            logger.info(f"Restoring binary save.")
            self.save()
        else:
            logger.warning(f"Previous binary {self._pickle_path} not found.")
            logger.warning(f"Previous csv {self._csv_path} not found.")
            logger.warning(f"Nothing loaded!")

        return self

    def _export_csv(self):
        """
        Export results as a csv.
        """
        with open(self._csv_path, mode="w", encoding="utf-8") as spp_file:
            # We don't want to save the index, as it's not especially meaningful, and makes life harder when trying to
            # restore the binary version from the csv (the index column would be imported and then need to be dropped).
            self.data.to_csv(spp_file, index=False)

    def import_csv(self):
        """
        Load from a CSV (use if the pickle gets lost or corrupted somehow.
        """
        self.data = pandas.read_csv(self._csv_path, converters={
            # Check if embedding size is the empty string,
            # as it would be for Count models
            "Embedding size": lambda v: int(v) if len(v) > 0 else nan
        })
