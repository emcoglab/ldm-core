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

import logging
import os
from abc import ABCMeta
from typing import List, Optional

import pandas

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

        self._save_dir: str = save_dir
        self._csv_path: str = os.path.join(self._save_dir, "evaluation_results.csv")

        self.data: pandas.DataFrame = pandas.DataFrame(columns=self.model_index_column_names + results_column_names)

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

    def save(self):
        """Save (and overwrite) data."""
        assert self.data is not None
        with open(self._csv_path, mode="w", encoding="utf-8") as spp_file:
            # We don't want to save the index, as it's not especially meaningful, and makes life harder when trying to
            # restore the binary version from the csv (the index column would be imported and then need to be dropped).
            self.data.to_csv(spp_file, index=False)
