"""
===========================
Evaluation of models.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2017, 2019
---------------------------
"""

from __future__ import annotations

import logging
import re
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from os import path
from typing import List, Optional

from numpy import nan
from pandas import DataFrame

from .test import Test, Tester
from .results import EvaluationResults
from ..corpus.indexing import LetterIndexing
from ..model.base import VectorSemanticModel, DistributionalSemanticModel
from ..model.ngram import NgramModel
from ..preferences.preferences import Preferences
from ..utils.exceptions import WordNotFoundError
from ..utils.maths import DistanceType, binomial_bayes_factor_one_sided

logger = logging.getLogger(__name__)


class SynonymResults(EvaluationResults):
    def __init__(self):
        super().__init__(
            results_column_names=[
                "Correct answers",
                "Total questions",
                "Score",
                "B10"
            ],
            save_dir=Preferences.synonym_results_dir
        )


class SynonymTest(Test, metaclass=ABCMeta):
    class TestColumn:
        prompt =     "Prompt"
        option =     "Option"
        is_correct = "Is correct"

    test_columns = [
        TestColumn.prompt,
        TestColumn.option,
        TestColumn.is_correct,
    ]

    def __init__(self, name: str):
        super().__init__(name)
        self.questions: List[SynonymTest.Question] = self._load()

        # Check all questions have the same number of answers
        for question in self.questions:
            assert len(question.answers) == self.n_options

    @property
    def n_options(self) -> int:
        return len(self.questions[0].answers)

    def questions_to_dataframe(self) -> DataFrame:
        return DataFrame.from_records(
            data=[
                {
                    SynonymTest.TestColumn.prompt: question.prompt_word,
                    SynonymTest.TestColumn.option: answer.word,
                    SynonymTest.TestColumn.is_correct: answer.is_correct,
                }
                for question in self.questions
                for answer in question.answers
            ],
            columns=SynonymTest.test_columns)

    @abstractmethod
    def _load(self) -> List[SynonymTest.Question]:
        raise NotImplementedError()

    @dataclass
    class Answer:
        word: str
        is_correct: bool

    @dataclass
    class Question:
        prompt_word: str
        answers: List[SynonymTest.Answer]


class ToeflTest(SynonymTest):
    """TOEFL test."""
    _n_options = 4

    def __init__(self):
        super().__init__("TOEFL")

    def _load(self) -> List[SynonymTest.Question]:
        prompt_re = re.compile(r"^"
                               r"(?P<question_number>\d+)"
                               r"\.\s+"
                               r"(?P<prompt_word>[a-z\-]+)"
                               r"\s*$")
        option_re = re.compile(r"^"
                               r"(?P<option_letter>[a-d])"
                               r"\.\s+"
                               r"(?P<option_word>[a-z\-]+)"
                               r"\s*$")
        answer_re = re.compile(r"^"
                               r"(?P<question_number>\d+)"
                               r"\s+\(a,a,-\)\s+\d+\s+"  # Who knows what
                               r"(?P<option_letter>[a-d])"
                               r"\s*$")

        # Get questions
        questions: List[SynonymTest.Question] = []
        with open(Preferences.toefl_question_path, mode="r", encoding="utf-8") as question_file:
            # Read groups of lines from file
            while True:
                prompt_line = question_file.readline().strip()

                # If we've reached the end of the file, stop reading
                if not prompt_line:
                    break

                prompt_match = re.match(prompt_re, prompt_line)
                # In the file, the questions are numbered 1-indexed, but we want 0-indexed
                question_number: int = int(prompt_match.group("question_number")) - 1
                prompt_word = prompt_match.group("prompt_word")

                options: List[SynonymTest.Answer] = []
                for option_i in range(self._n_options):
                    option_line = question_file.readline().strip()
                    option_match = re.match(option_re, option_line)
                    option_letter = option_match.group("option_letter")
                    assert option_i == LetterIndexing.letter2int(option_letter)
                    option_word = option_match.group("option_word")
                    # Set all answers' correctness to False initially, correct one will be set to True eventually
                    options.append(SynonymTest.Answer(option_word, False))
                questions.append(SynonymTest.Question(prompt_word, options))

                # There's a blank line after each question
                question_file.readline()

        # Get correct answers
        with open(Preferences.toefl_answer_path, mode="r", encoding="utf-8") as answer_file:
            for answer_line in answer_file:
                answer_line = answer_line.strip()
                # Skip empty lines
                if not answer_line:
                    continue

                answer_match = re.match(answer_re, answer_line)
                # In the file, the questions are numbered 1-indexed, but we want 0-indexed
                question_number: int = int(answer_match.group("question_number")) - 1
                option_letter = answer_match.group("option_letter")
                # 0-indexed
                answer_number = LetterIndexing.letter2int(option_letter)
                questions[question_number].answers[answer_number].is_correct = True

        # Verify each question has exactly 1 correct answer
        for question in questions:
            n_correct = 0
            for answer in question.answers:
                if answer.is_correct:
                    n_correct += 1
            assert n_correct == 1

        return questions


class EslTest(SynonymTest):
    """ESL test."""

    def __init__(self):
        super().__init__("ESL")

    def _load(self) -> List[SynonymTest.Question]:

        question_re = re.compile(r"^"
                                 r"(?P<prompt_word>[a-z\-]+)"
                                 r"\s+\|\s+"
                                 r"(?P<option_list>[a-z\-\s|]+)"
                                 r"\s*$")

        questions: List[SynonymTest.Question] = []
        with open(Preferences.esl_test_path, mode="r", encoding="utf-8") as test_file:
            for line in test_file:
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue
                # Skip comments
                if line.startswith("#"):
                    continue

                question_match = re.match(question_re, line)

                prompt = question_match.group("prompt_word")
                options = [option.strip() for option in question_match.group("option_list").split("|")]

                questions.append(SynonymTest.Question(
                    prompt_word=prompt,
                    answers=[SynonymTest.Answer(
                            word=option,
                            # first answer is always the right one
                            is_correct=option_i == 0)
                        for option_i, option in enumerate(options)
                    ]))

        return questions


class LbmMcqTest(SynonymTest):
    """
    MCQ test from Levy, Bullinaria and McCormick (2017).
    """
    _n_options = 4

    def __init__(self):
        super().__init__("LBM's new MCQ")

    def _load(self) -> List[SynonymTest.Question]:

        questions: List[SynonymTest.Question] = []
        with open(Preferences.mcq_test_path, mode="r", encoding="utf-8") as test_file:
            while True:
                prompt = test_file.readline().strip()

                # Stop at the last line
                if not prompt:
                    break

                questions.append(
                    SynonymTest.Question(prompt, [
                        SynonymTest.Answer(
                            word=test_file.readline().strip(),
                            # first answer is always the right one
                            is_correct=answer_number == 0)
                        # 0-indexed
                        for answer_number in range(self._n_options)
                    ]))

        return questions


class SynonymTester(Tester):
    """
    Administers synonym tests against models, saving all results as it goes.
    """

    def __init__(self, test: SynonymTest, save_progress: bool = True, force_refresh: bool = False):
        self.test: SynonymTest = test
        super().__init__(save_progress, force_refresh)

    @property
    def _save_path(self) -> str:
        return path.join(Preferences.synonym_results_dir, f"{self.test.name} data.csv")

    def _fresh_data(self) -> DataFrame:
        return self.test.questions_to_dataframe()

    def has_tested_model(self,
                         model: DistributionalSemanticModel,
                         distance_type: Optional[DistanceType] = None,
                         truncate_length: int = None) -> bool:
        return self.column_name_for_model(model, distance_type, truncate_length) in self._data.columns.values

    def column_name_for_model(self,
                              model: DistributionalSemanticModel,
                              distance_type: Optional[DistanceType],
                              truncate_vectors_at_length: int) -> str:
        self._validate_model_params(model, distance_type, truncate_vectors_at_length)
        if distance_type is None:
            return f"{model.name}"
        elif truncate_vectors_at_length is None:
            return f"{model.name}: {distance_type.name}"
        else:
            return f"{model.name}: {distance_type.name} ({truncate_vectors_at_length})"

    def column_name_for_model_guess(self,
                                    model: DistributionalSemanticModel,
                                    distance_type: Optional[DistanceType],
                                    truncate_vectors_at_length: int) -> str:
        return self.column_name_for_model(model, distance_type, truncate_vectors_at_length) + " guess"

    def column_name_for_model_correct(self,
                                      model: DistributionalSemanticModel,
                                      distance_type: Optional[DistanceType],
                                      truncate_vectors_at_length: int) -> str:
        return self.column_name_for_model(model, distance_type, truncate_vectors_at_length) + " correct"

    def administer_test(self,
                        model: DistributionalSemanticModel,
                        distance_type: Optional[DistanceType] = None,
                        truncate_vectors_at_length: int = None):
        """
        Administers a test against a model.

        :param model: Must be trained.
        :param distance_type:
        :param truncate_vectors_at_length:
        """

        if distance_type is not None:
            logger.info(f"Administering {self.test.name} test with {model.name} and {distance_type.name}")
        else:
            logger.info(f"Administering {self.test.name} test with {model.name}")

        # Validate args
        self._validate_model_params(model, distance_type, truncate_vectors_at_length)
        assert model.is_trained

        model_distance_col_name = self.column_name_for_model(model, distance_type, truncate_vectors_at_length)
        model_guess_col_name = self.column_name_for_model_guess(model, distance_type, truncate_vectors_at_length)
        model_correct_col_name = self.column_name_for_model_correct(model, distance_type, truncate_vectors_at_length)

        # Treat missing words as missing data
        def association_or_nan(word_pair):
            w1, w2 = word_pair
            try:
                if isinstance(model, NgramModel):
                    return model.association_between(w1, w2)
                elif isinstance(model, VectorSemanticModel):
                    return model.distance_between(w1, w2, distance_type, truncate_vectors_at_length)
                else:
                    raise NotImplementedError()
            except WordNotFoundError:
                return nan

        self._data[model_distance_col_name] = self._data[
            [SynonymTest.TestColumn.prompt, SynonymTest.TestColumn.option]
        ].apply(association_or_nan, axis=1)

        # Guess column is True where the model guesses, and is otherwise False
        self._data[model_guess_col_name] = False
        if isinstance(model, NgramModel):
            guesses = (self._data
                       # Reverse. Then idxmax returns the LAST row within a group which attains the minimum value.
                       # Then, in case of ties, we select the last option.
                       # In TOEFL this is unbiased (answers in a random order).
                       # In ESL and LBM this biases us AGAINST selecting the correct answer, as the correct answer is
                       # always the first option.
                       [::-1]
                       .groupby(SynonymTest.TestColumn.prompt, sort=False)
                       # idxmax to select the choice with the largest association value
                       .idxmax()
                       [model_distance_col_name])
        elif isinstance(model, VectorSemanticModel):
            guesses = (self._data
                       # Reverse. Then idxmin returns the LAST row within a group which attains the minimum value.
                       # Then, in case of ties, we select the last option.
                       # In TOEFL this is unbiased (answers in a random order).
                       # In ESL and LBM this biases us AGAINST selecting the correct answer, as the correct answer is
                       # always the first option.
                       [::-1]
                       .groupby(SynonymTest.TestColumn.prompt, sort=False)
                       # idxmin to select the choice with the smallest distance
                       .idxmin()
                       [model_distance_col_name])
        else:
            raise NotImplementedError()
        # In case prompt word was not found (or all options not found), all associations could be nan, so the guess
        # could be nan, which would cause a ValueError. So we drop any nans here (and the guess will be left as
        # False => unguessed).
        guesses.dropna(inplace=True)
        self._data[model_guess_col_name][guesses] = True

        # Mark guesses correct or not
        # TODO: nans next to non-guesses so can take mean and get score
        self._data[model_correct_col_name] = self._data[model_guess_col_name] & self._data[SynonymTest.TestColumn.is_correct]

        if self._save_progress:
            self._save_data()

    def results_for_model(self,
                          model: DistributionalSemanticModel,
                          distance_type: Optional[DistanceType] = None,
                          truncate_vectors_at_length: int = None) -> dict:
        """
        Save results based on the current self._data.
        Returns a dict ready to add to a SynonymResults.
        """
        assert self.has_tested_model(model, distance_type, truncate_vectors_at_length)

        model_correct_col_name = self.column_name_for_model_correct(model, distance_type, truncate_vectors_at_length)

        n_correct_answers = self._data[model_correct_col_name].sum()
        n_total_questions = len(self.test.questions)
        score = n_correct_answers / n_total_questions

        chance_level = 1 / self.test.n_options
        b10 = binomial_bayes_factor_one_sided(n_total_questions, n_correct_answers, chance_level)

        return {
            "Correct answers": n_correct_answers,
            "Total questions": n_total_questions,
            "Score":           score,
            "B10":             b10
        }

    @staticmethod
    def _validate_model_params(model: DistributionalSemanticModel, distance_type: Optional[DistanceType],
                               truncate_vectors_at_length: Optional[int]):
        if isinstance(model, NgramModel):
            assert distance_type is None
            assert truncate_vectors_at_length is None
        if isinstance(model, VectorSemanticModel):
            assert distance_type is not None
