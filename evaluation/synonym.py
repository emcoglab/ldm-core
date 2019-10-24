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
2017
---------------------------
"""

import logging
import math
import re
from abc import ABCMeta, abstractmethod
from copy import copy
from typing import List

from .results import EvaluationResults
from ..corpus.indexing import LetterIndexing
from ..model.base import VectorSemanticModel
from ..model.ngram import NgramModel
from ..utils.exceptions import WordNotFoundError
from ..utils.maths import DistanceType, binomial_bayes_factor_one_sided
from ..preferences.preferences import Preferences

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


class SynonymTestQuestion(object):
    """
    A synonym test question.
    """

    def __init__(self, prompt: str, options: List[str], correct_i: int):
        self.correct_i: int = correct_i
        self.options: List[str] = options
        self.prompt: str = prompt

    def __copy__(self):
        return SynonymTestQuestion(self.prompt, self.options.copy(), self.correct_i)

    def __str__(self):
        stars = []
        for option in self.options:
            stars.append(f'{option}*' if option is self.correct_answer_word else option)
        return f"{self.prompt}: {', '.join(stars)}"

    @property
    def correct_answer_word(self) -> str:
        return self.options[self.correct_i]


class AnsweredQuestion(object):
    """
    An answered synonym test question.
    """

    def __init__(self, question: SynonymTestQuestion, answer_i: int):
        self.answer_i = answer_i
        self.question = question

    @property
    def correct_answer_word(self) -> str:
        return self.question.options[self.answer_i]

    @property
    def is_correct(self):
        """
        Check whether an answer to the question is correct.
        """
        return self.question.correct_i == self.answer_i

    def __str__(self):
        if self.is_correct:
            mark = "CORRECT"
        else:
            mark = f"INCORRECT ({self.question.correct_answer_word})"
        return (f"Question: {self.question.prompt}?\t"
                f"Options: {' '.join(self.question.options)}.\t"
                f"Answer: {self.correct_answer_word}.\t"
                f"Mark: {mark}")


class AnswerPaper(object):
    """
    The list of answered questions.
    """

    def __init__(self, answers: List[AnsweredQuestion]):
        self.answers = answers

    @property
    def score(self) -> float:
        """
        The fraction of correct answers.
        """
        return self.n_correct_answers / len(self.answers)

    @property
    def n_correct_answers(self) -> int:
        """
        The number of correct answers.
        """
        return sum([int(answer.is_correct) for answer in self.answers])

    @property
    def n_incorrect_answers(self) -> int:
        """
        The number of correct answers.
        """
        return len(self.answers) - self.n_correct_answers

    def save_text_transcript(self, transcript_path: str):
        """
        Saves a text transcript of the results of the test.
        """
        # Validate filename
        if not transcript_path.endswith(".txt"):
            transcript_path += ".txt"

        with open(transcript_path, mode="w", encoding="utf-8") as transcript_file:
            transcript_file.write("-----------------------\n")
            transcript_file.write(f"Overall score: {100 * self.score}%\n")
            transcript_file.write("-----------------------\n")
            for answer in self.answers:
                transcript_file.write(str(answer) + "\n")


class SynonymTest(object, metaclass=ABCMeta):
    def __init__(self):
        # Backs self.question_list
        self.question_list: List[SynonymTestQuestion] = self._load()

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the test.
        """
        raise NotImplementedError()

    @abstractmethod
    def _load(self) -> List[SynonymTestQuestion]:
        raise NotImplementedError()


class ToeflTest(SynonymTest):
    """
    TOEFL test.
    """

    @property
    def name(self) -> str:
        return "TOEFL"

    def _load(self) -> List[SynonymTestQuestion]:

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
                               # Who knows what
                               r"\s+\(a,a,-\)\s+\d+\s+"
                               r"(?P<option_letter>[a-d])"
                               r"\s*$")

        # Get questions
        n_options = 4
        questions: List[SynonymTestQuestion] = []
        with open(Preferences.toefl_question_path, mode="r", encoding="utf-8") as question_file:
            # Read groups of lines from file
            while True:
                prompt_line = question_file.readline().strip()

                # If we've reached the end of the file, stop reading
                if not prompt_line:
                    break

                prompt_match = re.match(prompt_re, prompt_line)
                prompt_word = prompt_match.group("prompt_word")

                option_list = []
                for option_i in range(n_options):
                    option_line = question_file.readline().strip()
                    option_match = re.match(option_re, option_line)
                    option_list.append(option_match.group("option_word"))

                # Using -1 as an "unset" value
                questions.append(SynonymTestQuestion(prompt_word, option_list, -1))

                # There's a blank line after each question
                question_file.readline()

        # Get answers
        answers: List[int] = []
        with open(Preferences.toefl_answer_path, mode="r", encoding="utf-8") as answer_file:
            for answer_line in answer_file:
                answer_line = answer_line.strip()
                # Skip empty lines
                if not answer_line:
                    continue

                answer_match = re.match(answer_re, answer_line)
                option_letter = answer_match.group("option_letter")
                answer_i = LetterIndexing.letter2int(option_letter)
                answers.append(answer_i)

        # Add the correct answers
        for question_i, question in enumerate(questions):
            question.correct_i = answers[question_i]

        return questions


class EslTest(SynonymTest):
    """
    ESL test.
    """

    @property
    def name(self):
        return "ESL"

    def _load(self) -> List[SynonymTestQuestion]:
        question_re = re.compile(r"^"
                                 r"(?P<prompt_word>[a-z\-]+)"
                                 r"\s+\|\s+"
                                 r"(?P<option_list>[a-z\-\s|]+)"
                                 r"\s*$")

        questions: List[SynonymTestQuestion] = []
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

                # The first one is always the correct one
                questions.append(SynonymTestQuestion(prompt, options, correct_i=0))

        return questions


class LbmMcqTest(SynonymTest):
    """
    MCQ test from Levy, Bullinaria and McCormick (2017).
    """

    @property
    def name(self):
        return "LBM's new MCQ"

    def _load(self) -> List[SynonymTestQuestion]:

        n_options = 4
        questions: List[SynonymTestQuestion] = []
        with open(Preferences.mcq_test_path, mode="r", encoding="utf-8") as test_file:
            while True:
                prompt = test_file.readline().strip()
                # Stop at the last line
                if not prompt:
                    break

                options = []
                for i in range(n_options):
                    options.append(test_file.readline().strip())

                # The first one is always the correct one
                questions.append(SynonymTestQuestion(prompt, options, correct_i=0))

        return questions


# Static class
class SynonymTester(object):
    """
    Administers synonym tests against models.
    """

    @staticmethod
    def administer_test_with_distance(
            test: SynonymTest,
            model: VectorSemanticModel,
            distance_type: DistanceType,
            truncate_vectors_at_length: int = None
            ) -> SynonymResults:
        """
        Administers a test against a model
        :param test:
        :param model: Must be trained.
        :param distance_type:
        :param truncate_vectors_at_length:
        """

        assert model.is_trained

        results = SynonymResults()

        answers = []
        for question in test.question_list:
            answer = SynonymTester.attempt_question_with_distance(question, model, distance_type, truncate_vectors_at_length)

            answers.append(answer)

        answer_paper = AnswerPaper(answers)

        n_correct_answers = answer_paper.n_correct_answers
        n_total_questions = answer_paper.n_correct_answers + answer_paper.n_incorrect_answers

        chance_level = 0.25
        b10 = binomial_bayes_factor_one_sided(n_total_questions, n_correct_answers, chance_level)

        results.add_result(
            test.name, model, distance_type, {
                "Correct answers": n_correct_answers,
                "Total questions": n_total_questions,
                "Score":           answer_paper.score,
                "B10":             b10
            },
            append_to_model_name="" if truncate_vectors_at_length is None else f" ({truncate_vectors_at_length})")

        return results

    @staticmethod
    def administer_test_with_similarity(
            test: SynonymTest,
            model: NgramModel
            ) -> SynonymResults:
        """
        Administers a test against a model
        :param test:
        :param model: Must be trained.
        """

        assert model.is_trained

        results = SynonymResults()

        answers = []
        for question in test.question_list:
            answer = SynonymTester.attempt_question_with_similarity(question, model)

            answers.append(answer)

        answer_paper = AnswerPaper(answers)

        n_correct_answers = answer_paper.n_correct_answers
        n_total_questions = answer_paper.n_correct_answers + answer_paper.n_incorrect_answers

        chance_level = 0.25
        b10 = binomial_bayes_factor_one_sided(n_total_questions, n_correct_answers, chance_level)

        results.add_result(
            test.name, model, None, {
                "Correct answers": n_correct_answers,
                "Total questions": n_total_questions,
                "Score":           answer_paper.score,
                "B10":             b10
            })

        return results

    @staticmethod
    def attempt_question_with_distance(question: SynonymTestQuestion, model: VectorSemanticModel,
                                       distance_type: DistanceType,
                                       truncate_vectors_at_length: int = None) -> AnsweredQuestion:
        """
        Attempt a question.
        :param question:
        :param model:
        :param distance_type:
        :param truncate_vectors_at_length:
        :return: answer
        """
        # The current best guess
        best_guess_i = -1
        # _d for distance
        best_guess_d = math.inf

        # Record distances here so we can check for ties when a selection is finally made
        distances = []

        # List in reverse order, just to negate any possibility that we preference the first option, which is usually
        # the correct one.
        for option_i, option in reversed(list(enumerate(question.options))):
            try:
                guess_d = model.distance_between(question.prompt, option, distance_type,
                                                 truncate_vectors_at_length)
                distances.append(guess_d)
            except WordNotFoundError as er:
                logger.warning(er.message)
                # Make sure we don't pick this one, so we give it the largest possible distance
                guess_d = math.inf

            if guess_d < best_guess_d:
                best_guess_i = option_i
                best_guess_d = guess_d

        tied_entry_indices = [i for i, d in enumerate(distances) if d == best_guess_d]
        if len(tied_entry_indices) > 1 and best_guess_d > 0:
            tied_entries = [question.options[i].upper() for i in tied_entry_indices]
            logger.warning(f"{question.prompt.upper()}'s chosen synonym {question.options[best_guess_i]} "
                           f"was tied with {' and '.join(tied_entries)}.")

        return AnsweredQuestion(copy(question), best_guess_i)

    @staticmethod
    def attempt_question_with_similarity(question: SynonymTestQuestion, model: NgramModel) -> AnsweredQuestion:
        """
        Attempt a question.
        :param question:
        :param model:
        :return: answer
        """
        # The current best guess
        best_guess_i = -1
        # _a for association
        best_guess_a = -math.inf

        # Record associations here so we can check for ties when a selection is finally made
        associations = []

        # List in reverse order: in case we have all items identical we won't automatically pick the first one (which is
        # usually the correct one).
        for option_i, option in reversed(list(enumerate(question.options))):
            try:
                guess_a = model.association_between(question.prompt, option)
                associations.append(guess_a)
            except WordNotFoundError as er:
                logger.warning(er.message)
                # Make sure we don't pick this one, so we give it the smallest possible association
                guess_a = -math.inf

            if guess_a > best_guess_a:
                best_guess_i = option_i
                best_guess_a = guess_a

        tied_entry_indices = [i for i, a in enumerate(associations) if a == best_guess_a]
        if len(tied_entry_indices) > 1 and best_guess_a > 0:
            tied_entries = [question.options[i].upper() for i in tied_entry_indices]
            logger.warning(f"{question.prompt.upper()}'s chosen synonym {question.options[best_guess_i]} "
                           f"was tied with {' and '.join(tied_entries)}.")

        return AnsweredQuestion(copy(question), best_guess_i)
