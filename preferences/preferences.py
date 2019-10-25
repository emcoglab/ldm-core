"""
===========================
Global preference classes.
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
from collections import namedtuple
from os import path

from ..corpus.corpus import CorpusMetadata


class Preferences:
    """
    Preferences for models.
    """

    # Local import to prevent it being accidentally imported from this module
    from .config import Config
    
    # Static config
    _config: Config = Config()

    # Paths for intermediate processing steps
    bnc_processing_metas = dict(
        raw=CorpusMetadata(
            name=_config.value_by_key_path("corpora", "bnc-raw", "name"),
            path=_config.value_by_key_path("corpora", "bnc-raw", "path")),
        detagged=CorpusMetadata(
            name=_config.value_by_key_path("corpora", "bnc-detagged", "name"),
            path=_config.value_by_key_path("corpora", "bnc-detagged", "path")),
        tokenised=CorpusMetadata(
            name=_config.value_by_key_path("corpora", "bnc", "name"),
            path=_config.value_by_key_path("corpora", "bnc", "path"),
            freq_dist_path=_config.value_by_key_path("corpora", "bnc", "index")))
    bnc_text_processing_metas = dict(
        raw=CorpusMetadata(
            name=_config.value_by_key_path("corpora", "bnc-text-raw", "name"),
            path=_config.value_by_key_path("corpora", "bnc-text-raw", "path")),
        detagged=CorpusMetadata(
            name=_config.value_by_key_path("corpora", "bnc-text-detagged", "name"),
            path=_config.value_by_key_path("corpora", "bnc-text-detagged", "path")),
        tokenised=CorpusMetadata(
            name=_config.value_by_key_path("corpora", "bnc-text", "name"),
            path=_config.value_by_key_path("corpora", "bnc-text", "path"),
            freq_dist_path=_config.value_by_key_path("corpora", "bnc-text", "index")))
    bnc_speech_processing_metas = dict(
        raw=CorpusMetadata(
            name=_config.value_by_key_path("corpora", "bnc-speech-raw", "name"),
            path=_config.value_by_key_path("corpora", "bnc-speech-raw", "path")),
        detagged=CorpusMetadata(
            name=_config.value_by_key_path("corpora", "bnc-speech-detagged", "name"),
            path=_config.value_by_key_path("corpora", "bnc-speech-detagged", "path")),
        tokenised=CorpusMetadata(
            name=_config.value_by_key_path("corpora", "bnc-speech", "name"),
            path=_config.value_by_key_path("corpora", "bnc-speech", "path"),
            freq_dist_path=_config.value_by_key_path("corpora", "bnc-speech", "index")))
    bbc_processing_metas = dict(
        raw=CorpusMetadata(
            name=_config.value_by_key_path("corpora", "subtitles-raw", "name"),
            path=_config.value_by_key_path("corpora", "subtitles-raw", "path")),
        no_srt=CorpusMetadata(
            name=_config.value_by_key_path("corpora", "subtitles-no-srt", "name"),
            path=_config.value_by_key_path("corpora", "subtitles-no-srt", "path")),
        no_nonspeech=CorpusMetadata(
            name=_config.value_by_key_path("corpora", "subtitles-no-nonspeech", "name"),
            path=_config.value_by_key_path("corpora", "subtitles-no-nonspeech", "path")),
        replaced_symbols=CorpusMetadata(
            name=_config.value_by_key_path("corpora", "subtitles-replaced-symbols", "name"),
            path=_config.value_by_key_path("corpora", "subtitles-replaced-symbols", "path")),
        tokenised=CorpusMetadata(
            name=_config.value_by_key_path("corpora", "subtitles", "name"),
            path=_config.value_by_key_path("corpora", "subtitles", "path"),
            freq_dist_path=_config.value_by_key_path("corpora", "subtitles", "index")))
    ukwac_processing_metas = dict(
        raw=CorpusMetadata(
            name=_config.value_by_key_path("corpora", "ukwac-raw", "name"),
            path=_config.value_by_key_path("corpora", "ukwac-raw", "path")),
        no_urls=CorpusMetadata(
            name=_config.value_by_key_path("corpora", "ukwac-no-urls", "name"),
            path=_config.value_by_key_path("corpora", "ukwac-no-urls", "path")),
        partitioned=CorpusMetadata(
            name=_config.value_by_key_path("corpora", "ukwac-partitioned", "name"),
            path=_config.value_by_key_path("corpora", "ukwac-partitioned", "path")),
        tokenised=CorpusMetadata(
            name=_config.value_by_key_path("corpora", "ukwac", "name"),
            path=_config.value_by_key_path("corpora", "ukwac", "path"),
            freq_dist_path=_config.value_by_key_path("corpora", "ukwac", "index")))

    # The final locations of the processed corpora
    source_corpus_metas = namedtuple('SourceCorpusMetas', ['bnc', 'bbc', 'ukwac'])(
        bnc=bnc_processing_metas["tokenised"],
        bbc=bbc_processing_metas["tokenised"],
        ukwac=ukwac_processing_metas["tokenised"],
    )

    # We will test models with windows of each of these radii
    window_radii = [1, 3, 5, 10]

    # For the predict models, we will test a number of different embedding sizes
    # These sizes taken from Mandera et al. (2017)
    predict_embedding_sizes = [50, 100, 200, 300, 500]

    # The base directory for the models to be saved
    model_dir                 = _config.value_by_key_path("models", "directory")

    # TESTS

    test_dir                  = _config.value_by_key_path("tests", "directory")

    # Synonym tests
    toefl_question_path       = path.join(test_dir, "TOEFL_BrEng_&_substitutions", "toefl.qst")
    toefl_answer_path         = path.join(test_dir, "TOEFL_BrEng_&_substitutions", "toefl.ans")
    esl_test_path             = path.join(test_dir, "ESL_BrEng", "esl.txt")
    mcq_test_path             = path.join(test_dir, "LBM vocab MCQ", "newMCQ.txt")

    # Word association tests
    # similarity/relatedness judgements
    simlex_path               = path.join(test_dir, "SimLex-999 BrEng/SimLex-999.txt")
    wordsim_similarity_path   = path.join(test_dir, "WordSim353", "wordsim_similarity_goldstandard.txt")
    wordsim_relatedness_path  = path.join(test_dir, "WordSim353", "wordsim_relatedness_goldstandard.txt")
    men_path                  = path.join(test_dir, "MEN BrEng", "MEN_dataset_natural_form_full")
    # word association production
    colour_association_path   = path.join(test_dir, "Colour association", "Appendix 1 (cleaned).csv")
    thematic_association_path = path.join(test_dir, "Thematic relatedness", "13428_2015_679_MOESM2_ESM (corrected).csv")

    # Semantic priming data
    spp_path_csv              = path.join(test_dir, "SPP", "Hutchison et al. (2013) SPP combined.csv")  # Semantic priming data: source csv file
    spp_elexicon_csv          = path.join(test_dir, "SPP", "elexicon", "I148613.csv")  # Additional Elexicon predictors csv
    mandera_distances_csv     = path.join(test_dir, "Spp", "mandera", "mandera_cosine_cbow300_r6_ukwacsubtitles.csv")  # Distances computed from Brysbaert's SNAUT

    # Norms
    calgary_path_xlsx         = path.join(test_dir, "Calgary", "13428_2016_720_MOESM2_ESM BrEng.xlsx")
    calgary_elexicon_csv      = path.join(test_dir, "Calgary", "elexicon", "I150283.csv")

    # TEST RESULTS

    results_dir               = _config.value_by_key_path("results", "directory")

    synonym_results_dir       = path.join(results_dir, "synonyms/")
    association_results_dir   = path.join(results_dir, "association/")
    spp_results_dir           = path.join(results_dir, "SPP/")
    spp_path_pickle           = path.join(spp_results_dir, "Hutchison et al. (2013) SPP.pickle")  # Semantic priming data: pickled version for faster loading
    calgary_results_dir       = path.join(results_dir, "Calgary/")
    calgary_path_pickle       = path.join(calgary_results_dir, "13428_2016_720_MOESM2_ESM.pickle")

    mandera_results_csv       = path.join(spp_results_dir, "mandera_regression.csv")
    mandera_results_fa_csv    = path.join(spp_results_dir, "mandera_regression_first_assoc.csv")

    # FIGURES

    figures_dir               = _config.value_by_key_path("results", "figures-directory")
    summary_dir               = _config.value_by_key_path("results", "summary-directory")
