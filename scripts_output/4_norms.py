"""
===========================
Figures for similarity judgement tests.
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
import os
import sys

from glob import glob

import numpy
import pandas
import seaborn

from matplotlib import pyplot

from ..core.evaluation.association import ColourAssociation
from ..core.utils.logging import log_message, date_format
from ..core.utils.maths import DistanceType, CorrelationType
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def ensure_column_safety(df: pandas.DataFrame) -> pandas.DataFrame:
    return df.rename(columns=lambda col_name: col_name.replace(" ", "_").lower())


# TODO: essentially duplicated code
def main():

    test_name = ColourAssociation().name

    colour_results_df = load_data()
    colour_results_df = ensure_column_safety(colour_results_df)

    colour_results_df["model"] = colour_results_df.apply(
        lambda r:
        f"{r['corpus']} {r['distance_type']} {r['model_type']} {r['embedding_size']}"
        if not numpy.math.isnan(r['embedding_size'])
        else f"{r['corpus']} {r['distance_type']} {r['model_type']}",
        axis=1
    )

    logger.info(f"Making score-vs-radius figures for {test_name}")
    figures_score_vs_radius(colour_results_df, test_name)

    for radius in Preferences.window_radii:
        for distance_type in DistanceType:
            for correlation_type in CorrelationType:
                logger.info(f"Making model performance bargraph figures for r={radius}, d={distance_type.name}, c={correlation_type.name}")
                model_performance_bar_graphs(colour_results_df, window_radius=radius, distance_type=distance_type, correlation_type=correlation_type)


def model_performance_bar_graphs(colour_results_df: pandas.DataFrame, window_radius: int, distance_type: DistanceType, correlation_type: CorrelationType):

    figures_dir = Preferences.figures_dir
    seaborn.set_style("ticks")

    filtered_df: pandas.DataFrame = colour_results_df.copy()
    filtered_df = filtered_df[filtered_df["window_radius"] == window_radius]
    filtered_df = filtered_df[filtered_df["distance_type"] == distance_type.name]
    filtered_df = filtered_df[filtered_df["correlation_type"] == correlation_type.name]

    # Use absolute values of correlation
    filtered_df["correlation"] = abs(filtered_df["correlation"])

    # Model name doesn't need to include corpus or distance, since those are fixed
    filtered_df["model_name"] = filtered_df.apply(
        lambda r:
        f"{r['model_type']} {r['embedding_size']}"
        if not numpy.math.isnan(r['embedding_size'])
        else f"{r['model_type']}",
        axis=1
    )

    seaborn.set_context(context="paper", font_scale=1)
    grid = seaborn.FacetGrid(
        filtered_df,
        row="test_name", col="corpus",
        margin_titles=True,
        size=2.5,
        ylim=(0, 1))

    grid.set_xticklabels(rotation=-90)

    # Plot the bars
    plot = grid.map(seaborn.barplot, "model_name", "correlation", order=[
        "log n-gram",
        "Conditional probability",
        "Probability ratio",
        "PPMI",
        "Skip-gram 50",
        "Skip-gram 100",
        "Skip-gram 200",
        "Skip-gram 300",
        "Skip-gram 500",
        "CBOW 50",
        "CBOW 100",
        "CBOW 200",
        "CBOW 300",
        "CBOW 500",
    ])

    # TODO: this isn't working for some reason
    # Remove the "corpus = " from the titles
    # grid.set_titles(col_template='{col_name}', row_template="{row_name}")

    grid.set_ylabels("Correlation")

    pyplot.subplots_adjust(top=0.92)
    grid.fig.suptitle(f"Model {correlation_type.name} correlations for radius {window_radius} using {distance_type.name} distance")

    figure_name = f"similarity r={window_radius} {distance_type.name} corr={correlation_type.name}.png"

    # I don't know why PyCharm doesn't find this... it works...
    # noinspection PyUnresolvedReferences
    plot.savefig(os.path.join(figures_dir, figure_name), dpi=300)


def figures_score_vs_radius(colour_results_df: pandas.DataFrame, test_name: str):
    figures_dir = Preferences.figures_dir
    for distance in [d.name for d in DistanceType]:
        for corpus in ["BNC", "BBC", "UKWAC"]:
            figure_name = f"similarity {test_name} {corpus} {distance}.png"

            filtered_df: pandas.DataFrame = colour_results_df.copy()
            filtered_df = filtered_df[filtered_df["corpus"] == corpus]
            filtered_df = filtered_df[filtered_df["distance_type"] == distance]
            filtered_df = filtered_df[filtered_df["test_name"] == test_name]

            filtered_df = filtered_df.sort_values(by=["model", "window_radius"])
            filtered_df = filtered_df.reset_index(drop=True)

            filtered_df = filtered_df[[
                "model",
                "window_radius",
                "correlation"
            ]]

            plot = seaborn.factorplot(data=filtered_df,
                                      x="window_radius", y="correlation",
                                      hue="model",
                                      size=7, aspect=1.8,
                                      legend=False)

            plot.set(ylim=(-1, 1))

            # Put the legend out of the figure
            # resize figure box to -> put the legend out of the figure
            plot_box = plot.ax.get_position()  # get position of figure
            plot.ax.set_position([plot_box.x0, plot_box.y0, plot_box.width * 0.75, plot_box.height])  # resize position

            # Put a legend to the right side
            plot.ax.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), ncol=1)

            plot.savefig(os.path.join(figures_dir, figure_name))


def load_data() -> pandas.DataFrame:
    """
    Load a pandas.DataFrame from a collection of CSV fragments.
    """
    results_dir = Preferences.association_results_dir
    separator = ","

    header_filename = os.path.join(results_dir, " header.csv")
    data_filenames = glob(os.path.join(results_dir, "*.csv"))
    data_filenames.remove(header_filename)

    with open(os.path.join(results_dir, " header.csv"), mode="r", encoding="utf-8") as header_file:
        column_names = header_file.read().strip().split(separator)

    data = pandas.DataFrame(columns=column_names)

    for data_filename in data_filenames:
        partial_df = pandas.read_csv(data_filename, sep=separator, names=column_names)
        data = data.append(partial_df, ignore_index=True)

    return data


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
