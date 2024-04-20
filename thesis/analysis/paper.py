from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns

from thesis.constant import (
    ANALYSIS,
    CWD,
    FIGURES,
    HEIGHT,
    WIDTH
)


def plot_sixty_percent_overlap(
    dataframe: pd.DataFrame,
    save: bool = False
) -> None:
    amount = dataframe[dataframe.overlap == 60].amount.tolist()
    accuracies = dataframe[dataframe.overlap == 60].overall.tolist()

    figsize = (12, 6)

    fig = plt.figure(figsize=figsize)
    plt.plot(amount, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Model Accuracy vs. Number of Digits with 60\\% Overlap')
    plt.xlabel('Number of Digits')
    plt.ylabel('Model Accuracy (\\%)')
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)

    minimum = min(accuracies)
    maximum = max(accuracies)

    plt.ylim([
        minimum - minimum % 5,
        maximum + (5 - maximum % 5)
    ])

    tick = np.arange(
        minimum - minimum % 5,
        maximum + (5 - maximum % 5) + 1,
        5
    )

    plt.yticks(tick)

    filename = '60_overlap_accuracy.png'
    path = FIGURES.joinpath(filename)

    if save:
        plt.savefig(
            path,
            bbox_inches='tight',
            dpi=300,
            format='png'
        )
    else:
        figure_width, figure_height = fig.get_size_inches() * fig.dpi

        x = (WIDTH - figure_width) // 2
        y = (HEIGHT - figure_height) // 2
        y = y - 50

        plt.get_current_fig_manager().window.wm_geometry(f"+{int(x)}+{int(y)}")

        plt.tight_layout()
        plt.show()

    plt.close()


def plot_seventy_percent_overlap(
    dataframe: pd.DataFrame,
    save: bool = False
) -> None:
    amount = dataframe[dataframe.overlap == 70].amount.tolist()
    accuracies = dataframe[dataframe.overlap == 70].overall.tolist()

    figsize = (12, 6)

    fig = plt.figure(figsize=figsize)
    plt.plot(amount, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Model Accuracy vs. Number of Digits with 70\\% Overlap')
    plt.xlabel('Number of Digits')
    plt.ylabel('Model Accuracy (\\%)')
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)

    minimum = min(accuracies)
    maximum = max(accuracies)

    plt.ylim([
        minimum - minimum % 5,
        maximum + (5 - maximum % 5)
    ])

    tick = np.arange(
        minimum - minimum % 5,
        maximum + (5 - maximum % 5) + 1,
        5
    )

    plt.yticks(tick)

    filename = '70_overlap_accuracy.png'
    path = FIGURES.joinpath(filename)

    if save:
        plt.savefig(
            path,
            bbox_inches='tight',
            dpi=300,
            format='png'
        )
    else:
        figure_width, figure_height = fig.get_size_inches() * fig.dpi

        x = (WIDTH - figure_width) // 2
        y = (HEIGHT - figure_height) // 2
        y = y - 50

        plt.get_current_fig_manager().window.wm_geometry(f"+{int(x)}+{int(y)}")

        plt.tight_layout()
        plt.show()

    plt.close()


def plot_accuracies_across_overlaps(
    dataframe: pd.DataFrame,
    save: bool = False
) -> None:
    figsize = (12, 6)

    fig = plt.figure(figsize=figsize)
    sns.boxplot(x='overlap', y='overall', data=dataframe)
    plt.title('Accuracies Across Different Overlap Rates')
    plt.xlabel('Overlap Rate (\\%)')
    plt.ylabel('Model Accuracy (\\%)')

    filename = 'accuracies_across_different_overlap_rates.png'
    path = FIGURES.joinpath(filename)

    if save:
        plt.savefig(
            path,
            bbox_inches='tight',
            dpi=300,
            format='png'
        )
    else:
        figure_width, figure_height = fig.get_size_inches() * fig.dpi

        x = (WIDTH - figure_width) // 2
        y = (HEIGHT - figure_height) // 2
        y = y - 50

        plt.get_current_fig_manager().window.wm_geometry(f"+{int(x)}+{int(y)}")

        plt.tight_layout()
        plt.show()

    plt.close()


def plot_accuracies_across_amounts(
    dataframe: pd.DataFrame,
    save: bool = False
) -> None:
    figsize = (12, 6)

    fig = plt.figure(figsize=figsize)
    sns.boxplot(x='amount', y='overall', data=dataframe)
    plt.title('Accuracies vs Number of Digits')
    plt.xlabel('Number of Digits')
    plt.ylabel('Model Accuracy (\\%)')

    filename = 'accuracies_across_different_amounts.png'
    path = FIGURES.joinpath(filename)

    if save:
        plt.savefig(
            path,
            bbox_inches='tight',
            dpi=300,
            format='png'
        )
    else:
        figure_width, figure_height = fig.get_size_inches() * fig.dpi

        x = (WIDTH - figure_width) // 2
        y = (HEIGHT - figure_height) // 2
        y = y - 50

        plt.get_current_fig_manager().window.wm_geometry(f"+{int(x)}+{int(y)}")

        plt.tight_layout()
        plt.show()

    plt.close()


def plot_accuracies_individual_amount_overlap(
    dataframe: pd.DataFrame,
    save: bool = False
) -> None:
    figsize = (12, 6)

    fig = plt.figure(figsize=figsize)

    sns.lineplot(
        x='amount',
        y='overall',
        hue='overlap',
        data=dataframe,
        palette='tab20',
        marker='o'
    )

    plt.title('Accuracies Across Individual Number of Digits and Overlap Rates')
    plt.xlabel('Number of Digits')
    plt.ylabel('Model Accuracy (\\%)')
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)

    plt.legend(
        bbox_to_anchor=(1, 1),
        loc='upper left',
        title='Overlap Rate (\\%)'
    )

    filename = 'accuracies_across_individual_amount_and_overlap_rates.png'
    path = FIGURES.joinpath(filename)

    if save:
        plt.savefig(
            path,
            bbox_inches='tight',
            dpi=300,
            format='png'
        )
    else:
        figure_width, figure_height = fig.get_size_inches() * fig.dpi

        x = (WIDTH - figure_width) // 2
        y = (HEIGHT - figure_height) // 2
        y = y - 50

        plt.get_current_fig_manager().window.wm_geometry(f"+{int(x)}+{int(y)}")

        plt.tight_layout()
        plt.show()

    plt.close()


def plot_heatmap_accuracy_amount_overlap(
    dataframe: pd.DataFrame,
    save: bool = False
) -> None:
    heatmap = dataframe.pivot_table(index='amount', columns='overlap', values='overall')
    heatmap = heatmap.reindex(sorted(heatmap.columns), axis=1)

    figsize = (15, 8)

    fig = plt.figure(figsize=figsize)
    ax = sns.heatmap(heatmap, annot=True, fmt='.2f', cmap='YlGnBu')
    ax.tick_params(axis='both', which='both', length=0)

    plt.title('Model Accuracy vs. Number of Digits and Overlap Rate')
    plt.xlabel('Overlap Rate (\\%)')
    plt.ylabel('Number of Digits')

    filename = 'heatmap_of_overall_accuracy.png'
    path = FIGURES.joinpath(filename)

    if save:
        plt.savefig(
            path,
            bbox_inches='tight',
            dpi=300,
            format='png'
        )
    else:
        figure_width, figure_height = fig.get_size_inches() * fig.dpi

        x = (WIDTH - figure_width) // 2
        y = (HEIGHT - figure_height) // 2
        y = y - 50

        plt.get_current_fig_manager().window.wm_geometry(f"+{int(x)}+{int(y)}")

        plt.tight_layout()
        plt.show()

    plt.close()


def plot_violin_accuracy_distributions(
    dataframe: pd.DataFrame,
    save: bool = False
) -> None:
    figsize = (15, 8)

    fig = plt.figure(figsize=figsize)
    sns.violinplot(x='overlap', y='overall', data=dataframe)
    plt.title('Accuracy Distributions Across Overlap Rates')
    plt.xlabel('Overlap Rate (\\%)')
    plt.ylabel('Model Accuracy (\\%)')

    filename = 'accuracy_distributions_across_overlap_rates.png'
    path = FIGURES.joinpath(filename)

    if save:
        plt.savefig(
            path,
            bbox_inches='tight',
            dpi=300,
            format='png'
        )
    else:
        figure_width, figure_height = fig.get_size_inches() * fig.dpi

        x = (WIDTH - figure_width) // 2
        y = (HEIGHT - figure_height) // 2
        y = y - 50

        plt.get_current_fig_manager().window.wm_geometry(f"+{int(x)}+{int(y)}")

        plt.tight_layout()
        plt.show()

    plt.close()


def plot_bar_mean_accuracies(
    dataframe: pd.DataFrame,
    save: bool = False
) -> None:
    figsize = (15, 8)

    fig = plt.figure(figsize=figsize)
    sns.barplot(x='overlap', y='overall', data=dataframe)
    plt.title('Mean Accuracies vs. Overlap Rates')
    plt.xlabel('Overlap Rate (\\%)')
    plt.ylabel('Mean Model Accuracy (\\%)')

    filename = 'mean_accuracies_across_overlap_rates.png'
    path = FIGURES.joinpath(filename)

    if save:
        plt.savefig(
            path,
            bbox_inches='tight',
            dpi=300,
            format='png'
        )
    else:
        figure_width, figure_height = fig.get_size_inches() * fig.dpi

        x = (WIDTH - figure_width) // 2
        y = (HEIGHT - figure_height) // 2
        y = y - 50

        plt.get_current_fig_manager().window.wm_geometry(f"+{int(x)}+{int(y)}")

        plt.tight_layout()
        plt.show()

    plt.close()


def plot_accuracies(
    dataframe: pd.DataFrame,
    save: bool = False
) -> None:
    column = ['file', 'target', 'prediction', 'accuracy']
    dataframe = dataframe.drop(column, axis=1).reset_index()

    column = ['amount', 'overlap']
    dataframe = dataframe.groupby(column).mean().reset_index()

    figsize = (15, 8)

    fig = plt.figure(figsize=figsize)

    sns.lineplot(
        x='overlap',
        y='overall',
        hue='amount',
        data=dataframe,
        marker='o',
        palette='tab20'
    )

    plt.title('Model Accuracy vs. Overlap Rate for Different Number of Digits')
    plt.xlabel('Overlap Rate (\\%)')
    plt.ylabel('Model Accuracy (\\%)')
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)

    plt.legend(
        bbox_to_anchor=(1, 1),
        loc='upper left',
        title='Number of Digits'
    )

    filename = 'accuracy_across_digits.png'
    path = FIGURES.joinpath(filename)

    if save:
        plt.savefig(
            path,
            bbox_inches='tight',
            dpi=300,
            format='png'
        )
    else:
        figure_width, figure_height = fig.get_size_inches() * fig.dpi

        x = (WIDTH - figure_width) // 2
        y = (HEIGHT - figure_height) // 2
        y = y - 50

        plt.get_current_fig_manager().window.wm_geometry(f"+{int(x)}+{int(y)}")

        plt.tight_layout()
        plt.show()

    plt.close()


def main() -> None:
    save = False
    theme = 'dark'

    sns.set_theme(style='darkgrid', context='talk')
    plt.style.use('science')

    if theme == 'dark':
        path = CWD.joinpath('thesis', 'dark.mplstyle')
        plt.style.use(path)

    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16

    csv = ANALYSIS.joinpath('cassd.csv')
    dataframe = pd.read_csv(csv, index_col=0)

    dataframe['amount'] = dataframe['amount'] + 1
    dataframe['overall'] = dataframe['overall'] * 100
    dataframe['overall'] = dataframe['overall'].round(2)

    plot_accuracies(dataframe, save=save)
    plot_sixty_percent_overlap(dataframe, save=save)
    plot_seventy_percent_overlap(dataframe, save=save)
    plot_accuracies_across_overlaps(dataframe, save=save)
    plot_accuracies_across_amounts(dataframe, save=save)
    plot_accuracies_individual_amount_overlap(dataframe, save=save)
    plot_heatmap_accuracy_amount_overlap(dataframe, save=save)
    plot_violin_accuracy_distributions(dataframe, save=save)
    plot_bar_mean_accuracies(dataframe, save=save)


if __name__ == '__main__':
    main()
