import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import seaborn as sns
import logging
import sys

from utils.common import savesvg

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_parameter_scaling(
        df,
        outdir,
        x="parameters",
        y="gflops",
        xlabel='Model size (non-embedding parameters)',
        ylabel='GFLOPS',
        dataset_size=None,
        palette='muted',
        published_models_only=False,
        xlog=True,
        ylog=True,
        rgb='rgb',
        legend=[
            'Data (x, y, z, t, c)',
            '4D (224, 224, 112, 8, 3)',
            '3D (224, 224, 112, 1, 3)',
            '2D (224, 224, 1, 1, 3)',
            'Patch (x, y, z, t, c)',
            f'(16, 16, 16, 2, 3)',
        ],
        published_models_legend=[
            'Data (x, y, c)',
            '2D (224, 224, 3)',
            'Patch (x, y, c)',
            f'(16, 16, 3)',
        ],
):
    for background in ["default", "dark_background"]:
        plt.style.use(background)
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'axes.autolimit_mode': 'round_numbers'
        })

        fig, ax = plt.subplots(figsize=(8, 8))

        if published_models_only:
            data = df.loc[df['data'].str.match(r'2D\(rgb\)')]
        else:
            data = df.loc[df['data'].str.match(r'.*\(rgb\)')]

        data = data[data['px'] == 16]

        if published_models_only:
            g = sns.lineplot(
                data=data,
                x=x,
                y=y,
                hue='data',
                style="px",
                ax=ax,
                legend=True,
                markers=True,
                palette='Greys_r',
                markeredgecolor='dimgrey' if background == 'default' else 'lightgrey',
                markeredgewidth=.5
            )
        else:
            g = sns.lineplot(
                data=data,
                x=x,
                y=y,
                hue='data',
                hue_order=[f'4D({rgb})', f'3D({rgb})', f'2D({rgb})'],
                style="px",
                ax=ax,
                legend=True,
                markers=True,
                palette=palette,
                markeredgecolor='dimgrey' if background == 'default' else 'lightgrey',
                markeredgewidth=.5
            )

        d = data[(data['data'] == '2D(rgb)') & (data['px'] == 16)]

        for line in range(0, d.shape[0]):
            xx = d[x][line]
            yy = d[y][line]

            if published_models_only:
                if y == 'dataset_size':
                    y_text_offset = 100
                    x_text_offset = xx * .2
                elif y == 'training_volumes':
                    y_text_offset = .5
                    x_text_offset = xx * .1
                else:
                    y_text_offset = yy * .2
                    x_text_offset = xx * .2
            else:
                x_text_offset = 0
                if yy < 10:
                    y_text_offset = yy * .35
                elif yy < 50:
                    y_text_offset = yy * .25
                elif yy < 100:
                    y_text_offset = yy * .15
                else:
                    y_text_offset = yy * .25

            ax.annotate(
                d['class'][line].rstrip('/16'),
                (xx, yy),
                xytext=(xx - x_text_offset, yy + y_text_offset),
                arrowprops=dict(alpha=0),
            )

        if y == 'dataset_size':
            ax.set_ylim(-500, 5000)

        ax.grid(True, which="major", axis='both', lw=.05, ls='-', zorder=0)
        ax.grid(True, which="minor", axis='both', lw=.01, ls='-', zorder=0)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        if xlog:
            ax.set_xscale('log')

        if ylog:
            ax.set_yscale('log')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        legend_handles, _ = g.get_legend_handles_labels()

        if published_models_only:
            ax.legend(
                legend_handles, published_models_legend,
                loc='upper left', ncol=1, title="", frameon=False
            )
        else:
            ax.legend(
                legend_handles, legend,
                loc='upper left', ncol=1, title="", frameon=False
            )

        if dataset_size is not None:
            ax.set_title(f'Dataset: {dataset_size:,} volumes')
            savepath = Path(f'{outdir}/{y}_{dataset_size}_{background}')
        else:
            savepath = Path(f'{outdir}/{y}_{background}')

        plt.savefig(f'{savepath}.pdf', bbox_inches='tight', pad_inches=.25)
        plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
        savesvg(fig, f'{savepath}.svg')


def plot_data_parameter_scaling(
        df,
        outdir,
        x="parameters",
        xlabel='Model size (non-embedding parameters)',
        y="training_gflops_per_volume",
        ylabel="Training GFLOPs per volume",
        ytwin1="training_time_per_volume",
        ytwinlabel1="Training H100 seconds per volume",
        ytwin2=None,
        ytwinlabel2=None,
        ytwin3=None,
        ytwinlabel3=None,
        yscalelabel=None,
        dataset_size=None,
        palette='muted',
        published_models_only=False,
        xlog=True,
        ylog=True,
        patch_size=16,
        cost_h100_per_hr=6,
        rgb='rgb',
        legend=[
            'Data (x, y, z, t, c)',
                '4D (224, 224, 112, 8, 3)',
                '3D (224, 224, 112, 1, 3)',
                '2D (224, 224, 1, 1, 3)',
            'Patch (x, y, z, t, c)',
                f'(16, 16, 16, 2, 3)',
        ],
        published_models_legend=[
            'Data (x, y, c)',
                '2D (224, 224, 3)',
            'Patch (x, y, c)',
                f'(16, 16, 3)',
        ],
):
    for background in ["default", "dark_background"]:
        plt.style.use(background)
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'axes.autolimit_mode': 'round_numbers'
        })
        fig, ax = plt.subplots(figsize=(8, 8))

        if published_models_only:
            data = df.loc[df['data'].str.match(rf'2D\(rgb\)')]
        else:
            data = df.loc[df['data'].str.match(rf'.*\({rgb}\)')]

        data = data[data['px'] == patch_size]

        for ii, (yy, ll, cc, offset) in enumerate(zip(
                [y, ytwin1, ytwin2, ytwin3, ],
                [ylabel, ytwinlabel1, ytwinlabel2, ytwinlabel3],
                # [None, 'olive', 'magenta', 'r'],
                [None, None, None, None],
                [0, 0, .1, .2],
        )):
            if yy is not None:
                if ii == 0:
                    axis = ax
                else:
                    axis = ax.twinx()
                    if ii > 1:
                        axis.spines["right"].set_position(("axes", 1 + offset))

                if published_models_only:
                    g = sns.lineplot(
                        data=data,
                        x=x,
                        y=yy,
                        hue='data',
                        style="px",
                        ax=axis,
                        legend=True,
                        markers=True,
                        palette='Greens',
                        markeredgecolor='dimgrey' if background == 'default' else 'lightgrey',
                        markeredgewidth=.5
                    )
                else:
                    g = sns.lineplot(
                        data=data,
                        x=x,
                        y=yy,
                        hue='data',
                        hue_order=[f'4D({rgb})', f'3D({rgb})', f'2D({rgb})'],
                        style="px",
                        ax=axis,
                        legend=True,
                        markers=True,
                        palette=palette,
                        markeredgecolor='dimgrey' if background == 'default' else 'lightgrey',
                        markeredgewidth=.5
                    )

                axis.patch.set_visible(False)
                plt.setp(axis.spines.values(), visible=False)
                axis.spines["right"].set_visible(True)
                axis.spines["left"].set_visible(True)
                axis.spines["bottom"].set_visible(True)

                if cc is not None:
                    axis.tick_params(axis='y', colors=cc)
                    axis.spines["right"].set_edgecolor(cc)
                    axis.yaxis.label.set_color(cc)

                axis.set_ylabel(ll)
                if ytwin2 is not None and ii != 0:
                    axis.yaxis.set_label_coords(1 + offset, 1.05)

                if ylog:
                    axis.set_yscale('log')

                legend_handles, _ = g.get_legend_handles_labels()

                if published_models_only:
                    axis.legend(
                        legend_handles, published_models_legend,
                        loc='upper left', ncol=1, title="", frameon=False
                    )
                else:
                    axis.legend(
                        legend_handles, legend,
                        loc='upper left', ncol=1, title="", frameon=False
                    )

        if yscalelabel is not None:
            ann = ax.annotate(
                yscalelabel,
                xy=(0, 1.03),
                xycoords='axes fraction',
                clip_on=False,
                ha='center',
                rotation=90
            )

        d = data[(data['data'] == f'2D({rgb})') & (data['px'] == patch_size)]

        for line in range(0, d.shape[0]):
            xx = d[x][line]
            yy = d[y][line]

            if published_models_only:
                if y == 'dataset_size':
                    y_text_offset = 100
                    x_text_offset = xx * .2
                elif y == 'training_volumes':
                    y_text_offset = .5
                    x_text_offset = xx * .1
                else:
                    y_text_offset = yy * .2
                    x_text_offset = xx * .2
            else:
                x_text_offset = 0
                if yy < 10:
                    y_text_offset = yy * .35
                elif yy < 50:
                    y_text_offset = yy * .25
                elif yy < 100:
                    y_text_offset = yy * .15
                else:
                    y_text_offset = yy * .25

            label = d['class'][line].strip(f'/{patch_size}')
            ax.annotate(
                label,
                (xx, yy),
                xytext=(xx - x_text_offset, yy + y_text_offset * (-1 if label == 'G' else 1)),
                arrowprops=dict(alpha=0),
            )

        ax.grid(True, which="major", axis='both', lw=.05, ls='-', zorder=0)
        ax.grid(True, which="minor", axis='both', lw=.01, ls='-', zorder=0)
        ax.set_xlabel(xlabel)

        if xlog:
            ax.set_xscale('log')

        if dataset_size is not None:
            ax.set_title(f'Dataset: {dataset_size:,} volumes')
            savepath = Path(f'{outdir}/{y}_{dataset_size}_{background}')
        else:
            savepath = Path(f'{outdir}/{y}_{background}')

        plt.savefig(f'{savepath}.pdf', bbox_inches='tight', pad_inches=.25)
        plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
        savesvg(fig, f'{savepath}.svg')


def plot_individual_parameters(
    df,
    batch_size,
    outdir,
    cost_h100_per_hr=6,
    rgb='rgb',
    legend=[
        'Data (x, y, z, t, c)',
        '4D (224, 224, 112, 8, 3)',
        '3D (224, 224, 112, 1, 3)',
        '2D (224, 224, 1, 1, 3)',
        'Patch (x, y, z, t, c)',
        f'(16, 16, 16, 2, 3)',
    ],
):
    df["number_h100_for_batch"] = np.ceil(df["model_training_memory"] + (df["memory_per_volume"] * batch_size) / 80)
    df["cost_h100_for_batch"] = df["number_h100_for_batch"] * 37500
    df["training_h100_hours_per_step"] = batch_size * df["training_time_per_volume"] / 3600
    df["training_tflops_per_volume"] = df["training_gflops_per_volume"] / 1000

    fois = {
        f"training_gflops_per_volume": f"Training GFLOPs per volume",
        f"training_time_per_volume": f"Training H100 seconds per volume",
        f"number_h100_for_batch": f"Minimum number of H100s needed for a batch ({batch_size})",
        f"cost_h100_for_batch": f"Cost of H100s needed for a batch ({batch_size}, $37,500 each)",
        f"training_h100_hours_per_step": f"Training H100 hours per batch ({batch_size})",

    }
    for y, ylabel in fois.items():
        plot_parameter_scaling(
            df,
            outdir=outdir,
            x="parameters",
            xlabel="Model size (non-embedding parameters)",
            y=y,
            ylabel=ylabel,
            legend=legend,
        )

    fois = {
        f"training_h100_days_per_epoch": f"Training H100 days per epoch",
        f"training_h100_cost_per_epoch": f"H100 compute cost per epoch ($6/hr)",
        f"training_tflops_per_epoch": f"Training Tera-FLOPs (TFLOPs) per epoch",
    }

    for dataset_size in [1000000, 1281167, 14197122, 10000000, 100000000, 303000000, 1000000000]:
        df["training_h100_days_per_epoch"] = dataset_size * df["training_time_per_volume"] / 3600 / 24
        df["multigpu_training_days_per_epoch"] = df["training_h100_days_per_epoch"] / df["number_h100_for_batch"]
        df["multigpu_256_training_days_per_epoch"] = df["training_h100_days_per_epoch"] / 256
        df[f"training_h100_cost_per_epoch"] = df[f"training_h100_days_per_epoch"] * 24 * cost_h100_per_hr
        df[f"training_tflops_per_epoch"] = dataset_size * df["training_tflops_per_volume"]

        for y, ylabel in fois.items():
            plot_parameter_scaling(
                df,
                outdir=outdir,
                x="parameters",
                xlabel="Model size (non-embedding parameters)",
                y=y,
                ylabel=ylabel,
                dataset_size=dataset_size,
                legend=legend,
            )


def plot_published_models(
    df,
    outdir,
    models,
    cost_h100_per_hr=6,
    published_models_legend=[
        'Data (x, y, c)',
        '2D (224, 224, 3)',
        'Patch (x, y, c)',
        f'(16, 16, 3)',
    ],
):
    batch_size = 4096
    df["number_h100_for_batch"] = np.ceil(df["model_training_memory"] + (df["memory_per_volume"] * batch_size) / 80)
    df["cost_h100_for_batch"] = df["number_h100_for_batch"] * 37500
    df["training_h100_hours_per_step"] = batch_size * df["training_time_per_volume"] / 3600
    df["training_tflops_per_volume"] = df["training_gflops_per_volume"] / 1000

    cols = list(models['S'].keys())
    df[cols] = np.nan
    for k in models.keys():
        idx = df.loc[df['class'].str.match(k)].index
        df.loc[idx, cols] = models[k].values()

    df["training_volumes"] = df["steps"] * df["batch_size"] // 1000000000  # convert to billions
    df["dataset_size"] = df["dataset_size"] // 1000000  # convert to millions
    df["training_compute"] = df[f"training_tflops_per_volume"] * df["batch_size"] * df["steps"]
    df["training_time"] = df[f"training_time_per_volume"] * df["batch_size"] * df["steps"] / 3600 / 24
    df["training_cost"] = df[f"training_time"] * cost_h100_per_hr * 24

    fois = {
        f"dataset_size": f"Training dataset size (millions of volumes)",
        f"training_volumes": f"Training volumes seen (billions)",
        f"training_time": f"Training H100 days",
        f"training_compute": f"Training TFLOPs",
        f"training_cost": f"Training cost",
    }

    for y, ylabel in fois.items():
        plot_parameter_scaling(
            df,
            outdir=outdir,
            x="parameters",
            xlabel="Model size (non-embedding parameters)",
            y=y,
            ylabel=ylabel,
            published_models_only=True,
            ylog=False if y == "dataset_size" or y == "training_volumes" else True,
            published_models_legend=published_models_legend,
        )
