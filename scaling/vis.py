import matplotlib
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt

matplotlib.use("Agg")

import warnings

warnings.filterwarnings("ignore")

import logging
import re
import sys
from pathlib import Path

import numpy as np
import seaborn as sns

from cell_observatory_platform.utils.common import savesvg

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def days_to_formatter(days, pos):

    if days >= 365:
        years = np.ceil(days / 365)
        return f"{years:.0f}y"
    elif days >= 1:
        return f"{days:.0f}d"
    elif days >= 1 / 24:
        hours = np.ceil(days * 24)
        return f"{hours:.0f}h"
    elif days >= 1 / (24 * 60):
        minutes = np.ceil(days * 24 * 60)
        return f"{minutes:.0f}m"
    else:
        seconds = np.ceil(days * 24 * 60 * 60)
        return f"{seconds:.0f}s"


def savesvg(
    fig: plt.Figure,
    savepath: Union[Path, str],
    top: float = 0.9,
    bottom: float = 0.1,
    left: float = 0.1,
    right: float = 0.9,
    hspace: float = 0.35,
    wspace: float = 0.1,
):

    plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
    plt.savefig(savepath, bbox_inches="tight", dpi=300, pad_inches=0.25)

    if Path(savepath).suffix == ".svg":
        # Read in the file
        with open(savepath, "r", encoding="utf-8") as f:
            filedata = f.read()

        # Replace the target string
        filedata = re.sub('height="[0-9]+(\.[0-9]+)pt"', "", filedata)
        filedata = re.sub('width="[0-9]+(\.[0-9]+)pt"', "", filedata)

        # Write the file out again
        with open(savepath, "w", encoding="utf-8") as f:
            f.write(filedata)


def plot_parameter_scaling(
    df,
    outdir,
    x="parameters",
    y="gflops",
    xlabel="Trainable parameters (excluding input and head layers)",
    ylabel="GFLOPS",
    dataset_size=None,
    palette="muted",
    published_models_only=False,
    xlog=True,
    ylog=True,
    rgb="rgb",
    legend=[
        "Data (x, y, z, t, c)",
        "(224, 224, 112, 8, 3)",
        "(224, 224, 112, 1, 3)",
        "(224, 224, 1, 1, 3)",
        "Patch (x, y, z, t, c)",
        "(16, 16, 16, 2, 3)",
        "Model FLOPs Utilization",
        "MFU(0.3)",
        "MFU(0.6)",
        "MFU(0.9)",
    ],
    published_models_legend=[
        "Data (x, y, c)",
        "(224, 224, 3)",
        "Patch (x, y, c)",
        "(16, 16, 3)",
        "Model FLOPs Utilization",
        "MFU(0.3)",
        "MFU(0.6)",
        "MFU(0.9)",
    ],
):
    for background in ["default", "dark_background"]:
        plt.style.use(background)
        plt.rcParams.update(
            {
                #'font.family': 'Helvetica',
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 14,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 12,
                "axes.autolimit_mode": "round_numbers",
            }
        )

        fig, ax = plt.subplots(figsize=(8, 8))

        if published_models_only:
            data = df.loc[df["data"].str.match(r"2D\(rgb\)")]
        else:
            data = df.loc[df["data"].str.match(r".*\(rgb\)")]

        data = data[data["px"] == 16]
        data.reset_index(drop=True, inplace=True)

        if published_models_only:
            g = sns.lineplot(
                data=data,
                x=x,
                y=y,
                hue="data",
                size="px",
                style="mfu",
                ax=ax,
                legend=True,
                markers=True,
                palette="Greys_r",
                markeredgecolor="dimgrey" if background == "default" else "lightgrey",
                markeredgewidth=0.5,
            )
        else:
            g = sns.lineplot(
                data=data,
                x=x,
                y=y,
                hue="data",
                hue_order=[f"4D({rgb})", f"3D({rgb})", f"2D({rgb})"],
                size="px",
                style="mfu",
                ax=ax,
                legend=True,
                markers=True,
                palette=palette,
                markeredgecolor="dimgrey" if background == "default" else "lightgrey",
                markeredgewidth=0.5,
            )

        d = data[(data["data"] == "2D(rgb)") & (data["px"] == 16) & (data["mfu"] == 0.3)]

        for line in range(0, d.shape[0]):
            xx = d[x][line]
            yy = d[y][line]

            if published_models_only:
                if y == "dataset_size":
                    y_text_offset = 100
                    x_text_offset = xx * 0.2
                elif y == "training_volumes":
                    y_text_offset = 0.5
                    x_text_offset = xx * 0.1
                else:
                    y_text_offset = yy * 0.2
                    x_text_offset = xx * 0.2
            else:
                x_text_offset = 0
                if yy < 10:
                    y_text_offset = yy * 0.35
                elif yy < 50:
                    y_text_offset = yy * 0.25
                elif yy < 100:
                    y_text_offset = yy * 0.15
                else:
                    y_text_offset = yy * 0.25

            ax.annotate(
                d["class"][line].rstrip("/16"),
                (xx, yy),
                xytext=(xx - x_text_offset, yy + y_text_offset),
                arrowprops=dict(alpha=0),
            )

        if y == "dataset_size":
            ax.set_ylim(-500, 5000)

        ax.grid(True, which="major", axis="both", lw=0.05, ls="-", zorder=0)
        ax.grid(True, which="minor", axis="both", lw=0.01, ls="-", zorder=0)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        if xlog:
            ax.set_xscale("log")

        if ylog:
            ax.set_yscale("log")

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        legend_handles, _ = g.get_legend_handles_labels()

        if y == "training_time":
            formatter = ticker.FuncFormatter(days_to_formatter)
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.set_minor_formatter(formatter)
            ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=[0.25, 0.5]))

        if published_models_only:
            ax.legend(legend_handles, published_models_legend, loc="upper left", ncol=1, title="", frameon=False)
        else:
            ax.legend(legend_handles, legend, loc="upper left", ncol=1, title="", frameon=False)

        if dataset_size is not None:
            ax.set_title(f"Dataset: {dataset_size:,} volumes")
            savepath = Path(f"{outdir}/{y}_{dataset_size}_{background}")
        else:
            savepath = Path(f"{outdir}/{y}_{background}")

        plt.savefig(f"{savepath}.pdf", bbox_inches="tight", pad_inches=0.25)
        plt.savefig(f"{savepath}.png", dpi=300, bbox_inches="tight", pad_inches=0.25)
        savesvg(fig, f"{savepath}.svg")


def plot_data_parameter_scaling(
    df,
    outdir,
    x="parameters",
    xlabel="Trainable parameters (excluding input and head layers)",
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
    palette="muted",
    published_models_only=False,
    xlog=True,
    ylog=True,
    patch_size=16,
    cost_h100_per_hr=6,
    rgb="rgb",
    legend=[
        "Data (x, y, z, t, c)",
        "(224, 224, 112, 8, 3)",
        "(224, 224, 112, 1, 3)",
        "(224, 224, 1, 1, 3)",
        "Patch (x, y, z, t, c)",
        "(16, 16, 16, 2, 3)",
        "Model FLOPs Utilization",
        "MFU(0.3)",
        "MFU(0.6)",
        "MFU(0.9)",
    ],
    published_models_legend=[
        "Data (x, y, c)",
        "(224, 224, 3)",
        "Patch (x, y, c)",
        "(16, 16, 3)",
        "Model FLOPs Utilization",
        "MFU(0.3)",
        "MFU(0.6)",
        "MFU(0.9)",
    ],
):
    for background in ["default", "dark_background"]:
        plt.style.use(background)
        plt.rcParams.update(
            {
                #'font.family': 'Helvetica',
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 14,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 12,
                "axes.autolimit_mode": "round_numbers",
            }
        )
        fig, ax = plt.subplots(figsize=(8, 8))

        if published_models_only:
            data = df.loc[df["data"].str.match(rf"2D\(rgb\)")]
        else:
            data = df.loc[df["data"].str.match(rf".*\({rgb}\)")]

        data = data[data["px"] == patch_size]

        for ii, (yy, ll, cc, offset) in enumerate(
            zip(
                [
                    y,
                    ytwin1,
                    ytwin2,
                    ytwin3,
                ],
                [ylabel, ytwinlabel1, ytwinlabel2, ytwinlabel3],
                # [None, 'olive', 'magenta', 'r'],
                [None, None, None, None],
                [0, 0, 0.125, 0.25],
            )
        ):
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
                        hue="data",
                        size="px",
                        style="mfu",
                        ax=axis,
                        legend=True,
                        markers=True,
                        palette="Greens",
                        markeredgecolor="dimgrey" if background == "default" else "lightgrey",
                        markeredgewidth=0.5,
                    )
                else:
                    g = sns.lineplot(
                        data=data,
                        x=x,
                        y=yy,
                        hue="data",
                        hue_order=[f"4D({rgb})", f"3D({rgb})", f"2D({rgb})"],
                        size="px",
                        # style="mfu",
                        ax=axis,
                        legend=True,
                        markers=True,
                        palette=palette,
                        markeredgecolor="dimgrey" if background == "default" else "lightgrey",
                        markeredgewidth=0.5,
                    )

                axis.patch.set_visible(False)
                plt.setp(axis.spines.values(), visible=False)
                axis.spines["right"].set_visible(True)
                axis.spines["left"].set_visible(True)
                axis.spines["bottom"].set_visible(True)

                if cc is not None:
                    axis.tick_params(axis="y", colors=cc)
                    axis.spines["right"].set_edgecolor(cc)
                    axis.yaxis.label.set_color(cc)

                ll = ll.replace("days", "time")

                axis.set_ylabel(ll)
                if ytwin2 is not None and ii != 0:
                    axis.yaxis.set_label_coords(1 + offset, 1.07)

                if ylog:
                    axis.set_yscale("log")

                legend_handles, _ = g.get_legend_handles_labels()

                if published_models_only:
                    axis.legend(
                        legend_handles, published_models_legend, loc="upper left", ncol=1, title="", frameon=False
                    )
                else:
                    axis.legend(legend_handles, legend, loc="upper left", ncol=1, title="", frameon=False)

                if y == "training_time" or y.startswith("training_h100_days"):
                    formatter = ticker.FuncFormatter(days_to_formatter)
                    axis.yaxis.set_major_formatter(formatter)
                    axis.yaxis.set_minor_formatter(formatter)
                    axis.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=[0.25, 0.5]))

        if yscalelabel is not None:
            ann = ax.annotate(
                yscalelabel, xy=(0, 1.03), xycoords="axes fraction", clip_on=False, ha="center", rotation=90
            )

        d = data[(data["data"] == f"2D({rgb})") & (data["px"] == patch_size) & (data["mfu"] == 0.3)]

        for line in range(0, d.shape[0]):
            xx = d[x][line]
            yy = d[y][line]

            if published_models_only:
                if y == "dataset_size":
                    y_text_offset = 100
                    x_text_offset = xx * 0.2
                elif y == "training_volumes":
                    y_text_offset = 0.5
                    x_text_offset = xx * 0.1
                else:
                    y_text_offset = yy * 0.2
                    x_text_offset = xx * 0.2
            else:
                x_text_offset = 0
                if yy < 10:
                    y_text_offset = yy * 0.35
                elif yy < 50:
                    y_text_offset = yy * 0.25
                elif yy < 100:
                    y_text_offset = yy * 0.15
                else:
                    y_text_offset = yy * 0.25

            label = d["class"][line].rstrip(f"/{patch_size}")
            ax.annotate(
                label,
                (xx, yy),
                xytext=(xx - x_text_offset, yy + y_text_offset * (-1 if label == "G" else 1)),
                arrowprops=dict(alpha=0),
            )

        ax.grid(True, which="major", axis="both", lw=0.05, ls="-", zorder=0)
        ax.grid(True, which="minor", axis="both", lw=0.01, ls="-", zorder=0)
        ax.set_xlabel(xlabel)

        if xlog:
            ax.set_xscale("log")

        if dataset_size is not None:
            ax.set_title(f"Dataset: {dataset_size:,} volumes")
            savepath = Path(f"{outdir}/{y}_{dataset_size}_{background}")
        else:
            savepath = Path(f"{outdir}/{y}_{background}")

        plt.savefig(f"{savepath}.pdf", bbox_inches="tight", pad_inches=0.25)
        plt.savefig(f"{savepath}.png", dpi=300, bbox_inches="tight", pad_inches=0.25)
        savesvg(fig, f"{savepath}.svg")


def plot_individual_parameters(
    df,
    batch_size,
    outdir,
    cost_h100_per_hr=6,
    rgb="rgb",
    legend=[
        "Data (x, y, z, t, c)",
        "(224, 224, 112, 8, 3)",
        "(224, 224, 112, 1, 3)",
        "(224, 224, 1, 1, 3)",
        "Patch (x, y, z, t, c)",
        "(16, 16, 16, 2, 3)",
        "Model FLOPs Utilization",
        "MFU(0.3)",
        "MFU(0.6)",
        "MFU(0.9)",
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
            xlabel="Trainable parameters (excluding input and head layers)",
            y=y,
            ylabel=ylabel,
            legend=legend,
        )

    fois = {
        f"training_h100_days_per_epoch": f"Training H100 time per epoch",
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
                xlabel="Trainable parameters (excluding input and head layers)",
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
        "Data (x, y, c)",
        "(224, 224, 3)",
        "Patch (x, y, c)",
        "(16, 16, 3)",
        "Model FLOPs Utilization",
        "MFU(0.3)",
        "MFU(0.6)",
        "MFU(0.9)",
    ],
):
    batch_size = 4096
    df["number_h100_for_batch"] = np.ceil(df["model_training_memory"] + (df["memory_per_volume"] * batch_size) / 80)
    df["cost_h100_for_batch"] = df["number_h100_for_batch"] * 37500
    df["training_h100_hours_per_step"] = batch_size * df["training_time_per_volume"] / 3600
    df["training_tflops_per_volume"] = df["training_gflops_per_volume"] / 1000

    cols = list(models["S"].keys())
    df[cols] = np.nan
    for k in models.keys():
        idx = df.loc[df["class"].str.match(k)].index
        df.loc[idx, cols] = models[k].values()

    df["training_volumes"] = df["steps"] * df["batch_size"] // 1000000000  # convert to billions
    df["dataset_size"] = df["dataset_size"] // 1000000  # convert to millions
    df["training_compute"] = df[f"training_tflops_per_volume"] * df["batch_size"] * df["steps"]
    df["training_time"] = df[f"training_time_per_volume"] * df["batch_size"] * df["steps"] / 3600 / 24
    df["training_cost"] = df[f"training_time"] * cost_h100_per_hr * 24

    fois = {
        f"dataset_size": f"Training dataset size (millions of volumes)",
        f"training_volumes": f"Training volumes seen (billions)",
        f"training_time": f"Training H100 time",
        f"training_compute": f"Training TFLOPs",
        f"training_cost": f"Training cost",
    }

    for y, ylabel in fois.items():
        plot_parameter_scaling(
            df,
            outdir=outdir,
            x="parameters",
            xlabel="Trainable parameters (excluding input and head layers)",
            y=y,
            ylabel=ylabel,
            published_models_only=True,
            ylog=False if y == "dataset_size" or y == "training_volumes" else True,
            published_models_legend=published_models_legend,
        )


def plot_powerlaw(outdir):
    for background in ["default", "dark_background"]:
        plt.style.use(background)

        plt.rcParams.update(
            {
                #'font.family': 'Helvetica',
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 14,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 12,
                "axes.autolimit_mode": "round_numbers",
                "hatch.color": "k",
            }
        )

        x = np.logspace(0, 12, 10)
        exponents = np.arange(0, 1.1, 0.1).tolist()
        cmap = plt.get_cmap("nipy_spectral_r")
        colors = [cmap(i / (len(exponents) - 1)) for i in range(len(exponents))]

        fig, (ax, axe) = plt.subplots(figsize=(14, 6), ncols=2)
        opt = 6
        epsilon = 2
        baseline = 6

        for i, a in enumerate(exponents):
            lx = x ** (-a)
            ex = 100 * x ** (-a)

            try:
                idx = sum(ex <= opt - 1)
                if idx > 0:
                    ex[len(x) - idx :] = (opt - 1) + np.random.rand(idx)

                    if a == 0.2:
                        axe.annotate(
                            f"Diminishing returns for $x \\to \infty$" if a == 0.2 else f"",
                            xy=(x[idx], opt),
                            xytext=(0, -45) if a == 0.2 else (0, -35),
                            textcoords="offset points",
                            arrowprops=dict(arrowstyle="->", color=colors[i]),
                            color=colors[i],
                            ha="center",
                            va="center",
                            zorder=15,
                        )
            except IndexError:
                pass

            ax.loglog(x, lx, label=f"α={round(a, 2)}", color=colors[i])
            axe.loglog(x, ex, color=colors[i])

            if a == 0.1:
                for xx in range(baseline, 14, 1):
                    bvv = 100 * (10**baseline) ** (-a)
                    vv = 100 * (10**xx) ** (-a)
                    ifold = bvv / vv
                    xfold = 10**xx / 10**baseline

                    axe.scatter(10**xx, vv, color=colors[i], zorder=30)
                    axe.annotate(
                        f"$\\times${ifold:.1f}",
                        xy=(10**xx, vv),
                        xytext=(5, 15),
                        textcoords="offset points",
                        ha="center",
                        va="center",
                        color=colors[i],
                        zorder=20,
                    )
                    # axe.annotate(
                    #     f'$\\times10^{{{np.log10(xfold):.0f}}}x$' if xx >= 9 else f'$\\times{xfold:.0f}x$',
                    #     xy=(10**xx, vv),
                    #     xytext=(5, 30),
                    #     textcoords='offset points',
                    #     ha='center',
                    #     va='center',
                    #     color='k',
                    #     zorder=20
                    # )
                    axe.vlines(10**xx, 0, vv, color="gray", linestyle="--", linewidth=0.5, zorder=20)

        ax.set_xlabel("$x$")
        axe.set_xlabel("$x$")
        ax.set_ylabel("Pretraining $L(x)$")
        axe.set_ylabel("Benchmark $E(x)$")

        ax.spines["right"].set_visible(False)
        axe.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        axe.spines["top"].set_visible(False)

        ax.set_xlim(x.min(), x.max())
        axe.set_xlim(x.min(), x.max())
        ax.set_ylim(None, 1)
        axe.set_ylim(1, 100)
        axe.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 80, 100])
        axe.set_yticklabels(
            ["0", "$\\epsilon$", "", "", "", "", "", "", "", "10", "20", "30", "40", "50", "60", "80", "100"]
        )

        axe.fill_between(x, epsilon, opt, zorder=10, color="whitesmoke", alpha=0.5)
        axe.axhline(opt + 0.25, color="k", linestyle="--", zorder=10)
        axe.fill_between(x, 0, epsilon, zorder=10, hatch="/")
        axe.axhline(epsilon, color="k", linestyle=":", zorder=10)

        ax.legend(title="$L(x) = x^{{-\\alpha}}$", loc="lower left", frameon=False)
        axe.legend(title="$E(x) = 100 \\cdot x^{{-\\alpha}}$", loc="upper right", frameon=False)
        # ax.annotate(
        #     '',
        #     xy=(.26, .01),
        #     xytext=(.26, .6),
        #     xycoords='axes fraction',
        #     textcoords='axes fraction',
        #     arrowprops=dict(arrowstyle='->', color='black', linewidth=2),
        #     ha='center',
        #     va='center',
        #     rotation=90
        # )
        # ax.annotate(
        #     'Faster rates of diminishing returns',
        #     xy=(.28, .01),
        #     xytext=(.28, .3),
        #     xycoords='axes fraction',
        #     textcoords='axes fraction',
        #     ha='center',
        #     va='center',
        #     rotation=270,
        #     fontsize=10
        # )
        axe.annotate(
            "Saturation",
            xy=(-0.03, 0.01),
            xytext=(-0.03, 0.3),
            xycoords="axes fraction",
            textcoords="axes fraction",
            ha="center",
            va="center",
            rotation=90,
            fontsize=10,
        )
        axe.annotate(
            "Irreducible error",
            xy=(0.01, 0.01),
            xytext=(0.01, 0.17),
            xycoords="axes fraction",
            textcoords="axes fraction",
            color="k",
            ha="left",
            va="center",
            zorder=10,
            fontsize=12,
        )
        ax.grid(True, which="both", axis="both", lw=0.05, ls="-", zorder=0)
        axe.grid(True, which="both", axis="both", lw=0.05, ls="-", zorder=0)
        plt.savefig(f"{outdir}/powerlaw_{background}.pdf", bbox_inches="tight", pad_inches=0.25)
        plt.savefig(f"{outdir}/powerlaw_{background}.png", dpi=300, bbox_inches="tight", pad_inches=0.25)
        savesvg(fig, f"{outdir}/powerlaw_{background}.svg")


def plot_gpt_vit(outdir):
    for background in ["default", "dark_background"]:
        plt.style.use(background)

        plt.rcParams.update(
            {
                #'font.family': 'Helvetica',
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 14,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 12,
                "axes.autolimit_mode": "round_numbers",
                "hatch.color": "k",
            }
        )

        x = np.logspace(0, 12, 10)

        fig, (ax, axg, axe) = plt.subplots(figsize=(18, 6), ncols=3)

        ax.loglog(x, (x / 2.3 * 10**8) ** (-0.048), label="$L = {\dfrac{C}{2.3 \cdot 10^{8}}}^{{-0.048}}$", color="C0")
        axg.loglog(
            x,
            2.64 + (x / 1.6 * 10**-8) ** (-0.16),
            label="$L = 2.64 + {\dfrac{C}{1.6 \cdot 10^{-8}}}^{{-0.16}}$",
            color="C1",
        )
        axe.loglog(
            x, 100 * (0.09 + 0.26 * (x + 0.01) ** (-0.35)), label="$E = 0.09 + 0.26 (C + 0.01)^{{-0.35}}$", color="C2"
        )

        axe.set_ylabel("ImageNet finetune error rate (%)")
        axe.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))

        ax.set_xlabel("Compute")
        axg.set_xlabel("Compute")
        axe.set_xlabel("Compute")
        ax.set_ylabel("Loss")
        axg.set_ylabel("Loss")
        axe.set_ylabel("ImageNet finetune error rate")

        ax.spines["right"].set_visible(False)
        axg.spines["right"].set_visible(False)
        axe.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        axg.spines["top"].set_visible(False)
        axe.spines["top"].set_visible(False)

        ax.set_xlim(x.min(), x.max())
        axe.set_xlim(x.min(), x.max())
        ax.set_ylim(10**-1, 1)

        axg.set_ylim(1, 10**2)

        axe.set_yticks([5, 6, 7, 8, 9, 10, 20, 30, 40, 50])
        axe.set_ylim(5, 50)

        ax.legend(title="GPT [Kaplan et al. 2020] (Language)", loc="upper left", frameon=False)
        axg.legend(title="GPT [Henighan et al. 2020] (Image/16)", loc="upper left", frameon=False)
        axe.legend(title="ViT [Zhai et al. 2022] (Image/16)", loc="upper left", frameon=False)

        ax.grid(True, which="both", axis="both", lw=0.05, ls="-", zorder=0)
        axg.grid(True, which="both", axis="both", lw=0.05, ls="-", zorder=0)
        axe.grid(True, which="both", axis="both", lw=0.05, ls="-", zorder=0)

        plt.savefig(f"{outdir}/gpt_vit_{background}.pdf", bbox_inches="tight", pad_inches=0.25)
        plt.savefig(f"{outdir}/gpt_vit_{background}.png", dpi=300, bbox_inches="tight", pad_inches=0.25)
        savesvg(fig, f"{outdir}/gpt_vit_{background}.svg")
