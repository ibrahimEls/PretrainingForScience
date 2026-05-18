"""Plotting functions"""

import logging
from typing import Union

import awkward as ak
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import vector
from cycler import cycler
from matplotlib.lines import Line2D

import omnilearn_lightning.plotting.utils as plot_utils
from omnilearn_lightning.array_utils import combine_ak_arrays

from .histogram_utils import hist_ratio, hist_w_unc

vector.register_awkward()
logger = logging.getLogger(__name__)


def binclip(x, bins, dropinf=False):
    binfirst_center = bins[0] + (bins[1] - bins[0]) / 2
    binlast_center = bins[-2] + (bins[-1] - bins[-2]) / 2
    if dropinf:
        print("Dropping inf")
        print("len(x) before:", len(x))
        x = x[~np.isinf(x)]
        print("len(x) after:", len(x))
    return np.clip(x, binfirst_center, binlast_center)


DEFAULT_COLORS = [
    # "steelblue",
    # "orange",
    # "forestgreen",
    # "purple",
    # "firebrick",
    # "lightseagreen",
    # "yellowgreen",
    # "hotpink",
    # "dimgrey",
    # "olive",
    # list of 10 colors based on table 1 in https://arxiv.org/pdf/2107.02270
    "#3f90da",  # blue
    "#bd1f01",  # red
    "#ffa90e",  # orange (yellow-ish)
    "#832db6",  # purple
    "#b9ac70",  # olive
    "#92dadd",  # light blue
    # "#817175",  # grey (darker)
    "#a96b59",  # brown
    "#E76300",  # orange (red-ish)
    # "#a4a294",  # grey
]
DEFAULT_ALPHA = 0.95
rcParams = mpl.rcParams

params_to_update = {
    # --- axes ---
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    "axes.prop_cycle": cycler(
        "color",
        [
            mpl.colors.ColorConverter().to_rgba(col, DEFAULT_ALPHA)
            for col in DEFAULT_COLORS
        ],
    ),
    # ----------------------
    # Increase display resolution
    # "figure.dpi": 200,
    # Line styles
    "lines.linewidth": 1.5,
    # prevents lines from being jagged
    "lines.antialiased": True,
    #
    "lines.dashed_pattern": (2.8, 1.5),
    "lines.dashdot_pattern": (4.8, 1.5, 0.8, 1.5),
    # I have no idea what this does
    "lines.dotted_pattern": (1.1, 1.1),
    "lines.scale_dashes": True,
    # Default colors
    # 'axes.prop_cycle': cycler('color',['cornflowerblue','forestgreen','maroon','goldenrod','firebrick','mediumorchid','lightslategrey','darksalmon','olive'])
    # Fonts
    "font.family": "serif",
    "font.serif": "DejaVu Serif",
    "font.sans-serif": "DejaVu Sans",
    # "text.usetex": False,
    # Axes
    # "axes.linewidth": 1.0,
    # # "axes.labelsize": 25,
    # "axes.labelpad": 9.0,
    # # Title
    # "axes.titlepad": 10.0,
    # # Tick marks - the essence of life
    "xtick.top": True,
    "xtick.major.size": 5,
    "xtick.minor.size": 2.5,
    "xtick.major.width": 1.0,
    "xtick.minor.width": 0.75,
    "xtick.major.pad": 8,
    # "xtick.labelsize": 22,
    # default in mpl v2.0 is 'out'
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.right": True,
    "ytick.major.size": 5,
    "ytick.minor.size": 2.5,
    "ytick.major.width": 1.0,
    "ytick.minor.width": 0.75,
    "ytick.major.pad": 8,
    # "ytick.labelsize": 22,
    "ytick.minor.visible": True,
    # # Error bar plots
    # # default in mpl v2.0 is no caps on error bars
    # # "errorbar.capsize": 3,
    # # Legend
    # # "legend.fontsize": 22,
    "legend.frameon": True,
    "legend.framealpha": 0.8,
    "legend.edgecolor": "black",
    "legend.fancybox": True,
    "legend.borderpad": 0.4,  # border whitespace
    "legend.labelspacing": 0.5,  # the vertical space between the legend entries
    "legend.handlelength": 1.5,  # the length of the legend lines
    "legend.handleheight": 0.7,  # the height of the legend handle
    "legend.handletextpad": 0.5,  # the space between the legend line and legend text
    "legend.borderaxespad": 0.5,  # the border between the axes and legend edge
    "legend.columnspacing": 2.0,  # column separation
    # # "figure.figsize": (12, 8),
    "savefig.bbox": "tight",
    # # "savefig.pad_inches": 0.1,
    "savefig.dpi": 200,  # higher-res than default 100 dpi
}

MODE_STYLE = {
    "classifier": {
        "name": "Classifier",
        "color": "#3f90da",
    },
    "generator": {
        "name": "Generator",
        "color": "#bd1f01",
    },
    "mpmregress": {
        "name": "MPM",
        "color": "#ffa90e",
    },
    "classifier+generator": {
        "name": "Classifier + Generator",
        "color": "#832db6",
    },
    "classifier+generator-preturb": {
        "name": "Classifier + Generator",
        "color": "#832db6",
    },
    "classifier+mpmregress_perturb": {
        "name": "Classifier + MPM.",
        "color": "#b9ac70",
    },
    "generator+mpmregress": {
        "name": "Generator + MPM",
        "color": "#92dadd",
    },
    "pretrainregress_perturb": {
        "name": "Classifier + Generator + MPM",
        "color": "#a96b59",
    },
    "from_scratch": {
        "name": "From Scratch",
        "color": "#000000",
    },
}


def reset_mpl_style():
    """Reset matplotlib rcParams to default."""
    rcParams.update(mpl.rcParamsDefault)


def set_mpl_style(darkmode=False):
    """Set matplotlib rcParams to custom configuration."""
    reset_mpl_style()
    rcParams.update(params_to_update)


def get_bin_centers_and_bin_heights_from_hist(hist):
    """Return the bin centers and bin heights from a histogram.

    Parameters
    ----------
    hist : tuple
        The output of matplotlib hist.

    Returns
    -------
    bin_centers : array-like
        The bin centers.
    bin_heights : array-like
        The bin heights.
    """
    bin_centers = (hist[1][:-1] + hist[1][1:]) / 2
    bin_heights = hist[0]
    return bin_centers, bin_heights


def plot_features(
    ak_array_dict,
    names=None,
    label_prefix=None,
    flatten: Union[bool, list] = True,
    histkwargs: dict = None,
    legend_only_on: int = None,
    legend_kwargs: dict = {},
    fig_legend_kwargs: dict = None,
    ak_array_dict_labels: dict = None,
    ax_rows: int = 1,
    decorate_ax_kwargs: dict = {},
    bins_dict: dict = None,
    logscale_features: list[str] = None,
    colors: Union[list[str], dict[str, str]] = None,
    linestyles: Union[list[str], dict[str, str]] = None,
    ax_size: tuple[int, int] = (3, 2),
    ylabel: str = None,
    ratio: bool = False,
    ratio_range: tuple[float, float] = (0.5, 1.8),
    normed: bool = True,
    gridspec_hspace: float = 0.0,
    gridspec_wspace: float = 0.4,
    ratio_references: dict = None,
):
    """Plot the features of the constituents or jets.

    Parameters
    ----------
    ak_array_dict : dict of awkward array
        Dict with {"name": ak.Array, ...} of the constituents or jets to plot.
    names : list of str or dict, optional
        Names of the features to plot. Either a list of names, or a dict of {"name": "label", ...}.
    label_prefix : str, optional
        Prefix for the plot x-axis labels.
    flatten : bool or list, optional
        Whether to flatten the arrays before plotting. Default is True.
        If a list is given, only flatten the features in the list.
    histkwargs : dict, optional
        Keyword arguments passed to plt.hist.
    legend_only_on : int, optional
        Plot the legend only on the i-th subplot. Default is None.
        Ignored if fig_legend_kwargs is not None.
    legend_kwargs : dict, optional
        Keyword arguments passed to ax.legend.
        Ignored if fig_legend_kwargs is not None.
    fig_legend_kwargs : dict, optional
        Keyword arguments passed to fig.legend to create a figure-level legend.
        If not None, a single legend will be created for the entire figure instead
        of per-axis legends. The function automatically provides 'handles' and 'labels',
        so you can specify other kwargs like 'loc', 'ncol', 'bbox_to_anchor', 'title', etc.
        Common placements:
        - Top: {'loc': 'upper center', 'bbox_to_anchor': (0.5, 1.05), 'ncol': 3}
        - Right: {'loc': 'center left', 'bbox_to_anchor': (1.02, 0.5), 'ncol': 1}
        Default is None (use per-axis legends).
    ak_array_dict_labels: dict=None,
        Optional dict of {"array_name": "label", ...} to use custom labels for the arrays
        in the legend. If None, the keys of ak_array_dict will be used.
    ax_rows : int, optional
        Number of rows of the subplot grid. Default is 1.
    decorate_ax_kwargs : dict, optional
        Keyword arguments passed to `decorate_ax`.
    bins_dict : dict, optional
        Dict of {name: bins} for the histograms. `name` has to be the same as the keys in `names`.
    logscale_features : list, optional
        List of features to plot in log scale, of "all" to plot all features in log scale.
    colors : list or dict, optional
        List of colors for the histograms. Has to have the same length as the number of arrays.
        If shorter, the colors will be repeated.
        If a dict is given, it has to be of the form {"array_name": "color", ...}.
    linestyles : list or dict, optional
        List of linestyles for the histograms. Has to have the same length as the
        number of arrays. If shorter, the linestyles will be repeated.
        If a dict is given, it has to be of the form {"array_name": "linestyle", ...}.
    ax_size : tuple, optional
        Size of the axes. Default is (3, 2).
    ylabel : str, optional
        Label for the y-axis. Default is None, which will not set a label.
    ratio : bool, optional
        Whether to plot the ratio of the histograms. Default is False.
    ratio_range : tuple, optional
        Range of the y-axis for the ratio plot. Default is (0.5, 1.5).
    normed : bool, optional
        Whether to normalize the histograms. Default is True.
    gridspec_hspace : float, optional
        Spacing between the rows of the subplot grid. Default is 0.3.
    gridspec_wspace : float, optional
        Spacing between the columns of the subplot grid. Default is 0.2.
    ratio_references : dict, optional
        Dict of {name: baseline_array} for the ratio plots. If given, the ratios will be
        calculated with respect to the baseline arrays. This allows to have multiple
        references in the ratio plot. Default is None.
    """

    default_hist_kwargs = {
        "density": False,
        "histtype": "step",
        "bins": 50,
        "linewidth": 1.5,
    }

    if ak_array_dict_labels is None:
        ak_array_dict_labels = {key: key for key in ak_array_dict.keys()}

    # setup colors
    if colors is not None:
        if len(colors) < len(ak_array_dict):
            print(
                "Warning: colors list is shorter than the number of arrays. "
                "Will use default colors for remaining ones."
            )
            colors = colors + [f"C{i}" for i in range(len(ak_array_dict) - len(colors))]

    if histkwargs is None:
        histkwargs = default_hist_kwargs
    else:
        if "density" in histkwargs:
            histkwargs["density"] = False
            logger.warning(
                "The 'density' keyword argument is deprecated and will be ignored. "
                "Use 'normed' instead, which is set to True by default."
            )
        histkwargs = default_hist_kwargs | histkwargs

    # create the bins dict
    if bins_dict is None:
        bins_dict = {}
    # loop over all names - if the name is not in the bins_dict, use the default bins
    for name in names:
        if name not in bins_dict:
            bins_dict[name] = histkwargs["bins"]

    # if ratio_references is given, check that all labels are in the dict
    if ratio and ratio_references is not None:
        for i, label in enumerate(ak_array_dict.keys()):
            if label not in ratio_references:
                raise ValueError(
                    f"Label '{label}' not found in ratio_references. "
                    "Please provide a baseline for each label."
                )

        # check that no reference is called before it's plotted itself
        for i, label in enumerate(ak_array_dict.keys()):
            ref_label = ratio_references[label]
            if ref_label not in list(ak_array_dict.keys())[: i + 1]:
                raise ValueError(
                    f"The array {ref_label} is used as ratio reference before "
                    "it's plotted itself. Change the order."
                )

    # remove default bins from histkwargs
    histkwargs.pop("bins")

    if isinstance(names, list):
        names = {name: name for name in names}

    if len(names) == 1 and ax_rows > 1:
        ax_rows = 1
        logger.info("Setting ax_rows to 1 since only one feature is being plotted.")

    ax_cols = (
        len(names) // ax_rows + 1 if len(names) % ax_rows > 0 else len(names) // ax_rows
    )

    # fig, axarr = plt.subplots(
    #     ax_rows, ax_cols, figsize=(ax_size[0] * ax_cols, ax_size[1] * ax_rows)
    # )
    height_factor = 1.3 if not ratio else 1.8
    fig, axarr = plot_utils.get_ax_and_fig(
        rows=ax_rows,
        cols=ax_cols,
        ratio=ratio,
        figsize=(ax_size[0] * ax_cols * 1.3, ax_size[1] * ax_rows * height_factor),
        gridspec_hspace=gridspec_hspace,
        gridspec_wspace=gridspec_wspace,
    )
    axarr_main = axarr[:, 0] if ratio else axarr
    axarr_ratio = axarr[:, 1] if ratio else None

    legend_handles = []
    legend_labels = []

    if ratio_references is not None:
        base_label = list(set(ratio_references.values())) if ratio else []
    else:
        base_label = next(iter(ak_array_dict.keys()))
        base_label = [base_label] if ratio else []
        if len(base_label) > 0:
            ratio_references = {label: base_label[0] for label in ak_array_dict.keys()}

    base_values_dict = {base_name: {} for base_name in base_label}

    # for features where no bins are specified, use the default number of bins but
    # the same range for all plotted histograms
    for feat in names:
        if feat in bins_dict:
            if isinstance(bins_dict[feat], (list, np.ndarray)):
                # if bins_dict[feat] is a list or array, use it as is
                continue
            elif isinstance(bins_dict[feat], int):
                n_bins = bins_dict[feat]
            else:
                n_bins = histkwargs["bins"]
        # go through all ak arrays and get the min/max values for the feature
        logger.info(f"Calculating binning range for feature '{feat}'")
        min_val = np.inf
        max_val = -np.inf
        for ak_array in ak_array_dict.values():
            if not hasattr(ak_array, feat):
                logger.info(
                    f"Feature '{feat}' not found in array '{ak_array}', skipping."
                )
                continue
            values = getattr(ak_array, feat)
            if isinstance(flatten, list):
                if feat in flatten:
                    values = ak.flatten(values)
            elif flatten:
                values = ak.flatten(values)

            min_val = min(min_val, np.nanmin(values.to_numpy()))
            max_val = max(max_val, np.nanmax(values.to_numpy()))
        bins_dict[feat] = np.linspace(min_val, max_val, n_bins + 1)

    for i_label, (label, ak_array) in enumerate(ak_array_dict.items()):
        if colors is not None:
            color = colors[i_label] if isinstance(colors, list) else colors[label]
        else:
            color = f"C{i_label}"

        if linestyles is not None:
            linestyle = (
                plot_utils.get_good_linestyles(linestyles[i_label])
                if isinstance(linestyles, list)
                else plot_utils.get_good_linestyles(linestyles[label])
            )
            if "linestyle" in histkwargs:
                logger.warning(
                    "The 'linestyle' keyword argument in histkwargs is being overridden "
                    "by the linestyles argument."
                )
            histkwargs["linestyle"] = linestyle
        else:
            histkwargs["linestyle"] = "solid"

        legend_labels.append(ak_array_dict_labels[label])
        for i, (feat, feat_label) in enumerate(names.items()):
            # if multiple ratio references are use: draw horizontal line
            if ratio and i_label == 0 and len(base_label) > 1:
                axarr_ratio[i].axhline(
                    1,
                    ls="--",
                    color="dimgray",
                    linewidth=histkwargs.get("linewidth", 1),
                )

            if not hasattr(ak_array, feat):
                logger.info(f"Feature '{feat}' not found in array '{label}', skipping.")
                continue

            if isinstance(flatten, list):
                if feat in flatten:
                    values = ak.flatten(getattr(ak_array, feat))
                else:
                    values = getattr(ak_array, feat)
            elif flatten:
                values = ak.flatten(getattr(ak_array, feat))
            else:
                values = getattr(ak_array, feat)

            if not isinstance(bins_dict[feat], int) and bins_dict[feat] is not None:
                values = binclip(values, bins_dict[feat])

            # calculate the histogram itself and the corresponding uncertainties
            bin_edges, hist, unc, band = hist_w_unc(
                arr=values.to_numpy(),
                bins=bins_dict[feat],
                normed=normed,
            )

            # draw the histogram on the main axes
            _, bins, patches = axarr_main[i].hist(
                # values, **histkwargs, bins=bins_dict[feat], color=color
                x=bin_edges[:-1],
                bins=bin_edges,
                weights=hist,
                color=color,
                **histkwargs,
            )

            # draw the uncertainty band
            bottom_error = np.array([band[0], *band.tolist()])
            top_error = band + 2 * unc
            top_error = np.array([top_error[0], *top_error.tolist()])

            axarr_main[i].fill_between(
                x=bin_edges,
                y1=bottom_error,
                y2=top_error,
                color=color,
                alpha=0.3,
                zorder=1,
                step="pre",
                edgecolor="none",
            )
            axarr_main[i].set_xlabel(
                feat_label if label_prefix is None else f"{label_prefix} {feat_label}"
            )
            if ylabel is not None:
                axarr_main[i].set_ylabel(ylabel)
            if i == 0:
                legend_handles.append(
                    Line2D(
                        [],
                        [],
                        color=patches[0].get_edgecolor(),
                        lw=patches[0].get_linewidth(),
                        label=ak_array_dict_labels[label],
                        linestyle=patches[0].get_linestyle(),
                    )
                )

            if ratio:
                if label in base_label:
                    base_values_dict[label][feat] = hist
                    # if multiple baselines are given, don't plot the ratio for
                    # the baselines, but a horizontal dashed line at 1
                    if len(base_label) > 1:
                        continue

                # calculate the ratio of the histogram to the base histogram
                ratio_values, ratio_unc = hist_ratio(
                    numerator=hist,
                    denominator=base_values_dict[ratio_references[label]][feat],
                    numerator_unc=unc,
                    step=False,
                )

                # draw the ratio
                axarr_ratio[i].step(
                    bins,
                    # repeat the first entry to avoid a gap in the plot (due to step plot)
                    np.append(np.array(ratio_values[0]), np.array(ratio_values)),
                    where="pre",
                    color=color,
                    linewidth=histkwargs.get("linewidth", 1),
                    linestyle=histkwargs.get("linestyle", "solid"),
                )

                # calculate the uncertainty band for the ratio
                ratio_bottom_error = ratio_values - ratio_unc
                ratio_top_error = ratio_values + ratio_unc

                # draw the uncertainty band
                axarr_ratio[i].fill_between(
                    bins,
                    # repeat the first entry to avoid a gap in the plot (due to step plot)
                    np.append(np.array(ratio_bottom_error[0]), ratio_bottom_error),
                    np.append(np.array(ratio_top_error[0]), ratio_top_error),
                    color=color,
                    alpha=0.3,
                    zorder=1,
                    step="pre",
                    edgecolor="none",
                )

                axarr_ratio[i].set_xlabel(
                    feat_label
                    if label_prefix is None
                    else f"{label_prefix} {feat_label}"
                )
                axarr_ratio[i].set_ylabel("Ratio")
                axarr_main[i].set_xlabel(None)
                axarr_ratio[i].set_ylim(*ratio_range)
                # find out where the ratio is larger than the upper limit
                ratio_over = ratio_values > ratio_range[1]
                # find out where the ratio is smaller than the lower limit
                ratio_under = ratio_values < ratio_range[0]
                # bin centers
                bin_centers = (bins[:-1] + bins[1:]) / 2
                # plot the over and under limit regions
                ratio_overunderflow_kwargs = dict(
                    clip_on=False, markersize=5, color=color
                )
                axarr_ratio[i].plot(
                    bin_centers[ratio_over],
                    ratio_range[1] * np.ones_like(bin_centers)[ratio_over],
                    "^",
                    **ratio_overunderflow_kwargs,
                )
                axarr_ratio[i].plot(
                    bin_centers[ratio_under],
                    ratio_range[0] * np.ones_like(bin_centers)[ratio_under],
                    "v",
                    **ratio_overunderflow_kwargs,
                )

    legend_kwargs["handles"] = legend_handles
    legend_kwargs["labels"] = legend_labels
    legend_kwargs["frameon"] = False

    # Create figure-level legend if fig_legend_kwargs is provided
    if fig_legend_kwargs is not None:
        # Merge handles and labels with user-provided kwargs
        fig_legend_kwargs_merged = {
            "handles": legend_handles,
            "labels": legend_labels,
            **fig_legend_kwargs,
        }
        fig.legend(**fig_legend_kwargs_merged)
    else:
        # Create per-axis legends (original behavior)
        for i, (_ax, feat_name) in enumerate(zip(axarr_main, names.keys())):
            if legend_only_on is None:
                _ax.legend(**legend_kwargs)
            else:
                if i == legend_only_on:
                    _ax.legend(**legend_kwargs)

            if (logscale_features is not None and feat_name in logscale_features) or (
                logscale_features == "all"
            ):
                _ax.set_yscale("log")
            plot_utils.decorate_ax(_ax, **decorate_ax_kwargs)

    # Apply log scale and decorations for all axes (in case fig_legend_kwargs was used)
    for i, (_ax, feat_name) in enumerate(zip(axarr_main, names.keys())):
        if (logscale_features is not None and feat_name in logscale_features) or (
            logscale_features == "all"
        ):
            _ax.set_yscale("log")
        if fig_legend_kwargs is not None:
            # Only apply decorations if we didn't already do it above
            plot_utils.decorate_ax(_ax, **decorate_ax_kwargs)

    # make empty plots invisible
    for i in range(len(names), len(axarr_main)):
        axarr_main[i].axis("off")
        if ratio:
            axarr_ratio[i].axis("off")

    # align the ratio and main plot x-ranges
    if ratio:
        fig, axarr = plot_utils.align_main_and_ratio_xrange(fig, axarr)

    # crop the borders
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.00)

    fig.tight_layout()
    return fig, axarr_main


def plot_features_pairplot(
    ak_arr_dict,
    names: dict,
    flatten: bool = True,
    pairplot_kwargs: dict = None,
    figsize: tuple = (5, 5),
    y_labels_xcoord: float = None,
    subplots_adjust_kwargs: dict = None,
    hide_upper_right: bool = True,
    colors={},
):
    """Plot the features of the constituents or jets using a pairplot.

    Parameters
    ----------
    ak_arr_dict : awkward array or numpy array
        Constituents or jets.
    names : list or dict
        List of names of the features to plot, or dict of {"name": "label", ...}.
    flatten : bool, optional
        Whether to flatten the arrays before plotting. Default is True.
    pairplot_kwargs : dict, optional
        Keyword arguments passed to sns.pairplot. Default is None, which uses
        the default settings of sns.pairplot with `kind="hist"`.
    figsize : tuple, optional
        Size of the figure (width, height), default is (5, 5).
    y_labels_xcoord : float, optional
        x-coordinate of the y-axis labels. Default is None. Usually setting this
        to -0.4 works well.
    subplots_adjust_kwargs : dict, optional
        Keyword arguments passed to fig.subplots_adjust when hue is used. Default is None.
        If None, the function will default to `dict(top=0.9)`.
    colors : dict, optional
        Dict of {group_name: color} for the histograms. Default is None. If is an empty dict,
        the function will use the default colors from the gabbro.plotting.utils module.
    hide_upper_right : bool, optional
        Whether to hide the upper right plots of the pairplot. Default is True.

    Returns
    -------
    pairplot : seaborn.axisgrid.PairGrid
        Pairplot object of the features.
    """

    # setting default values
    if pairplot_kwargs is None:
        pairplot_kwargs = {}
    if "kind" not in pairplot_kwargs:
        pairplot_kwargs["kind"] = "hist"

    # convert list of names to dict
    if isinstance(names, list):
        names = {name: name for name in names}

    # --- Array preparation ---
    is_hue = len(list(ak_arr_dict.keys())) > 1
    hue_labels = {}
    new_ak_arr_dict = {}

    color_palette = {}
    # combine the arrays if there are multiple groups, and add a dummy group
    # to the array to be able to color the groups in the pairplot
    for i, (group_name, arr) in enumerate(ak_arr_dict.items()):
        if flatten:
            arr = ak.Array(
                {
                    field_name: ak.flatten(getattr(arr, field_name))
                    for field_name in arr.fields
                }
            )
        if is_hue:
            hue_labels[str(i)] = group_name
            color_palette[str(i)] = colors.get(group_name, plot_utils.DEFAULT_COLORS[i])
            pairplot_kwargs["hue"] = "dummy_group"
            names["dummy_group"] = "dummy_group"
            # add a dummy group to the array
            tmp_len = len(arr[arr.fields[0]])
            arr = combine_ak_arrays(
                arr,
                ak.Array({"dummy_group": np.ones(tmp_len, dtype=int) * i}),
            )
        new_ak_arr_dict[group_name] = arr

    # concatenate the arrays to have one array for the pairplot (with the
    # dummy group added)
    arr = ak.concatenate(list(new_ak_arr_dict.values()), axis=0)
    # convert the awkward array to a pandas dataframe for seaborn
    df = pd.DataFrame(
        {feat_label: getattr(arr, feat) for feat, feat_label in names.items()}
    )
    if is_hue:
        # convert the dummy group to a string
        df["dummy_group"] = df["dummy_group"].astype(str)

    # --- Plotting ---

    plot_utils.set_mpl_style()
    # sns.set_style("ticks")

    g = sns.pairplot(
        df,
        palette=color_palette if is_hue else None,
        **pairplot_kwargs,
    )

    # set axes.axisbelow to False to make the grid lines visible
    plt.rcParams["axes.axisbelow"] = False

    # set the figure size
    g.figure.set_figheight(figsize[1])
    g.figure.set_figwidth(figsize[0])

    # align the y-axis labels
    if y_labels_xcoord is not None:
        for ax in g.axes.flatten():
            ax.yaxis.set_label_coords(y_labels_xcoord, 0.5)

    g.figure.tight_layout()

    if "hue" in pairplot_kwargs:
        # get the handles and labels from the legend (drawn by sns.pairplot)
        handles = g._legend_data.values()
        labels = g._legend_data.keys()
        # replace the labels with the ones from the specified hue_labels
        if len(list(hue_labels.keys())) > 0:
            labels = [hue_labels.get(str(label), label) for label in labels]
        # remove the old legend and create a new legend
        g._legend.remove()
        g.figure.legend(
            handles=handles,
            labels=labels,
            loc="upper center",
            ncol=len(list(labels)),
            frameon=False,
        )
        # adjust the subplots to make space for the legend
        if subplots_adjust_kwargs is None:
            subplots_adjust_kwargs = dict(top=0.9)
        g.figure.subplots_adjust(**subplots_adjust_kwargs)

    # TODO: is this even necessary? can't we just use the pairplot kwarg `corner=True`?
    if hide_upper_right:
        for i, j in zip(*np.triu_indices_from(g.axes, 1)):
            if g.axes[i, j] is not None:
                g.axes[i, j].set_visible(False)

    # put back the axes frame (spine)
    sns.despine(
        fig=g.figure,
        ax=None,
        top=False,
        right=False,
        left=False,
        bottom=False,
        offset=None,
        trim=False,
    )
    plt.show()

    # reset the style
    plt.rcdefaults()

    return g
