import warnings
from enum import Enum
from typing import Union, Optional, Tuple, List
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib import cm, colors


class ImageVisualizationMethod(Enum):
    heatmap = 1
    blended_heat_map = 2
    original_image = 3
    masked_image = 4
    alpha_scaling = 5


class VisualizeSign(Enum):
    positive = 1
    absolute_value = 2
    negative = 3
    all = 4

class MergeChannels(Enum):
    none = 1
    average = 2
    v5_only = 3


def _normalize_scale(attr: ndarray, scale_factor: float, clip: bool = True):
    #assert scale_factor != 0, "Cannot normalize by scale factor = 0"
    if abs(scale_factor) < 1e-5:
        warnings.warn(
            "Attempting to normalize by value approximately 0, visualized results"
            "may be misleading. This likely means that attribution values are all"
            "close to 0."
        )
    
    if scale_factor != 0:
        attr_norm = attr / scale_factor
    else:
        attr_norm = attr
        
    if clip:
        attr_norm = np.clip(attr_norm, -1, 1)

    return attr_norm


def _cumulative_sum_threshold(values: ndarray, percentile: Union[int, float]):
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]


def _normalize_attr(
    attr: ndarray,
    sign: str,
    outlier_perc: Union[int, float] = 2,
    clip_values: bool = True
):

    assert len(attr.shape) == 2, f'attr should be (C, N), got shape {attr.shape}'

    _attr = attr
    #if merge_channels:
    #    _attr = np.sum(_attr, axis=0)
    
    # Choose appropriate signed values and rescale, removing given outlier percentage.
    if VisualizeSign[sign] == VisualizeSign.all:
        threshold = _cumulative_sum_threshold(np.abs(_attr), 100 - outlier_perc)

    elif VisualizeSign[sign] == VisualizeSign.positive:
        _attr = (_attr > 0) * _attr
        threshold = _cumulative_sum_threshold(_attr, 100 - outlier_perc)

    elif VisualizeSign[sign] == VisualizeSign.negative:
        _attr = (_attr < 0) * _attr
        threshold = -1 * _cumulative_sum_threshold(
            np.abs(_attr), 100 - outlier_perc
        )

    elif VisualizeSign[sign] == VisualizeSign.absolute_value:
        _attr = np.abs(_attr)
        threshold = _cumulative_sum_threshold(_attr, 100 - outlier_perc)

    else:
        raise AssertionError("Visualize Sign type is not valid.")

    return _normalize_scale(_attr, threshold, clip=clip_values)


def visualise_attr_1d(
    attr: ndarray,
    data: ndarray,
    merge_channels: str = 'none',
    method: str = 'heatmap',
    sign: str = 'absolute_value',
    normalise: bool = True,
    outlier_perc: Union[int, float] = 10,
    cmap_name: Optional[str] = None,
    alpha_overlay: float = 0.7,
    show_colorbar: bool = False,
    title: str = '',
    figsize: Optional[Tuple[int]] = None,
    savefile: Optional[Union[str, List]] = None,
    show: bool = True
):  

    assert len(data.shape) == 2
    assert len(attr.shape) == 2

    lead_v5_index = 10
    
    # Merge or select channels
    if MergeChannels[merge_channels] == MergeChannels.average:

        # Override sign if averaging attribution values, so they don't cancel out
        if VisualizeSign[sign] == VisualizeSign.all:
            print('Warning: Can\'t average with sign=all, setting to absolute_value')
            sign = 'absolute_value'
            _attr = np.abs(attr)

        _attr = np.mean(attr, axis=0, keepdims=True)
        _data = data[np.newaxis, lead_v5_index, :]
        default_figsize = (8, 2)

    elif MergeChannels[merge_channels] == MergeChannels.v5_only:
        _attr = attr[np.newaxis, lead_v5_index, :]
        _data = data[np.newaxis, lead_v5_index, :]
        default_figsize = (8, 2)

    else:
        _attr = attr
        _data = data 
        default_figsize = (8, 6)

    if figsize is None:
        figsize = default_figsize

    # Normalise
    if normalise:
        _attr = _normalize_attr(_attr, sign, outlier_perc)


    # Set default colormap and bounds based on sign
    if VisualizeSign[sign] == VisualizeSign.all:
        default_cmap_name = 'RdBu'
        #default_cmap_name = 'PiYG'
        vmin, vmax = -1, 1
    elif VisualizeSign[sign] == VisualizeSign.positive:
        default_cmap_name = "Greens"
        vmin, vmax = 0, 1
    elif VisualizeSign[sign] == VisualizeSign.negative:
        default_cmap_name = "Reds"
        vmin, vmax = 0, 1
    elif VisualizeSign[sign] == VisualizeSign.absolute_value:
        default_cmap_name = "Blues"
        vmin, vmax = 0, 1
    else:
        raise AssertionError("Visualize Sign type is not valid.")

    cmap_name = cmap_name if cmap_name is not None else default_cmap_name
    cmap = cm.get_cmap(cmap_name)
    cm_norm = colors.Normalize(vmin, vmax)

    if ImageVisualizationMethod[method] != ImageVisualizationMethod['heatmap']:
        raise NotImplementedError

    # Show visualisation
    num_channels = _attr.shape[0]
    plt_fig, plt_axis = plt.subplots(nrows=num_channels, ncols=1, sharex=True, figsize=figsize)
    if not isinstance(plt_axis, np.ndarray):
        plt_axis = np.array([plt_axis])

    xvals = np.arange(_data.shape[1])

    lead_labels = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    for i in range(num_channels):

        # Plot data
        plt_axis[i].plot(xvals, _data[i, :], color='black')
        
        if num_channels == 1:
            plt_axis[i].set_ylabel('V5')
        else:
            plt_axis[i].set_ylabel(lead_labels[i])
        
        plt_axis[i].set_yticks([])

        # Plot attributions
        half_col_width = (xvals[1] - xvals[0]) / 2.0
        for icol, col_center in enumerate(xvals):
            left = col_center - half_col_width
            right = col_center + half_col_width
            plt_axis[i].axvspan(
                xmin=left,
                xmax=right,
                facecolor=(
                    cmap(
                        cm_norm(_attr[i, icol])
                    )
                ),
                edgecolor=None,
                alpha=alpha_overlay,
            )

    if show_colorbar:

        plt_fig.colorbar(
            cm.ScalarMappable(norm=None, cmap=cmap),
            ax=plt_axis[:],
            orientation='vertical',
            #location='right',
            fraction=0.05
        )
        plt.subplots_adjust(
            left=0.06 * figsize[0],
            right=0.80 * figsize[0],
            top=0.93 * figsize[1],
            bottom=0.07 * figsize[1],
            hspace=0
        )

    else:
        plt.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.12, hspace=0)
    
    plt.xlim([xvals[0], xvals[-1]])
    if title != '':
        plt.suptitle(title)

    if savefile is not None:
        if isinstance(savefile, list):
            for sf in savefile:
                plt.savefig(sf)
        else:
            plt.savefig(savefile)

    if show:
        plt.show()
    else:
        plt.close()


 