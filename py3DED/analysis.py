import dataclasses
import inspect

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from py3ded.config import BaseConfig, parse_arguments


@dataclasses.dataclass
class AnalysisConfig(BaseConfig):
    g_max: float = 4.0
    include_000: bool = False
    averaging_depth: float = 10.0
    tol: float = 1e-5


def abtem_data_to_xarray(data_bw, data_ms):
    kx = data_bw.positions[:, 0, :, 0]  # removed in future
    ky = data_bw.positions[:, 0, :, 1]  # removed in future
    kz = data_bw.positions[:, 0, :, 2]  # removed in future
    data_bw = data_bw.to_data_array()
    data_bw = data_bw.assign_coords(kx=(("x_rotation", "hkl"), kx))  # removed in future
    data_bw = data_bw.assign_coords(ky=(("x_rotation", "hkl"), ky))  # removed in future
    data_bw = data_bw.assign_coords(kz=(("x_rotation", "hkl"), kz))  # removed in future

    kx = data_ms.positions[:, 0, :, 0]  # removed in future
    ky = data_ms.positions[:, 0, :, 1]  # removed in future
    kz = data_ms.positions[:, 0, :, 2]  # removed in future
    data_ms = data_ms.to_data_array()
    data_ms = data_ms.assign_coords(kx=(("x_rotation", "hkl"), kx))  # removed in future
    data_ms = data_ms.assign_coords(ky=(("x_rotation", "hkl"), ky))  # removed in future
    data_ms = data_ms.assign_coords(kz=(("x_rotation", "hkl"), kz))  # removed in future

    data = xr.concat(
        (data_bw, data_ms),
        dim=xr.DataArray(["BW", "MS"], dims="algorithm"),
        fill_value=0.0,
    )

    # need to assign k, kx, ky to missing data
    data = data.assign_coords(k=("hkl", data_bw.k.combine_first(data_ms.k).data))
    data = data.assign_coords(
        kx=(("x_rotation", "hkl"), data_bw.kx.combine_first(data_ms.kx).data)
    )
    data = data.assign_coords(
        ky=(("x_rotation", "hkl"), data_bw.ky.combine_first(data_ms.ky).data)
    )
    data = data.assign_coords(
        kz=(("x_rotation", "hkl"), data_bw.kz.combine_first(data_ms.kz).data)
    )
    return data


def filter_data(data, config):
    if config.g_max is not None:
        data = data.sel(hkl=(data.k < config.g_max))

    if not config.include_000:
        data = data.sel(hkl=data.hkl != "0 0 0")

    if config.averaging_depth:
        dz = (data.z[1] - data.z[0]).data
        averaging_window = int(config.averaging_depth / dz)

        data = data.rolling(z=averaging_window).mean()
        data = data.isel(z=slice(averaging_window - 1, None))

    if config.tol is not None:
        data = data.compute()
        data = data.sel(
            hkl=(data.max("z") > config.tol).any(("algorithm", "x_rotation"))
        )

    return data


def move_legend(obj, loc, **kwargs):
    """
    Recreate a plot's legend at a new location.
    Extracted from seaborn/utils.py
    """
    if isinstance(obj, mpl.axes.Axes):
        old_legend = obj.legend_
        legend_func = obj.legend
    elif isinstance(obj, mpl.figure.Figure):
        if obj.legends:
            old_legend = obj.legends[-1]
        else:
            old_legend = None
        legend_func = obj.legend
    else:
        err = "`obj` must be a matplotlib Axes or Figure instance."
        raise TypeError(err)

    if old_legend is None:
        err = f"{obj} has no legend attached."
        raise ValueError(err)

    # Extract the components of the legend we need to reuse
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]

    # Extract legend properties that can be passed to the recreation method
    # (Vexingly, these don't all round-trip)
    legend_kws = inspect.signature(mpl.legend.Legend).parameters
    props = {k: v for k, v in old_legend.properties().items() if k in legend_kws}

    # Delegate default bbox_to_anchor rules to matplotlib
    props.pop("bbox_to_anchor")

    # Try to propagate the existing title and font properties; respect new ones too
    title = props.pop("title")
    if "title" in kwargs:
        title.set_text(kwargs.pop("title"))
    title_kwargs = {k: v for k, v in kwargs.items() if k.startswith("title_")}
    for key, val in title_kwargs.items():
        title.set(**{key[6:]: val})
        kwargs.pop(key)

    # Try to respect the frame visibility
    kwargs.setdefault("frameon", old_legend.legendPatch.get_visible())

    # Remove the old legend and create the new one
    props.update(kwargs)
    old_legend.remove()
    new_legend = legend_func(handles, labels, loc=loc, **props)
    new_legend.set_title(title.get_text(), title.get_fontproperties())


def calculate_r_factor(a, b):
    r_factor = np.abs(np.sqrt(a) - np.sqrt(b)).sum("hkl") / np.sqrt(a).sum("hkl") * 100
    r_factor.attrs["units"] = "%"
    r_factor.attrs["long_name"] = "R"
    return r_factor


def calculate_r_factor_diffs(a, b):
    denom = np.sqrt(a).sum("hkl") - np.sqrt(a)
    num = np.abs(np.sqrt(a) - np.sqrt(b)).sum("hkl") - np.abs(np.sqrt(a) - np.sqrt(b))
    r_missing = num / denom * 100
    r_full = calculate_r_factor(a, b)
    r_diffs = r_full - r_missing
    return r_diffs


def get_hkl_integrated_intensity_order(data, z):
    integrated = data.max("algorithm")[:, :].sum("x_rotation")
    order = np.argsort(-integrated.sel(z=z, method="nearest"))
    hkl_order = integrated.hkl[order.data]
    return hkl_order.data


def get_hkl_max_intensity_order(data, x_rotation):
    data = data.sel(x_rotation=x_rotation, method="nearest").max(("z", "algorithm"))
    order = np.argsort(-data)
    hkl_order = data.hkl[order.data]
    return hkl_order.data


def plot_diffraction_pattern(
    data,
    scale_factor,
    power=1,
    threshold_annotation=0.0,
    ax=None,
    text_kwargs=None,
):
    if ax is None:
        ax = plt.subplot()
    ax.scatter(data.kx, data.ky, s=data.data**power * scale_factor, c="k")
    ax.set_aspect(aspect=1, adjustable="box")

    if text_kwargs is None:
        text_kwargs = {}

    for kxi, kyi, hkli, datai in zip(data.kx, data.ky, data.hkl, data.data):
        if datai > threshold_annotation:
            ax.text(
                x=kxi.data,
                y=kyi.data,
                s=hkli.data,
                ha="center",
                clip_on=True,
                **text_kwargs,
            )


def plot_intensity_vs_rotation(data, hkl, ncols, figsize, z):
    nplots = len(hkl)
    nrows = -(-nplots // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)

    for i, (ax, hkl) in enumerate(zip(axes.ravel(), hkl)):
        data.sel(hkl=hkl).sel(z=z, method="nearest").plot(
            hue="algorithm", ax=ax, add_legend=i == 0
        )
        ax.set_xlabel(None)
        ax.set_ylabel(None)

    fig.supylabel("Intensity [arb. unit]")
    fig.supxlabel("Rotation [rad]")
    plt.tight_layout()


def plot_intensity_vs_depth(data, hkl, ncols, figsize, x_rotation):
    nplots = len(hkl)
    nrows = -(-nplots // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)

    for i, (ax, hkl) in enumerate(zip(axes.ravel(), hkl)):
        data.sel(hkl=hkl).sel(x_rotation=x_rotation, method="nearest").plot(
            hue="algorithm", ax=ax, add_legend=i == 0
        )
        ax.set_xlabel(None)
        ax.set_ylabel(None)

    fig.supylabel("Intensity [arb. unit]")
    fig.supxlabel("Rotation [rad]")
    plt.tight_layout()
