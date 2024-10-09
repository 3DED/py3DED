import numpy as np
from abtem.detectors import PixelatedDetector
from abtem.core.backend import get_array_module
from abtem.waves import Waves
from scipy.signal import windows


class WindowedPixelatedDetector(PixelatedDetector):
    def __init__(
        self,
        max_angle: str | float = "valid",
        resample: bool = False,
        reciprocal_space: bool = True,
        to_cpu: bool = True,
        url: str = None,
        margin: float = 0.0,
        window_func: str = "hann",
    ):
        """
        Parameters
        ----------
        max_angle : str or float, optional
            The maximum angle value for the method. Can be either a string or a float.
            If provided as a string, the options are 'valid', 'same', or 'full'.
            Default is 'valid'.
        resample : bool, optional
            Whether to resample the data. Default is False.
        reciprocal_space : bool, optional
            Whether to use reciprocal space. Default is True.
        to_cpu : bool, optional
            Whether to run the method on CPU. Default is True.
        url : str, optional
            The URL for the method. Default is None.
        margin : float, optional
            The margin value for the method. Default is 0.0.
        window_func : str, optional
            The window function for the method. Should be one of the window functions
            available in `scipy.signal.windows`. Default is 'hann'.
        """
        self._margin = margin
        self._window_func = window_func

        super().__init__(
            max_angle=max_angle,
            resample=resample,
            reciprocal_space=reciprocal_space,
            to_cpu=to_cpu,
            url=url,
        )

    def _crop(self, waves):
        if not self._margin:
            return waves
        
        cropped = crop(
            waves,
            offset=(self._margin, self._margin),
            extent=(waves.extent[0] - 2 * self._margin, waves.extent[1] - 2 * self._margin),
        )
        return cropped

    def angular_limits(self, waves) -> tuple[float, float]:
        return super().angular_limits(self._crop(waves))

    def _out_base_axes_metadata(self, waves, index=0):
        return super()._out_base_axes_metadata(self._crop(waves))

    def _out_base_shape(self, waves, index: int = 0) -> tuple[int, int]:
        return super()._out_base_shape(self._crop(waves))

    def _calculate_new_array(self, waves):
        cropped = self._crop(waves)

        window_x = windows.get_window(self._window_func, cropped.shape[-2])
        window_y = windows.get_window(self._window_func, cropped.shape[-1])

        window = window_x[:, None] * window_y[None, :]

        xp = get_array_module(cropped.array)

        window = xp.asarray(window)

        cropped = cropped.copy()
        cropped._array = cropped._array * window

        return super()._calculate_new_array(cropped)


def crop(
    waves: Waves, extent: tuple[float, float], offset: tuple[float, float] = (0.0, 0.0)
):
    """
    Crop images to a smaller extent.

    Parameters
    ----------
    extent : tuple of float
        Extent of rectangular cropping region in `x` and `y` [Å].
    offset : tuple of float
        Lower corner of cropping region in `x` and `y` [Å] (default is (0,0)).

    Returns
    -------
    cropped_images : Images
        The cropped images.
    """

    offset = (
        int(np.round(waves.base_shape[0] * offset[0] / waves.extent[0])),
        int(np.round(waves.base_shape[1] * offset[1] / waves.extent[1])),
    )
    new_shape = (
        int(np.round(waves.base_shape[0] * extent[0] / waves.extent[0])),
        int(np.round(waves.base_shape[1] * extent[1] / waves.extent[1])),
    )

    array = waves.array[
        ...,
        offset[0] : offset[0] + new_shape[0],
        offset[1] : offset[1] + new_shape[1],
    ]

    kwargs = waves._copy_kwargs(exclude=("array",))
    kwargs["array"] = array
    return waves.__class__(**kwargs)
