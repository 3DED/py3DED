import argparse
import dataclasses
from typing import Any, Dict


@dataclasses.dataclass
class BaseConfig:
    _original_values: Dict[str, Any] = dataclasses.field(
        init=False, repr=False, default_factory=dict
    )

    def __enter__(self, **kwargs):
        # Store original values and update attributes with new values
        for key, value in kwargs.items():
            if hasattr(self, key):
                self._original_values[key] = getattr(self, key)
                setattr(self, key, value)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Revert attributes back to their original values
        for key, value in self._original_values.items():
            setattr(self, key, value)
        self._original_values.clear()

    def ctx(self, **kwargs):
        return self.__enter__(**kwargs)


def parse_arguments(config: type(BaseConfig)):
    """
    Parse the command line arguments and return an instance of ConfigSingleAxis.

    Returns
    -------
    config : BaseConfig
        An instance of BaseConfig.
    """

    parser = argparse.ArgumentParser()
    for field in dataclasses.fields(config):
        if getattr(field.type, "__args__", None):
            first_type = field.type.__args__[0]
        else:
            first_type = field.type

        parser.add_argument(f"--{field.name}", default=field.default, type=first_type)

    args = vars(parser.parse_args())

    new_config = config(**args)
    return new_config
