"""Config for Birdy project."""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml
from dataclass_wizard import fromdict

from birdy.utils import parse_path


@dataclass
class SpectrogramConfig:
    """Config for spectrogram transformation."""

    mel: bool = True
    n_fft: int = 400
    win_length: int | None = None
    hop_length: int | None = None
    pad: int = 0
    power: float = 2.0
    center: bool = True
    pad_mode: str = "reflect"
    onesided: bool = True


@dataclass
class DataConfig:
    """Config for training."""

    birdclef2026_path: Path
    spectrogram: SpectrogramConfig


@dataclass
class TrainConfig:
    """Config for training."""

    batch_size: int = 32


@dataclass
class Config:
    """Config for Birdy project."""

    data: DataConfig
    train: TrainConfig


class ConfigStrFormat(Enum):
    """Enum of config file format."""

    YAML = 0
    JSON = 1
    OTHER = 99


def load_config_str(
    config_str: str,
    config_format: ConfigStrFormat = ConfigStrFormat.JSON,
) -> Config:
    """Load config from a config string.

    Args:
        config_str: config string
        config_format: format of config file

    Returns:
        a Config object containing configuration for the Birdy project.
    """
    if config_format == ConfigStrFormat.JSON:
        config_data = json.loads(config_str)
    elif config_format == ConfigStrFormat.YAML:
        config_data = yaml.safe_load(config_str)
    else:
        msg = f'config_format="{config_format.name}" is invalid'
        raise ValueError(msg)

    return fromdict(Config, config_data)


def load_config(
    config_path: Path | str,
    config_format: ConfigStrFormat = ConfigStrFormat.JSON,
) -> Config | None:
    """Load config from a config file.

    Args:
        config_path: path to the config file
        config_format: format of config file

    Returns:
        a Config object containing configuration for the Birdy project.
        If config_path does not exists, return None.
    """
    config_path = parse_path(config_path)
    if not config_path.exists():
        return None

    return load_config_str(config_path.read_text(), config_format)
