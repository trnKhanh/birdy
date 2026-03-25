"""Data classes for the BirdCLEF 2026 competition."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, override

import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch import nn
from torch.utils.data import Dataset

from birdy.config import SpectrogramConfig

logger = logging.getLogger(__name__)


class BirdCLEFMetadata:
    """BirdCLEF2026 metadata loader.

    Data is available here: [Kaggle](https://www.kaggle.com/competitions/birdclef-2026/data).
    CSV files are read on first access and cached for subsequent use.
    """

    def __init__(self, data_path: Path) -> None:
        """Initialize metadata loader.

        Args:
            data_path: Path to the competition data directory following the format from
        """
        self._data_path: Path = data_path

        self._taxonomy_path: Path = data_path / "taxonomy.csv"
        self._sample_submission_path: Path = data_path / "sample_submission.csv"
        self._train_soundscapes_labels_path: Path = data_path / "train_soundscapes_labels.csv"
        self._train_path: Path = data_path / "train.csv"

        self._taxonomy: pd.DataFrame | None = None
        self._sample_submission: pd.DataFrame | None = None
        self._train_soundscapes_labels: pd.DataFrame | None = None
        self._train: pd.DataFrame | None = None
        self._train_taxonomy: pd.DataFrame | None = None

    @property
    def taxonomy(self) -> pd.DataFrame:
        """Taxonomy of targeted birds."""
        if self._taxonomy is None:
            self._taxonomy = pd.read_csv(self._taxonomy_path)

        return self._taxonomy

    @property
    def sample_submission(self) -> pd.DataFrame:
        """Sample submission data."""
        if self._sample_submission is None:
            self._sample_submission = pd.read_csv(self._sample_submission_path)

        return self._sample_submission

    @property
    def train_soundscapes_labels(self) -> pd.DataFrame:
        """Labels for train soundscapes."""
        if self._train_soundscapes_labels is None:
            self._train_soundscapes_labels = pd.read_csv(
                self._train_soundscapes_labels_path,
            )

        return self._train_soundscapes_labels

    @property
    def train(self) -> pd.DataFrame:
        """Training data."""
        if self._train is None:
            self._train = pd.read_csv(self._train_path)

        return self._train

    @property
    def train_taxonomy(self) -> pd.DataFrame:
        """Training data with taxonomy."""
        if self._train_taxonomy is None:
            self._train_taxonomy = self.train.merge(self.taxonomy, on="primary_label")

        return self._train_taxonomy


class BirdCLEFItem(TypedDict):
    """A single train birdclef sample."""

    wave: torch.Tensor
    spectrogram: torch.Tensor
    sample_rate: int


class BirdCLEFDataset(Dataset[BirdCLEFItem]):
    """BirdCLEF2026 dataset following torch interface.

    Data is available here: [Kaggle](https://www.kaggle.com/competitions/birdclef-2026/data).
    """

    TRAIN_DIR: str = "train_audio"

    def __init__(
        self,
        data_path: Path,
        spectrogram_config: SpectrogramConfig,
        transform: Callable[[BirdCLEFItem], BirdCLEFItem] | None = None,
    ) -> None:
        """Initialize BirdCLEF data.

        Args:
            data_path: Path to the competition data directory following the format from
            spectrogram_config: config for spectrogram transformation.
            transform: transformation to apply to the data after return (Optional)
        """
        self._data_path: Path = data_path
        self._metadata: BirdCLEFMetadata = BirdCLEFMetadata(self._data_path)
        self._spec_transform: nn.Module = nn.Module()
        self._transform: Callable[[BirdCLEFItem], BirdCLEFItem] | None = transform

        if spectrogram_config.mel:
            self._spec_transform = T.MelSpectrogram(
                n_fft=spectrogram_config.n_fft,
                win_length=spectrogram_config.win_length,
                hop_length=spectrogram_config.hop_length,
                pad=spectrogram_config.pad,
                power=spectrogram_config.power,
                center=spectrogram_config.center,
                pad_mode=spectrogram_config.pad_mode,
                onesided=spectrogram_config.onesided,
            )
        else:
            self._spec_transform = T.Spectrogram(
                n_fft=spectrogram_config.n_fft,
                win_length=spectrogram_config.win_length,
                hop_length=spectrogram_config.hop_length,
                pad=spectrogram_config.pad,
                power=spectrogram_config.power,
                center=spectrogram_config.center,
                pad_mode=spectrogram_config.pad_mode,
                onesided=spectrogram_config.onesided,
            )

    @override
    def __getitem__(self, index: int) -> BirdCLEFItem:
        sample_metadata = self._metadata.train.iloc[index]

        file_path = self._data_path / BirdCLEFDataset.TRAIN_DIR / sample_metadata.filename

        wave, sr = torchaudio.load(file_path)
        spec: torch.Tensor = self._spec_transform(wave)

        item = BirdCLEFItem(wave=wave, spectrogram=spec, sample_rate=sr)

        if self._transform is not None:
            item = self._transform(item)

        return item

    def __len__(self) -> int:
        """Length of the dataset."""
        return self._metadata.train.shape[0]
