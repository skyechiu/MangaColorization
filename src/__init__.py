"""Manga colorization using Pix2Pix GAN.

This package contains model architectures and dataset utilities
for training and evaluating the manga colorization system.
"""

from .model import UNetGenerator, PatchGANDiscriminator
from .dataset import MangaDataset

__all__ = ["UNetGenerator", "PatchGANDiscriminator", "MangaDataset"]
__version__ = "1.0.0"
