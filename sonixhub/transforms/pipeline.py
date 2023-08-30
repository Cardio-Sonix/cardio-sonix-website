"""
Contains a ``ETLPipeline`` class which is a pipeline
for data preprocessing, feature extraction and augmentation.
"""

__all__ = ["ETLPipeline"]

from typing import Literal, Any, Optional
import torch
from torch.nn import Module

from .extractors import MFCCExtractor
from .preprocessing import Upload


class ETLPipeline(Module):
    """
    ETLPipeline is a high-level API that implements methods for
    data preprocessing, feature extraction and augmentation.
    Thus, it allows you to transform the data before issuing it.
    This class must be integrated into any ``Dataset`` subclass
    to process an audio sample before it is emitted, along with a class label.

    Args:
        etl_pipeline_params: (ETLPipelineParams) subclass of ``BaseModel``
            containing parameters (configuration) for ``ETLPipeline`` initialization.

        stage: (Literal["train", "val", "test"]) depending on the stage,
            the transformations that will take place with the data will be determined.
            The stage argument passed will determine the transformations that will be applied to the data.
            For example, with stage ``train``, we want to apply data augmentation,
            but during ``validation``, we want to check the accuracy of the model
            on a clean data without additional augmentation.
            Stages: ``training``, ``validation``, ``testing``.
    """
    def __init__(self,
                 extractor: Literal["MFCC"],
                 extractor_kwargs: dict,
                 sample_rate: Optional[int] = 22050,
                 duration: Optional[int] = 10,
                 mono: Optional[bool] = True
                 ):
        super().__init__()
        self.extractors = {"MFCC": MFCCExtractor}
        self.upload = Upload(sample_rate, duration, mono)
        self.extractor = self.extractors[extractor](
            sample_rate=sample_rate, **extractor_kwargs
        )

    def forward(self, filepath: str) -> torch.Tensor:
        """
        Performs loading of an audio sample, data preprocessing, augmentation and feature extraction step by step.
        Finally returns a set of features cast to a tensor type

        Args:
            filepath: (str) path to the audio sample file
        """

        waveform = self.upload(filepath)
        features = self.extractor(waveform)
        return features.unsqueeze(0)
