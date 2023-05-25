import unittest

from tests.models.beit.test_modeling_beit import *
from transformers import BeitAdapterModel
from transformers.testing_utils import require_torch

from .methods import (
   BottleneckAdapterTestMixin,
   CompacterTestMixin,
   IA3TestMixin,
   LoRATestMixin,
   PrefixTuningTestMixin,
   UniPELTTestMixin,
)
from .test_adapter import VisionAdapterTestBase, make_config
from .test_adapter_backward_compability import CompabilityTestMixin
from .test_adapter_composition import ParallelAdapterInferenceTestMixin, ParallelTrainingMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_embeddings import EmbeddingTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin
from .test_common import AdapterModelTesterMixin


@require_torch
class BeitAdapterModelTest(AdapterModelTesterMixin, BeitModelTest):
    all_model_classes = (
        BeitAdapterModel,
    )
    fx_compatible = False


class BeitAdapterTestBase(VisionAdapterTestBase):
    config_class = BeitConfig
    config = make_config(
        BeitConfig,
        image_size=224,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
    )
    feature_extractor_name = 'microsoft/beit-base-patch16-224-pt22k'


@require_torch
class BeitAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    PredictionHeadModelTestMixin,
    BeitAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class BeitClassConversionTest(
    ModelClassConversionTestMixin,
    BeitAdapterTestBase,
    unittest.TestCase,
):
    pass
