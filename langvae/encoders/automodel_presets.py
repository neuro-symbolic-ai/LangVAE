from dataclasses import dataclass
from typing import Type
from enum import StrEnum, auto
from transformers import AutoModel, AutoModelForTextEncoding


class PoolingMethod(StrEnum):
    MEAN = auto()
    LAST = auto()
    CLS = auto()


@dataclass
class AutoModelPreset:
    """
    Predefined settings class for encoder models loaded with AutoModel.

    Attributes:
        cls (str): Name of the class used for loading thge model [AutoModel | AutoModelForTextEncoding].
        pooling_method (PoolingMethod): Method used for pooling the token embeddings [MEAN | LAST | CLS].
        normalize (bool): Whether the embeddings are to be normalized or not.
    """
    cls: str = "AutoModelForTextEncoding"
    pooling_method: PoolingMethod = PoolingMethod.MEAN
    normalize: bool = False

    @property
    def cls_type(self) -> Type:
        return {
            "AutoModel": AutoModel,
            "AutoModelForTextEncoding": AutoModelForTextEncoding
        }[self.cls]


AUTOMODEL_MAP = {
    "Salesforce/SFR-Embedding-2_R": {"cls": "AutoModel", "pooling_method": PoolingMethod.LAST, "normalize": True},
    "intfloat/multilingual-e5-large-instruct": {"cls": "AutoModel", "pooling_method": PoolingMethod.MEAN, "normalize": True},
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": {"cls": "AutoModel", "pooling_method": PoolingMethod.LAST, "normalize": True},
    "NovaSearch/stella_en_1.5B_v5": {"cls": "AutoModel", "pooling_method": PoolingMethod.MEAN, "normalize": True}
}