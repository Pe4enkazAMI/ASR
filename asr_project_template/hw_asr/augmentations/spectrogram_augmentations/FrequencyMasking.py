import torchaudio.transforms as tat 
from torch import Tensor
import random
from hw_asr.augmentations.base import AugmentationBase


class FrequencyMasking(AugmentationBase):
    def __init__(self, p, tmp=1, *args, **kwargs):
        self.p = p 
        self._aug = tat.FrequencyMasking(*args, p=self.tmp, **kwargs)

    def __call__(self, data: Tensor):
        if random.random() < self.p:
            x = data.unsqueeze(1)
            return self._aug(x).squeeze(1)
        else: 
            return data
