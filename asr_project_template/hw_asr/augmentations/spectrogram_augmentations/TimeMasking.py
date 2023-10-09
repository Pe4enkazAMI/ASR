import torchaudio.transforms as tat 
from torch import Tensor
import random
from hw_asr.augmentations.base import AugmentationBase


class TimeMasking(AugmentationBase):
    def __init__(self, p, tmp=1, *args, **kwargs):
        self.p = p 
        self.tmp = tmp
        self._aug = tat.TimeMasking(*args, time_mask_param=80, p=self.tmp, **kwargs)

    def __call__(self, data: Tensor):
        if random.random() < self.p:
            x = data.unsqueeze(1)
            return self._aug(x).squeeze(1)
        else: 
            return data
