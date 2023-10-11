from typing import List

import torch
from torch import Tensor
import numpy as np

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_cer


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)

class BeamSearchCER(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, beam_size=100, *args, **kwargs):
        super.__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size
    def __call__(self, probs, probs_len, text, **kwargs):
        preds = probs.detach().cpu().numpy()
        lens = probs_len.detach().numpy()
        metric = []

        for probs, len_, target in zip(preds, lens, text):
            target = self.text_encoder.normalize_text(target)
            hypos = self.text_encoder.ctc_beam_search(probs[:len_], self.beam_size)
            pred_text = hypos[0].text

            metric.append(calc_cer(predicted_text=pred_text, target_text=target))
        return sum(metric) / len(metric)
