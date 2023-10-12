from typing import List, NamedTuple

import torch
from collections import defaultdict
from scipy.special import softmax
from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        last_char = self.EMPTY_TOK
        text = []
        for idx in inds:
            if last_char == self.ind2char[idx]:
                continue
            else:
                text.append(self.ind2char[idx])
            last_char = self.ind2char[idx]
        return ("".join(text)).replace(self.EMPTY_TOK, "")
    
    def _extend_beam(self, beam, prob):
        if len(beam) == 0:
            for i in range(len(prob)):
                last_char = self.ind2char[i]
                beam[('', last_char)] += prob[i]
            return beam
        
        new_beam = defaultdict(float)
        
        for (text, last_char), v in beam.items():
            for i in range(len(prob)):
                if self.ind2char[i] == last_char:
                    new_beam[(text, last_char)] += v * prob[i]
                else:
                    new_last_char = self.ind2char[i]
                    new_text = (text + last_char).replace(self.EMPTY_TOK, "")
                    new_beam[(new_text, new_last_char)] += v * prob[i]
        return new_beam
    def _cut_beam(self, beam, beam_size):
        return dict(sorted(beam.items(), key=lambda x: -x[1])[:beam_size])

# спиздил с сема
    def ctc_beam_search(self, probs: torch.tensor,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        beam = defaultdict(float)
        probs = softmax(probs, axis=1)

        for prob in probs:
            beam = self._extend_beam(beam, prob)
            beam = self._cut_beam(beam, beam_size)

        final_beam = defaultdict(float)

        for (text, last_char), v in beam.items():
            final_text = (text + last_char).replace(self.EMPTY_TOK, "")
            final_beam[final_text] += v

        sorted_beam = sorted(final_beam.items(), key=lambda x: -x[1])
        result = [Hypothesis(text, v) for text, v in sorted_beam]
        return result
