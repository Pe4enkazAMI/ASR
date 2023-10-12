import logging
from typing import List
import torch
logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    batch_size = len(dataset_items)
    chanel_size= dataset_items[0]["spectrogram"].shape[0]

    # spectogram utils
    spec_freq = dataset_items[0]["spectrogram"].shape[1]
    spec_len = [item["spectrogram"].shape[2] for item in dataset_items]
    max_spec_len = max(spec_len)

    # text utils
    texts = [item["text"] for item in dataset_items]
    text_enc_len = [item["text_encoded"].shape[1] for item in dataset_items]
    max_text_enc_len = max(text_enc_len)

    audios = [item["audio"] for item in dataset_items]
    audios_paths = [item["audio_path"] for item in dataset_items]
    audio_len = [item.shape[1] for item in audios]
    max_audio_len = max([item["audio"].shape[1] for item in dataset_items]) 

    spec_batch = torch.zeros(size=(batch_size, spec_freq, max_spec_len))
    text_batch = torch.zeros(size=(batch_size, max_text_enc_len))

    audios_padded = torch.zeros(size=(batch_size, max_audio_len))  
    for el in range(batch_size):

        spec_batch[el, ..., :spec_len[el]] = dataset_items[el]["spectrogram"].squeeze(0)
        text_batch[el, ..., :text_enc_len[el]] = dataset_items[el]["text_encoded"].squeeze(0)
        audios_padded[el, :audio_len[el]] = dataset_items[el]["audio"].squeeze(0) 

    result_batch = {
        "spectrogram": spec_batch,
        "text_encoded": text_batch,
        "text_encoded_length": torch.tensor(text_enc_len),
        "spectrogram_length": torch.tensor(spec_len),
        "text": texts,
        "audio": audios_padded,
        "audio_path": audios_paths
    }
    return result_batch