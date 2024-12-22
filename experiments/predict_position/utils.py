import random
import numpy as np 
from typing import List, Tuple
from tqdm import tqdm
import logging 

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(__name__ + '.log'),
        logging.StreamHandler()
    ],
)

logger = logging.getLogger(__name__)


class PositionDataset(Dataset):
    def __init__(self, max_length: int = 256, vocab_size: int = 256, dataset_size: int = 1000, offline=False, T=16.):
        self.vocab_size = vocab_size
        self.vocab = "".join([chr(ord("a") + x) for x in range(vocab_size)])
        self.max_length = max_length
        self.dataset_size = dataset_size
        self.data: List[Tuple[torch.Tensor, torch.Tensor]] = []
        if offline:
            self._regenerate_data()
        self.offline = offline
        self.T = T

    def _generate_random_string(self) -> str:
        return ''.join(random.choice(self.vocab) for _ in range(self.max_length))

    def _generate_ground_truth(self, sequence: torch.Tensor) -> torch.Tensor:
        return F.softmax(torch.arange(len(sequence)).to(torch.float32) / self.T, dim=0)

    def _generate_sample(self):
        sequence = self._generate_random_string()
        sequence_tensor = torch.tensor([self.vocab.index(char) for char in sequence], dtype=torch.long)
        label_tensor = self._generate_ground_truth(sequence_tensor)
        return (sequence_tensor, label_tensor)

    def _regenerate_data(self):
        self.data = []
        for _ in tqdm(range(self.dataset_size)):
            self.data.append(self._generate_sample())

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.offline:
            return self.data[index]
        else:
            return self._generate_sample()

    def __len__(self) -> int:
        return self.dataset_size