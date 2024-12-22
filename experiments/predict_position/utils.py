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
    

def train_transformer(
    model, 
    optimizer, 
    loss_fn, 
    train_loader, 
    test_loader, 
    device, 
    scheduler=None, 
    epochs=25,
    model_path = 'model.pt'
):
    train_losses, eval_losses = [], []
    best_eval_loss = np.inf
    for epoch in range(epochs):
        train_loss = []
        model.train()
        for seqs, labels in tqdm(train_loader):
            seqs = seqs.to(device)
            labels = labels.to(device)

            preds = F.log_softmax(model(seqs).squeeze(-1), 1)
            optimizer.zero_grad()
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            if not scheduler is None:
                scheduler.step()
            train_loss.append(loss.item())

        eval_loss = []
        model.eval()
        with torch.no_grad():
            for seqs, labels in tqdm(test_loader):
                seqs = seqs.to(device)
                labels = labels.to(device)

                preds = F.log_softmax(model(seqs).squeeze(-1), 1)
                loss = loss_fn(preds, labels)
                eval_loss.append(loss.item())

        logger.info(f'Loss_epoch {epoch + 1}: {np.mean(train_loss)}')
        logger.info(f'Loss_eval {epoch + 1}: {np.mean(eval_loss)}')
        train_losses.append(np.mean(train_loss))
        eval_losses.append(np.mean(eval_loss))

        if np.mean(eval_loss) < best_eval_loss:
            best_eval_loss = np.mean(eval_loss)
            torch.save(model.state_dict(), model_path)
        
    return train_losses, eval_losses


def kendall(x):
    """
    Computes Kendall Correlation Coefficient with ideal order
    """
    n = x.shape[0]
    ref = torch.ones(n, n).tril(diagonal=0) - torch.ones(n, n).tril(diagonal=0).permute(1, 0)
    return x.expand(n, n).T.sub(x).sign_().mul_(ref).sum().div(n * (n-1))
