import random
import numpy as np 
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm
import logging 

import torch
from torch.utils.data import Dataset


logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(__name__ + '.log'),
        logging.StreamHandler()
    ],
)

logger = logging.getLogger(__name__)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class InductionDataset(Dataset):
    def __init__(self, max_length: int = 256, vocab_size: int = 16, dataset_size: int = 1000, offline=False, bos=False):
        self.vocab_size = vocab_size
        self.vocab = "".join([chr(ord("a") + x) for x in range(vocab_size)])
        self.max_length = max_length
        self.dataset_size = dataset_size
        self.data: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.bos = bos
        if self.bos:
            self.vocab_size += 1
        if offline:
            self._regenerate_data()
        self.offline = offline

    def _generate_random_string(self) -> str:
        return ''.join(random.choice(self.vocab) for _ in range(self.max_length))

    def _generate_ground_truth(self, sequence: torch.Tensor) -> torch.Tensor:
        last_seen = {}
        result = []
        
        for i, char in enumerate(sequence):
            char = int(char)
            prev_index = last_seen.get(char, -1)
            if prev_index > -1:
                result.append(sequence[prev_index + 1])
            else:
                result.append(-1)
            last_seen[char] = i
        
        return torch.tensor(result, dtype=torch.long)

    def _generate_sample(self):
        sequence = self._generate_random_string()
        if self.bos:
            sequence_tensor = torch.tensor([self.vocab_size - 1] + [self.vocab.index(char) for char in sequence], dtype=torch.long)
            label_tensor = self._generate_ground_truth(sequence_tensor)
        else:
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
    

def train(
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
    accuracies = []
    best_eval_acc = 0
    for epoch in range(epochs):
        train_loss = []
        model.train()
        for seqs, labels in tqdm(train_loader):
            seqs = seqs.to(device)
            labels = labels.to(device)

            preds = model(seqs)
            optimizer.zero_grad()
            loss = loss_fn(preds.view(-1, preds.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            if not scheduler is None:
                scheduler.step()
            train_loss.append(loss.item())

        eval_loss = []
        accuracy = 0 
        num_obs = 0
        model.eval()
        with torch.no_grad():
            for seqs, labels in tqdm(test_loader):
                seqs = seqs.to(device)
                labels = labels.to(device)

                preds = model(seqs)
                loss = loss_fn(preds.view(-1, preds.size(-1)), labels.view(-1))
                eval_loss.append(loss.item())
                
                preds = preds.max(2)[1]
                correct = 1.0 * (preds == labels)
                mask = (labels != -1)
                accuracy += correct[mask].sum().item()
                num_obs += mask.sum().item()

        accuracy = accuracy / num_obs
        logger.info(f'Loss_epoch {epoch + 1}: {np.mean(train_loss)}')
        logger.info(f'Loss_eval {epoch + 1}: {np.mean(eval_loss)}')
        logger.info(f'Accuracy {epoch + 1}: {accuracy}')
        train_losses.append(np.mean(train_loss))
        eval_losses.append(np.mean(eval_loss))
        accuracies.append(accuracy)

        if accuracy > best_eval_acc:
            best_eval_acc = accuracy
            torch.save(model.state_dict(), model_path)
        
    return train_losses, eval_losses, accuracies


def eval_accuracy(model, test_loader, device, min_len=16, max_len=1024, ignore_index=-1):
    accuracy = torch.zeros((max_len - min_len)).to(device)
    num_obs = torch.zeros((max_len - min_len)).to(device)
    model.eval()
    with torch.no_grad():
        for seqs, labels in tqdm(test_loader):
            seqs = seqs.to(device)
            labels = labels.to(device)

            preds = model(seqs).max(2)[1]
            correct = 1.0 * (preds == labels)[:, min_len:max_len]
            mask = (labels != ignore_index)[:, min_len:max_len]
            correct = torch.where(mask, correct, torch.zeros_like(correct))
            num_obs += mask.sum(0)
            accuracy += correct.sum(0)
    return accuracy / num_obs