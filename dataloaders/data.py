import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

# Set the number of workers based on CPU cores
NUM_WORKERS = os.cpu_count()

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text
        tokens = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')

        # Remove batch dimension (1, max_length) -> (max_length)
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)

        return input_ids, attention_mask, label


def create_dataloaders(train_texts, train_labels, test_texts, test_labels, batch_size, tokenizer_name='bert-base-uncased', max_length=128, num_workers=NUM_WORKERS):
    """Creates DataLoaders for sentence embedding tasks.

    Args:
    train_texts: List of training sentences.
    train_labels: List of training labels.
    test_texts: List of testing sentences.
    test_labels: List of testing labels.
    batch_size: Number of samples per batch.
    tokenizer_name: Pre-trained tokenizer name (from HuggingFace's transformers).
    max_length: Maximum length for tokenized sequences.
    num_workers: Number of CPU workers for DataLoader.

    Returns:
    A tuple of (train_dataloader, test_dataloader).
    """
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    # Create dataset instances
    train_dataset = TextDataset(texts=train_texts, labels=train_labels, tokenizer=tokenizer, max_length=max_length)
    test_dataset = TextDataset(texts=test_texts, labels=test_labels, tokenizer=tokenizer, max_length=max_length)

    # Create DataLoader instances
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, test_dataloader
