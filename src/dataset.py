import torch
from torch.utils.data import Dataset, DataLoader

class ABSADataset(Dataset):
    def __init__(self, encodings, sentiment_labels, aspect_labels):
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        self.sentiment_labels = torch.tensor(sentiment_labels, dtype=torch.long)
        self.aspect_labels = torch.tensor(aspect_labels, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'sentiment_labels': self.sentiment_labels[idx],
            'aspect_labels': self.aspect_labels[idx],
        }

def get_dataloader(encodings, sentiment_labels, aspect_labels, batch_size, shuffle=False):
    dataset = ABSADataset(encodings, sentiment_labels, aspect_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)