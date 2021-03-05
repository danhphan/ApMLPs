import torch

class ImdbDataset:
    def __init__(self, reviews, targets):
        self.reviews = reviews
        self.targets = targets

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        review = self.reviews[index, :]
        target = self.targets[index]
        return {"review": torch.tensor(review, dtype=torch.long),
                "target": torch.tensor(target, dtype=torch.float)
            }
    
