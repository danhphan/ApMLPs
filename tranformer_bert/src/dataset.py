import config
import torch

class BERTDataset:
    def __init__(self, review, target):
        """
        :param review: list or numpy array of strings
        :param targets: list or numpy array of binary
        """
        self.review = review
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.review)

    def __getitem__(self, index):
        review = str(self.review[index])
        review = " ".join(review.split())
        # encode_plus from hugginface's transformers
        inputs = self.tokenizer.encode_plus(review, None, add_special_tokens=True, 
            max_length=self.max_len, pad_to_max_length=True,)
        # Get ids of generated tokens
        ids = inputs["input_ids"]
        # mask 1 with input, 0 with padding
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[index], dtype=torch.float)
        }

    


