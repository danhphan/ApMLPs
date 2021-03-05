import config
import transformers
import torch.nn as nn

class BertBaseUncased(nn.Module):
    def __init__(self):
        super(BertBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        # Simple bert base
        bo = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo = bo.pooler_output
        bo = self.dropout(bo)
        output = self.out(bo)
        return output

