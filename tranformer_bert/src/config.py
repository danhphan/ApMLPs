import transformers

MAX_LEN = 512

BATCH_SIZE = 4

EPOCHS = 3

BERT_PATH = "bert-base-uncased"

MODEL_PATH = "./model.bin"

TRAINING_FILE = "../data/imdb.csv"

TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
