import transformer

MAX_LEN = 512

BATCH_SIZE = 32

EPOCHS = 10

BERT_PATH = "../data/bert_base_uncased/"

MODEL_PATH = "model.bin"

TRAINING_FILE = "../data/imdb.csv"

TOKENIZER = transformer.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
