import torch

import config
from tokenizer import PadTokenizer
from model import ReviewAnalyzeModel

tokenizer = PadTokenizer.from_vocab(config.VOCAB_FILE)

model = ReviewAnalyzeModel(vocab_size=tokenizer.vocab_size, padding_index=tokenizer.pad_index)

model.load_state_dict(torch.load(config.MODEL_FILE))
model.eval()

if __name__ == "__main__":

    history = []
    while True:
        sentence = input("Input your sentence (in Chinese): ")

        output = model(torch.tensor([tokenizer.encode_to_seq_len(sentence, seq_len=config.SEQ_LEN)]))

        prob = torch.sigmoid(output)

        print("Predicts:", prob)

