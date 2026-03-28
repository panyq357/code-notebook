import torch

import config
from tokenizer import MyTokenizer
from model import MyModel

tokenizer = MyTokenizer.from_vocab(config.VOCAB_FILE)

model = MyModel(vocab_size=tokenizer.vocab_size)
model.load_state_dict(torch.load(config.MODEL_FILE))
model.eval()

if __name__ == "__main__":

    history = []
    while True:
        print("History:", [tokenizer.index2word.get(x) for x in history])
        sentence = input("Input your sentence (in Chinese): ")
        history.extend(tokenizer.encode(sentence))

        if len(history) < 5:
            # Input need to be a 2D tensor to match model input size.
            batch_with_one_row = torch.tensor([
                history + [0] * (5-len(history))
            ])
        else:
            batch_with_one_row = torch.tensor([history])

        output = model(batch_with_one_row)

        print("Predicts:", [tokenizer.index2word.get(x) for x in torch.topk(output[0, ], 5).indices.tolist()])

